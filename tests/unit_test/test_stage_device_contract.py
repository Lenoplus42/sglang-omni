# SPDX-License-Identifier: Apache-2.0
"""Every GPU-placed stage factory must take device/gpu_id and honor gpu_id."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

import sglang_omni.platforms as platforms
from sglang_omni.utils.imports import import_string

_MODELS_DIR = Path(importlib.import_module("sglang_omni.models").__file__).parent

# note (lennox): zonos2's preprocessing is CPU-only but declares gpu=0 to share
# the pipeline process with tts_engine.
_CPU_ONLY_GPU_PLACED = {("zonos2", "preprocessing")}


def _iter_stages():
    for config_path in sorted(_MODELS_DIR.glob("*/config.py")):
        model = config_path.parent.name
        module = importlib.import_module(f"sglang_omni.models.{model}.config")
        topologies = {}
        if getattr(module, "EntryClass", None) is not None:
            topologies["default"] = module.EntryClass
        topologies.update(getattr(module, "Variants", None) or {})
        for label, config_cls in topologies.items():
            for stage in config_cls(model_path="unused").stages:
                yield model, label, stage


def _factory_parameters(dotted: str) -> dict[str, object]:
    try:
        return {
            name: (... if p.default is inspect.Parameter.empty else p.default)
            for name, p in inspect.signature(import_string(dotted)).parameters.items()
        }
    except ImportError:
        pass
    module_name, _, func_name = dotted.rpartition(".")
    source = (
        _MODELS_DIR.parent / (module_name.replace(".", "/") + ".py")
    ).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            args = node.args
            positional = args.posonlyargs + args.args
            defaults = [...] * (len(positional) - len(args.defaults)) + [
                ast.literal_eval(d) for d in args.defaults
            ]
            params = dict(zip((a.arg for a in positional), defaults))
            for a, d in zip(args.kwonlyargs, args.kw_defaults):
                params[a.arg] = ... if d is None else ast.literal_eval(d)
            return params
    raise AssertionError(f"factory {dotted} not found in {module_name}")


def _gpu_stage_ids():
    return [
        pytest.param(model, label, stage, id=f"{model}-{label}-{stage.name}")
        for model, label, stage in _iter_stages()
        if stage.gpu is not None and (model, stage.name) not in _CPU_ONLY_GPU_PLACED
    ]


@pytest.mark.parametrize("model,label,stage", _gpu_stage_ids())
def test_gpu_stage_factories_declare_device_and_gpu_id(model, label, stage):
    params = _factory_parameters(stage.factory)
    assert "gpu_id" in params, (
        f"{stage.factory} is placed on a GPU (stage.gpu={stage.gpu}) but has "
        "no gpu_id parameter"
    )
    assert params["gpu_id"] is None, (
        f"{stage.factory}: gpu_id defaults to {params['gpu_id']!r}, should use None"
    )
    assert "device" in params, f"{stage.factory} has no device parameter"
    assert params["device"] is None, (
        f"{stage.factory}: device defaults to {params['device']!r}, should use None"
    )


@pytest.mark.parametrize(
    "model,label,stage",
    [pytest.param(m, l, s, id=f"{m}-{l}-{s.name}") for m, l, s in _iter_stages()],
)
def test_config_device_never_carries_an_index(model, label, stage):
    device = (stage.factory_args or {}).get("device")
    if device is None:
        return
    assert ":" not in str(device), (
        f"stage {stage.name!r} of {model}/{label} sets device={device!r}; "
        "device must not contain index, set in stage.gpu"
    )


class _Settled(Exception):
    def __init__(self, device, index):
        self.device = device
        self.index = index


def _arm_device_spec_resolvers(monkeypatch):
    import sglang_omni.utils.device as device_mod
    from sglang_omni.scheduling.engine_factory import SGLangGenerationEngineBuilder

    def _capture(device, index=None):
        raise _Settled(device, index)

    monkeypatch.setattr(device_mod, "resolve_device_spec", _capture)
    monkeypatch.setattr(device_mod, "place_device_spec", _capture)
    monkeypatch.setattr(
        SGLangGenerationEngineBuilder,
        "resolve_checkpoint",
        lambda self, model_path: model_path,
    )


@pytest.mark.parametrize("model,label,stage", _gpu_stage_ids())
def test_gpu_stage_factories_forward_gpu_id_into_device_spec_resolution(
    monkeypatch, model, label, stage
):
    try:
        factory = import_string(stage.factory)
    except ImportError as exc:
        pytest.skip(f"optional dependency missing: {exc}")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    _arm_device_spec_resolvers(monkeypatch)
    try:
        factory(model_path="unused", device=None, gpu_id=2)
    except _Settled as settled:
        assert settled.index == 2, (
            f"{stage.factory} reached device-spec resolution but dropped gpu_id "
            f"(index={settled.index!r})"
        )
        assert settled.device is None
    except ModuleNotFoundError as exc:
        pytest.skip(f"optional dependency missing inside factory body: {exc}")
    except RuntimeError as exc:
        if isinstance(exc.__cause__, ImportError):
            pytest.skip(f"optional dependency missing inside factory body: {exc}")
        raise
    except TypeError as exc:
        pytest.fail(
            f"{stage.factory} rejected the standard (device, gpu_id) call: {exc}"
        )
    else:
        pytest.fail(
            f"{stage.factory} returned without ever consulting "
            "resolve_device_spec/place_device_spec; should not resolve device and "
            "gpu_id by hand (or ignore them)"
        )


# note (lennox): forwarding into device-spec resolution is not the same as
# binding its result. code below drives the real build() chain and asserts the
# device/gpu_id the builder fixes at pre_infra_setup, before any weight loading.
_ENGINE_FACTORIES = {
    "arkasr": (
        "sglang_omni.models.arkasr.stages.create_sglang_arkasr_executor",
        "sglang_omni.models.arkasr.engine_builder",
        "ArkasrEngineBuilder",
    ),
    "whisper_asr": (
        "sglang_omni.models.whisper_asr.stages.create_sglang_whisper_asr_executor",
        "sglang_omni.models.whisper_asr.engine_builder",
        "WhisperASREngineBuilder",
    ),
    "fun_asr": (
        "sglang_omni.models.fun_asr.stages.create_sglang_fun_asr_executor",
        "sglang_omni.models.fun_asr.engine_builder",
        "FunASREngineBuilder",
    ),
    "moss_transcribe_diarize": (
        "sglang_omni.models.moss_transcribe_diarize.stages."
        "create_sglang_moss_transcribe_diarize_executor",
        "sglang_omni.models.moss_transcribe_diarize.engine_builder",
        "MossTranscribeDiarizeEngineBuilder",
    ),
    "qwen3_asr": (
        "sglang_omni.models.qwen3_asr.stages.create_sglang_qwen3_asr_executor",
        "sglang_omni.models.qwen3_asr.engine_builder",
        "Qwen3ASREngineBuilder",
    ),
    "dots_tts": (
        "sglang_omni.models.dots_tts.stages.create_sglang_latent_engine_executor",
        "sglang_omni.models.dots_tts.engine_builder",
        "DotsTTSEngineBuilder",
    ),
    "moss_tts": (
        "sglang_omni.models.moss_tts.stages.create_sglang_tts_engine_executor",
        "sglang_omni.models.moss_tts.engine_builder",
        "MossTtsEngineBuilder",
    ),
    "moss_tts_local": (
        "sglang_omni.models.moss_tts_local.stages.create_sglang_tts_engine_executor",
        "sglang_omni.models.moss_tts_local.engine_builder",
        "MossTtsLocalEngineBuilder",
    ),
    "ming_tts": (
        "sglang_omni.models.ming_tts.stages.create_sglang_tts_engine_executor",
        "sglang_omni.models.ming_tts.engine_builder",
        "MingTtsEngineBuilder",
    ),
    "voxtral_tts": (
        "sglang_omni.models.voxtral_tts.pipeline.stages.create_generation_executor",
        "sglang_omni.models.voxtral_tts.pipeline.engine_builder",
        "VoxtralTtsEngineBuilder",
    ),
    "fishaudio_s2_pro": (
        "sglang_omni.models.fishaudio_s2_pro.stages.create_sglang_tts_engine_executor",
        "sglang_omni.models.fishaudio_s2_pro.engine_builder",
        "FishS2ProEngineBuilder",
    ),
    "higgs_tts": (
        "sglang_omni.models.higgs_tts.stages.create_sglang_tts_engine_executor",
        "sglang_omni.models.higgs_tts.engine_builder",
        "HiggsTtsEngineBuilder",
    ),
    "minimax_music3": (
        "sglang_omni.models.minimax_music3.stages.create_ar_executor",
        "sglang_omni.models.minimax_music3.engine_builder",
        "MiniMaxMusic3EngineBuilder",
    ),
    "zonos2": (
        "sglang_omni.models.zonos2.stages.create_sglang_omni_tts_engine_executor",
        "sglang_omni.models.zonos2.engine_builder",
        "Zonos2EngineBuilder",
    ),
}


@pytest.mark.parametrize("model", sorted(_ENGINE_FACTORIES))
def test_engine_factories_bind_the_placed_gpu(monkeypatch, model):
    factory_path, builder_module, builder_class = _ENGINE_FACTORIES[model]
    try:
        factory = import_string(factory_path)
        builder = getattr(importlib.import_module(builder_module), builder_class)
    except ImportError as exc:
        pytest.skip(f"optional dependency missing: {exc}")

    final: dict[str, object] = {}

    class _Stop(Exception):
        pass

    def capture(self, checkpoint_dir):
        del checkpoint_dir
        final["device"] = self.device
        final["gpu_id"] = self.gpu_id
        raise _Stop

    monkeypatch.setattr(
        builder, "resolve_checkpoint", lambda self, model_path: model_path
    )
    monkeypatch.setattr(builder, "pre_infra_setup", capture)
    # note (lennox): pinned to "cuda" so the assertion below is
    # host-independent.
    monkeypatch.setattr(
        platforms.current_platform, "device_type", "cuda", raising=False
    )

    with pytest.raises(_Stop):
        factory(model_path="unused", device=None, gpu_id=2)
    assert final == {"device": "cuda:2", "gpu_id": 2}

    final.clear()
    with pytest.raises(_Stop):
        factory(model_path="unused", device="cuda", gpu_id=2)
    assert final == {"device": "cuda:2", "gpu_id": 2}


# note (lennox): predates this file's model x topology x stage sweep above (PR
# #994, #1628) and checks something the sweep does not: what an *unspecified*
# device (no gpu_id either) resolves to for the small set of factories that were
# already contract-compliant before this refactor. The sweep's own T3
# (test_gpu_stage_factories_forward_gpu_id_into_device_spec_resolution) always
# passes gpu_id=2, so it never exercises this ambient-platform path.
_MODELS = sorted(p.parent.name for p in _MODELS_DIR.glob("*/config.py"))

# Every stage that relies on device=None. Adding one means adding a test below that
# proves the factory resolves it.
_NONE_DEVICE_STAGES = {
    ("qwen3_asr", "asr"),
    ("qwen3_omni", "audio_encoder"),
    ("qwen3_omni", "code2wav"),
    ("qwen3_omni", "image_encoder"),
}


def _stages_with_device(model: str):
    """Every stage of every shipped topology, not just the default one."""
    module = importlib.import_module(f"sglang_omni.models.{model}.config")
    topologies = {}
    entry = getattr(module, "EntryClass", None)
    if entry is not None:
        topologies["default"] = entry
    topologies.update(getattr(module, "Variants", None) or {})

    found = []
    for label, config_class in topologies.items():
        for stage in config_class(model_path="unused").stages:
            if "device" in (stage.factory_args or {}):
                found.append((label, stage))
    return found


def test_only_the_qualified_stages_pass_device_none() -> None:
    """A new None-passing stage must come with a resolution test of its own."""
    found = {
        (model, stage.name)
        for model in _MODELS
        for _, stage in _stages_with_device(model)
        if stage.factory_args["device"] is None
    }

    assert found == _NONE_DEVICE_STAGES


@pytest.mark.parametrize("factory_name", ["image_encoder", "audio_encoder"])
def test_qwen3_omni_encoder_stages_resolve_none_to_the_platform(
    monkeypatch: pytest.MonkeyPatch, factory_name: str
) -> None:
    from sglang_omni.models.qwen3_omni import stages
    from sglang_omni.scheduling import simple_scheduler

    built: dict[str, object] = {}

    class _Encoder:
        def __init__(
            self,
            *,
            model_path,
            device,
            dtype,
            enable_layer_cuda_graph: bool | None = None,
        ):
            del model_path, dtype
            built["device"] = device
            built["enable_layer_cuda_graph"] = enable_layer_cuda_graph

        def __getattr__(self, name):
            del name
            return 2

    encoder_attr = {
        "image_encoder": "Qwen3OmniImageEncoder",
        "audio_encoder": "Qwen3OmniAudioEncoder",
    }[factory_name]
    monkeypatch.setattr(stages, encoder_attr, _Encoder)
    monkeypatch.setattr(
        simple_scheduler, "SimpleScheduler", lambda *a, **k: SimpleNamespace()
    )

    getattr(stages, f"create_{factory_name}_executor")("unused", device=None)

    assert built["device"] == platforms.current_platform.device_type
    assert built["enable_layer_cuda_graph"] is (
        False if factory_name == "audio_encoder" else None
    )


def test_qwen3_omni_code2wav_resolves_none_to_a_concrete_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.qwen3_omni.components import code2wav_scheduler

    model = SimpleNamespace(
        total_upsample=1, config=SimpleNamespace(num_quantizers=4), eval=lambda: None
    )
    model.eval = lambda: model
    monkeypatch.setattr(
        code2wav_scheduler, "load_code2wav_model", lambda *a, **k: model
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler("unused", device=None)

    assert scheduler._device.type == platforms.current_platform.device_type
    if platforms.current_platform.device_type != "cpu":
        # Placement was not requested, so the backend's current card is bound.
        # A cpu device correctly carries no index.
        assert scheduler._device.index is not None


def test_qwen3_asr_stage_forwards_none_to_the_shared_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The factory must hand None down rather than substitute a literal.

    Patching the base builder's build() also proves it is the builder in play: a
    factory using an unrelated builder would leave this spy untouched. What build()
    then does with None is covered in test_server_args_builder_device.py.
    """
    from sglang_omni.models.qwen3_asr import stages
    from sglang_omni.scheduling import engine_factory

    seen: dict[str, object] = {}

    def spy_build(self, model_path, **kwargs):
        del self, model_path
        seen.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        engine_factory.SGLangGenerationEngineBuilder, "build", spy_build
    )

    stages.create_sglang_qwen3_asr_executor("unused", device=None, gpu_id=1)

    assert "device" in seen, "the factory did not route through the shared builder"
    assert seen["device"] is None
    # Placement injects gpu_id only when the signature declares it. Without it the
    # builder resolved a bare accelerator and told SGLang card 0.
    assert seen["gpu_id"] == 1

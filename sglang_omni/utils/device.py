# SPDX-License-Identifier: Apache-2.0
"""Resolve the device spec a pipeline stage runs on. Never initializes a device."""

from __future__ import annotations


def _with_index(dev_type: str, raw_index: str, index: int | None) -> str:
    if raw_index:
        raise ValueError(
            f"device={f'{dev_type}:{raw_index}'!r} names an index; device can"
            " only name the type"
        )
    if dev_type == "cpu" or index is None:
        return dev_type
    return f"{dev_type}:{int(index)}"


def resolve_device_spec(device: str | None, index: int | None = None) -> str:
    """Return the concrete device spec for a stage."""
    from sglang_omni.platforms import current_platform

    platform_type = current_platform.device_type

    if device is None:
        return _with_index(platform_type, "", index)

    dev_type, _, raw_index = str(device).strip().partition(":")
    dev_type = dev_type.lower()
    if dev_type not in ("cpu", platform_type):
        raise ValueError(
            f"device={device!r} names {dev_type!r}, but this host resolved to "
            f"{platform_type!r}. Pass device=None to run on whatever the host "
            f"provides, or 'cpu'/'{platform_type}' explicitly."
        )
    return _with_index(dev_type, raw_index, index)


def place_device_spec(device: str, index: int | None = None) -> str:
    """Apply a placement index to an explicit device spec, keeping its own type.

    The caller named the platform, so honor it rather than validate it: a CUDA-only
    stage keeps CUDA on any host, and an explicit xpu or cpu is never rewritten.
    """
    dev_type, _, raw_index = str(device).strip().partition(":")
    return _with_index(dev_type.lower(), raw_index, index)


def resolve_concrete_device(device: str | None, index: int | None = None) -> "torch.device":
    """Resolve device/index to a concrete torch.device with an index.

    Falls back to asking the host which card this process is already on
    when neither the caller nor placement supplied an index, rather than
    assuming 0.
    """
    import torch

    from sglang_omni.platforms import current_platform

    spec = (
        resolve_device_spec(device, index)
        if device is None
        else place_device_spec(device, index)
    )
    concrete = torch.device(spec)
    if concrete.type != "cpu" and concrete.index is None:
        concrete = current_platform.get_device(torch.get_device_module().current_device())
    return concrete


__all__ = ["place_device_spec", "resolve_concrete_device", "resolve_device_spec"]

"""Reuse the project ASR runner with explicit request and scoring languages."""

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.metrics.wer import SampleOutput
from benchmarks.tasks.asr import (
    apply_wer,
    build_asr_eval_results,
    run_asr_transcription,
)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--request-language", default="auto")
    parser.add_argument("--score-language", choices=["en", "zh"], default="en")
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", action="store_true")
    arguments = parser.parse_args()
    if arguments.repeats <= 0 or min(arguments.concurrencies) <= 0:
        parser.error("Repetitions and concurrency must be positive")
    else:
        samples = load_seedtts_samples(arguments.meta)
    if not samples or len({sample.sample_id for sample in samples}) != len(samples):
        parser.error("Require nonempty metadata with unique sample identifiers")
    else:
        pass
    apply_wer(SampleOutput(target_text="hello"), "hello", arguments.score_language)
    arguments.output.mkdir(parents=True, exist_ok=False)
    (arguments.output / "config.json").write_text(
        json.dumps(
            {
                **vars(arguments),
                "output": str(arguments.output),
                "sample_ids": [sample.sample_id for sample in samples],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    failures = 0
    for concurrency in arguments.concurrencies:
        for repeat in range(0 if arguments.warmup else 1, arguments.repeats + 1):
            outputs, elapsed_seconds = await run_asr_transcription(
                samples,
                port=arguments.port,
                model_path=arguments.model,
                lang=arguments.request_language,
                concurrency=concurrency,
                request_timeout_s=300,
            )
            result = build_asr_eval_results(
                samples,
                outputs,
                elapsed_seconds,
                arguments.score_language,
                model_path=arguments.model,
                concurrency=concurrency,
            )
            result["request_language"] = arguments.request_language
            result["repeat"] = repeat
            result["warmup"] = repeat == 0
            (arguments.output / f"c{concurrency}-r{repeat}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2)
            )
            with (arguments.output / f"c{concurrency}-r{repeat}-raw.jsonl").open(
                "w"
            ) as handle:
                for output in outputs:
                    handle.write(json.dumps(asdict(output), ensure_ascii=False) + "\n")
            print(json.dumps(result["summary"], ensure_ascii=False), flush=True)
            if not all(output.is_success for output in outputs):
                failures += sum(not output.is_success for output in outputs)
            else:
                pass
    if failures:
        raise RuntimeError(f"{failures} HTTP requests failed; inspect saved outputs")
    else:
        pass


if __name__ == "__main__":
    asyncio.run(main())
else:
    pass

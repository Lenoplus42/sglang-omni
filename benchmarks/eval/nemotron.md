# Nemotron ASR reproduction

These optional entry points reproduce the **measurement method** used to review
[#1748](https://github.com/sgl-project/sglang-omni/pull/1748#issuecomment-5861686350).
They do not register CI jobs or change model inference. Run commands from the
repository root after installing the project and its benchmark dependencies.
The native client also needs `websockets>=14`. Outputs and audio belong outside
source control, for example under `benchmarks/results/`.

## Prepare identical inputs

SeedTTS supplies recordings and reference transcripts here; no TTS is generated.
The loader preserves dataset rows, including repeated recordings with distinct
sample identifiers. Do not describe the row count as the number of unique clips.

```bash
python -m benchmarks.eval.prepare_nemotron_audio \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --language en --max-samples 20 \
  --output benchmarks/results/nemotron-input
```

Omit `--max-samples` for the full split; use `--language zh` for Chinese. A local
`--meta /path/to/meta.lst` also works. Each line is
`sample_id|reference_text|audio_path|target_text`; paths are relative to the
metadata file. Reference text is used for scoring. Identifiers must be unique.
Preparation converts audio to mono 16 kHz PCM16 before either implementation
runs, and writes `meta.lst`, WAV files and a manifest with source revision,
ordered sample identifiers, row count and unique source-path count. The latter
is not a content-based deduplication count.

## Start the service

Resolve the checkpoint once and use the same local snapshot for both runtimes:

```bash
MODEL_PATH=$(hf download nvidia/nemotron-3.5-asr-streaming-0.6b \
  --revision ea30d66debe3740a08b573244286791d423d6b3e --quiet)
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
sgl-omni serve --model-path "$MODEL_PATH" \
  --model-name nvidia/nemotron-3.5-asr-streaming-0.6b \
  --enable-realtime --host 127.0.0.1 --port 8000 \
  --asr.factory.dtype float32 --asr.factory.num_lookahead_tokens 3 \
  --asr.factory.max_batch_size 8 --asr.factory.max_batch_wait_ms 2 \
  --asr.factory.session_max_concurrency 8
```

Record `git rev-parse HEAD`, `python -m pip freeze`, the server command and
`nvidia-smi` alongside results. The original run used one H100 80 GB, Python
3.12.3, PyTorch 2.13.0+cu130 and Transformers 5.12.1 in the serving environment.
Those historical numbers are not results for a newer checkout or a different GPU.

## Measure HTTP and native streaming

```bash
python -m benchmarks.eval.benchmark_nemotron_http \
  --meta benchmarks/results/nemotron-input/meta.lst \
  --model nvidia/nemotron-3.5-asr-streaming-0.6b \
  --request-language auto --score-language en \
  --concurrencies 1 4 --repeats 3 --warmup \
  --output benchmarks/results/nemotron-http

python -m benchmarks.eval.benchmark_nemotron_native \
  --meta benchmarks/results/nemotron-input/meta.lst \
  --score-language en --max-samples 20 \
  --concurrencies 1 4 --repeats 3 --warmup \
  --packet-milliseconds 20 --timeout-seconds 300 \
  --output benchmarks/results/nemotron-native-paced
```

Set native `--max-samples` to the prepared row count for a full run. For Chinese,
use `--score-language zh` on both clients while retaining request language `auto`.
The HTTP entry point reuses the project ASR runner and scorer; its thin wrapper
separates request language from scoring language. The general SeedTTS HTTP CLI
uses one language for both, which would change the reviewed experiment.

Native sends audio at recording speed by default. Add `--burst` and use a new
output directory to send packets as quickly as possible. Each concurrency is
the number of in-flight requests, not a guaranteed model batch size. Repeat 0
is warmup and must be excluded from performance comparisons. Outputs use fresh
directories to avoid mixing experiments. HTTP writes per-repeat summaries and
raw request JSONL; native writes configuration, per-request text/errors, received
wire events, packet timestamps and measured summaries. Failed requests remain
in the raw output and cause a nonzero exit status.

| Metric | Meaning |
|---|---|
| WER / CER | Project-normalized corpus word error rate for English; character error rate for Chinese. Native `wer_percent` is a percentage for either scoring language. |
| Success / scored / skipped | Request completion is separate from scoring coverage. A transcript whose reference normalizes to empty can succeed but be skipped for scoring. |
| Audio seconds per second (RTFx) | Successfully processed audio duration divided by the whole run's wall time, including connection setup and failed requests. |
| Requests per second | Successful requests divided by the same run wall time. |
| First text | First nonempty text delta received minus first packet send time; null when no nonempty delta arrives. |
| EOS to final / drained | Time from sending end-of-input to final text / confirmation that input was processed. |
| Request wall time / RTF | Native first-send-to-drained duration; divide by that request's audio duration for RTF. HTTP uses the shared runner's request timer. |

Native latency values are seconds; percentiles use nearest rank and include
successful requests only. Every percentile includes its observation count.
Report each repeat separately, especially with small cohorts. Under burst input,
EOS-to-final includes processing of queued audio and is **not** latency after a
person stops speaking. HTTP file transcription and native streaming use different
inference paths; their throughput ratio is not an implementation speedup.

## Independent Transformers transcripts

Use a separate environment with `transformers==5.13.0`, NumPy and the same PyTorch
version as the service. Do not upgrade the serving environment to run the reference.
Stop the service before loading the reference on a capacity-constrained GPU.

```bash
/path/to/reference-env/bin/python -m benchmarks.eval.nemotron_reference \
  --model-path "$MODEL_PATH" \
  --meta benchmarks/results/nemotron-input/meta.lst \
  --language auto --lookahead 3 --repeats 2 \
  --output benchmarks/results/nemotron-reference.jsonl
```

The reference imports the official Transformers model, not Omni's vendored model.
It runs offline and streaming inference at batch size 1, retains tokens and raw
text, and pads the final streaming feature window. Compare repetitions within
one implementation before comparing matching sample identifiers and modes across
implementations. Compare normalized text with `benchmarks.tasks.asr.normalize_text`
in the serving environment and retain raw mismatches. Do not compare HTTP output
to the streaming reference. Its diagnostic timing is not a service benchmark.

This small reproduction entry point does **not** reproduce the historical paired
A/A and A/B performance experiments or batch-matched reference diagnostics.
At concurrency 4, dynamically formed service batches can differ from reference
batch size 1; a transcript mismatch alone does not identify a model integration
bug. The original review also retained a few unresolved Pipecat mismatches and
one failure among 12 longer burst requests at concurrency 4 (`outbound event
budget exhausted`). That sample does not estimate a general long-audio failure
rate. See the linked review for the complete historical results and limits.

## Packaged-script smoke result (2026-10-06 UTC)

On one H100 80 GB, the packaged scripts completed 80/80 measured service requests
and 32/32 warmup requests on 8 distinct English recordings (41.191 s total).
The code base was `5ed8a8d7af86d3a486875a8deedaf0b4bf0c0687` plus these reproduction
scripts. Serving used PyTorch 2.13.0+cu130, SGLang 0.5.21, Transformers 5.12.1
and the FP32 server settings above. The independent reference used Transformers
5.13.0 and the same checkpoint. All 112 normalized service transcripts matched
the corresponding official mode. Both reference repetitions produced identical
tokens (32 generations: 8 recordings × 2 modes × 2 repeats).

| Path | Concurrency | Measured requests | Audio seconds / second | WER |
|---|---|---|---|---|
| HTTP | 1 | 16/16 | 73.08–73.10 | 5.833% |
| HTTP | 4 | 16/16 | 184.77–186.21 | 5.833% |
| Native burst | 1 | 16/16 | 9.99–10.04 | 6.667% |
| Native burst | 4 | 16/16 | 12.78–14.50 | 6.667% |
| Native paced | 1 | 8/8 | 0.995 | 6.667% |
| Native paced | 4 | 8/8 | 3.252 | 6.667% |

HTTP and burst used one excluded warmup plus two measured repeats per concurrency;
paced input used one measured repeat without a separate warmup, on the already
warm server. Paced first-text p50 was 901.6 / 904.7 ms and EOS-to-final p50 was
21.4 / 22.3 ms at concurrency 1 / 4. Ranges span the two measured repeats, not
confidence intervals. These are small-cohort execution checks, not new
corpus-quality, capacity or long-audio claims.

To select the same recordings, prepare the first 20 English dataset rows with the
pinned revision above, then retain row indices `0, 2, 4, 6, 8, 10, 12, 14` from
`meta.lst` (zero-based). Those rows contain eight distinct PCM recordings. Use
that metadata for all four entry points. Set `--max-samples 8` for native,
`--repeats 2 --warmup` for HTTP/burst, and `--repeats 1` without `--warmup`
for paced input. Run the official reference with `--repeats 2` after stopping
the service.

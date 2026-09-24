# Controlled llama.cpp Comparison Methodology

Comparison Prompt 7 begins only after the AIR release-candidate architecture is frozen.
The comparison harness is evidence tooling. It does not change model execution,
scheduling, KV policy, kernel selection, or qualification behavior.

## Primary lane: identical raw requests

AIR and llama.cpp are loaded from the exact same GGUF file. The harness records its
full SHA-256 and size. Both servers receive the same raw prompt text, greedy sampling,
and requested generation length. Chat templates are not involved.

Before measuring a workload, the harness asks both servers to evaluate it and rejects
the workload if their reported prompt-token counts differ. This catches tokenizer,
BOS, or request-shaping differences before any performance ratio is computed.

Both servers remain resident during measured rounds, but only one request is active at
a time. Engine order is randomized within every paired round from a persisted seed.
This reduces systematic thermal/order bias while avoiding model-load time in request
latency. Isolated idle process VRAM is measured separately before the paired run.

The same Python HTTP/SSE client measures the primary comparable metrics:

- first-token wall latency (TTFT),
- total request wall latency,
- generated tokens per wall-clock second.

Paired AIR/llama ratios use the per-round log ratio and a two-sided 95% Student-t
interval. A confidence interval spanning 1 is not called a demonstrated difference.

## Secondary lane: engine-native timing

AIR reports internal prefill and decode compute timing. llama.cpp reports its own
prompt and predicted-token timing. These are preserved as useful evidence, but their
internal timer boundaries are not assumed to be identical. They are therefore kept
separate from the primary client-measured comparison.

`air-bench` and `llama-bench` are also run as a secondary length-controlled
microbenchmark lane. llama-bench uses synthetic tokens of the same prompt length, so
these results must never be presented as the exact-request lane.

## Prompt-7 workloads

Prompt 7 intentionally remains concurrency 1. Scaling is Comparison Prompt 8.

- `decode`: a very short raw prompt with 256 requested new tokens,
- `balanced`: approximately 128 prompt tokens with 128 new tokens,
- `prefill`: approximately 1024 prompt tokens with 32 new tokens.

The exact generated prompt strings and exact AIR/llama prompt-token counts are stored
with every evidence archive.

## Runtime configuration

AIR runs its static CUDA backend with adaptive manifests and prefix caching disabled.
This isolates the frozen CUDA execution engine from planner decisions.

llama.cpp runs the same model on the same GPU with one parallel slot, native model
context, all layers offloaded, and Flash Attention enabled. llama.cpp's default KV
representation is retained because it is part of that runtime's normal design. AIR's
paged KV representation is likewise retained. Memory differences are therefore a
runtime-design result, not normalized away.

## Correctness lane

Before timing, `air-verify` produces teacher-forced AIR reference/CUDA evidence and
`verify-llama-teacher.py` feeds the exact numeric token history to llama.cpp's raw
`/completion` endpoint. Top-k rankings and first external disagreement are preserved.
The comparison does not require exact generated-text identity when a previously
characterized near-tie makes two correct floating-point implementations choose
different argmax tokens.

## Thermal and memory evidence

GPU temperature, power, SM clock, utilization, and memory are sampled around every
request. `nvidia-smi dmon` is retained in the raw evidence archive. Start-state thermal
imbalance between engines must be reported rather than silently ignored.

Process VRAM is also measured with each server resident alone. This is separate from
the paired-run total, in which both models are resident to make request order genuinely
interleavable.

## Publication rules

1. Publish the full model SHA-256 and exact software revisions.
2. Publish raw JSON/JSONL evidence alongside summary tables.
3. Never compare AIR client wall time against llama.cpp internal kernel time.
4. Never present the secondary microbenchmark lane as an exact-prompt comparison.
5. Do not optimize AIR because of a single workload or single round.
6. A result whose paired 95% interval spans parity is reported as inconclusive.
7. Prompt 7 is a baseline. Context/concurrency/model scaling belongs to Prompt 8.

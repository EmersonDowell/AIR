# Benchmarking

`air-bench` executes through the same `InferenceService` used by the server. There is no benchmark-only inference path.

## Metrics

Benchmark schema `air.benchmark.v3` reports:

- prompt/generated/reused token counts;
- queue time;
- prefill time and compute tok/s;
- TTFT;
- decode time and compute tok/s;
- total request time;
- P50/P95 TTFT and total latency;
- aggregate generated tok/s;
- requests/s;
- KV/device high-water marks;
- selected prefill/decode quantized-linear tactics;
- planned scheduler prefill quantum and KV page geometry.

Percentiles use shared linear interpolation semantics.

## Cold-prefix default

Prefix caching is disabled by default in `air-bench`. This prevents a repeated benchmark prompt from accidentally turning prompt-processing measurements into warm-prefix measurements.

Use `--prefix-cache N` only when intentionally measuring exact-prefix reuse and only on a backend that advertises that capability.

## Concurrency

`--concurrency N` submits overlapping requests through the production scheduler. It measures serving behavior, not an assumed fused CUDA batch. CUDA now advertises a qualified native greedy decode width of eight compatible sequences. Concurrency measurements therefore distinguish scheduler-only overlap from actual physical decode batching using benchmark-native physical batch counters.

## Comparison discipline

For external runtime comparisons, hold constant:

- exact GGUF bytes/hash;
- prompt token counts;
- generated-token counts;
- sampling policy;
- GPU/device;
- warmup/repetition count;
- thermal/power context where practical.

Publish raw evidence alongside summarized results.

## AIR 0.9.6 evidence schema

`air.benchmark.v9` records the current execution tactic vocabulary after rejecting `shared-tile8` and adding `dense-f32-cublas`. Peak-device evidence is especially important for the dense candidate because its prepared FP32 matrices deliberately trade substantial VRAM for the mature-GEMM diagnostic.

# Comparison Prompt 8: Scaling Matrix

Prompt 8 asks whether the Prompt 7 result is specific to one 0.5B workload or persists as prompt length, concurrency, and compatible Qwen2/Qwen2.5 model size change.

This is an evidence expansion only. It does not authorize changes to AIR model execution, scheduling ownership, KV policy, qualification, or kernel selection.

## Matrix dimensions

Default machine-validation profile:

- prompt targets: 32, 256, and 1024 tokens;
- concurrency: 1, 2, and 4 simultaneous requests;
- generation: 32 greedy tokens per request;
- paired rounds: 3 per cell;
- models: up to 3 compatible Qwen2/Qwen2.5 GGUF files selected across available file sizes.

All values are configurable from the validator environment. The default is deliberately bounded so the matrix can run on the primary RTX 3080 Laptop validation machine without turning Prompt 8 into an unbounded benchmark campaign.

## Same primary methodology

Every model is identified by full SHA-256. AIR and llama.cpp load the exact same file for a pair. Raw prompt text and greedy generation limits are identical.

For each prompt length, AIR constructs one deterministic natural-language prompt and records the exact tokenization. Both resident servers must report the same prompt-token count before any cell for that prompt is admitted.

Within a cell, the requested number of clients are released together. One engine runs at a time. Engine order is randomized inside every paired round from a persisted seed. Both engines remain resident during a model run so model-load latency is excluded and order remains interleavable.

Primary comparable measurements are client-side:

- median and P95 request TTFT inside each concurrent group;
- median and P95 request total latency;
- group wall time;
- aggregate generated tokens per wall second;
- completed requests per wall second.

Paired AIR/llama ratios are computed from per-round log ratios with two-sided 95% Student-t confidence intervals. An interval spanning 1 is inconclusive.

## Concurrency interpretation

AIR and llama.cpp are allowed to retain their normal runtime designs. AIR uses its frozen scheduler and physical paged KV implementation. llama.cpp uses its normal slot/batching implementation with Flash Attention enabled. Prompt 8 is testing those runtime designs, not trying to force them into an artificial common implementation.

The server is provisioned for the maximum concurrency in the selected matrix and then exercised at each lower concurrency. This avoids relaunching with a different runtime shape for every cell.

## Correctness

Correctness is rechecked independently for every selected model before performance cells run:

1. AIR reference and CUDA execution are compared through `air-verify`.
2. The resulting exact numeric teacher-forced history is sent to llama.cpp.
3. External top-k decisions and any disagreement are retained.

A model that cannot execute through the supported frozen Qwen2 surface, or that fails the fixed reference/CUDA max-absolute-error admission gate, is rejected rather than being silently substituted.

## Model discovery

`scale-matrix.py` accepts explicit model paths and may discover local GGUF files. Discovery never downloads a model. `air-cli inspect` is used only for metadata and architecture identification. AIR's own `air-verify` reference/CUDA path is the authority for execution support and applies the same fixed admission tolerance before a model enters the performance matrix. The harness does not maintain a duplicate tensor-format support list.

Every candidate inventory entry records its SHA-256. When more candidates exist than the configured maximum, the harness selects a size-spread sample rather than picking only favorable models. Explicitly supplied models are retained first.

If fewer than two compatible models are available, the prompt/concurrency matrix remains valid for the available model but the aggregate report marks model-size generalization incomplete.

## Secondary evidence

At concurrency 1, `air-bench` and `llama-bench` are retained as a secondary length-controlled lane. As in Prompt 7, these engine-native benchmark boundaries are not treated as equivalent to the exact-request HTTP/SSE lane.

GPU start/end telemetry is recorded for every concurrent group, and raw `nvidia-smi dmon` evidence is retained for the full model run. Isolated idle process VRAM is measured with each server configured for the maximum matrix concurrency.

## Publication rules

- Keep every valid matrix cell, including unfavorable results.
- Do not infer cross-model generalization from a single model.
- Do not infer concurrency scaling from concurrency 1.
- Do not compare AIR client wall time to llama.cpp internal timing.
- Do not treat file size as a parameter count; it is only a reproducible model-size descriptor unless model metadata provides more.
- Do not change the frozen AIR architecture from Prompt 8 evidence. Optimization decisions belong after the scaling evidence is interpreted.

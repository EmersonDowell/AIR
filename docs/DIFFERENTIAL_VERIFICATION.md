# Differential Verification

`air-verify` compares the real reference and CUDA executors under one teacher-forced token history. It is an observer over production executor code, not a second transformer implementation.

## Decision verification

At each decision AIR compares:

- full logits;
- maximum/mean/RMS absolute error;
- top-1 token parity;
- top-k ranking;
- top1/top2 margin.

The reference top token is then forced into both executors for the next step. This preserves diagnostic value after a potential mismatch by preventing the two autoregressive histories from drifting apart.

## Stage tracing

Optional trace windows capture canonical internal boundaries such as embedding, normalized inputs, Q/K/V, post-RoPE values, attention output, residuals, FFN stages, final norm, and logits.

Tracing introduces CUDA synchronization/readback and must not be used for performance measurement.

## External llama.cpp comparison

`scripts/verify-llama-teacher.py` can feed exact numeric token arrays to llama.cpp's raw completion endpoint and compare next-token ranking/margins. llama.cpp does not expose AIR-compatible internal stage vectors, so external claims remain decision/top-k comparisons.

## Destruction coverage

The repository includes:

- randomized scalar-oracle tests for every supported quantized block format;
- a deterministic Qwen2 fixture that mixes all supported quantized execution formats;
- exact-token boundary tests around KV/prefill boundaries;
- context-edge and overflow tests.

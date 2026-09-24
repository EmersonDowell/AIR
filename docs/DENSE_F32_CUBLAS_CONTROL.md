# Dense FP32 cuBLAS Control

`dense-f32-cublas` is a Prompt 13 R3 diagnostic/Pareto tactic, not an AIR default.

The tactic exists to isolate **execution geometry and reduction behavior** from the FP16 conversion error observed in Prompt 13 R2.

## Ownership

- `ModelDefinition` and canonical GGUF bytes remain authoritative.
- Rank-2 transformer execution matrices are lazily dequantized into a disposable FP32 device arena owned by `PreparedModel`.
- Embedding lookup and final output selection remain on their existing paths.
- Baseline/reuse8 plans never allocate the dense arena.
- The dense arena is prepared before capacity admission so its VRAM residency participates in AIR's device-memory guarantees.

## Math

The candidate uses FP32 prepared weights, FP32 activations, FP32 output and `CUBLAS_COMPUTE_32F_PEDANTIC`. It intentionally avoids FP16/TF32 arithmetic so a correctness failure cannot be blamed on half-precision conversion.

This is a diagnostic control. Nsight must determine the actual cuBLAS kernel used. AIR does not label this a Tensor Core tactic.

## Gate

The existing `atol=0.001` verifier runs before any competitive timing. If the tactic fails, no performance benchmark or profiler comparison is admitted.

## Tactic scope

`dense-f32-cublas` applies to prepared rank-2 transformer-block linear matrices.
`token_embd.weight` and the terminal vocabulary projection (`output.weight`, or
tied embedding fallback) remain on AIR's canonical low-residency lookup/quantized
projection path. This boundary is enforced inside CUDA matmul dispatch so native
physical batching cannot accidentally request an unprepared dense terminal
projection. Treat terminal-output acceleration as a separate tactic question.

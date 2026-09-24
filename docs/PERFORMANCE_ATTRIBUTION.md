# AIR Performance Prompt 9/16: Attribution Protocol

Status: **CLOSED**. See `PROMPT9_ATTRIBUTION_RESULT.md` for measured results and the Prompt 10 decision.

Prompt 9 is diagnostic. It does not authorize optimized kernels.

## Questions

Measure enough evidence to distinguish these hypotheses:

1. quantized linear execution spends dominant time decoding GGUF blocks and reading weights;
2. prefill fails to reuse decoded/loaded weight work across prompt items;
3. scheduler prefill quantum materially limits native CUDA batch width;
4. paged prefill/decode attention is a dominant context-length cost;
5. FP32 KV bandwidth/residency is material at the tested contexts;
6. Prompt 12 resolved the previous decode-width-1 limitation with qualified native batching up to width 8;
7. kernel-launch overhead is material on small models;
8. H2D/D2H transfers or logit readback are unexpectedly material;
9. larger-model numerical divergence enters primarily during native prefill versus decode.

Do not elevate any hypothesis to a conclusion without profiler or differential evidence.

## Instrumentation strategy

AIR 0.9 profiling builds add opt-in NVTX v3 ranges only. Default builds compile with NVTX disabled.

Ranges identify:

- `air.prefill` / `air.prefill.chunk` / `air.prefill.layer`;
- `air.decode.step` / `air.decode.layer`;
- matrix-vector and matrix-matrix calls;
- prefill/decode attention;
- RMS normalization;
- RoPE;
- residual additions;
- KV append;
- logit readback and device greedy selection.

Kernel timings, launch counts, CUDA API costs, and GPU memory-copy evidence remain profiler-owned rather than being reimplemented as a second timing system inside AIR.

## Primary profiler

Nsight Systems is used first because it can report:

- CUDA kernel launch counts and total kernel-family time;
- CUDA API costs;
- device memory transfer time/size;
- NVTX range aggregation;
- timeline evidence for serialization and gaps.

Prompt 9 captures short, medium, and long prompt cases on the production `InferenceService` path through `air-bench`.

## Tactic sensitivity without optimization

The current executor already supports scheduler prefill quanta up to its reported native width. Prompt 9 measures 32, 64, and 128 token quanta using the unchanged executor.

This is not a new fast path. It determines whether part of the observed deficit is scheduling geometry versus kernel implementation.

## Optional Nsight Compute

After Nsight Systems identifies dominant kernel families, targeted Nsight Compute collection may inspect:

- SpeedOfLight;
- LaunchStats;
- Occupancy;
- MemoryWorkloadAnalysis;
- ComputeWorkloadAnalysis.

Do not profile every launch with the full metric set. Replay overhead can distort or dramatically lengthen execution.

## Correctness scale diagnostics

The 0.5B model remains the qualified performance subject.

The 1.5B and 7B models are diagnostics until they pass the frozen correctness gate. Prompt 9 records:

- full final-logit differential reports;
- top-1 parity and margins;
- max/mean/RMS error;
- one traced decode transition where feasible.

Do not use unqualified larger models in competitive performance claims.

## Exit criteria

Prompt 9 closes only when evidence can rank the dominant costs well enough to choose Prompt 10 work without guessing.

The expected output is an attribution report, not a faster AIR binary.

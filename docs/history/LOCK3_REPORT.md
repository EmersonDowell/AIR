# AIR Lock 3 Report

Lock 3 is the CUDA hot-path and evidence-system pass for the AIR 0.8 architecture-lock line. It preserves the Lock-2 execution/resource contract and changes only semantically equivalent execution paths or the way AIR gathers and reports evidence.

## Entry evidence

Lock 2 was validated on an NVIDIA RTX 3080 Laptop GPU with CUDA 13 before this pass began. The machine validation established that:

- the real CUDA build and six characterization suites passed;
- token-by-token CUDA/reference top-1 parity held;
- native multi-token CUDA prefill preserved top-1 parity with small numerical error;
- physical paged KV removed the previous full-context-per-request allocation behavior;
- concurrency no longer caused the old roughly 768 MiB KV reservation per active request;
- decode width remained one, so concurrency throughput was not yet expected to scale.

Those results allowed Lock 3 to focus on hot-path cost and evidence quality rather than reopening Lock-2 resource semantics.

## CUDA hot-path changes

### Prepared typed matrix dispatch

CUDA tensor preparation assigns each supported resident tensor to one explicit kernel family. F16, BF16, Q4_0, Q5_0, Q8_0, Q4_K, and Q6_K execution therefore no longer performs a runtime tensor-type switch for every decoded matrix element.

The production tensor bytes remain derived from the canonical immutable `ModelDefinition`; prepared kernel identity is backend state, not a second model source of truth.

### Warp-per-row quantized execution

The specialized non-F32 path uses one warp per output row, with four warps per 128-thread block. Threads stride the input dimension and use warp-shuffle reduction. This removes the old eight-warp/shared-memory reduction used for every output row.

F32 remains on the cuBLAS path. The current typed kernels are intentionally simple and auditable; tensor-core layouts, permanent repacking, larger fused kernels, and CUDA Graph capture remain deferred until profiling justifies them.

### Outputless intermediate prefill

Serving may divide a long prompt into multiple scheduler/native-prefill submissions. Intermediate CUDA prefill batches now commit transformer/KV state without running final RMSNorm, vocabulary projection, or host-logit readback when no token decision is required yet.

Only the final prompt position produces an output used for token selection.

### Device-side deterministic selection

Temperature-zero CUDA requests now use an exact first-maximum argmax on device. The normal deterministic production path reads back the selected token and a finite-value guard instead of the complete vocabulary logit vector.

Stochastic sampling remains owned by the shared host `Sampler`; Lock 3 does not create a second sampling policy in CUDA.

### Hot-path evidence counters

CUDA exposes cumulative counters for:

- F32 matrix-vector calls;
- specialized matrix-vector calls;
- F32 matrix-matrix calls;
- specialized matrix-matrix calls;
- full-logit readbacks;
- device-greedy token readbacks;
- outputless prefill chunks.

They are atomic, intentionally coarse, and designed to prove which path executed without adding per-kernel synchronization to production inference.

## Correctness gates

`air-cli cuda-compare` now covers two independent CUDA behaviors:

1. ordinary token-step logits against the reference executor;
2. native multi-token CUDA prefill against reference prefill.

It additionally validates the device-greedy result against the top-1 token from full logits. This prevents the new optimized output path from escaping the existing correctness oracle.

The real RTX/NVCC validation remains authoritative for these CUDA gates and is performed by `scripts/validate-lock3-machine.sh` after installation.

## Evidence-system changes

### Reproducibly interleaved qualification

Qualification now executes one measured sample per candidate per round and deterministically shuffles candidate order inside every round. A persisted evidence seed makes the order reproducible while balancing thermal, power, and time drift across candidates.

`--seed N` explicitly controls that sequence; seed zero derives a stable seed from model, hardware, and workload fingerprints.

### One causal variable per candidate comparison

Physical KV page geometry is held fixed while scheduler prefill quantum is varied. Lock 3 therefore does not repeat the earlier mistake of comparing candidates that changed two unrelated execution dimensions at once.

### Conservative paired selection

AIR stores per-candidate sample variance and compares the apparent best candidate to the canonical default using paired score differences. Short qualification runs use a two-sided 95% Student-t interval rather than a normal approximation.

AIR selects a non-default strategy only when:

- there are at least three paired samples;
- the mean score advantage is at least 0.02; and
- the lower bound of the paired 95% interval remains above zero.

Otherwise the canonical default is retained and the reason is persisted. “Indistinguishable” is a valid qualification result.

### Manifest schema v3

Manifest schema v3 persists the evidence seed, sample standard deviations, selection reason, paired margin, and confidence half-width. Schema v1/v2 remain readable for inspection/migration but cannot drive adaptive execution. Evidence methodology is part of manifest validity, not merely file syntax.

### Benchmark schema v2 and shared percentiles

Benchmark percentile semantics were corrected to linear interpolation rather than truncating fractional ranks. The same internal percentile helper is used by benchmark reports and live runtime snapshots so P50/P95 cannot silently disagree across surfaces.

Because the meaning changed, new benchmark JSON uses `air.benchmark.v2` rather than silently redefining v1.

## Source validation completed before packaging

The final source tree is required to pass:

- clean GCC Release build and all six tests;
- clean Clang Release build and all six tests with no compiler warnings;
- Clang ASan + UBSan build and all six tests with leak detection and halt-on-error enabled;
- CUDA translation-unit syntax parsing through an independent CUDA-language parser gate;
- fresh synthetic GGUF inspect, benchmark, qualification, manifest-v3 write/read, adaptive manifest reload, `/health`, `/runtime`, and `/generate` smoke;
- `git diff --check`;
- stale operator-semantic and TODO/FIXME/HACK scans;
- source-only archive extraction and file-hash comparison.

The container validation cannot replace a real `nvcc`/GPU run. The Lock-3 release package includes `scripts/validate-lock3-machine.sh` for that authority gate.

## Lock-3 machine gate

The machine validation script records system/model hashes and NVIDIA telemetry, then performs:

- CUDA/reference step parity;
- native-prefill parity;
- device-greedy parity;
- a long deterministic request that must show zero full-logit readbacks, non-zero greedy-token readbacks, and non-zero outputless-prefill chunks;
- short-prompt, long-prefill, and concurrency benchmarks;
- five-round seeded concurrent qualification writing manifest schema v3;
- automatic evidence packaging under `~/Downloads`.

Lock 3 is not considered hardware-qualified until that archive passes on the target NVIDIA machine.

## Explicit non-goals

Lock 3 does not implement:

- multi-sequence CUDA decode batching;
- stochastic GPU sampling;
- pressure-safe persistent CUDA prefix-cache eviction;
- tensor-core-specific packed layouts;
- broad kernel fusion;
- CUDA Graph capture;
- speculative decoding;
- new model architectures or backends.

Those remain outside this lock unless later destructive testing proves one is required for correctness or resource safety.

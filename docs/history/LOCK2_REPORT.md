# Lock 2 report: real CUDA prefill, paged KV, capacity-aware admission

## Objective

Replace three structural limitations measured on the RTX 3080 Laptop validation host without widening AIR's model/backend scope:

1. prompt processing was decode repeated once per token;
2. every CUDA request allocated full-context KV regardless of actual length;
3. request-count admission could overcommit future KV because physical allocation happened before resource ownership was modeled.

## Architectural changes

- CUDA prefill is a native bounded token-matrix path (max width 128), not a loop over `step()`.
- CUDA KV is an executor-owned physical page pool with per-sequence device block tables.
- Decode and prefill attention address pages directly; pages are never flattened into a contiguous compatibility buffer.
- Sequence checkpoints share committed CUDA pages and use copy-on-write for a shared partial tail.
- Capacity admission reserves page-rounded worst-case KV plus page-table bytes while physical pages remain lazy.
- Decode-ready sequences are scheduled before prefill-ready sequences.
- Runtime observability distinguishes prefill execution width from decode execution width and exposes admission and page-pool state.

## Destructive-review decisions

Two initially tempting behaviors were deliberately rejected:

- Physical KV page boundaries do not constrain native prefill batches. Allocation geometry and execution geometry are separate concepts.
- Persistent CUDA prefix reuse is not advertised yet. Page sharing/checkpoint mechanics exist, but enabling a persistent cache before pressure eviction is tested would let cache-only references undermine admission guarantees.

## Current truth

```text
CUDA prefill execution: native-batch, max 128
CUDA decode execution:  width 1
CUDA KV storage:        physically paged
CUDA checkpointing:     implemented
CUDA persistent prefix: disabled pending pressure eviction
CUDA sampling:          host-side full-logit transfer
```

## Validation performed in the packaging environment

- clean CPU/stub Release build;
- six test binaries including direct capacity/decode-first scheduler tests;
- Clang CUDA-language host syntax parse against a minimal CUDA/cuBLAS contract surface;
- stale semantic/name scans;
- process smoke with synthetic GGUF, benchmark, qualification, server/runtime/generate;
- clean GCC, Clang, ASan/UBSan gates before packaging.

The packaging environment has no real NVIDIA CUDA toolkit/device. Real `nvcc`, real CUDA execution, numerical parity, memory scaling, and performance remain mandatory machine gates on the RTX validation host.

## Final packaging gates

The exact source tree packaged for Lock 2 additionally passed:

- GCC Release: six of six tests;
- Clang 17 Release: six of six tests;
- ASan + UBSan Debug with leak detection and halt-on-error: six of six tests;
- fresh synthetic GGUF inspect/benchmark/qualification;
- schema-v2 manifest write/read smoke;
- live `/health`, `/runtime`, and `/generate` smoke;
- no TODO/FIXME/HACK/XXX markers;
- no compiled artifacts outside build directories;
- only intentional schema-v1 `prefill_chunk_tokens` migration references remain.

`scripts/validate-lock2-machine.sh MODEL.gguf` is the canonical real-NVIDIA gate. It collects CUDA/reference decode and native-prefill parity, a 1/2/4/8 concurrency memory/throughput sweep, CUDA qualification, system information, and packages the evidence into `~/Downloads`.

Real `nvcc` compilation and real CUDA execution remain mandatory before Lock 2 is considered machine-qualified. The source package is architecture-frozen pending those hardware gates; failures should be fixed inside Lock 2 rather than deferred into hot-path Lock 3.

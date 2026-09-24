# AIR Release-Candidate Boundary

The release-candidate architecture freeze begins after the reliability destruction phase.

## Frozen architecture

```text
GGUF
  -> ModelDefinition                  canonical state
  -> PreparedModel                    backend-derived state
       -> SequenceState
       -> physical KV ownership

InferenceService
  -> capacity admission               worst-case logical reservation
  -> micro-scheduling                 decode-first + phase round robin
  -> backend execution
  -> bounded stream delivery

Reference executor                    correctness oracle
CUDA executor                         optimized implementation
Qualification                         measured execution evidence
Verification                          opt-in differential observer
```

The following ownership rules are frozen:

1. `ModelDefinition` is the only canonical model definition.
2. Prepared backends may derive execution state but may not duplicate model truth.
3. The scheduler owns request ordering, not transformer math.
4. Backends own physical execution/KV geometry, not transport code.
5. Capacity admission reserves future resource requirements before sequence allocation.
6. Cancellation and shutdown release backend resources before completion becomes externally final.
7. Verification observes the real executors; it is not a second transformer implementation.
8. Benchmarks execute through the production `InferenceService` path.
9. Qualification manifests are evidence and are valid only for matching model/hardware/runtime semantics.

## Release-candidate change rule

After this boundary, comparison testing may trigger:

- correctness fixes;
- reliability/resource fixes;
- measurement fixes;
- documentation corrections.

It may not introduce new execution architecture solely to improve a benchmark. Architectural research moves to `ROADMAP.md` unless a demonstrated defect makes it necessary for correctness or release safety.

## Frozen schema versions

- execution manifest: v3
- benchmark report: `air.benchmark.v2`
- verification report: `air.verification.v1`

## Exit into comparison phase

The RC is ready for controlled comparison when:

- clean GCC and Clang builds pass;
- ASan/UBSan pass;
- CUDA source parses and real NVCC validation passes;
- parser fuzz, compute-sanitizer, serving soak, cancellation, resource reclamation and repeated lifecycle gates pass;
- installed CMake package builds a downstream consumer;
- public protocol rejects unknown/unsupported fields explicitly;
- current docs contain no obsolete lock-stage semantics.

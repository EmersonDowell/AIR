# AIR 0.9 Execution Tactic Contract

This document constrains later performance work. It deliberately does not add a second inference pipeline.

## Core rule

`ModelDefinition` remains canonical model truth.

A tactic is derived execution behavior owned by a `PreparedModel`. A tactic may change representation and execution mechanics, but it may not change model semantics, request semantics, scheduler ownership, KV ownership, or correctness criteria.

## Examples of tactic dimensions

Future CUDA preparation may expose qualified alternatives for:

- decode linear kernel family;
- prefill linear kernel family;
- prepared/repacked weight layout;
- prefill attention implementation;
- decode attention implementation;
- KV storage precision and physical page geometry;
- native prefill width;
- native multi-sequence decode width;
- selected safe fusion groups;
- CUDA Graph capture shape;
- prefix-cache policy.

This is not a commitment to implement every dimension.

## Ownership

A tactic belongs behind the existing prepared-backend boundary:

```text
ModelDefinition (canonical)
        |
        v
PreparedModel (derived)
        |
        +-- tactic A
        +-- tactic B
        `-- tactic C
```

Prepared/repacked tensors are disposable derived state. They must be reconstructible from the canonical model plus tactic identity.

The serving layer must not branch on kernel names, quant formats, or GPU models.

## Capability truth

A prepared backend may advertise only tactics it physically implements for the loaded model and device.

Unsupported combinations are absent, not emulated through hidden fallback.

A selected tactic must never cause an explicit CUDA request to silently execute the reference backend.

## Correctness gate

Performance qualification happens only after correctness qualification.

A candidate tactic must:

1. produce finite outputs;
2. pass the applicable reference/CUDA differential criteria;
3. preserve deterministic greedy tie semantics;
4. preserve KV/checkpoint/resource invariants;
5. pass cancellation, pressure, and reclamation tests relevant to the tactic.

A faster tactic that fails correctness is rejected.

## Evidence

Qualification evidence should eventually identify at least:

- model digest;
- hardware digest;
- AIR version;
- tactic identifier;
- workload identity;
- latency/throughput distribution;
- memory high-water behavior;
- correctness evidence identity;
- measurement uncertainty;
- selection reason.

Do not select a candidate when its apparent advantage is inside measurement uncertainty.

## Objectives

AIR may later qualify against explicit user goals such as:

- interactive latency;
- balanced;
- maximum throughput;
- minimum VRAM.

These are optimization objectives, not alternate inference implementations.

## Rejected designs

Do not create:

- `FastCudaExecutor` beside `CudaExecutor`;
- benchmark-only optimized execution;
- a second model representation that becomes authoritative;
- kernel-specific branches in the serving layer;
- tactic configuration that bypasses capability validation;
- hidden correctness fallback that makes benchmark results uninterpretable.

The existing executor evolves internally. The contracts remain stable unless empirical evidence proves a contract defect.

## Prompt 10 concrete contract

AIR 0.9.1 makes quantized-linear tactic identity explicit in `ExecutionPlan::linear`. CUDA currently advertises `baseline`, `batch-reuse4`, and `batch-reuse8` for prefill, while decode remains baseline-only. Static experiment selection uses the same validated plan path as future adaptive manifests.

Execution manifest schema v4 persists the tactic and benchmark schema `air.benchmark.v3` records it in every evidence report. Tactic identity is therefore evidence, not an out-of-band benchmark flag.

## Prompt 11 extension: attention tactics

AIR 0.9.2 adds `ExecutionPlan::attention` beside the existing linear policy. CUDA advertises `baseline` and `online-softmax` for prefill attention while decode attention remains baseline-only. Attention tactic identity is capability-validated, persisted in execution manifest schema v5, and emitted in benchmark schema `air.benchmark.v4`.

The attention policy changes execution only. It does not own KV state, scheduling, model identity, or request semantics.

## AIR 0.9.8 operation scope and prepared-artifact lifecycle

Prompt 13 rejected `shared-tile8` and `dense-f16-cublas`; their negative evidence remains historical. `dense-f32-cublas` remains a diagnostic/Pareto research tactic for transformer-block linear operations.

Linear tactic identity is operation-scoped rather than phase-global:

- `prefill_block`;
- `decode_block`;
- `decode_output`.

Backend capabilities expose the same three surfaces. Generic CUDA matrix dispatch contains no hidden tensor-name fallback. The terminal vocabulary projection receives its own explicit plan tactic and is currently not eligible for dense FP32 execution.

Optional PreparedModel artifacts are forecast before capacity admission. Only after the prospective artifact plus sequence reservation fits does AIR materialize the plan. While the backend is idle, optional artifacts not required by the incoming plan may be trimmed. This makes high-residency and low-residency strategies reversible rather than process-lifetime side effects.

Preparation time and bytes are evidence. Current runtime/benchmark telemetry records preparation cost and current prepared-artifact residency.

Execution manifest schema v9 requires all operation-scoped tactic fields and benchmark schema `air.benchmark.v9` records them. Missing current-schema tactic identity is rejected rather than silently defaulted.

# AIR Execution Contract

The execution contract is the boundary between planning/scheduling and backend implementation.

## Prepared backend

A prepared backend owns derived execution state for one canonical `ModelDefinition` and reports `BackendCapabilities`.

Capabilities include:

- backend kind;
- prefill execution kind;
- physical KV storage kind;
- maximum native prefill width;
- maximum decode width;
- sequence checkpoint support;
- exact-prefix reuse support;
- device-side deterministic greedy selection;
- supported prefill/decode quantized-linear execution tactics.

Capabilities are factual. A planner may not claim a behavior the prepared backend does not implement.

## ExecutionPlan

An `ExecutionPlan` selects:

- backend identity;
- strategy identifier;
- scheduler prefill quantum;
- physical KV page geometry for paged backends;
- selected prefill and decode quantized-linear tactics.

`validate_execution_plan()` rejects plans incompatible with backend capabilities.

## Sequence contract

`SequenceState` owns one request's backend state and provides:

- prefill returning logits;
- outputless prefill;
- device-greedy prefill when supported;
- decode returning logits;
- device-greedy decode when supported;
- committed resource accounting;
- checkpoint creation.

Serving code does not know the concrete KV implementation.

## CUDA release-candidate capabilities

- native-batch prefill, width up to 128;
- physical paged KV;
- qualified compatible-sequence greedy decode width 8;
- sequence checkpoints;
- deterministic device-side greedy selection;
- persistent exact-prefix serving reuse disabled pending pressure-aware eviction.

## Admission

`estimate_sequence_device_bytes()` computes a logical worst-case reservation for the request's maximum committed length. `sequence_capacity_bytes()` reports the backend sequence-memory budget. Admission happens before sequence allocation.

Physical pages remain demand allocated. Logical reservation and current residency are intentionally separate metrics.

## Evidence manifest

Manifest schema v4 stores typed backend identity, scheduler quantum, physical KV geometry, selected quantized-linear tactics, workload identity, measured means/variance, evidence seed, and conservative selection rationale.

AIR reads and writes schema v4 only. Earlier schemas are rejected and require requalification.

# AIR Strategy Lab — Prompt 14/16

Status: AIR 0.9.11 candidate implementation. Product selection is not qualified until the WolfCat machine workflow returns valid transition evidence.

## Purpose

Strategy Lab chooses an existing `ExecutionPlan`; it does not execute inference. ModelDefinition remains canonical model truth, PreparedModel owns disposable execution artifacts, CapacityScheduler remains resource authority, and InferenceService remains the only serving path.

Prompt 13 established two qualified transformer-block Pareto candidates on the controlled RTX 3080 Laptop / Qwen2.5-0.5B workload: `batch-reuse8` for low residency and `dense-f32-cublas` for higher throughput at an additional 1,431,306,240 prepared bytes. Strategy Lab exists to decide when changing execution state is worth that resource and transition cost.

## Planner designs considered

### Bucket winner

The first design stores one winner per broad workload bucket. It is simple but loses the Pareto frontier, cannot represent a tactic that is optimal only under a particular VRAM budget, and tends to hide transition cost. Rejected.

### Pareto candidate planner

The selected design persists multiple strictly qualified candidates with exact operation-scoped plans, measured performance distributions, confidence bounds, resource costs, transition costs, model/hardware identity, and evidence provenance. At request planning time it filters infeasible/unqualified candidates and evaluates conservative request-horizon economics. Selected.

## Selection inputs

The planner considers:

- strict correctness qualification;
- exact workload region;
- operation-scoped tactic support;
- free device memory and optional prepared-state budget;
- current prepared artifact residency;
- measured incremental preparation and eviction costs;
- expected workload horizon;
- conservative performance bounds;
- whether a candidate's useful artifact is already hot;
- the requested objective.

Objectives are `interactive`, `balanced`, `maximum-throughput`, and `minimum-vram`.

## Transition semantics

Unknown transition cost is not zero. A cold optional candidate is ineligible for automatic product selection until preparation cost has been measured. Switching away from resident optional state is similarly blocked until eviction cost has been measured. A qualification-only probe can bootstrap the first eviction measurement through the normal InferenceService path; that override is not exposed by benchmark/server product CLIs.

A hot prepared artifact naturally removes its preparation term. This creates hysteresis through real state-transition economics rather than timers.

The planner estimates conservative horizon cost using lower confidence bounds for throughput and upper confidence bounds for transition latency. Higher-memory candidates may also be rejected when their measured performance confidence overlaps a lower-memory qualified alternative.

## Evidence model

Execution manifest schema v10 stores multiple `QualifiedStrategy` entries. Every current entry contains:

- exact workload region and `ExecutionPlan`;
- strict-qualification state;
- latency/throughput statistics and confidence half-widths;
- KV/device/prepared bytes;
- preparation and eviction statistics plus separate measured/not-measured flags;
- evidence seed/id;
- model, hardware, AIR and schema identity at manifest level.

Benchmark schema `air.benchmark.v11` records selected-strategy telemetry, actual preparation/eviction time, and the full per-candidate decision trace. Server status and the strategy probe expose the same rejection/eligibility information.

## Qualification workflow

`validate-performance14-machine.sh` performs, in order:

1. real CUDA build plus the full research test suite;
2. exact model/hardware fingerprint;
3. strict reuse8/dense correctness and native-decode equality gates;
4. fresh-process cold dense preparation measurements with no warmup;
5. randomized paired steady-state p54/p262/p1042/c8 evidence;
6. bootstrap manifest creation with dense eviction still unknown;
7. repeated qualification-only dense-to-reuse product-path transitions to measure eviction;
8. final schema-v10 manifest creation;
9. product Strategy Lab probes with the bootstrap override disabled, including minimum VRAM, explicit budget rejection, cold-long dense selection, hot dense reuse, short reuse selection, and final measured eviction;
10. optional Nsight Systems profiling and evidence packaging.

Evidence regions are exact measured prompt/concurrency points. Strategy Lab does not infer unmeasured workload ranges during Prompt 14.

## Product-enabled versus experimental

Product-selectable candidate tactics remain `batch-reuse8` and `dense-f32-cublas`, subject to a valid schema-v10 manifest and strict qualification. `online-softmax` remains the qualified prefill-attention tactic. Decode output remains `batch-reuse8`.

`shared-tile8`, `dense-f16-cublas`, and `q5q8-dp4a-hybrid` are historical research results, not selectable Strategy Lab tactics. The rejected packed/integer research may remain in research-only source to preserve reproducibility.

## Known limitations

Prompt 14 does not generalize evidence outside explicitly measured regions. The concurrent region is recorded from c8 evidence but request-local active-sequence snapshots can vary while a batch is forming; Prompt 15 must destructively test this behavior. The longer-history 1.5B numerical-generalization issue is also intentionally deferred to Prompt 15. Decode-output optimization remains deferred because Prompt 13 measured it as material but not dominant.

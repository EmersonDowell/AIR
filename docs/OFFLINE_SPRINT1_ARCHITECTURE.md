# AIR 0.9.8 Offline Performance Sprint 1

## Purpose

This sprint does not claim a new CUDA speed result. It hardens the execution and resource contracts that AIR Strategy Lab will depend on before additional hardware-native tactics are added.

The trigger was Prompt 13 R3/R4. Dense FP32 proved numerically viable on the 0.5B model, but the first R3 decode qualification exposed that a phase-wide linear tactic could leak into the terminal vocabulary projection even though that operation was intentionally outside the dense prepared representation.

A second source audit found a more serious future Strategy Lab defect: optional PreparedModel state was materialized before capacity admission and was not evictable. A rejected dense request could therefore allocate a large derived arena, permanently reduce free VRAM, and make a later "minimum VRAM" strategy selection false in practice.

## 1. Operation-scoped linear policy

AIR no longer models one universal prefill/decode linear tactic.

`ExecutionPlan::linear` now has three explicit operation surfaces:

- `prefill_block`
- `decode_block`
- `decode_output`

`BackendCapabilities` advertises support separately for the same surfaces.

For the CUDA backend in this sprint:

- prefill block: baseline, reuse4, reuse8, dense-f32-cublas
- decode block: baseline, reuse8, dense-f32-cublas
- decode output: baseline, reuse8

The generic CUDA matrix dispatcher performs exactly the tactic requested by the operation owner. It contains no tensor-name fallback.

This is an important distinction. `output.weight` is not "special because its name is output.weight" inside the kernel dispatcher. It is the terminal vocabulary projection because the model executor says that operation is the terminal vocabulary projection, and its execution plan has its own capability-validated tactic.

## 2. Evidence contract

Execution manifest schema v9 persists:

- `prefill_block_quantized_linear`
- `decode_block_quantized_linear`
- `decode_output_quantized_linear`

These fields are required by the current schema. Missing tactic identity is a data error rather than an implicit baseline.

Benchmark schema `air.benchmark.v9` separately records the same three tactic surfaces and reports current optional prepared-artifact residency.

Canonical CLI names are likewise operation explicit:

- `--cuda-prefill-block-linear`
- `--cuda-decode-block-linear`
- `--cuda-decode-output-linear`

Historical evidence remains historical; the active development surface no longer uses ambiguous phase-wide names.

## 3. Prepared artifacts participate in admission before allocation

`PreparedModel` now separates three operations:

1. forecast incremental plan-preparation device bytes;
2. trim optional artifacts that an incoming plan does not require, only while the backend is idle;
3. materialize required plan artifacts after prospective capacity admission succeeds.

`CapacityScheduler` accounts for prospective prepared-artifact bytes before materialization.

Prepared artifacts must fit genuinely free device memory. Existing free KV-pool pages are valid sequence capacity but are not counted as memory available for an unrelated model artifact.

After materialization AIR rechecks capacity so an external change in free device memory cannot silently create overcommit between forecast and allocation.

## 4. Reversible strategy residency

The CUDA dense FP32 arena can now be released when the backend is idle and the next plan does not require it.

This makes the future choice:

```
low VRAM / quantized
versus
higher VRAM / prepared dense
```

a real reversible resource choice rather than a one-way process-lifetime allocation.

This first implementation deliberately allows artifact trimming only with zero active sequences. AIR does not attempt to invalidate global prepared state underneath live sequences.

## 5. Preparation cost is evidence

Per-request metrics now include:

- `plan_preparation_bytes`
- `plan_preparation_ms`

Runtime snapshots expose:

- `current_prepared_artifact_bytes`

Benchmark summary evidence includes:

- `prepared_artifact_bytes`

This is necessary because steady-state throughput alone is not sufficient for Strategy Lab. A tactic that costs substantial time/VRAM to prepare may be rational for a long-lived model or large batch and irrational for a single short request.

## 6. Why this came before another kernel

AIR's competitive thesis depends on selecting strategies under real constraints. That requires truthful answers to:

- what operation a tactic actually owns;
- what state it materializes;
- how much memory that state costs;
- whether that state can be removed;
- how much preparation costs;
- whether the selected plan is actually admissible before allocation.

Without those properties, an optimizer would be selecting labels rather than executable resource states.

## 7. Dense FP32 resource scale on the current 0.5B model

From the current Qwen2.5-0.5B tensor geometry, the transformer-block rank-2 matrices targeted by dense FP32 preparation represent roughly 1.33 GiB of derived FP32 device state. The terminal output projection would add roughly another 0.51 GiB if independently expanded, which is one reason it remains a separate operation/tactic surface.

The exact device allocation remains machine evidence and should be taken from `prepared_artifact_bytes`, not from this estimate.

## 8. Next research questions

No custom Tensor-Core quantized MMA tactic is authorized by this sprint.

The next offline sprint should establish the numerical contract for lower-precision / integer-MMA candidates before implementation. In particular it should study:

- current GGUF Q5_0, Q4_K, Q6_K and Q8_0 block semantics;
- activation quantization as a separate source of approximation;
- reduction-order error versus quantization error;
- absolute, relative, RMS, top-k and margin-aware equivalence measures;
- model-depth and model-size scaling using the existing 0.5B / 1.5B / 7B evidence;
- criteria fixed before performance numbers are observed.

Only after that should AIR prototype a packed quantized Ampere representation.

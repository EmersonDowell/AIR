# Prompt 15/16 — Competitive Destruction and Generalization

Status: machine qualification pending.

AIR 0.9.12 is an integrity-hardening release of the qualified AIR 0.9.11 adaptive runtime. Prompt 15 does not add an execution tactic or optimize a CUDA kernel. Production execution policy is frozen except for defects discovered by destructive tests.

## Integrity defect discovered before machine collection

The first Prompt-15 characterization test demonstrated that the schema-v10 manifest loader accepted five semantically invalid conditions:

- duplicate `strategy_id` rows;
- duplicate `evidence_id` rows;
- negative performance statistics;
- strict-qualified strategies with no usable performance evidence;
- optional prepared-state strategies without measured preparation evidence.

The pre-fix characterization failed all five cases. AIR 0.9.12 hardens only the manifest semantic boundary. The loader and writer now reject duplicate identities, non-finite/negative measured statistics, missing strict performance evidence, empty required identity fields, and optional prepared-state entries without measured preparation cost. Bootstrap manifests may still carry `eviction_measured=false`; the first eviction can only be established by a real product-path transition.

No executor, scheduler, Strategy Lab scoring rule, attention tactic, linear tactic, KV contract, model representation, or inference path changed as part of the fix.

## Destructive gates

The machine collector executes ten gates in causal order:

1. **Manifest integrity:** corrupted schema/identity/tactic/region/statistical/evidence variants must fail closed.
2. **Evidence-region generalization:** p128/p512/p2048 and c2/c4/c12 probes must not silently inherit exact p54/p262/p1042/c8 evidence.
3. **Memory pressure:** prepared-state budgets are swept around the exact 1,431,306,240-byte dense artifact requirement.
4. **Oscillation/hysteresis:** one `InferenceService` alternates p1042 and p54 for repeated real prepare/evict transitions.
5. **Concurrent dynamics:** concurrency 1/2/4/8/12 is measured with explicit separation between calibrated c8 and out-of-region cases.
6. **Resource stress:** transition latency, prepared bytes, current/peak device bytes and process RSS are tracked over repeated cycles.
7. **Model-scale numerical generalization:** 0.5B/1.5B/7B use a shared longer prompt and teacher-forced history under the unchanged `atol=0.001` gate.
8. **Frozen 0.8 control provenance:** an actual 0.8 artifact is used only if one exists; otherwise archived Prompt-8 evidence is explicitly historical control.
9. **Startup vs transition economics:** fresh process readiness, first-use execution, dense preparation after a reuse8 warm request, and hot dense repetition are measured separately.
10. **External runtime comparison:** local llama.cpp comparison runs only if manifest integrity and current 0.5B correctness survive.

## Stress probe

`air-strategy-probe` now has evidence-only scenarios `oscillation` and `hot-stress`. They reuse the normal `InferenceService` and record every request's strategy decision, preparation/eviction timing, before/after service snapshots, device residency, KV state and process RSS. This is test instrumentation, not a second inference implementation.

## External comparison discipline

`scripts/compare-llama.py` accepts fixed AIR server tactic arguments. Prompt 15 uses this to freeze the AIR profile before each paired lane. Isolated process-VRAM measurement warms each server once so lazy AIR PreparedModel state is included in residency rather than reporting only idle model load.

The external comparator uses only local binaries and identical model bytes. The final analysis must distinguish the adaptive product's exact evidence regions from exploratory fixed-tactic cells.

## Stop rule

The collector writes a preliminary non-promotional summary only. Competitive or production claims are deferred until the returned archive is analyzed. If AIR fails correctness/integrity or cannot establish a repeatable performance/resource/adaptive-runtime niche, Prompt 15 must report the negative result and block a normal public-release claim.

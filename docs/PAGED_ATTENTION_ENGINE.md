# AIR 0.9.2 — Paged Attention Engine Experiment

## Evidence basis

After Prompt 10's `batch-reuse8` prefill-linear tactic, paged prefill attention became the largest measured GPU kernel family on the long-context profile. The baseline prefill attention kernel currently performs QK work in three logical passes: maximum, denominator, and value accumulation.

## Approaches considered

### A. Materialize score tiles and run separate softmax/value kernels

This can improve parallelism but adds a large intermediate score surface and more global-memory traffic. It also moves AIR away from its existing bounded-workspace attention design before evidence shows that materialization is necessary.

### B. One-pass online-softmax paged attention

Keep one block per `(query token, query head)`, preserve paged KV ownership, compute each QK dot once, and maintain a numerically stable running maximum and denominator while rescaling the output numerator online.

This directly attacks the measured redundant QK work with a minimal contract change.

### C. Fully tiled FlashAttention-style query/KV tiles

This is the likely higher-ceiling design, but it is significantly more complex. It should be justified only if the simpler online tactic leaves attention dominant after qualification.

## Prompt 11 first experiment

Prompt 11 implements B as `online-softmax` behind `ExecutionPlan::attention.prefill`.

The baseline attention path remains intact as a correctness/performance control. Decode attention remains baseline-only.

The new tactic must pass the same differential correctness gate before performance collection. Benchmarks fix the prefill linear tactic to `batch-reuse8` so the attention comparison is isolated.

## Ownership

`ModelDefinition` remains canonical. Attention tactics are derived execution behavior. Paged KV ownership and transaction semantics do not change. No second inference executor or benchmark-only attention path is introduced.

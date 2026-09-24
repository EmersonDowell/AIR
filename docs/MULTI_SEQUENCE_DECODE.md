# AIR 0.9 Native Multi-Sequence Decode

## Why this exists

Prompt 8 demonstrated that increasing request concurrency did not improve AIR aggregate CUDA throughput, while llama.cpp gained substantial throughput. AIR's scheduler was concurrent, but the physical CUDA decode width remained one sequence.

Prompt 12 tests whether multiple compatible greedy sequences can share the same model-weight work without changing request, KV, or scheduler ownership.

## Ownership

The serving layer owns grouping only. It may form a bounded batch of compatible ready decode sequences.

`PreparedModel` owns the physical batch capability and validates that the selected backend/tactics support it.

`CudaExecutor` owns actual batched execution.

Each sequence keeps its own:

- `SequenceState`;
- paged KV cache;
- transaction boundary;
- cancellation state;
- output tokens;
- request metrics.

No flattened or synthetic shared sequence is created.

## Initial CUDA tactic

The first native width is bounded to 8 sequences.

For a native batch, AIR batches operations whose model weights are naturally shared across requests:

- embedding load;
- RMS normalization;
- Q/K/V projections;
- attention output projection;
- FFN gate/up/down projections;
- final output projection.

Each sequence still performs position-dependent RoPE and decode attention against its own paged KV history. KV append and commit remain per-sequence transactions.

This is deliberately narrower than a fully fused batched-attention engine. It tests the already-demonstrated hypothesis that concurrency is failing to reuse model-weight work.

## Tactic contract

The control uses:

```text
decode linear = baseline
```

and remains serial at the physical executor.

The candidate uses:

```text
decode linear = batch-reuse8
```

and permits physical decode batches up to backend capability width.

The scheduler does not branch on CUDA kernel names. It only groups requests when the prepared backend advertises native width > 1 and all requests have compatible backend/tactic/sampling semantics.

## Correctness and evidence

`air.benchmark.v5` records the exact generated token IDs for every measured request plus:

- `native_decode_batches`;
- `native_decode_sequences`.

Prompt 12 qualification requires:

1. real NVCC build and complete tests;
2. exact greedy output equality between serial and native modes for paired workloads;
3. physical batch counters > 0 for native concurrency > 1;
4. zero physical-batch counters for the serial control;
5. randomized paired performance evidence;
6. Nsight confirmation that execution geometry changed as claimed.

A scheduler-only throughput change does not qualify as native batching.

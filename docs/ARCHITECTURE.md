# AIR Architecture

AIR separates canonical model state, backend-derived execution state, resource admission, scheduling, serving, and evidence.

## Core pipeline

```text
GGUF
  -> GgufFormat
  -> ModelDefinition
      -> ReferenceExecutor
      -> PreparedModel
          -> SequenceState
          -> backend KV ownership

InferenceService
  -> Planner / ExecutionPlan
  -> CapacityScheduler
  -> MicrobatchScheduler
  -> PreparedModel / SequenceState
  -> sampling / streaming / metrics
```

`ModelDefinition` is the single source of truth for model metadata, tensor descriptors, tokenizer metadata, and mapped model storage. Prepared backends may own transformed execution state such as device residency, workspaces, page pools, and kernel-family classification, but they are derived from the canonical model and never become a second model definition.

## Architecture contract

Qwen2 structural validation is shared in one internal contract. Both reference and CUDA execution consume the same validated model geometry and required tensor names. Executor-specific shape tables are intentionally avoided.

The current execution scope is documented in `SUPPORT_MATRIX.md`.

## Reference path

The CPU reference executor prioritizes inspectability and deterministic numerical behavior over speed. It is the correctness oracle used by CUDA differential verification.

## CUDA path

The CUDA backend owns:

- model residency;
- prepared tensor/kernel-family metadata;
- native multi-token prefill;
- device-side paged KV;
- reusable page pools;
- deterministic greedy selection;
- specialized matrix kernels;
- CUDA execution counters.

Compressed GGUF tensors remain compressed in VRAM. AIR does not materialize a duplicate full-model FP16/FP32 representation.

## Resource ownership

Capacity admission and physical allocation are deliberately separate. Admission reserves the worst-case sequence footprint implied by the request before sequence creation. Physical KV pages are then acquired on demand as tokens commit.

A sequence transaction may write uncommitted state, but token count and externally visible state advance only after successful execution. Checkpoint restoration shares committed pages by reference; a shared partial tail is copied on write.

## Scheduling

The capacity scheduler decides whether a request can safely become active. The micro-scheduler decides which admitted sequence receives work next.

Decode-ready work has phase priority over prefill-ready work. Within each phase, round-robin rotation prevents fixed-slot starvation.

Three quantities are intentionally independent:

```text
scheduler prefill quantum != backend native prefill width != physical KV page size
```

CUDA decode execution width is currently one. Admitting multiple requests therefore provides fairness/interleaving rather than fused multi-sequence execution.

## Serving

`InferenceService` is the single production inference boundary used by the server, benchmark system, and qualification workflow. Transport logic does not own model execution.

Streaming uses bounded per-request delivery queues so slow clients cannot block the inference scheduler. Cancellation is checked before admission and between bounded execution slices. Shutdown cancels queued/active work and joins the scheduler cleanly.

## Evidence

Benchmarks use the production service path. Qualification produces execution manifests from measured paired rounds. Verification observes the real reference/CUDA executors under one teacher-forced token history.

There is no benchmark-only executor and no verification-only transformer implementation.

## Release-candidate invariants

- one canonical model definition;
- one production serving path;
- planner decides, executor executes;
- backend capabilities describe physical behavior only;
- resource ownership is explicit and reclaimable;
- unsupported model/protocol behavior fails explicitly;
- evidence schemas are versioned and not silently reinterpreted.

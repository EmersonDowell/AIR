# Serving and Scheduling

AIR has one production request path while separating transport admission, model-resource admission, execution ordering, and response delivery.

## Transport admission

`air-server` bounds simultaneously accepted handlers with `--max-connections` and applies socket I/O timeouts. HTTP worker count and model active-request count are separate controls.

## Model admission

Before creating a sequence, the capacity scheduler checks:

- active-request limit;
- current logical admission reservations;
- prepared-backend sequence capacity;
- worst-case bytes required for `prompt + max_output`.

This prevents lazy physical allocation from hiding future overcommit.

## Micro-scheduling

Admitted requests are scheduled in bounded cycles. Decode-ready sequences run before prefill-ready sequences to protect already-streaming latency. Phase-local round-robin rotation prevents early slots from monopolizing a small cycle budget.

The scheduler prefill quantum controls how much prompt work one sequence may consume before yielding. It does not define CUDA page size or native prefill width.

## Cancellation

Cancellation is part of the request state, not an I/O side effect. It is checked before admission and between bounded prefill/decode work.

Service shutdown marks queued and active requests cancelled, wakes the scheduler, releases request-owned sequence/admission resources, and joins the worker.

## Streaming backpressure

The inference scheduler writes token events to a bounded per-request queue. The caller/HTTP worker owns potentially blocking delivery. If the queue cannot be drained or a client disconnects, only that request is cancelled.

## Prefix reuse

Reference execution supports serving-level exact-prefix reuse. CUDA checkpoint/page-sharing mechanics exist, but persistent CUDA prefix reuse is disabled until a pressure-aware eviction policy can be validated without weakening admission guarantees.

## Observability

`/runtime` and `/metrics` expose queue/active counts, completion/cancellation/failure totals, token counts, latency percentiles, current/peak KV, current/peak device bytes, admission reservations, page-pool state, and backend execution widths/capabilities.

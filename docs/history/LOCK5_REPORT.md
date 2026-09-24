# Lock 5 Report: Scheduler, Memory, Cancellation, and Failure Destruction

Lock 5 is a reliability lock. It does not add a second inference pipeline or optimize CUDA mathematics. It pressure-tests the production serving path and changes architecture only where destructive tests reveal a correctness, ownership, or bounded-resource defect.

## Findings that required architectural changes

### Cancellation was not a first-class request state

Before Lock 5, a caller could not cancel a queued or active library request and `InferenceService` shutdown drained outstanding generation naturally. Lock 5 introduces `CancellationSource` / `CancellationToken`, a distinct `ErrorCode::cancelled`, cancellation checks before admission and between bounded execution slices, explicit cancellation accounting, and bounded shutdown that cancels queued and active work before joining the scheduler.

Sequence state and admission reservations are released before the request future becomes ready. A caller therefore cannot observe a completed/cancelled request that still owns live logical KV or admission capacity.

### Streaming callbacks could block global inference

Before Lock 5, the scheduler thread invoked the user/network streaming callback directly. A slow callback or socket could stall every active request.

Lock 5 separates inference from delivery. The scheduler only appends decoded stream events to a bounded per-request queue. The request's caller thread drains that queue and performs the potentially blocking callback. Queue capacity is explicit (`stream_queue_capacity`, server `--stream-queue`). If a consumer cannot keep up, AIR cancels only that request rather than blocking global inference or growing memory without bound.

Model execution completion and stream delivery failure are separately observable. `completed_requests` describes completed inference, `cancelled_requests` describes inference cancelled before completion, and `stream_delivery_failures` describes callback/socket delivery failure.

### Decode-first scheduling was not fair under a small cycle budget

Lock 2 correctly prioritized decode over prefill, but stable ordering within the decode phase could starve later sequences when `token_budget_per_cycle` was smaller than the number of ready decoders. Lock 5 retains decode priority while adding phase-local round-robin rotation. The same fairness rule applies to prefill candidates.

### HTTP transport admission was unbounded

The inference scheduler was bounded, but accepted HTTP handlers could accumulate in the transport thread-pool queue. Lock 5 adds an independent `--max-connections` bound. Transport admission and model-resource admission remain separate ownership domains.

Accepted sockets also receive bounded read/write timeouts (`--io-timeout`) so a stalled peer cannot hold a transport worker indefinitely. SIGINT/SIGTERM now take an interruptible accept path, cancel inference through `InferenceService::shutdown()`, drain HTTP workers, and exit cleanly.

## Resource semantics

CUDA KV pages are allocator-owned reusable memory. A completed request must leave:

- zero active scheduler work;
- zero queued scheduler work;
- zero logical admission reservation;
- zero request-owned committed KV;
- no referenced CUDA page that belongs only to the completed request.

The CUDA page pool may retain free pages at its high-water mark. That is allocator cache, not live request ownership. `/runtime` and `/metrics` expose allocated and free pool bytes separately so validation can require `allocated == free` at idle when CUDA prefix caching is disabled.

Persistent CUDA prefix caching remains disabled in Lock 5. Page-sharing checkpoint mechanics are valid, but a pressure-aware cache eviction policy is intentionally not introduced during a reliability prompt.

## Failure policy

- malformed optional execution manifests are reported and fall back to static planning;
- malformed required manifests prevent startup;
- requests that cannot fit device reservation capacity fail before sequence allocation;
- context overflow fails before admission;
- callback/socket disconnect is cancellation/delivery failure, not a silent background generation request;
- service shutdown cancels outstanding work and is idempotent;
- malformed GGUF files may be accepted if a mutation remains valid or rejected with a status, but may not crash or hang the process.

## Destructive validation added

The source test suite now characterizes:

- cancellation before submission;
- cancellation while queued;
- cancellation while active;
- admission/KV reclamation after cancellation;
- bounded service shutdown;
- slow-stream isolation from the inference scheduler;
- bounded stream-backpressure cancellation;
- repeated service create/generate/shutdown cycles;
- malformed optional/required manifest behavior;
- single-request over-capacity rejection;
- phase-local round-robin scheduling under a one-token cycle budget.

`scripts/fuzz-gguf-smoke.py` performs deterministic bounded mutation testing of a small known-good GGUF and fails on crash/signal/timeout.

`scripts/stress-server.py` drives mixed concurrent HTTP requests, malformed API requests, early streaming disconnects, and idle-resource checks.

`scripts/validate-lock5-machine.sh` combines parser fuzzing, compute-sanitizer, manifest destruction, CUDA serving soak, live resource-baseline checks, GPU telemetry, disconnect pressure, and repeated CUDA server start/stop cycles into one evidence archive under `~/Downloads`.

## Lock boundary

Lock 5 does not implement multi-sequence CUDA decode, persistent CUDA prefix-cache eviction, speculative decoding, a new model architecture, or performance tuning. Those are outside this reliability lock.

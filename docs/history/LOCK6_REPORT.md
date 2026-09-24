# AIR Lock 6 Report: Release Candidate Convergence and Architecture Freeze

Lock 6 establishes the first public release-candidate boundary. It is a convergence and deletion pass, not a new inference architecture pass.

## Entry evidence

Lock 5 closed on the NVIDIA RTX 3080 Laptop validation machine with all destructive reliability gates passing:

- 200 malformed-GGUF mutations: 21 accepted, 179 rejected, zero crashes/timeouts;
- compute-sanitizer: zero errors;
- optional corrupt manifest: explicit static fallback;
- required corrupt manifest: startup rejection;
- 96/96 normal CUDA stress requests completed;
- four deliberate disconnects became four cancellations / stream-delivery failures;
- failed requests: zero;
- queued and active requests returned to zero;
- current KV ownership returned to zero;
- admission reservation returned to zero;
- all 6,684,672 allocated KV-pool bytes were free after the run;
- eight of eight CUDA server lifecycle iterations reached healthy state and exited zero.

During sustained load the GPU reached 89 C and was frequently power-limited, but NVIDIA telemetry reported no thermal-violation condition. The reliability results therefore remain valid under a demanding operating state.

## Architecture decision

The core architecture survived the destructive RC review. Lock 6 does not replace:

- `ModelDefinition` as canonical model truth;
- `PreparedModel` as backend-derived execution state;
- capacity admission and the micro-scheduler;
- physical paged KV ownership;
- the reference executor as correctness oracle;
- CUDA as an optimized implementation behind the same execution contract;
- `InferenceService` as the production embedding/serving boundary;
- qualification manifests as measured evidence;
- differential verification as an observer over the real executors.

After this lock, new execution architecture is deferred unless a demonstrated correctness or reliability defect requires it.

## Deleted prototype surface

Lock 6 removes public types and compatibility code that were not meaningful release contracts:

- unused `RuntimeConfig`;
- unused `ServerConfig`;
- unused `RuntimeMetrics`;
- unused abstract `ModelFormat` base;
- legacy `--prefill-chunk` vocabulary;
- manifest schema-v1/v2 migration branches.

Pre-release manifest schemas are explicitly rejected. Requalification is the migration path because manifests are evidence rather than durable user configuration.

## Public protocol hardening

The OpenAI-shaped endpoints are intentionally strict. Unknown fields are rejected rather than silently ignored. The RC adds or locks:

- full `uint64` seed propagation;
- explicit `n=1` support only;
- rejection of conflicting `max_tokens` / `max_completion_tokens`;
- strict chat-message fields (`role`, `content` only);
- HTTP 501 for recognized protocol shape that AIR does not support;
- HTTP 400 for invalid argument combinations.

The optional `model` field is accepted only for single-model client compatibility and does not imply routing.

## Frozen evidence contracts

- execution manifest: schema v3 only;
- benchmark report: `air.benchmark.v2`;
- differential verification report: `air.verification.v1`.

The fast local model fingerprint now incorporates model metadata, tensor geometry, file size, edge samples, and file modification time. It is explicitly non-cryptographic. Public comparison evidence records the model's full SHA-256 separately.

## Build and package integration

The README's canonical build path now exists as `scripts/build.sh`.

Installed AIR exports CMake package targets:

```cmake
find_package(AIR CONFIG REQUIRED)
target_link_libraries(app PRIVATE AIR::core AIR::cuda)
```

A downstream out-of-tree consumer was configured, built, linked, and executed against an isolated installation during the source freeze.

## Documentation convergence

Current operator/developer documentation describes the present runtime rather than the development locks. Historical lock reports and old validators are retained under `docs/history/` and `scripts/history/` for provenance only.

The release-candidate support boundary, public contracts, and deferred research are now explicit in:

- `docs/SUPPORT_MATRIX.md`;
- `docs/PUBLIC_CONTRACTS.md`;
- `docs/RELEASE_CANDIDATE.md`;
- `ROADMAP.md`.

## Source-side freeze gates

The exact final source tree passed:

- GCC Release: 7/7 tests;
- Clang 17 Release: 7/7 tests, no warnings;
- ASan + UBSan + leak detection: 7/7 tests;
- fresh tiny-GGUF inspect/reference/benchmark/qualification;
- benchmark schema-v2 check;
- manifest schema-v3 check and adaptive reload;
- live `/health`, `/runtime`, `/generate` smoke;
- strict unsupported-field HTTP smoke;
- public CLI help/stale-name scan;
- shell and Python syntax checks;
- isolated install plus downstream CMake consumer.

The CUDA translation unit is unchanged from the Lock-5 tree that passed real NVCC, compute-sanitizer, and RTX destruction validation. Lock 6 changes build/export/protocol/evidence identity around it; the RC machine validator therefore repeats the real NVCC/package/runtime authority gate on the target NVIDIA system.

## Post-freeze rule

Comparison testing may still produce correctness, reliability, measurement, packaging, or documentation fixes. It may not introduce new execution architecture solely to win a benchmark. New architecture belongs in `ROADMAP.md` unless evidence shows it is required for release safety or correctness.

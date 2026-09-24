# AIR Seven-Prompt Build Plan

1. **Foundation and contracts - complete**
2. **GGUF, tokenizer, and model representation - complete**
3. **Reference transformer execution - complete**
4. **CUDA execution and memory - complete**
5. **Production runtime, HTTP server, and browser UI - complete**
6. **Scheduler, paged reference KV, exact prefix sharing, and benchmarks - complete**
7. **Qualification, adaptive planner, diagnostics, packaging, and final release audit - complete**

Each phase preserves one canonical model representation and one request-to-execution path. Parsing support, tokenizer support, tensor execution, device execution, paging, and prefix reuse are reported as separate capabilities rather than inferred from one another.

Historical note: Prompt 6 correctly reported CUDA KV as contiguous. The 0.8 Lock-2 work later replaces that allocation with a physical device page pool and native CUDA prefill. This build-plan document describes the original seven-prompt sequence; current semantics are documented in `EXECUTION_CONTRACT.md`, `LOCK2_REPORT.md`, and `LOCK3_REPORT.md`.

## Architecture lock and public-release phase

AIR 0.8 begins a bounded architecture-lock phase driven by real CUDA measurements rather than feature accumulation.

1. **Execution contract and resource architecture - complete**
   - typed backend capabilities and execution plans;
   - derived `PreparedModel` boundary;
   - backend-neutral sequence/checkpoint ownership;
   - capacity admission separated from micro-scheduling;
   - scheduler prefill quantum separated from native prefill execution;
   - manifest schema v2 removes false CUDA paging semantics.
2. **Real CUDA prefill, paged KV, and capacity-aware admission - complete**
   - native multi-token CUDA prefill;
   - executor-owned shared physical CUDA KV pages;
   - demand allocation, copy-on-write checkpoints, and logical admission reservations;
   - decode-first micro-scheduling;
   - separate prefill/decode execution widths and truthful runtime telemetry.
3. **CUDA hot path and evidence/qualification hardening - complete**
   - outputless intermediate prefill chunks;
   - device-side deterministic argmax with token-sized readback;
   - prepared format-specific CUDA dispatch and warp-per-row reductions;
   - reproducibly interleaved paired qualification evidence with variance and Student-t selection gates;
   - manifest schema v3 and benchmark schema v2;
   - corrected percentile interpolation and reproducible qualification seeds.
4. **Differential correctness destruction - complete**
   - one teacher-forced reference/CUDA history with full-logit and top-k evidence;
   - opt-in canonical layer-stage snapshots over the real executors;
   - exact numeric-token reproduction and boundary testing;
   - independent randomized quantization-block oracles and an all-format CUDA execution fixture;
   - raw-token llama.cpp next-token/top-k comparison without chat/template ambiguity.
5. **Scheduler, memory, cancellation, and failure destruction - source complete, hardware qualification pending**
   - first-class queued/active cancellation and bounded service shutdown;
   - bounded per-request stream delivery queues isolated from inference scheduling;
   - decode-first phase-local round-robin fairness;
   - bounded HTTP handler admission and socket I/O timeouts;
   - explicit live-resource, cancellation, and delivery-failure telemetry;
   - parser mutation, manifest-failure, lifecycle, and concurrent serving stress validation.
6. **Release-candidate convergence and architecture freeze**
7. **Controlled llama.cpp correctness and performance comparison**
8. **Scaling matrix**
9. **Evidence report and final release candidate**
10. **GitHub public release**
11. **Superior Mind Labs launch report**

After step 6, new architectural features are deferred unless testing exposes a correctness, safety, or resource-lifetime defect. Performance ideas discovered later are recorded in the roadmap instead of silently expanding the first-release scope.

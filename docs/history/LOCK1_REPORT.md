# AIR Lock Prompt 1 Report

Version line: 0.8.0 development

## Goal

Replace ambiguous execution and resource contracts before optimizing the CUDA hot path. Preserve one canonical model representation and one request-to-execution path.

## Structural changes

- Added typed backend, prefill, and KV capability contracts.
- Added a derived `PreparedModel` boundary. `ModelDefinition` remains canonical.
- Added backend-neutral sequence state, resource accounting, and checkpoint contracts.
- Separated capacity admission from micro-scheduling policy.
- Moved serving prefill through a backend `prefill(span<TokenId>)` boundary.
- Renamed the old prefill chunk concept to a scheduler prefill quantum. It no longer claims native batched execution.
- Made physical KV page geometry conditional on a backend actually using paged KV.
- Migrated execution manifests to schema v2 while retaining a v1 reader.
- Old CUDA v1 `kv_page_tokens` values are discarded during migration because they did not describe physical CUDA paging.
- Qualification no longer compares inert CUDA page geometries for ordinary workloads.
- Runtime capability reporting is derived from the prepared backend rather than inferred from a backend name.

## Verification completed in this environment

- GCC Release CPU/stub build: pass.
- Clang Release CPU/stub build: pass, no compiler warnings found.
- ASan + UBSan Debug CPU/stub build: pass, all five tests green.
- Existing five test binaries: all pass.
- Fresh tiny GGUF inspection: pass.
- Benchmark through `InferenceService`: pass.
- Schema-v2 reference qualification and manifest write: pass.
- Adaptive server manifest reload: pass.
- `/health`, `/runtime`, and `/generate` process smoke: pass.

## Real CUDA validation still required

This environment does not provide `nvcc`, so the refactored 0.8.0 CUDA translation unit was not compiled here. The source retains the CUDA 13 fixes discovered on the target RTX 3080 Laptop machine, including `<math_constants.h>` and the Q5_0 execution support previously validated there. The next gate is a clean CUDA rebuild and the existing parity test on that machine before Lock Prompt 2 begins.

## Deliberate non-goals of Prompt 1

- no physical CUDA paged KV yet;
- no native multi-token CUDA prefill yet;
- no true multi-sequence CUDA batch execution yet;
- no GPU sampling acceleration yet;
- no new model architectures or non-CUDA GPU backends.

Those changes belong behind the truthful contracts established here rather than being mixed into the contract refactor itself.

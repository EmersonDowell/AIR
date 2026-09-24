# AIR Lock 4 Report: Differential Correctness Destruction

Lock 4 turns numerical correctness from a final-logit check into a first-divergence diagnostic system while preserving one production execution path.

## Entry evidence

Lock 3 passed its RTX 3080 Laptop validation. Reference/CUDA decode top-1 parity remained exact across 32 teacher-forced positions with maximum absolute logit error `6.38962e-05`. Native-prefill top-1 also matched with maximum absolute error `1.72615e-04`.

The Lock-3 deterministic hot path performed zero full-logit host readbacks, 64 device-greedy token readbacks, and four outputless prefill chunks. The exact concurrency-4 workload improved over Lock 2 by approximately:

- +33.7% mean prefill tok/s;
- +23.1% mean decode tok/s;
- +27.7% aggregate output tok/s;
- -23.3% P50 TTFT;
- -21.3% P50 total latency;

with essentially unchanged peak device residency. These measurements were recorded while the laptop GPU reached high sustained temperature, so Lock 4 does not reinterpret them as an uncontrolled cooling advantage.

## Changes

### One opt-in verification observer

Reference and CUDA step execution now expose `step_verified(...)`, which invokes the same executor implementation as production with an optional `VerificationTrace`. The observer is read-only. Normal `step(...)` passes no trace.

Stage identity is shared in `air/verification.hpp`; vector comparison and deterministic top-k ranking are centralized in `src/runtime/verification.cpp`.

### `air-verify`

The new diagnostic executable:

- accepts text prompts or exact numeric token IDs;
- teacher-forces reference-selected tokens through both executors;
- records full-logit error and top-k evidence for every decision;
- can trace a narrow range of transitions at canonical internal Qwen2 stages;
- persists versioned `air.verification.v1` JSON;
- never changes the teacher-forced history after a mismatch;
- permits an explicit `--atol` gate but does not invent one by default.

Full stage tracing is range-selectable so diagnostics do not require thousands of unnecessary CUDA synchronizations.

### External llama.cpp evidence

`verify-llama-teacher.py` supplies llama.cpp with exact numeric prompt tokens and one-token raw completions. It records top-k log-probability evidence and the first top-1 divergence. This removes chat-template and retokenization ambiguity from the earlier raw-completion comparison.

### Quantization property tests

Randomized blocks for Q4_0, Q5_0, Q8_0, Q4_K, and Q6_K are checked against independent scalar test oracles.

A small mixed-quant Qwen2 GGUF fixture exercises all five formats together through complete reference/CUDA transformer execution on the validation machine.

### Exact execution boundaries

The validation workflow tests prompt lengths around CUDA KV page and native-prefill boundaries using numeric token IDs, including an expected context-overflow rejection.

## Scope discipline

Lock 4 deliberately does not change kernels, KV geometry, scheduling policy, qualification scoring, sampling, or transport behavior. A correctness defect discovered by the differential tests may still require a Lock-4 patch; benchmark opportunities do not.

## Freeze criteria

Lock 4 can close only after:

- GCC/Clang/sanitizer suites pass;
- randomized quantization properties pass;
- CUDA source compiles with real nvcc;
- main-model internal teacher-forced top-1 parity is preserved;
- all-format quantized fixture reference/CUDA parity passes;
- exact boundary suite passes and context overflow fails explicitly;
- external llama.cpp teacher-forced evidence identifies whether the historical divergence persists and records top-k margins;
- any internal first divergence can be localized to a canonical stage;
- compute-sanitizer passes when available.

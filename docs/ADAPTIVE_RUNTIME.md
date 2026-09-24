# Adaptive Runtime

AIR's adaptive layer selects only between execution plans that correspond to real backend behavior.

## Workload classes

Requests are classified as:

- small: prompt <= 256 tokens;
- medium: 257-2048 tokens;
- large: > 2048 prompt tokens;
- concurrent: more than one active sequence.

## Qualification

Qualification compares candidate scheduler quanta while keeping physical KV geometry fixed for each backend. Candidate order is reproducibly shuffled in paired rounds to reduce systematic thermal/time-order bias.

Each candidate records:

- TTFT;
- total latency;
- prefill throughput;
- decode throughput;
- KV/device high-water marks;
- score mean and sample standard deviation.

Candidate differences are compared against the canonical default with a paired Student-t confidence interval and a minimum meaningful score margin. If the evidence is insufficient or statistically indistinguishable, AIR keeps the canonical default.

A persisted evidence seed reproduces candidate ordering.

## Manifest validity

Execution manifest schema is v4.

A manifest is valid only when these match the current runtime:

- schema version;
- AIR version;
- model fingerprint digest;
- hardware fingerprint digest;
- backend capabilities required by every stored strategy.

Manifest v1-v3 are earlier development formats and are intentionally rejected. Run `air-qualify` to create current evidence.

## Automatic backend behavior

Automatic mode prefers CUDA when a valid CUDA prepared backend exists. If CUDA cannot be prepared, automatic qualification/runtime can fall back to the reference backend. Explicit `--backend cuda` never silently falls back.

## Fingerprint scope

The manifest model fingerprint is intentionally a fast non-cryptographic local identity: it includes model/config/tokenizer/tensor metadata, mapped file size, file modification timestamp, and deterministic edge samples. Public benchmark artifacts should additionally record a full external SHA-256 of the GGUF.

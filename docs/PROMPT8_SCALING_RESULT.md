# AIR Performance Prompt 8/16 Result

Status: **closed with a correctness generalization gate failure**.

Prompt 8 was intended to measure Qwen2.5 Q4_K_M scaling across 0.5B, 1.5B, and 7B models, prompt length, and concurrency. The harness correctly stopped before benchmarking a model that failed AIR's frozen differential-correctness tolerance.

## What completed

The Qwen2.5 0.5B model completed the full 3 x 3 matrix:

- actual prompt lengths: 54, 262, 1042 tokens;
- concurrency: 1, 2, 4;
- three randomized paired AIR/llama.cpp rounds per cell;
- identical prompt-token counts in AIR, AIR CLI, and llama.cpp;
- AIR reference/CUDA verification: 16/16 top-1 matches, finite, worst final-logit max-absolute error 8.16e-4, within atol=1e-3;
- AIR/llama.cpp teacher forcing: 16/16 top-1 matches for this scaling prompt.

Every primary client-measured performance confidence interval excluded parity and favored llama.cpp.

### Client-measured aggregate output throughput

| prompt | concurrency | AIR tok/s | llama.cpp tok/s | AIR / llama |
|---:|---:|---:|---:|---:|
| 54 | 1 | 60.99 | 270.82 | 0.204 |
| 54 | 2 | 60.96 | 405.96 | 0.150 |
| 54 | 4 | 54.08 | 553.39 | 0.097 |
| 262 | 1 | 17.49 | 243.40 | 0.072 |
| 262 | 2 | 16.23 | 352.86 | 0.046 |
| 262 | 4 | 15.25 | 444.87 | 0.034 |
| 1042 | 1 | 3.81 | 182.40 | 0.021 |
| 1042 | 2 | 3.61 | 233.36 | 0.015 |
| 1042 | 4 | 3.47 | 275.41 | 0.013 |

### Concurrency scaling

From concurrency 1 to 4:

- AIR aggregate output throughput changed by 0.887x at 54 prompt tokens, 0.872x at 262, and 0.912x at 1042.
- llama.cpp aggregate output throughput scaled by 2.043x, 1.828x, and 1.510x respectively.

This demonstrates that AIR's scheduler admits and interleaves concurrent work but the current CUDA executor does not convert concurrency into cross-sequence GPU throughput. That is consistent with the factual capability `max_decode_batch_width = 1`.

### Prefill shape

At concurrency 1, AIR's engine-native prefill rate declined as the prompt grew:

- 54 tokens: ~227 tok/s median;
- 262 tokens: ~193 tok/s median;
- 1042 tokens: ~138 tok/s median.

The corresponding llama.cpp server-reported prefill rates increased strongly with prompt size:

- ~5.4k tok/s;
- ~11.9k tok/s;
- ~17.0k tok/s.

The separate native benchmark lane points in the same direction but is not treated as boundary-equivalent: AIR `air-bench` reported ~169 / 147 / 121 prefill tok/s, while `llama-bench` reported ~6.4k / 15.2k / 17.3k prompt tok/s.

This is strong evidence that AIR's prefill implementation is failing to obtain the matrix-matrix / weight-reuse efficiency that llama.cpp obtains as batch width grows. Prompt 9 must profile before assigning causality to a particular kernel.

## Larger-model gate

All three models were structurally readable Qwen2 GGUFs and the one-token execution probes had finite top-1 reference/CUDA parity.

However, the admission probe used top-1 execution success without applying the later fixed atol=1e-3 gate. That evidence-tooling inconsistency is corrected after Prompt 8.

### 1.5B

The full 16-decision verification stopped the matrix before performance testing:

- top-1 reference/CUDA parity: 16/16;
- finite: yes;
- worst final-logit max-absolute error: 9.41e-3;
- requested atol: 1e-3;
- result: FAIL.

The largest error occurred at decision 0, immediately after native prefill. Later teacher-forced decode decisions remained top-1 identical and generally had smaller errors. This is evidence for investigating prefill numerical accumulation separately from decode; it is not yet proof of a specific faulty operation.

### 7B

The matrix never reached the 7B performance run because the 1.5B correctness gate stopped the orchestrator. Its one-decision execution probe did show:

- top-1 reference/CUDA match;
- finite output;
- max-absolute error ~8.30e-3.

Because the admission probe did not apply atol=1e-3, 7B was listed as selected even though that first decision already exceeds the frozen tolerance. No 7B performance claim is permitted from Prompt 8.

## Model-size conclusion

Performance scaling across model size remains **untested**.

What Prompt 8 did establish is a separate and important scale result: AIR 0.8's fixed numerical parity tolerance that passes 0.5B does not currently generalize to the 1.5B Qwen2.5 Q4_K_M workload, and the 7B probe shows a similar magnitude signal.

Do not relax the tolerance after seeing the result. Prompt 9 must first identify where the numerical error enters and whether the fixed absolute threshold is exposing an implementation defect, expected reduction-order scaling, or an inappropriate invariant.

## Resource and thermal interpretation

Prompt 8 does not establish an AIR VRAM advantage. The archive did not retain comparable isolated per-process VRAM values for both engines. AIR did expose exact internal resource state: current KV returned to zero, the reusable CUDA page pool remained owned by the executor, and peak KV reached ~105.5 MB under the completed matrix. Current CUDA KV storage is FP32.

AIR and llama.cpp usually began paired cells at similar temperatures and clocks. AIR's much longer executions then drove the laptop GPU to ~86-87 C and frequently lower ending clocks, while llama.cpp often completed before reaching sustained thermal pressure. Thermal/DVFS effects can amplify AIR's deficit but cannot explain the order-of-magnitude differences: randomized paired rounds, similar starting conditions, and every primary confidence interval all preserve the result.

## Prompt 8 close decision

Prompt 8 is closed as an informative gate failure, not a successful three-model scaling pass.

Demonstrated:

1. the 0.5B performance deficit generalizes strongly across prompt length;
2. it also generalizes across concurrency and worsens relatively under concurrency;
3. AIR's current CUDA path does not gain aggregate throughput from concurrent requests;
4. prefill becomes relatively worse with prompt length;
5. the fixed 1e-3 reference/CUDA numerical gate does not generalize to 1.5B;
6. 7B performance remains untested.

Prompt 9 therefore has two responsibilities before optimization:

- performance attribution on the qualified 0.5B path;
- correctness-scale attribution on 1.5B/7B.

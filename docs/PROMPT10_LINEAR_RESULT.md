# Prompt 10/16 Result — Quantized Linear Engine

Prompt 10 qualified `batch-reuse8` as the current preferred native prefill linear tactic on the tested RTX 3080 Laptop / Qwen2.5-0.5B Q4_K_M workload.

Five paired randomized rounds showed:

- p1042 prefill: 270.35 tok/s reuse8 vs 113.26 tok/s baseline, paired ratio 2.387, 95% CI [2.371, 2.403].
- p262 prefill: 608.11 tok/s reuse8 vs 167.83 tok/s baseline, paired ratio 3.634, 95% CI [3.388, 3.880].

All tested tactics passed the frozen `atol=0.001` correctness gate. Nsight Systems showed quantized matmul GPU time falling from roughly 6.51 seconds under baseline to roughly 1.44 seconds under reuse8 on the long profile. Paged prefill attention then became the dominant kernel family at roughly 57% of GPU kernel time.

Prompt 10 therefore closes because its target bottleneck was materially reduced and another subsystem is now the rational next target. Reuse4 remains an available measured tactic for future hardware qualification, but it is not preferred on the tested machine because reuse8 dominated it in both tested prompt classes.

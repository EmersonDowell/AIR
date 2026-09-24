# Prompt 13 R3 dense-FP32 control result

R3 established that `dense-f32-cublas` is numerically compatible with the frozen
0.001 differential gate for the tested 0.5B model at prefill widths 1, 8, and 64.
The worst observed max-absolute logit error for the dense control was about
4.12e-5 at width 64, with finite logits and exact top-1 parity.

Competitive timing did not run because the subsequent concurrency-4 greedy
decode qualification exposed a tactic-scope defect: native batched decode routed
the terminal `output.weight` vocabulary projection through the globally selected
`dense-f32-cublas` matmul tactic, while `output.weight` was deliberately excluded
from dense prepared state. The run therefore failed with:

`dense FP32 cuBLAS tactic has no prepared matrix: output.weight`

This is not a numerical failure and is not performance evidence. R4 fixes the
scope contract centrally in CUDA matmul dispatch: dense FP32 applies only to the
prepared transformer-block matrices, while token embedding and terminal output
projection remain on the qualified low-residency quantized path.

R3 therefore contributes only numerical evidence. No dense-FP32 competitive
performance claim is valid until R4 passes exact decode qualification and the
paired benchmark/profiler lanes.

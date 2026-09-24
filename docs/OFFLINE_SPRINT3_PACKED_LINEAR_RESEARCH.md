# AIR Offline Research Sprint 3 - Packed Quantized Linear Candidate

Status: AIR 0.9.10 research candidate. Prompt 13 remains open until GPU qualification returns.

## Question

Can AIR preserve canonical GGUF integer weight semantics while changing only derived execution geometry enough to make integer hardware useful on an Ampere consumer GPU?

Sprint 3 deliberately does not claim Tensor Core execution. The first bounded GPU candidate is `q5q8-dp4a-hybrid`, a calibration path built around signed integer dot products using DP4A for Q5_0 and Q8_0 transformer-block matrices. Q4_K and Q6_K remain on the already-qualified `batch-reuse8` path. The terminal vocabulary projection remains separately controlled by `decode_output` and also remains on `batch-reuse8` during this experiment.

## Why DP4A first

Three implementation directions were compared.

1. Direct `mma.sync` / WMMA integer Tensor Core code could offer greater peak throughput, but it couples packing, tile geometry, instruction layout, activation quantization, scale reconstruction, and epilogue design in the first experiment.
2. CUTLASS INT8 GEMM provides mature Ampere primitives, but AIR's GGUF block scales vary inside the K dimension. AIR would still need to own block boundary reconstruction and cannot treat Q4_K/Q6_K as one ordinary dense INT8 matrix.
3. DP4A can consume the same signed Q8 activation codes and preserved GGUF integer weight codes with a much smaller implementation surface. It is therefore the better first hardware calibration of the representation itself.

Sprint 3 chooses option 3. A positive result justifies moving the same representation toward Ampere Tensor Core MMA. A negative result separates representation/activation-quantization issues from MMA integration complexity.

## Prepared representation

Canonical GGUF remains authoritative. Prepared state is disposable and operation-scoped.

For eligible transformer-block Q5_0 and Q8_0 matrices AIR prepares:

- one signed int8 code for every canonical weight value;
- one FP32 canonical block scale per 32 weights;
- no second weight quantization;
- no change to ModelDefinition.

The execution-time activation representation uses 32-value Q8 blocks:

- scale = max(abs(x)) / 127;
- signed int8 activation codes in [-127, 127];
- FP32 scale per 32 values.

The research CPU representation also retains each activation block's integer code sum because a future Q4_K implementation requires it for the affine minimum correction. The bounded Q5_0/Q8_0 CUDA candidate does not compute that unused sum.

## Exact block identities

For Q5_0 and Q8_0:

    w_i = d_w * q_i
    a_i ~= d_a * r_i

    dot ~= d_w * d_a * sum(q_i * r_i)

The only new approximation in this path is activation Q8 quantization. Original GGUF weight codes and block scales are preserved.

Sprint 2 already proved the Q4_K and Q6_K identities needed for later expansion. They are deliberately not folded into this first kernel because their scale/minimum epilogues are a separate implementation variable.

## Integer accumulator safety

The CPU property suite fixes conservative per-group worst-case bounds before GPU timing:

- Q5_0 x Q8 activation, K=32: 65,024
- Q8_0 x Q8 activation, K=32: 520,192
- centered Q4_K x Q8 activation, K=32: 32,512
- Q6_K x Q8 activation, K=16: 65,024

All are far below signed INT32 capacity. The GPU candidate performs one INT32 DP4A block dot at a time and converts the block partial to FP32 before applying canonical scales and accumulating across K groups.

## CPU/property evidence

Research mode builds `air-packed-quantized-tests` and `air-packed-model-check`.

The property suite validates thousands of randomized cases for:

- Q8 activation code/sum construction;
- Q5_0 packed code reconstruction;
- Q8_0 packed code reconstruction;
- Q4_K affine centered-code identity;
- Q6_K per-16 scale identity;
- accumulator bounds.

The generated 896-wide Qwen-like mixed-quant fixture produced:

- packed-vs-Q8-dequant max absolute error: 7.406370699e-07
- packed-vs-Q8-dequant mean absolute error: 1.14042546e-07
- packed-vs-Q8-dequant RMS error: 1.811417431e-07
- Q8-activation-vs-float max absolute error: 0.1187910438
- Q8-activation-vs-float mean absolute error: 0.01516509244
- Q8-activation-vs-float RMS error: 0.02476399779

The important separation is that packing/reconstruction error is negligible. Activation quantization is the material numerical risk and must be judged by the unchanged end-to-end product gate.

The full CPU/research build passes 9/9 tests.

## Prepared residency

For the inspected Qwen2.5-0.5B model, eligible Q5_0/Q8_0 transformer-block weights contain approximately 253.2 million values.

The prepared hybrid arena is estimated at approximately 271.7 MiB including signed codes, per-block FP32 scales, and bounded activation workspace. Relative to the already-resident compressed Q5_0/Q8_0 GGUF storage, incremental duplicate residency is approximately 105 MiB.

By contrast, the dense FP32 transformer-block calibration tactic is roughly 1.3 GiB of derived matrix state on this model. Exact device residency is still measured by the machine collector rather than assumed from these estimates.

Prepared state participates in the Sprint-1 lifecycle contract:

- prospective bytes are forecast before materialization;
- admission sees the prospective cost;
- state is materialized only after admission;
- unused optional artifacts may be trimmed while idle;
- preparation bytes/time and current prepared residency are observable.

## Operation scope

`q5q8-dp4a-hybrid` is intentionally not universal.

Eligible surfaces:

- prefill_block
- decode_block

Excluded surface:

- decode_output

Within eligible block-linear operations:

- Q5_0 and Q8_0 use the new prepared DP4A path;
- Q4_K and Q6_K deliberately fall back to the qualified `batch-reuse8` kernel.

The tactic name contains `hybrid` so persisted evidence cannot be confused with full-format coverage.

Core tests explicitly reject this tactic on `decode_output`.

## GPU qualification contract

The consolidated WolfCat collector fixes all correctness gates before timing.

Product gate remains unchanged:

- finite outputs;
- exact greedy top-1;
- maximum absolute logit error <= 0.001.

The candidate must first pass 0.5B differential verification at token widths 1, 8, and 64. Width 64 is the hard competitive gate.

It must then pass native concurrency-4 qualification:

- exact generated token equality versus `batch-reuse8`;
- physical native batching > 0;
- `decode_output` remains `batch-reuse8`;
- prepared artifact residency > 0.

Only strict-qualified tactics enter competitive timing.

Randomized paired lanes are:

- p262 prefill;
- p1042 prefill;
- concurrency-8 native decode.

Controls/candidates are:

- `batch-reuse8` low-VRAM qualified control;
- `dense-f32-cublas` high-VRAM mature-GEMM calibration if R4 qualification passes;
- `q5q8-dp4a-hybrid` low-memory integer candidate if strict qualification passes.

Nsight Systems is collected for every qualified tactic. Nsight Compute is optional and diagnostic only.

## Decision after GPU evidence

The DP4A candidate is not retained just for passing correctness.

Retain/escalate only if:

1. strict correctness passes;
2. exact native decode behavior passes;
3. paired performance shows useful value or profiler evidence reveals a clear next hardware bottleneck;
4. its preparation and VRAM cost are acceptable;
5. profiler attribution confirms the intended path actually executed.

Possible outcomes:

- If DP4A materially wins, keep the representation and investigate moving its integer partial products to Ampere Tensor Core/CUTLASS MMA.
- If DP4A is numerically sound but slower, use profiler evidence to decide whether the primitive is the limitation. The representation may still justify an MMA implementation.
- If activation quantization fails the strict numerical gate, do not time or promote it. Use Sprint-2 research classification only diagnostically. Product qualification remains unchanged.
- If dense FP32 has large speed headroom while DP4A remains weak, that strengthens the case for a more serious hardware-native packed MMA engine.
- If mature dense GEMM has little headroom, do not build a complex MMA engine merely because the hardware supports it.

Prompt 13 closes only after these hardware-native linear questions have a defensible evidence-backed answer.

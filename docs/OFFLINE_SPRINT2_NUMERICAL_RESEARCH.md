# Offline Research Sprint 2/3 — Numerical Equivalence

AIR version: 0.9.9

Prompt 13/16 remains open. No new CUDA performance claim is made in this sprint.

## Purpose

Sprint 2 establishes the numerical science required before AIR attempts a lower-precision Ampere integer-MMA tactic.

AIR now contains a research-only numerical harness (`AIR_BUILD_RESEARCH=ON`) that loads actual GGUF fixtures through AIR's production GGUF loader and uses `ReferenceTensorReader` to obtain AIR's real Q5_0/Q4_K/Q6_K/Q8_0 decoding semantics. The experiment changes arithmetic only after the canonical GGUF tensor has been decoded.

This is important: the study measures execution-induced error, not the unknown quantization error between the GGUF and an unavailable upstream full-precision model.

## Methods tested

The CPU experiment separates:

- double-accumulation oracle;
- sequential FP32 accumulation;
- reverse-order FP32 accumulation;
- pairwise FP32 reduction;
- 32-lane warp/tree-style FP32 reduction;
- FP16 weight conversion only;
- FP16 activation conversion only;
- FP16 weights + activations;
- TF32-style 10-bit mantissa rounding;
- blockwise symmetric Q8 activation quantization;
- Q8_1-like activation quantization with FP16-stored activation scale;
- a deliberately pessimistic Q8xQ8 re-quantization proxy.

The Q8xQ8 proxy is not the proposed AIR MMA design because it requantizes already-canonical GGUF weights and therefore adds an unnecessary error source. It is retained as a pessimistic comparison.

## Width scaling

Fixtures were generated at inner dimensions representative of Qwen2.5 model-scale reductions: 256, 896, 1536, and 3584. K-quant research tensors use the next block-aligned width where required by their canonical block geometry.

Worst sampled absolute dot error:

| arithmetic | 256 | 896 | 1536 | 3584 |
|---|---:|---:|---:|---:|
| FP32 warp/tree | 2.54e-6 | 4.04e-6 | 7.49e-6 | 1.56e-5 |
| FP32 sequential | 1.21e-5 | 2.23e-5 | 1.55e-5 | 4.79e-5 |
| FP16 both | 4.96e-3 | 9.51e-3 | 1.23e-2 | 1.71e-2 |
| TF32-style both | 4.96e-3 | 9.51e-3 | 1.23e-2 | 1.71e-2 |
| Q8_1-like activations | 9.06e-2 | 1.60e-1 | 4.27e-1 | 3.69e-1 |
| Q8xQ8 proxy | 1.51e-1 | 2.53e-1 | 3.89e-1 | 4.19e-1 |

Worst normalized RMS dot error:

| arithmetic | 256 | 896 | 1536 | 3584 |
|---|---:|---:|---:|---:|
| FP32 warp/tree | 9.52e-8 | 1.36e-7 | 1.86e-7 | 2.57e-7 |
| FP32 sequential | 3.43e-7 | 8.18e-7 | 9.62e-7 | 1.37e-6 |
| FP16 both | 3.14e-4 | 3.73e-4 | 4.16e-4 | 3.01e-4 |
| Q8_1-like activations | 7.09e-3 | 6.40e-3 | 1.14e-2 | 6.74e-3 |
| Q8xQ8 proxy | 7.99e-3 | 8.05e-3 | 1.14e-2 | 8.39e-3 |

### Interpretation

1. **FP32 reduction order is small at the isolated dot level.** Even at width 3584, the sampled warp/tree error is ~1.6e-5 and sequential FP32 ~4.8e-5. Therefore the previously observed ~0.008-0.009 larger-model end-to-end CUDA/reference drift cannot be explained by a single long FP32 dot product alone. Depth, repeated residual operations, attention, normalization, or another implementation difference must contribute.

2. **FP16 magnitude matches the failed R2 experiment.** FP16 conversion produces isolated-dot errors in the 1e-2 range at larger widths, consistent in scale with dense-f16-cublas failing at ~0.0161/~0.00918. This strengthens the conclusion that R2 was a precision problem rather than evidence against mature GEMM geometry.

3. **Activation Q8 is the principal numerical risk for an integer MMA design.** Direct Q8_1-like activation quantization yields roughly 0.6%-1.1% normalized RMS dot error in these fixtures. An integer dot itself can be exact in int32; activation quantization and scale representation are the approximation.

4. **TF32 and FP16 happened to coincide in these bounded fixture samples because both retain roughly 10 mantissa bits and the sampled values did not exercise FP16 exponent-range failures. They are not generally equivalent formats.**

## Residual-depth experiment

The research harness repeatedly applies an actual AIR-decoded square quantized matrix through a bounded residual/nonlinear chain. This is not a transformer simulation, but it demonstrates error accumulation without GPU effects.

At width 3584 after 12 repeated residual applications:

- FP32 warp/tree L-infinity state error: ~7.45e-8;
- FP16 both: ~9.15e-5;
- Q8_1-like activations: ~6.19e-4;
- pessimistic Q8xQ8 proxy: ~2.11e-3.

The experiment shows why a per-dot error that looks modest can accumulate toward or through a strict final-state bound. It does not justify changing AIR's logit threshold.

## Mathematically honest integer mapping

Sprint 2 adds property-tested integer identities for the canonical GGUF code structure.

### Q5_0 / Q8_0

For one block:

`w_i = d_w * q_i`, `a_i = d_a * r_i`

therefore:

`dot = d_w * d_a * sum(q_i * r_i)`.

If AIR preserves the original GGUF integer codes, no new weight requantization is required.

### Q4_K

For a 32-value subgroup:

`w_i = d * s * q_i - d_min * m`

and Q8 activation `a_i = d_a * r_i`:

`dot = d_a * (d*s*sum(q_i*r_i) - d_min*m*sum(r_i))`.

Thus a mathematically honest packed path must preserve the activation block sum in addition to Q8 codes. This is consistent with the reason Q8_1-style intermediate representations carry a block sum. A signed-centered `q_i - 8` form is algebraically equivalent and can make signed integer MMA practical while retaining an affine sum correction.

### Q6_K

For each 16-value subgroup:

`w_i = d * s_g * q_i`

so:

`dot = d * s_g * d_a * sum(q_i*r_i)`.

The per-subgroup scale must be applied before combining K groups.

Thousands of randomized CPU property cases verify these identities against direct dequantized arithmetic.

## Implication for Ampere MMA design

Ampere exposes signed/unsigned INT8 MMA with K=16/K=32 instruction shapes. AIR's quantization boundaries are unusually compatible with those K dimensions:

- Q5_0/Q8_0 blocks: 32 values;
- Q4_K scale/min subgroups: 32 values;
- Q6_K scale subgroups: 16 values.

That does **not** mean one monolithic INT8 GEMM over the full K dimension is correct. The GGUF scales vary across blocks, so AIR must obtain integer partial tiles at the quantization-scale boundary, apply the canonical scale/offset correction, and accumulate those partial results into FP32 output state.

This is the leading Sprint 3 design hypothesis.

## Prior art implication

llama.cpp's CUDA MMQ path quantizes activations into Q8_1-like representations and provides dedicated Q4_K/Q6_K-to-Q8_1 dot paths rather than treating K-quants as a generic dense INT8 matrix. This supports AIR's conclusion that the right abstraction is block-aware integer partial products plus scale reconstruction, not blind conversion of GGUF into a generic INT8 GEMM.

## Numerical contract outcome

`docs/NUMERICAL_EQUIVALENCE_RESEARCH.md` freezes the proposed research-only classification before Sprint 3 performance exists.

The existing release gate remains untouched.

A future lower-precision tactic cannot become product-selectable without passing the same strict release gate. A separate `research-margin-certified` category is defined only to reason about non-strict experiments using a mathematically grounded top-1 margin certificate; it is not enabled in product behavior.

## Strategy Lab outcome

Sprint 2 also freezes a conservative transition model. Strategy Lab must account for tactic preparation/eviction cost and expected workload horizon rather than selecting solely by steady-state throughput. Prepared packed weights in Sprint 3 must therefore expose preparation bytes/time and be safely evictable; otherwise the adaptive runtime would not have truthful resource choices.

## Sprint 3 target

Sprint 3 should prototype one prepared quantized representation that:

1. keeps canonical GGUF bytes authoritative;
2. reorders original integer codes without requantizing them;
3. creates Q8_1-like activation tiles with explicit scale/sum semantics;
4. aligns Q5_0/Q8_0/Q4_K work around 32-value K groups and Q6_K around 16-value groups;
5. accumulates integer partial products exactly in int32;
6. applies GGUF block scales/minimum corrections into FP32 accumulation between K groups;
7. exposes preparation residency and preparation latency through the existing PreparedModel lifecycle;
8. remains capability-validated and operation-scoped;
9. is property-tested on CPU before a CUDA kernel is authorized.

The final CUDA mapping may use Ampere MMA/WMMA/CUTLASS primitives, but the mathematical representation is owned by AIR, not by a technology name.

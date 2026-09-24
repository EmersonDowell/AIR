# AIR Numerical Equivalence Research Contract

Status: research-only, AIR 0.9.9.

This document does not change AIR's release gate. The production qualification rule remains finite outputs, exact greedy top-1 agreement, and max-absolute logit error <= 0.001 on the required differential histories.

## 1. What is canonical

For a GGUF runtime, canonical model truth is the tensor represented by the GGUF bytes and AIR's validated GGUF decoding semantics. The pre-quantization full-precision model is generally unavailable and therefore its quantization error cannot be reconstructed from the GGUF alone.

AIR numerical research therefore separates:

1. model-format quantization already encoded in canonical GGUF weights;
2. execution-induced arithmetic error relative to those canonical weights;
3. decision-level effects in logits/token selection.

Only (2) and (3) are execution-runtime responsibilities.

## 2. Strict release equivalence remains unchanged

A decision is `release-strict` only when:

- all candidate outputs are finite;
- the candidate top-1 token equals the oracle top-1 token;
- max absolute logit error is <= 0.001.

A tactic that fails this gate is not a qualified runtime tactic. Sprint 2 does not relax this rule.

## 3. Research-only margin evidence

For diagnostics only, define:

- epsilon = observed max absolute logit error;
- m = oracle top-1 minus oracle second-place logit margin;
- rho = m / (2 * epsilon).

If m > 2*epsilon, then no independent perturbation bounded by epsilon in L-infinity can reverse that observed two-class top-1 decision. AIR calls rho the `greedy safety ratio`.

Research classification:

- `release-strict`: existing production gate passes;
- `research-margin-certified`: strict gate fails, top-1 is still exact, all values are finite, and rho >= 2.0;
- `research-inconclusive`: top-1 matches but rho < 2.0;
- `rejected`: non-finite result or any top-1 mismatch.

The rho >= 2.0 research threshold is deliberately twice the mathematical stability boundary. It is an a-priori safety factor for research classification, not a product tolerance.

`research-margin-certified` does not make a tactic selectable by Strategy Lab and does not permit public performance claims as if the tactic were release-qualified. It only allows explicitly labeled experimental investigation if a future research plan opts into that policy.

## 4. Near-tie adversarial rule

Matching top-1 alone is insufficient. An adversarial near-tie suite is required for any future lower-precision study.

Property tests in `research/numerical_methods_tests.cpp` verify:

- margins strictly greater than 2*epsilon survive worst-case opposing L-infinity perturbations;
- margins below 2*epsilon can be flipped by such perturbations;
- exact top-1 mismatches always reject regardless of average/RMS error.

This prevents a low average error from hiding a meaningful token-decision instability.

## 5. Diagnostic metrics

The research harness records:

- max absolute error;
- mean absolute error;
- RMS absolute error;
- normalized RMS error relative to oracle output RMS;
- repeated residual-chain L-infinity and RMS state error;
- top-1 margin safety evidence.

Normalized/RMS metrics are diagnostics, not substitutes for the strict logit gate.

## 6. Lower-precision tactic prerequisites

Before any lower-precision hardware tactic can become product-qualified it must:

1. preserve canonical GGUF storage semantics;
2. pass format-level property tests for every supported quant type;
3. remain finite;
4. preserve exact greedy token trajectories on designated verification histories;
5. pass the existing strict `atol=0.001` gate;
6. pass adversarial near-tie tests;
7. expose its prepared-resource state and preparation cost;
8. only then enter paired performance qualification.

If baseline CUDA itself fails strict equivalence on a model size, lower-precision tactics cannot be product-qualified on that model until the baseline discrepancy is independently resolved. The larger-model 1.5B/7B issue therefore remains a separate correctness research problem.

## 7. A-priori research benchmark rule for Sprint 3

Sprint 3 may build an experimental packed-integer candidate, but performance evidence must be clearly separated into two classes:

- `qualified`: strict release equivalence passes before timing;
- `research-only`: only if a separately explicit experiment enables it, every generated token is exact and every non-strict decision is `research-margin-certified` with rho >= 2.0.

Any `research-inconclusive` or rejected decision blocks research timing for that candidate/history. Product selection continues to require `qualified` status.

This rule is fixed before the future MMA benchmark and must not be changed after seeing its speed.

# AIR Prompt 10/16: Quantized Linear Engine

Prompt 9 demonstrated that quantized linear kernels dominate the measured CUDA workload. Prompt 10 changes execution only behind the existing prepared CUDA backend.

## Designs considered

### A. Expanded FP16/FP32 prepared weights + cuBLAS

Advantages:

- immediately accesses mature GEMV/GEMM implementations;
- likely strong prefill throughput;
- useful future tactic for memory-rich devices.

Costs:

- multiplies resident weight memory relative to GGUF quantization;
- weakens AIR's consumer-GPU memory thesis if made the default;
- changes the performance question from quantized execution to expansion/residency.

This remains a legitimate future tactic, not the first implementation.

### B. Native GGUF batch-reuse kernel — selected first

Keep quantized GGUF bytes resident. For prefill, assign a warp to one output row and a small tile of prompt items. Decode each quantized weight once and accumulate it into 4 or 8 item sums before advancing to the next weight.

The baseline kernel effectively performs:

```text
for item:
    for output_row:
        for input_col:
            decode weight
            accumulate(weight * input[item,col])
```

The candidate performs the equivalent logical work as:

```text
for item_tile:
    for output_row:
        for input_col:
            weight = decode once
            for item in item_tile:
                accumulate[item](weight * input[item,col])
```

This is intentionally a small first tactic. It tests the measured reuse hypothesis without introducing a second model owner or a large prepared-weight allocation.

## Current tactic contract

`QuantizedLinearExecutionKind` currently exposes:

- `baseline`
- `batch-reuse4`
- `batch-reuse8`

CUDA advertises the three tactics for prefill and only `baseline` for decode. Reference execution advertises baseline only.

The selected tactic lives in `ExecutionPlan::linear`. Static CUDA tests may request it with `--cuda-prefill-linear`; adaptive strategy selection will consume the same contract later rather than inventing another control path.

Execution manifest schema v4 persists the tactic. Benchmark schema v3 records the tactic and planned execution geometry so an evidence file is self-describing.

## Qualification rule

A candidate is admitted to performance comparison only after differential verification passes. Prompt 10 initially qualifies against the already-qualified Qwen2.5 0.5B model. Larger-model claims remain blocked by the separate numerical-equivalence question identified in Prompt 9.

Performance collection uses randomized candidate order and preserves raw per-round evidence. The baseline remains available and unchanged.

## Rejection rule

If batch reuse is slower, unstable, or correctness-invalid, remove it rather than carrying dead tactics forward. Prompt 10 may then test the expanded/cuBLAS design or another profiler-justified linear tactic behind the same contract.

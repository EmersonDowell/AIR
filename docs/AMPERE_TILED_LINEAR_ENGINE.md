# Ampere hardware-native linear research

Prompt 13 evaluates how AIR should use the RTX 3080 Laptop's Ampere matrix hardware without sacrificing AIR's ownership and qualification contracts.

## Shared-memory tiling result

`shared-tile8` was the first geometry-only candidate. It preserved canonical GGUF bytes, direct dequantization, float accumulation, exact native-decode outputs, and the frozen differential gate. It nevertheless lost in every paired lane:

- p1042 prefill ratio 0.9657 [0.9516, 0.9799]
- p262 prefill ratio 0.9509 [0.9337, 0.9680]
- c8 decode aggregate ratio 0.9131 [0.9026, 0.9236]

The candidate is removed. Shared-memory staging and synchronization cost more than the activation-load reuse saved for this layout.

## Why not jump directly to custom INT8/INT4 MMA?

AIR's execution model contains mixed GGUF formats (Q5_0, Q8_0, Q4_K, Q6_K on the current test model), each with block scaling semantics. A mathematically honest integer-MMA path requires prepared packing, activation quantization, block-scale handling, and explicit numerical qualification. That is substantial code.

Before committing to it, AIR should establish whether mature matrix hardware offers a large enough benefit.

## Dense FP32 cuBLAS calibration

AIR 0.9.6 therefore tests `dense-f32-cublas` first. It:

- keeps ModelDefinition/GGUF canonical;
- lazily expands rank-2 execution matrices into FP32 PreparedModel state;
- keeps batched activations in FP32;
- uses pedantic FP32 cuBLAS GEMM;
- records the resulting device-memory increase;
- keeps scheduler, KV, attention, sequence ownership, and service path unchanged.

The tactic name does not assert Tensor Core execution. Nsight must show which cuBLAS kernels actually ran.

The candidate must pass the existing `atol=0.001` differential gate before it can enter competitive paired timing. If it fails, AIR will not change the gate retroactively.

If this tactic demonstrates a large qualified speedup, the next decision is whether its VRAM cost is acceptable as a selectable high-throughput tactic and whether a lower-memory packed integer-MMA tactic is justified.

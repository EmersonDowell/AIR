# Prompt 13 shared-tile8 result

Prompt 13 first tested whether shared-memory activation tiling alone could improve AIR's qualified `batch-reuse8` quantized linear tactic without changing quantization semantics or arithmetic precision.

The candidate passed correctness and preserved exact native-decode outputs, but it lost performance in every measured lane on the RTX 3080 Laptop:

- p1042 prefill: shared-tile8 / reuse8 = 0.9657, 95% CI [0.9516, 0.9799]
- p262 prefill: shared-tile8 / reuse8 = 0.9509, 95% CI [0.9337, 0.9680]
- concurrency-8 decode aggregate throughput: shared-tile8 / reuse8 = 0.9131, 95% CI [0.9026, 0.9236]

Peak device memory was unchanged. Nsight showed the candidate's quantized linear kernels taking more total GPU time rather than less. The extra synchronization/shared-memory staging did not pay for itself for this geometry.

`shared-tile8` is therefore rejected and removed from the live tactic vocabulary. The experiment remains documented as negative evidence.

The next Prompt 13 iteration uses `dense-f32-cublas` as a deliberately memory-heavier calibration tactic. It expands execution matrices into derived FP32 prepared state and executes batched linear work through cuBLAS. This tests mature dense-GEMM geometry without conflating it with FP16/Tensor-Core precision before AIR commits to a substantially more complex packed integer-MMA implementation.

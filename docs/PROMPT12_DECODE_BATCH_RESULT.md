# Prompt 12 Result: Native Multi-Sequence Decode

AIR 0.9.3 converted scheduler concurrency into bounded physical CUDA decode batches while preserving independent request, sequence, KV, cancellation, and metric ownership.

On Qwen2.5-0.5B Q4_K_M with prefill fixed to `batch-reuse8` + `online-softmax`, five randomized paired rounds showed:

| Concurrency | Serial aggregate tok/s | Native aggregate tok/s | Native / serial | 95% CI |
|---:|---:|---:|---:|---:|
| 1 | 93.038 | 96.216 | 1.0395 | [0.9046, 1.1744] |
| 2 | 91.143 | 113.375 | 1.2439 | [1.2378, 1.2500] |
| 4 | 91.067 | 191.487 | 2.1028 | [2.0777, 2.1278] |
| 8 | 91.613 | 289.705 | 3.1624 | [3.1221, 3.2028] |

Every paired output-token sequence matched exactly. Concurrency greater than one recorded native physical batches while the serial control recorded none. At concurrency eight, mean p50 total latency fell from about 5.54 s to 1.75 s and p95 total latency from about 5.58 s to 1.77 s. Device and KV high-water values were effectively unchanged.

The per-request decode-tokens/second field decreases under native batching because several sequences share one physical execution interval. Aggregate generated throughput and request latency are the relevant service metrics for this experiment.

Nsight Systems at concurrency eight showed approximately 2.27 s of serial GPU-kernel time versus 0.79 s under native batching. CUDA kernel launches fell from about 124k to 45k. The serial path was dominated by quantized matvec kernels. The native path converted that work to `batch-reuse8` matmul, which became roughly 72% of native GPU-kernel time. This moves the next performance target back to quantized linear execution under hardware-native batch geometry.

GPU telemetry reached 88 C and 114 W. During samples with >=90% SM utilization, mean SM clock was about 1.40 GHz with a range around 1.11-1.73 GHz. Power-limit activity was present; the thermal-violation flag remained zero. Randomized paired ordering and very large confidence-separated effects make the native-batching conclusion robust to those conditions.

## Decision

Native multi-sequence decode is retained. The evidence establishes that physical batching, not scheduler concurrency alone, is necessary for AIR to convert concurrent requests into GPU throughput.

Prompt 13 should optimize the now-dominant batched quantized linear geometry. Attention, KV precision, and CUDA Graph work are not the first target after this profile.

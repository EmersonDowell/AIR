# AIR Prompt 12/16 Decode Batching Analysis

## Conclusion

Prompt 12 is closed. Native multi-sequence decode earned retention.

On Qwen2.5-0.5B Q4_K_M, AIR converted scheduler-level concurrency into real physical CUDA batching while preserving exact greedy outputs, independent sequence/KV ownership, and essentially unchanged device/KV high-water memory.

## Paired randomized evidence

Five paired rounds were run at each concurrency level.

| Concurrency | Serial aggregate tok/s | Native aggregate tok/s | Native / serial | 95% CI | Physical batch seen | Exact outputs |
|---:|---:|---:|---:|---:|:---:|:---:|
| 1 | 93.038 | 96.216 | 1.0395 | [0.9046, 1.1744] | no | yes |
| 2 | 91.143 | 113.375 | 1.2439 | [1.2378, 1.2500] | yes | yes |
| 4 | 91.067 | 191.487 | 2.1028 | [2.0777, 2.1278] | yes | yes |
| 8 | 91.613 | 289.705 | 3.1624 | [3.1221, 3.2028] | yes | yes |

Concurrency one is correctly inconclusive because no physical batch can exist there. At concurrency two, four, and eight the confidence intervals exclude parity strongly.

The serial control recorded zero native physical batches in every run. The native candidate recorded deterministic batch evidence:

- c2: 252 native batches / 504 batched decode-sequence participations per benchmark;
- c4: 126 / 504;
- c8: 64 / 504.

The 504 sequence participations correspond to 8 requests x 63 native decode steps after the first token is obtained from prefill.

## Latency

Native batching improves service latency as concurrency grows.

At concurrency eight:

- serial p50 total: ~5.54 s;
- native p50 total: ~1.75 s;
- serial p95 total: ~5.58 s;
- native p95 total: ~1.77 s.

TTFT is comparable and slightly better for native batching on average. The native path therefore does not obtain aggregate throughput by merely making individual requests wait longer.

The per-request `mean_decode_tokens_per_second` metric becomes lower under physical batching because several logical sequences share the same physical execution interval. It is not the appropriate measure of service throughput for this experiment. Aggregate generated tokens/s, request rate, wall time, and request latency are the meaningful quantities.

## Resource behavior

Device/KV high-water values are essentially unchanged between paired serial/native modes at the same concurrency.

At concurrency eight, mean peak device memory was about 517 MB in both modes and peak KV about 15.85 MB. Native batching therefore did not buy throughput through a substantial memory expansion.

## Nsight Systems attribution at concurrency eight

The serial profile was dominated by quantized matrix-vector execution. The major matvec families consumed roughly 1.76 seconds of GPU kernel time.

The native path converted the model-weight work into `batch-reuse8` matmul. Its major batched quantized-linear families consumed roughly 0.58 seconds.

Approximate total GPU-kernel time inferred from the Nsight kernel summaries:

- serial: ~2.27 s;
- native: ~0.79 s.

CUDA kernel launch count fell from approximately 123,968 in the serial profile to 45,197 in the native profile.

The native profile is again dominated by quantized linear work: the Q5_0/Q6_K/Q4_K/Q8_0 `batch-reuse8` kernels comprise about 72% of GPU kernel time. Decode attention is no longer the dominant concurrency bottleneck.

This provides the attribution for Prompt 13: optimize batched quantized linear geometry before CUDA Graphs, KV precision, or more attention work.

## Transfers and telemetry

Both profiles moved about 485 MB host-to-device, corresponding primarily to model residency/setup. Device-to-host traffic remained tiny. Transfers are not the dominant hot-path explanation.

Across the full run, GPU telemetry reached approximately 88 C and 114 W. For samples with >=90% SM utilization, mean SM clock was about 1.40 GHz, ranging roughly 1.11-1.73 GHz. Power-limit activity occurred; the thermal-violation flag remained zero. The randomized paired order and large confidence-separated effect sizes make the native-batching conclusion robust despite laptop DVFS.

## Architecture decision

Retain native multi-sequence decode.

The implementation validates the separation between:

- scheduler ownership of grouping compatible work;
- PreparedModel capability declaration;
- CUDA ownership of physical batch execution;
- independent SequenceState and paged-KV ownership per request.

No flattened synthetic sequence or second executor was needed.

## Next target

Prompt 13 begins hardware-native quantized-linear work.

The first candidate should isolate tile/data-movement geometry without simultaneously lowering numerical precision. A shared-memory activation tile is therefore a cleaner first experiment than immediately introducing FP16/TF32/INT8 Tensor Core MMA. If tiling alone is insufficient, Ampere Tensor Core tactics remain strongly justified by the measured dominance of batched quantized linear execution.

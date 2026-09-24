# AIR Prompt 9/16 Performance Attribution Result

Status: **CLOSED — sufficient attribution to select quantized linear execution for Prompt 10.**

AIR 0.8 remains the frozen correctness-first control. Prompt 9 measured AIR 0.9.0 with opt-in NVTX/Nsight instrumentation and did not introduce optimized execution.

## Hardware and evidence boundary

The evidence archive was collected on the target RTX 3080 Laptop GPU (16 GB), driver 595.84, CUDA runtime 13.2, with Nsight Systems 2025.3.2. The CUDA build and all seven project tests passed.

The GPU was thermally stressed during long cases, reaching roughly 88-92 C with sustained clocks frequently around 990-1110 MHz. Existing AIR/llama server processes were resident before collection. Absolute latency therefore carries a thermal/environment caveat. Kernel-family proportions, launch counts, and the large ordering of costs are still strong enough to choose the next engineering target. Prompt 10 must improve benchmark isolation.

## Tactic-sensitivity result

Changing only scheduler prefill quantum did not rescue the current executor.

| Prompt | q32 prefill tok/s | q64 | q128 | q32 aggregate tok/s | q128 aggregate tok/s |
|---:|---:|---:|---:|---:|---:|
| 54 | 180.27 | 123.15 | 139.90 | 49.05 | 37.75 |
| 262 | 140.60 | 132.07 | 123.58 | 13.34 | 11.86 |
| 1042 | 114.67 | 109.56 | 108.34 | 3.14 | 2.98 |

The ordered sweep favored q32. Under Nsight, q128 and q32 moved closer and thermal/order effects were visible. The scientifically safe conclusion is not that q32 is universally optimal. It is that scheduler quantum alone is not the source of AIR's large prefill deficit and cannot substitute for better GPU execution.

## GPU kernel attribution

### Long prompt, 1042 tokens, q128

- Q5_0 quantized matmul: 43.0% of GPU kernel time
- Q6_K quantized matmul: 16.1%
- Q4_K quantized matmul: 15.8%
- Q8_0 quantized matmul: 0.2%
- paged batch attention: 21.8%
- attention value accumulation: 1.5%

Quantized matmul therefore accounts for about **75%** of GPU kernel time. Paged attention accounts for roughly **23-24%**.

### Long prompt, 1042 tokens, q32

The ordering is stable:

- Q5_0 matmul: 41.6%
- Q6_K matmul: 15.6%
- Q4_K matmul: 15.2%
- paged batch attention: 24.3%
- attention value accumulation: 1.5%

Quantized matmul remains roughly **72-73%** of GPU kernel time and attention roughly **26%**.

### Short prompt, 54 tokens, q32

Quantized linear work is even more dominant when matmul and decode matvec are considered together:

- Q5_0 matmul: 35.5%
- Q5_0 matvec: 13.9%
- Q4_K matmul: 13.7%
- Q6_K matmul: 13.4%
- Q8_0 matvec: 6.3%
- Q6_K matvec: 4.7%
- Q4_K matvec: 4.5%

Attention is small in this short-context case.

## Launch/API attribution

Small-model launch overhead is real but secondary.

- p54/q32/c1: 8,193 `cudaLaunchKernel` API calls, about 73.6 ms total CPU launch API time.
- p1042/q32/c1: 18,480 launches, about 150.3 ms launch API time while GPU execution is on the order of 9 seconds.
- p1042/q128/c1 reduces launch count to 7,512, but does not close the throughput gap.

CUDA Graphs/fusion remain justified later, especially for small models, but the long-prefill evidence does not support making them Prompt 10.

`cudaStreamSynchronize` dominates long-profile CUDA API wall time because the host waits for GPU execution. It is not independently the dominant compute cause.

## Transfer attribution

Each profile records roughly 485 MB of H2D traffic, close to the 0.5B model's resident weight footprint. The evidence is consistent with model initialization/upload rather than repeated hot-path weight transfers. D2H traffic is tiny. Device-side greedy selection is successfully avoiding full-vocabulary host readback in the hot generation path.

H2D/D2H optimization is therefore not a Prompt 10 priority.

## Concurrency hypothesis

Prompt 8 already demonstrated that admitted concurrency does not improve AIR aggregate throughput, while llama.cpp gains substantially. Prompt 9 source/capability inspection confirms CUDA decode width remains one. This is demonstrated as a runtime limitation, but Prompt 9 profiling ranks quantized linear execution ahead of multi-sequence work for the current single-model hot path. Physical multi-sequence decode remains Prompt 12.

## Numerical scaling diagnosis

The 0.5B model passes the frozen `atol=0.001` gate. The larger models preserve greedy top-1 decisions in the short diagnostic but exceed that absolute threshold:

- 1.5B: short diagnostic worst max-abs error about 0.00236; prior longer diagnostic reached about 0.00941.
- 7B: short diagnostic worst max-abs error about 0.00138; prior probe reached about 0.00830.

Stage traces show small differences entering early and accumulating with depth. Source inspection establishes a material oracle difference: the CPU reference matrix-vector path accumulates decoded products in `double`, while CUDA quantized kernels accumulate per lane in `float` and perform parallel warp reductions. The evidence therefore supports **scale-dependent numerical accumulation/reduction-order drift** as a strong explanation. It does not prove every observed difference is harmless and does not justify silently loosening the frozen gate.

A later numerical-equivalence contract should distinguish high-precision-oracle numerical distance from semantic/top-k failure using explicit, justified criteria. Until then, larger models remain excluded from performance claims.

## Hypothesis disposition

| Hypothesis | Prompt 9 disposition |
|---|---|
| Quantized linear execution dominates | **Demonstrated** |
| Prefill lacks effective cross-token weight reuse | **Source-supported and consistent with profiler; specific speedup unproven until Prompt 10** |
| q32 scheduler quantum causes the main deficit | **Rejected** |
| Paged attention is a major long-context cost | **Demonstrated, second-ranked** |
| FP32 KV is a current dominant cost | **Not demonstrated** |
| Decode width 1 prevents physical concurrency gains | **Demonstrated capability limitation; performance effect consistent with Prompt 8** |
| Launch overhead is material | **Demonstrated for small workloads, secondary for long prefill** |
| H2D/D2H/readback is a hot-path bottleneck | **Rejected for the measured workload** |
| Larger-model drift is primarily a gross CUDA semantic defect | **Not supported; numerical accumulation/reduction-order drift is the stronger current explanation** |

## Prompt 10 decision

Prompt 10 remains **Quantized Linear Engine**.

The first candidate must preserve canonical GGUF residency and the existing `CudaExecutor`. It should test whether a prompt-item tile can reuse each decoded quantized weight across multiple input rows before AIR considers a memory-heavier expanded-weight/cuBLAS tactic.

Every candidate remains optional, capability-declared, correctness-gated, and benchmarked against the unchanged baseline. A candidate that is slower or numerically invalid is rejected rather than becoming permanent complexity.

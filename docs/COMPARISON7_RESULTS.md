# Comparison Prompt 7 Results

This document records the controlled 0.5B baseline that motivates Prompt 8. It is evidence, not an optimization directive.

## Identity and method

- AIR version: 0.8.0
- llama.cpp build: 10376 (`47635d703`)
- model file: `qwen2.5-0.5b-instruct-q4_k_m.gguf`
- exact GGUF SHA-256: `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`
- GGUF file size: 491,400,032 bytes
- paired rounds per workload: 5
- engine order seed: 1337
- sampling seed: 424242
- primary concurrency: 1

AIR and llama.cpp loaded the same file path. The raw HTTP/SSE prompts were identical. Prompt-token counts agreed exactly before measurement:

- decode: 5 tokens;
- balanced: 132 tokens;
- prefill: 1042 tokens.

The paired comparison uses per-round AIR/llama log ratios and two-sided 95% Student-t confidence intervals. Client-measured wall timing is primary. Engine-native timing is secondary.

## Correctness

AIR reference versus AIR CUDA:

- 32/32 top-1 decisions matched;
- all compared logits were finite;
- every decision remained within the requested 0.001 absolute tolerance;
- worst observed full-logit-vector max absolute error across the 32 decisions was approximately `1.56e-4`.

AIR versus llama.cpp teacher forcing:

- 31/32 exact top-1 decisions matched;
- the only disagreement was decision 30;
- AIR ranked token 7407 immediately above token 279 with a top-1/top-2 margin of about 0.0433 logits;
- llama.cpp ranked token 279 immediately above token 7407 with a score separation of about 0.0109;
- both systems therefore identified the same two leading candidates and disagreed only at a numerically fragile near-tie.

This is not evidence of broad semantic or execution divergence. It is also not exact cross-runtime top-1 parity and should not be reported as such.

## Primary exact-request performance

Ratios are AIR/llama. For latency, lower than 1 favors AIR. For throughput, greater than 1 favors AIR.

| Workload | AIR TTFT median | llama TTFT median | TTFT ratio 95% CI | AIR total median | llama total median | Total ratio 95% CI | AIR output tok/s | llama output tok/s | Throughput ratio 95% CI |
|---|---:|---:|---|---:|---:|---|---:|---:|---|
| decode, 5 + 256 | 31.51 ms | 12.29 ms | 2.339x [1.263, 4.333] | 3212.61 ms | 781.53 ms | 4.139x [3.932, 4.356] | 79.69 | 327.56 | 0.242x [0.230, 0.254] |
| balanced, 132 + 128 | 606.62 ms | 17.71 ms | 33.504x [27.792, 40.389] | 2358.88 ms | 415.94 ms | 5.725x [5.452, 6.012] | 54.26 | 307.74 | 0.175x [0.166, 0.183] |
| prefill, 1042 + 32 | 7753.69 ms | 69.42 ms | 114.806x [105.616, 124.795] | 8687.75 ms | 174.37 ms | 51.336x [47.539, 55.436] | 3.68 | 183.52 | 0.019x [0.018, 0.021] |

Every primary confidence interval excludes parity. On this model and this hardware, llama.cpp is demonstrably faster in all three Prompt 7 workloads.

The shape of the gap matters more than a single aggregate number. AIR is roughly four times slower in decode-heavy total latency, roughly six times slower in the balanced workload, and more than fifty times slower in the long-prefill total-latency workload. The prefill path is therefore the largest measured relative deficit in this baseline.

## Secondary engine-native timing

Engine-native timers do not have identical boundaries, so these values are supporting evidence only.

Prompt 7 reported median native throughput ratios:

- decode workload: AIR prefill 0.319x llama; AIR decode 0.239x llama;
- balanced workload: AIR prefill 0.026x llama; AIR decode 0.223x llama;
- prefill workload: AIR prefill 0.008x llama; AIR decode 0.103x llama.

The separate `air-bench` / `llama-bench` lane also points in the same direction, while using synthetic length-controlled prompts on the llama-bench side:

- decode generation: AIR mean decode about 71.2 tok/s versus llama-bench about 314.5 tok/s;
- balanced prefill: AIR about 119 tok/s versus llama-bench about 10.5k tok/s;
- balanced decode: AIR about 61.1 tok/s versus llama-bench about 331 tok/s;
- 1042-token prefill: AIR about 107.8 tok/s versus llama-bench about 18.3k tok/s;
- associated decode: AIR about 29.8 tok/s versus llama-bench about 358 tok/s.

Because benchmark boundaries and prompt construction differ, these numbers must not replace the exact-request primary lane. They do, however, independently reinforce the conclusion that the observed performance deficit is not a client-side measurement artifact.

## Memory

With each server resident alone before request execution:

- AIR process VRAM: approximately 646 MiB;
- llama.cpp process VRAM: approximately 1010 MiB.

AIR therefore used about 364 MiB less isolated process VRAM, approximately 36% less than llama.cpp in this idle configuration.

This is an architectural difference rather than a normalized memory benchmark. AIR retains demand-allocated paged KV with zero current KV bytes at idle. llama.cpp retains its normal server/slot/KV design. The comparison intentionally preserves those runtime choices.

## Thermal, power, clocks, and PCIe evidence

Start-state balance was generally good after cooldown. The balanced and prefill workloads began within roughly one degree Celsius between engines and at the same 1545 MHz starting SM clock in every paired round. The decode workload had one anomalous first pair in which AIR began at 87 C and 1140 MHz while llama.cpp began at 78 C and 1545 MHz; the other four decode pairs began at matched temperatures and clocks. The decode result remains clearly separated even with that imbalance retained rather than removed.

AIR's longer requests commonly drove the GPU into roughly 86-87 C and ended with SM clocks around 1.1-1.3 GHz. llama.cpp often completed balanced and prefill work before comparable sustained heating occurred and typically retained higher clocks. Across the raw `nvidia-smi dmon` stream the GPU reached 90 C, power reached about 98 W, and SM utilization reached 100%. The thermal-violation flag remained zero in the retained samples, while transient power-violation percentages were observed.

Thermal/DVFS behavior is therefore a real part of the measured machine result, especially for AIR's longer sustained runs, but it does not plausibly explain the order-of-magnitude prefill gap by itself. Prompt 8 must preserve thermal evidence and randomized order rather than trying to normalize it away after the fact.

The raw dmon stream also retains PCIe receive/transmit telemetry, with observed peaks around 745 MB/s receive and 1951 MB/s transmit during the full paired run. Because that stream was not timestamp-correlated to individual request boundaries in Prompt 7, these peaks are retained as run-level telemetry and are not assigned to one engine.

## Neutral conclusion

Prompt 7 establishes three things simultaneously:

1. AIR's frozen CUDA execution remains numerically well aligned with its reference path and closely aligned with llama.cpp under teacher forcing.
2. llama.cpp is decisively faster on the tested Qwen2.5 0.5B workload, with AIR's largest relative weakness in long-prompt prefill.
3. AIR shows a lower isolated idle VRAM footprint under its demand-paged design, but this does not offset the measured latency/throughput deficit for this workload.

No frozen AIR architecture change follows from one model and concurrency level. Prompt 8 exists to determine whether the performance shape survives prompt-length, concurrency, and model-size scaling before any optimization target is selected.

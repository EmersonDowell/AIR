# AIR Release-Candidate Support Matrix

## Model format

| Area | Supported |
|---|---|
| Container | GGUF v3 |
| Architecture metadata | `qwen2` |
| Tested family | Qwen2.5 GGUF |
| Tokenizer | byte-level GPT-2 BPE / Qwen2 pre-tokenization |
| Chat rendering | explicit Qwen2 ChatML |

## Execution tensor encodings

Both the CPU reference executor and CUDA executor support:

- F32
- F16
- BF16
- Q4_0
- Q5_0
- Q8_0
- Q4_K
- Q6_K

The GGUF inspector can describe additional encodings, but execution rejects unsupported tensor types explicitly.

## Qwen2 execution constraints

Supported:

- grouped-query attention
- optional Q/K/V biases
- standard RMSNorm
- SwiGLU FFN
- full even head-dimension RoPE
- tied output embeddings when `output.weight` is absent

Not supported in the first release candidate:

- RoPE scaling types/factors other than the unscaled form
- sliding-window attention
- partial rotary dimensions
- other model architectures

## Backends

### Reference

- CPU correctness path
- physical paged KV
- sequence checkpoints
- serving-level exact-prefix reuse
- scalar/readable execution prioritized over speed

### NVIDIA CUDA

- native multi-token prefill, maximum width 128
- physical executor-owned paged KV
- demand allocation and page reuse
- capacity-aware logical reservations
- checkpoint/page sharing with copy-on-write partial tails
- device-side deterministic greedy token selection
- specialized F16/BF16/quantized matrix kernels
- native greedy decode execution width 8 for compatible sequences
- persistent serving-level prefix reuse disabled pending pressure-aware eviction

## Platforms

Primary validation platform for the first release candidate:

- Linux
- NVIDIA CUDA
- x86-64

The CPU/reference build is portable C++20 but the project does not yet claim a tested macOS/Windows release matrix.

## AIR 0.9.6 experimental CUDA tactic

`shared-tile8` was rejected by Prompt 13 paired evidence and is no longer a live tactic.

`dense-f32-cublas` is supported only by the CUDA prepared backend as an explicit research tactic for batched linear prefill/native decode. It lazily allocates a derived FP32 matrix arena, is not a reference-backend capability, and is not automatically selected.

# Reference Execution

The CPU reference executor is AIR's inspectable numerical oracle for Qwen2 execution.

It implements the same validated model contract as CUDA while favoring readable scalar/vector code over performance.

## Supported tensor encodings

- F32
- F16
- BF16
- Q4_0
- Q5_0
- Q8_0
- Q4_K
- Q6_K

Randomized block-property tests compare quantized decoding against separate scalar test formulas.

## Transformer path

The reference path implements:

- token embedding lookup;
- RMSNorm;
- Q/K/V projection with optional Q/K/V biases;
- unscaled full-head RoPE;
- grouped-query causal attention;
- attention output projection and residual;
- SwiGLU feed-forward network and residual;
- final RMSNorm;
- tied embedding output when `output.weight` is absent;
- optional output bias.

Unsupported RoPE scaling, partial rotary dimensions, and sliding-window attention fail explicitly during executor construction.

## KV

Reference KV uses physical pages and supports checkpoints/exact-prefix reuse. Its page implementation is intentionally straightforward and is not a performance model for CUDA.

## Sampling

Sampling is kept outside transformer math. `temperature <= 0` selects greedy argmax. Positive temperature supports top-k/top-p sampling with a seeded `std::mt19937_64` generator.

Keeping sampling separate allows logits to be compared independently of stochastic token choice.

# CUDA Execution

AIR's CUDA backend is a prepared execution implementation of the shared Qwen2 contract.

## Weight residency

Execution tensors remain in their GGUF representation on the device. Supported tensor families are classified when the model is prepared, allowing hot matrix operations to dispatch to format-specific kernels without runtime datatype branching inside each scalar decode.

## Prefill

CUDA prefill accepts bounded token matrices and executes native multi-token transformer work. Intermediate serving prefill chunks may be outputless: they update KV without running the final vocabulary projection when no token decision is required.

The current maximum native prefill width is 128 tokens.

## Decode

Decode currently executes one sequence at a time through the shared CUDA workspace. Multiple requests can be admitted and interleaved, but AIR does not call this fused/batched decode.

## Paged KV

The executor owns a reusable physical page pool. A sequence stores a logical table of page references and acquires pages on demand. Attention reads pages directly; AIR does not flatten paged KV into a contiguous compatibility buffer.

Checkpoint forks share committed pages. A shared partial tail is copied before append. Failed execution releases pages acquired by the failed transaction.

## Deterministic output

For `temperature <= 0`, AIR performs argmax on device and reads back the selected token rather than transferring the full vocabulary logits to the host. Stochastic sampling retains the shared host sampler.

## Hot-path evidence

`CudaExecutionStats` exposes cumulative counters for model residency, page-pool state, host/device traffic, matrix kernel families, full-logit readbacks, greedy-token readbacks, and outputless prefill chunks.

These counters avoid timing synchronization and are intended to verify that the expected execution path actually ran.

## Current limitations

- qualified compatible-sequence greedy decode width 8;
- no persistent CUDA prefix cache under memory pressure;
- no CUDA graph capture;
- no speculative decoding;
- no multi-GPU execution;
- only tensor encodings listed in `SUPPORT_MATRIX.md`.

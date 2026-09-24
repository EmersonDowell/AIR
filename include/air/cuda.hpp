#pragma once

#include "air/device.hpp"
#include "air/execution.hpp"
#include "air/generation.hpp"
#include "air/model.hpp"
#include "air/result.hpp"
#include "air/verification.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace air {

struct CudaExecutionStats {
    std::uint64_t resident_model_bytes{0};
    std::uint64_t workspace_bytes{0};
    // Logical bytes required to hold the model's full context for one sequence.
    // Paged KV does not reserve this amount per sequence.
    std::uint64_t kv_logical_context_bytes{0};
    // Physical page geometry and current executor-owned pool residency.
    std::uint64_t kv_page_bytes{0};
    std::uint64_t kv_pool_allocated_bytes{0};
    std::uint64_t kv_pool_free_bytes{0};
    std::uint64_t host_to_device_bytes{0};
    std::uint64_t device_to_host_bytes{0};

    // Hot-path evidence counters. These are cumulative executor counters and
    // intentionally avoid timing synchronization overhead.
    std::uint64_t f32_matvec_calls{0};
    std::uint64_t specialized_matvec_calls{0};
    std::uint64_t f32_matmul_calls{0};
    std::uint64_t specialized_matmul_calls{0};
    std::uint64_t batch_reuse4_matmul_calls{0};
    std::uint64_t batch_reuse8_matmul_calls{0};
    std::uint64_t q5q8_dp4a_hybrid_matmul_calls{0};
    std::uint64_t q5q8_dp4a_hybrid_fallback_calls{0};
    std::uint64_t dense_f32_cublas_matmul_calls{0};
    std::uint64_t q5q8_dp4a_hybrid_prepared_bytes{0};
    std::uint64_t prepared_linear_bytes{0};
    std::uint64_t full_logit_readbacks{0};
    std::uint64_t greedy_token_readbacks{0};
    std::uint64_t target_logprob_readbacks{0};
    std::uint64_t outputless_prefill_chunks{0};
    std::uint64_t prefill_batch_calls{0};
    std::uint64_t prefill_batched_sequences{0};
    std::uint64_t prefill_batched_tokens{0};
    std::uint64_t decode_batch_calls{0};
    std::uint64_t decode_batched_sequences{0};
};

[[nodiscard]] bool cuda_compiled() noexcept;
[[nodiscard]] Result<std::vector<DeviceInfo>> cuda_devices();

class CudaKvCache final {
public:
    ~CudaKvCache();
    CudaKvCache(CudaKvCache&&) noexcept;
    CudaKvCache& operator=(CudaKvCache&&) noexcept;
    CudaKvCache(const CudaKvCache&) = delete;
    CudaKvCache& operator=(const CudaKvCache&) = delete;

    [[nodiscard]] std::uint64_t size() const noexcept;
    [[nodiscard]] std::uint64_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t page_tokens() const noexcept;
    [[nodiscard]] std::uint64_t committed_bytes() const noexcept;
    [[nodiscard]] std::uint64_t resident_bytes() const noexcept;
    [[nodiscard]] Result<std::unique_ptr<CudaKvCache>> fork(std::uint64_t prefix_tokens) const;
    void reset() noexcept;

private:
    struct Impl;
    explicit CudaKvCache(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class CudaExecutor;
};



enum class CudaPrefillBatchOutput : std::uint8_t {
    discard = 0,
    greedy,
    logits,
};

struct CudaPrefillBatchItem {
    CudaKvCache* cache{nullptr};
    std::span<const TokenId> tokens;
    CudaPrefillBatchOutput output{CudaPrefillBatchOutput::discard};
};

struct CudaPrefillBatchItemResult {
    std::optional<TokenId> greedy_token;
    std::vector<float> logits;
};

struct CudaPrefillBatchExecution {
    std::vector<CudaPrefillBatchItemResult> items;
    std::uint32_t physical_batches{0};
    std::uint64_t physical_sequence_participations{0};
    std::uint64_t physical_tokens{0};
    std::uint32_t max_sequences{0};
};

class CudaExecutor final {
public:
    ~CudaExecutor();
    CudaExecutor(CudaExecutor&&) noexcept;
    CudaExecutor& operator=(CudaExecutor&&) noexcept;
    CudaExecutor(const CudaExecutor&) = delete;
    CudaExecutor& operator=(const CudaExecutor&) = delete;

    [[nodiscard]] static Result<std::unique_ptr<CudaExecutor>> create(
        std::shared_ptr<const ModelDefinition> model, int device_ordinal = 0);

    [[nodiscard]] const ModelDefinition& model() const noexcept;
    [[nodiscard]] int device_ordinal() const noexcept;
    [[nodiscard]] CudaExecutionStats stats() const noexcept;
    [[nodiscard]] Status prepare_linear_tactic(QuantizedLinearExecutionKind kind);
    [[nodiscard]] std::uint64_t estimate_linear_tactic_preparation_bytes(
        QuantizedLinearExecutionKind kind) const noexcept;
    [[nodiscard]] Status trim_linear_tactics(
        std::span<const QuantizedLinearExecutionKind> required);
    [[nodiscard]] Result<std::unique_ptr<CudaKvCache>> create_kv_cache(
        std::uint32_t page_tokens = 16U) const;
    [[nodiscard]] Result<std::vector<float>> step(
        TokenId token, CudaKvCache& cache,
        QuantizedLinearExecutionKind linear = QuantizedLinearExecutionKind::baseline,
        AttentionExecutionKind attention = AttentionExecutionKind::baseline);
    [[nodiscard]] Result<std::vector<float>> step(
        TokenId token, CudaKvCache& cache,
        QuantizedLinearExecutionKind block_linear,
        QuantizedLinearExecutionKind output_linear,
        AttentionExecutionKind attention);
    [[nodiscard]] Result<std::vector<float>> step_verified(
        TokenId token, CudaKvCache& cache, VerificationTrace& trace,
        QuantizedLinearExecutionKind block_linear = QuantizedLinearExecutionKind::baseline,
        QuantizedLinearExecutionKind output_linear = QuantizedLinearExecutionKind::baseline,
        AttentionExecutionKind attention = AttentionExecutionKind::baseline);
    [[nodiscard]] Result<TokenId> step_greedy(
        TokenId token, CudaKvCache& cache,
        QuantizedLinearExecutionKind linear = QuantizedLinearExecutionKind::baseline,
        AttentionExecutionKind attention = AttentionExecutionKind::baseline);
    [[nodiscard]] Result<TokenId> step_greedy(
        TokenId token, CudaKvCache& cache,
        QuantizedLinearExecutionKind block_linear,
        QuantizedLinearExecutionKind output_linear,
        AttentionExecutionKind attention);
    [[nodiscard]] Result<std::vector<float>> step_target_logprobs(
        TokenId token, std::span<const TokenId> target_tokens, CudaKvCache& cache,
        QuantizedLinearExecutionKind block_linear = QuantizedLinearExecutionKind::batch_reuse8,
        QuantizedLinearExecutionKind output_linear = QuantizedLinearExecutionKind::batch_reuse8,
        AttentionExecutionKind attention = AttentionExecutionKind::baseline);
    [[nodiscard]] Result<std::vector<TokenId>> step_greedy_batch(
        std::span<const TokenId> tokens, std::span<CudaKvCache*> caches,
        QuantizedLinearExecutionKind block_linear = QuantizedLinearExecutionKind::batch_reuse8,
        QuantizedLinearExecutionKind output_linear = QuantizedLinearExecutionKind::batch_reuse8,
        AttentionExecutionKind attention = AttentionExecutionKind::baseline);
    [[nodiscard]] Result<std::vector<float>> prefill(
        std::span<const TokenId> tokens, CudaKvCache& cache,
        QuantizedLinearExecutionKind linear = QuantizedLinearExecutionKind::baseline,
        AttentionExecutionKind attention = AttentionExecutionKind::baseline);
    [[nodiscard]] Status prefill_discard(
        std::span<const TokenId> tokens, CudaKvCache& cache,
        QuantizedLinearExecutionKind linear = QuantizedLinearExecutionKind::baseline,
        AttentionExecutionKind attention = AttentionExecutionKind::baseline);
    [[nodiscard]] Result<TokenId> prefill_greedy(
        std::span<const TokenId> tokens, CudaKvCache& cache,
        QuantizedLinearExecutionKind linear = QuantizedLinearExecutionKind::baseline,
        AttentionExecutionKind attention = AttentionExecutionKind::baseline);
    [[nodiscard]] Result<std::vector<float>> prefill_target_logprobs(
        std::span<const TokenId> tokens, std::span<const TokenId> target_tokens,
        CudaKvCache& cache,
        QuantizedLinearExecutionKind linear = QuantizedLinearExecutionKind::batch_reuse8,
        AttentionExecutionKind attention = AttentionExecutionKind::online_softmax);
    [[nodiscard]] Result<CudaPrefillBatchExecution> prefill_batch(
        std::span<const CudaPrefillBatchItem> items,
        QuantizedLinearExecutionKind linear = QuantizedLinearExecutionKind::baseline,
        AttentionExecutionKind attention = AttentionExecutionKind::baseline);
    [[nodiscard]] Result<GenerationResult> generate(std::span<const TokenId> prompt,
                                                    const GenerationConfig& config);

private:
    enum class FinalOutput : std::uint8_t { logits = 0, discard, greedy, target_logprobs };
    struct Impl;
    explicit CudaExecutor(std::unique_ptr<Impl> impl);
    [[nodiscard]] Result<std::vector<float>> step_impl(
        TokenId token, CudaKvCache& cache, FinalOutput output, TokenId* greedy_token,
        std::span<const TokenId> target_tokens, std::vector<float>* target_logprobs,
        QuantizedLinearExecutionKind block_linear,
        QuantizedLinearExecutionKind output_linear,
        AttentionExecutionKind attention, VerificationTrace* trace = nullptr);
    [[nodiscard]] Result<std::vector<float>> prefill_impl(
        std::span<const TokenId> tokens, CudaKvCache& cache, FinalOutput output,
        TokenId* greedy_token, std::span<const TokenId> target_tokens,
        std::vector<float>* target_logprobs, QuantizedLinearExecutionKind linear,
        AttentionExecutionKind attention);
    std::unique_ptr<Impl> impl_;
};

} // namespace air

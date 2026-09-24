#include "runtime/backend.hpp"

#include "air/cuda.hpp"
#include "air/reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace air::runtime_detail {
namespace {

[[nodiscard]] Result<std::vector<float>> target_logprobs_from_logits(
    std::span<const float> logits,
    std::span<const TokenId> target_tokens) {
    if (logits.empty()) return Status::data_error("target logprob requires non-empty logits");
    const auto maximum = *std::max_element(logits.begin(), logits.end());
    double exp_sum = 0.0;
    for (const auto value : logits) exp_sum += std::exp(static_cast<double>(value - maximum));
    if (!std::isfinite(exp_sum) || exp_sum <= 0.0) {
        return Status::data_error("target logprob normalization is non-finite");
    }
    const double log_z = static_cast<double>(maximum) + std::log(exp_sum);
    std::vector<float> out;
    out.reserve(target_tokens.size());
    for (const auto token : target_tokens) {
        if (token < 0 || static_cast<std::size_t>(token) >= logits.size()) {
            return Status::invalid_argument("target token is outside model vocabulary");
        }
        out.push_back(static_cast<float>(static_cast<double>(
            logits[static_cast<std::size_t>(token)]) - log_z));
    }
    return out;
}

class ReferenceCheckpoint final : public SequenceCheckpoint {
public:
    explicit ReferenceCheckpoint(ReferenceKvCache cache) : cache_(std::move(cache)) {}

    [[nodiscard]] BackendKind backend() const noexcept override { return BackendKind::reference; }
    [[nodiscard]] std::uint64_t tokens() const noexcept override { return cache_.size(); }
    [[nodiscard]] SequenceResources resources() const noexcept override {
        return SequenceResources{cache_.size(), cache_.resident_bytes(), 0U};
    }
    [[nodiscard]] const ReferenceKvCache& cache() const noexcept { return cache_; }

    [[nodiscard]] Result<std::unique_ptr<SequenceCheckpoint>> clone() const override {
        auto forked = cache_.fork(cache_.size());
        if (!forked) return forked.status();
        return std::unique_ptr<SequenceCheckpoint>(
            std::make_unique<ReferenceCheckpoint>(std::move(forked).value()));
    }

private:
    ReferenceKvCache cache_;
};

class ReferenceSequence final : public SequenceState {
public:
    ReferenceSequence(const ReferenceExecutor& executor, ReferenceKvCache cache)
        : executor_(executor), cache_(std::move(cache)) {}

    [[nodiscard]] BackendKind backend() const noexcept override { return BackendKind::reference; }

    [[nodiscard]] Result<std::vector<float>> prefill(std::span<const TokenId> tokens) override {
        return executor_.prefill(tokens, cache_);
    }

    [[nodiscard]] Result<std::vector<float>> prefill_target_logprobs(
        std::span<const TokenId> tokens,
        std::span<const TokenId> targets) override {
        auto logits = executor_.prefill(tokens, cache_);
        if (!logits) return logits.status();
        return target_logprobs_from_logits(logits.value(), targets);
    }

    [[nodiscard]] Result<std::vector<float>> decode(TokenId token) override {
        return executor_.step(token, cache_);
    }

    [[nodiscard]] Result<std::vector<float>> decode_target_logprobs(
        TokenId token, std::span<const TokenId> targets) override {
        auto logits = executor_.step(token, cache_);
        if (!logits) return logits.status();
        return target_logprobs_from_logits(logits.value(), targets);
    }

    [[nodiscard]] SequenceResources resources() const noexcept override {
        return SequenceResources{cache_.size(), cache_.resident_bytes(), 0U};
    }

    [[nodiscard]] Result<std::unique_ptr<SequenceCheckpoint>> checkpoint() const override {
        auto forked = cache_.fork(cache_.size());
        if (!forked) return forked.status();
        return std::unique_ptr<SequenceCheckpoint>(
            std::make_unique<ReferenceCheckpoint>(std::move(forked).value()));
    }

private:
    const ReferenceExecutor& executor_;
    ReferenceKvCache cache_;
};

class ReferencePreparedModel final : public PreparedModel {
public:
    ReferencePreparedModel(std::shared_ptr<const ModelDefinition> model,
                           std::unique_ptr<ReferenceExecutor> executor)
        : model_(std::move(model)), executor_(std::move(executor)) {
        capabilities_.backend = BackendKind::reference;
        capabilities_.prefill_execution = PrefillExecutionKind::serial;
        capabilities_.kv_storage = KvStorageKind::paged;
        capabilities_.max_prefill_batch_width = 1U;
        capabilities_.max_decode_batch_width = 1U;
        capabilities_.sequence_checkpointing = true;
        capabilities_.exact_prefix_reuse = true;
    }

    [[nodiscard]] BackendKind backend() const noexcept override { return BackendKind::reference; }
    [[nodiscard]] const BackendCapabilities& capabilities() const noexcept override { return capabilities_; }
    [[nodiscard]] const ModelDefinition& model() const noexcept override { return *model_; }

    [[nodiscard]] Result<std::unique_ptr<SequenceState>> create_sequence(
        const ExecutionPlan& plan) override {
        const auto valid = validate_execution_plan(plan, capabilities_);
        if (!valid) return valid;
        const auto& config = model_->config();
        const auto head_dimension = config.embedding_size / config.attention_head_count;
        ReferenceKvCache cache(config.layer_count,
                               config.kv_head_count,
                               head_dimension,
                               config.context_length,
                               *plan.kv.page_tokens);
        return std::unique_ptr<SequenceState>(
            std::make_unique<ReferenceSequence>(*executor_, std::move(cache)));
    }

    [[nodiscard]] Result<std::unique_ptr<SequenceState>> restore_sequence(
        const ExecutionPlan& plan, const SequenceCheckpoint& checkpoint) override {
        const auto valid = validate_execution_plan(plan, capabilities_);
        if (!valid) return valid;
        if (checkpoint.backend() != BackendKind::reference) {
            return Status::invalid_argument("reference backend cannot restore a checkpoint from another backend");
        }
        const auto* reference = dynamic_cast<const ReferenceCheckpoint*>(&checkpoint);
        if (!reference) return Status::invalid_argument("reference checkpoint type is invalid");
        auto cloned = reference->cache().fork(reference->cache().size());
        if (!cloned) return cloned.status();
        return std::unique_ptr<SequenceState>(
            std::make_unique<ReferenceSequence>(*executor_, std::move(cloned).value()));
    }

    [[nodiscard]] std::uint64_t resident_device_bytes() const noexcept override { return 0U; }
    [[nodiscard]] std::uint64_t estimate_sequence_device_bytes(
        const ExecutionPlan&, std::uint64_t) const noexcept override { return 0U; }
    [[nodiscard]] std::optional<std::uint64_t> sequence_capacity_bytes() const override { return std::nullopt; }
    [[nodiscard]] std::optional<std::uint64_t> free_device_bytes() const override { return std::nullopt; }
    [[nodiscard]] std::uint64_t kv_pool_allocated_bytes() const noexcept override { return 0U; }
    [[nodiscard]] std::uint64_t kv_pool_free_bytes() const noexcept override { return 0U; }

private:
    std::shared_ptr<const ModelDefinition> model_;
    std::unique_ptr<ReferenceExecutor> executor_;
    BackendCapabilities capabilities_{};
};

class CudaCheckpoint final : public SequenceCheckpoint {
public:
    explicit CudaCheckpoint(std::unique_ptr<CudaKvCache> cache) : cache_(std::move(cache)) {}

    [[nodiscard]] BackendKind backend() const noexcept override { return BackendKind::cuda; }
    [[nodiscard]] std::uint64_t tokens() const noexcept override { return cache_ ? cache_->size() : 0U; }
    [[nodiscard]] SequenceResources resources() const noexcept override {
        if (!cache_) return {};
        return SequenceResources{cache_->size(), cache_->committed_bytes(), cache_->resident_bytes()};
    }
    [[nodiscard]] const CudaKvCache& cache() const noexcept { return *cache_; }

    [[nodiscard]] Result<std::unique_ptr<SequenceCheckpoint>> clone() const override {
        if (!cache_) return Status::invalid_state("CUDA checkpoint has no cache");
        auto forked = cache_->fork(cache_->size());
        if (!forked) return forked.status();
        return std::unique_ptr<SequenceCheckpoint>(
            std::make_unique<CudaCheckpoint>(std::move(forked).value()));
    }

private:
    std::unique_ptr<CudaKvCache> cache_;
};

class CudaSequence final : public SequenceState {
public:
    CudaSequence(CudaExecutor& executor, std::unique_ptr<CudaKvCache> cache,
                 QuantizedLinearExecutionKind prefill_block_linear,
                 QuantizedLinearExecutionKind decode_block_linear,
                 QuantizedLinearExecutionKind decode_output_linear,
                 AttentionExecutionKind prefill_attention,
                 AttentionExecutionKind decode_attention)
        : executor_(executor), cache_(std::move(cache)),
          prefill_block_linear_(prefill_block_linear),
          decode_block_linear_(decode_block_linear),
          decode_output_linear_(decode_output_linear),
          prefill_attention_(prefill_attention), decode_attention_(decode_attention) {}

    [[nodiscard]] BackendKind backend() const noexcept override { return BackendKind::cuda; }

    [[nodiscard]] Result<std::vector<float>> prefill(std::span<const TokenId> tokens) override {
        return executor_.prefill(tokens, *cache_, prefill_block_linear_, prefill_attention_);
    }

    [[nodiscard]] Status prefill_discard(std::span<const TokenId> tokens) override {
        return executor_.prefill_discard(tokens, *cache_, prefill_block_linear_, prefill_attention_);
    }

    [[nodiscard]] Result<TokenId> prefill_greedy(std::span<const TokenId> tokens) override {
        return executor_.prefill_greedy(tokens, *cache_, prefill_block_linear_, prefill_attention_);
    }

    [[nodiscard]] Result<std::vector<float>> prefill_target_logprobs(
        std::span<const TokenId> tokens,
        std::span<const TokenId> targets) override {
        return executor_.prefill_target_logprobs(
            tokens, targets, *cache_, prefill_block_linear_, prefill_attention_);
    }

    [[nodiscard]] Result<std::vector<float>> decode(TokenId token) override {
        return executor_.step(token, *cache_, decode_block_linear_, decode_output_linear_, decode_attention_);
    }

    [[nodiscard]] Result<std::vector<float>> decode_target_logprobs(
        TokenId token, std::span<const TokenId> targets) override {
        return executor_.step_target_logprobs(
            token, targets, *cache_, decode_block_linear_, decode_output_linear_, decode_attention_);
    }

    [[nodiscard]] Result<TokenId> decode_greedy(TokenId token) override {
        return executor_.step_greedy(token, *cache_, decode_block_linear_, decode_output_linear_, decode_attention_);
    }

    [[nodiscard]] CudaKvCache& cache() noexcept { return *cache_; }
    [[nodiscard]] QuantizedLinearExecutionKind prefill_block_linear() const noexcept { return prefill_block_linear_; }
    [[nodiscard]] AttentionExecutionKind prefill_attention() const noexcept { return prefill_attention_; }
    [[nodiscard]] QuantizedLinearExecutionKind decode_block_linear() const noexcept { return decode_block_linear_; }
    [[nodiscard]] QuantizedLinearExecutionKind decode_output_linear() const noexcept { return decode_output_linear_; }
    [[nodiscard]] AttentionExecutionKind decode_attention() const noexcept { return decode_attention_; }

    [[nodiscard]] SequenceResources resources() const noexcept override {
        return SequenceResources{cache_->size(), cache_->committed_bytes(), cache_->resident_bytes()};
    }

    [[nodiscard]] Result<std::unique_ptr<SequenceCheckpoint>> checkpoint() const override {
        auto forked = cache_->fork(cache_->size());
        if (!forked) return forked.status();
        return std::unique_ptr<SequenceCheckpoint>(
            std::make_unique<CudaCheckpoint>(std::move(forked).value()));
    }

private:
    CudaExecutor& executor_;
    std::unique_ptr<CudaKvCache> cache_;
    QuantizedLinearExecutionKind prefill_block_linear_{QuantizedLinearExecutionKind::baseline};
    QuantizedLinearExecutionKind decode_block_linear_{QuantizedLinearExecutionKind::baseline};
    QuantizedLinearExecutionKind decode_output_linear_{QuantizedLinearExecutionKind::baseline};
    AttentionExecutionKind prefill_attention_{AttentionExecutionKind::baseline};
    AttentionExecutionKind decode_attention_{AttentionExecutionKind::baseline};
};

class CudaPreparedModel final : public PreparedModel {
public:
    CudaPreparedModel(std::shared_ptr<const ModelDefinition> model,
                      std::unique_ptr<CudaExecutor> executor,
                      int device_ordinal)
        : model_(std::move(model)), executor_(std::move(executor)), device_ordinal_(device_ordinal) {
        capabilities_.backend = BackendKind::cuda;
        capabilities_.prefill_execution = PrefillExecutionKind::native_batch;
        capabilities_.kv_storage = KvStorageKind::paged;
        capabilities_.max_prefill_batch_width = 128U;
        capabilities_.max_decode_batch_width = 8U;
        capabilities_.sequence_checkpointing = true;
        // Physical sharing is implemented, but persistent CUDA prefix caching
        // stays disabled until the cache has a pressure-eviction policy. This
        // keeps admission guarantees truthful under memory pressure.
        capabilities_.exact_prefix_reuse = false;
        capabilities_.device_greedy_selection = true;
        capabilities_.prefill_block_quantized_linear = {
            QuantizedLinearExecutionKind::baseline,
            QuantizedLinearExecutionKind::batch_reuse4,
            QuantizedLinearExecutionKind::batch_reuse8,
            QuantizedLinearExecutionKind::dense_f32_cublas,
        };
        capabilities_.decode_block_quantized_linear = {
            QuantizedLinearExecutionKind::baseline,
            QuantizedLinearExecutionKind::batch_reuse8,
            QuantizedLinearExecutionKind::dense_f32_cublas,
        };
        capabilities_.decode_output_quantized_linear = {
            QuantizedLinearExecutionKind::baseline,
            QuantizedLinearExecutionKind::batch_reuse8,
        };
        capabilities_.prefill_attention = {
            AttentionExecutionKind::baseline,
            AttentionExecutionKind::online_softmax,
        };
        capabilities_.decode_attention = {
            AttentionExecutionKind::baseline,
        };
    }

    [[nodiscard]] BackendKind backend() const noexcept override { return BackendKind::cuda; }
    [[nodiscard]] const BackendCapabilities& capabilities() const noexcept override { return capabilities_; }
    [[nodiscard]] const ModelDefinition& model() const noexcept override { return *model_; }

    [[nodiscard]] Result<std::unique_ptr<SequenceState>> create_sequence(
        const ExecutionPlan& plan) override {
        const auto valid = validate_execution_plan(plan, capabilities_);
        if (!valid) return valid;
        auto cache = executor_->create_kv_cache(*plan.kv.page_tokens);
        if (!cache) return cache.status();
        return std::unique_ptr<SequenceState>(
            std::make_unique<CudaSequence>(*executor_, std::move(cache).value(),
                                           plan.linear.prefill_block, plan.linear.decode_block,
                                           plan.linear.decode_output,
                                           plan.attention.prefill, plan.attention.decode));
    }

    [[nodiscard]] Result<std::unique_ptr<SequenceState>> restore_sequence(
        const ExecutionPlan& plan, const SequenceCheckpoint& checkpoint) override {
        const auto valid = validate_execution_plan(plan, capabilities_);
        if (!valid) return valid;
        if (checkpoint.backend() != BackendKind::cuda) {
            return Status::invalid_argument("CUDA backend cannot restore a checkpoint from another backend");
        }
        const auto* cuda_checkpoint = dynamic_cast<const CudaCheckpoint*>(&checkpoint);
        if (!cuda_checkpoint) return Status::invalid_argument("CUDA checkpoint type is invalid");
        if (cuda_checkpoint->cache().page_tokens() != *plan.kv.page_tokens) {
            return Status::invalid_argument("CUDA checkpoint page geometry does not match execution plan");
        }
        auto cloned = cuda_checkpoint->cache().fork(cuda_checkpoint->cache().size());
        if (!cloned) return cloned.status();
        return std::unique_ptr<SequenceState>(
            std::make_unique<CudaSequence>(*executor_, std::move(cloned).value(),
                                           plan.linear.prefill_block, plan.linear.decode_block,
                                           plan.linear.decode_output,
                                           plan.attention.prefill, plan.attention.decode));
    }

    [[nodiscard]] std::uint64_t estimate_plan_preparation_device_bytes(
        const ExecutionPlan& plan) const noexcept override {
        const QuantizedLinearExecutionKind tactics[] = {
            plan.linear.prefill_block,
            plan.linear.decode_block,
            plan.linear.decode_output,
        };
        std::uint64_t total = 0U;
        for (std::size_t i = 0; i < std::size(tactics); ++i) {
            bool duplicate = false;
            for (std::size_t j = 0; j < i; ++j) duplicate = duplicate || tactics[j] == tactics[i];
            if (duplicate) continue;
            const auto bytes = executor_->estimate_linear_tactic_preparation_bytes(tactics[i]);
            if (bytes > std::numeric_limits<std::uint64_t>::max() - total) {
                return std::numeric_limits<std::uint64_t>::max();
            }
            total += bytes;
        }
        return total;
    }

    [[nodiscard]] Status trim_plan_artifacts(const ExecutionPlan& plan) override {
        const QuantizedLinearExecutionKind tactics[] = {
            plan.linear.prefill_block,
            plan.linear.decode_block,
            plan.linear.decode_output,
        };
        return executor_->trim_linear_tactics(tactics);
    }

    [[nodiscard]] Status prepare_plan(const ExecutionPlan& plan) override {
        const auto valid = validate_execution_plan(plan, capabilities_);
        if (!valid) return valid;
        const QuantizedLinearExecutionKind tactics[] = {
            plan.linear.prefill_block,
            plan.linear.decode_block,
            plan.linear.decode_output,
        };
        for (std::size_t i = 0; i < std::size(tactics); ++i) {
            bool duplicate = false;
            for (std::size_t j = 0; j < i; ++j) duplicate = duplicate || tactics[j] == tactics[i];
            if (duplicate) continue;
            auto status = executor_->prepare_linear_tactic(tactics[i]);
            if (!status) return status;
        }
        return Status::ok();
    }

    [[nodiscard]] std::uint64_t estimate_transition_temporary_device_bytes(
        const ExecutionPlan& plan, std::size_t sequence_count) const noexcept override {
        const auto page_tokens = plan.kv.page_tokens.value_or(0U);
        if (page_tokens == 0U || sequence_count == 0U) return 0U;
        const auto max_pages = (model_->config().context_length + page_tokens - 1U) / page_tokens;
        if (max_pages > std::numeric_limits<std::uint64_t>::max() / (2U * sizeof(float*))) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        const auto table_bytes = max_pages * 2U * sizeof(float*);
        // Atomic migration holds one checkpoint page table and one restored
        // sequence page table per sequence before committing over old sessions.
        if (sequence_count > std::numeric_limits<std::uint64_t>::max() /
                                 (2U * table_bytes)) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        return static_cast<std::uint64_t>(sequence_count) * 2U * table_bytes;
    }

    [[nodiscard]] Result<PrefillBatchExecution> prefill_batch(
        std::span<const PrefillBatchItem> items) override {
        if (items.size() < 2U) {
            return Status::invalid_argument("CUDA prefill batch requires at least two sequences");
        }
        std::vector<CudaPrefillBatchItem> cuda_items;
        cuda_items.reserve(items.size());

        QuantizedLinearExecutionKind linear = QuantizedLinearExecutionKind::baseline;
        AttentionExecutionKind attention = AttentionExecutionKind::baseline;
        bool first = true;
        for (const auto& item : items) {
            auto* sequence = dynamic_cast<CudaSequence*>(item.sequence);
            if (!sequence) {
                return Status::invalid_argument("CUDA prefill batch contains a non-CUDA sequence");
            }
            if (item.tokens.empty()) {
                return Status::invalid_argument("CUDA prefill batch contains an empty token slice");
            }
            if (first) {
                linear = sequence->prefill_block_linear();
                attention = sequence->prefill_attention();
                first = false;
            } else if (sequence->prefill_block_linear() != linear ||
                       sequence->prefill_attention() != attention) {
                return Status::invalid_argument(
                    "CUDA prefill batch requires identical prefill execution tactics");
            }

            CudaPrefillBatchOutput output = CudaPrefillBatchOutput::discard;
            if (item.output == PrefillBatchOutput::greedy) {
                output = CudaPrefillBatchOutput::greedy;
            } else if (item.output == PrefillBatchOutput::logits) {
                output = CudaPrefillBatchOutput::logits;
            }
            cuda_items.push_back(CudaPrefillBatchItem{
                &sequence->cache(), item.tokens, output});
        }

        auto executed = executor_->prefill_batch(cuda_items, linear, attention);
        if (!executed) return executed.status();

        PrefillBatchExecution result;
        result.physical_batches = executed.value().physical_batches;
        result.physical_sequence_participations =
            executed.value().physical_sequence_participations;
        result.physical_tokens = executed.value().physical_tokens;
        result.max_sequences = executed.value().max_sequences;
        result.items.reserve(executed.value().items.size());
        for (auto& item : executed.value().items) {
            result.items.push_back(PrefillBatchItemResult{
                item.greedy_token, std::move(item.logits)});
        }
        return result;
    }

    [[nodiscard]] Result<std::vector<TokenId>> decode_greedy_batch(
        std::span<const GreedyDecodeBatchItem> items) override {
        if (items.empty()) return Status::invalid_argument("CUDA decode batch must not be empty");
        if (items.size() > capabilities_.max_decode_batch_width) {
            return Status::unsupported("CUDA decode batch exceeds native width");
        }
        std::vector<TokenId> tokens;
        std::vector<CudaKvCache*> caches;
        tokens.reserve(items.size());
        caches.reserve(items.size());
        QuantizedLinearExecutionKind block_linear = QuantizedLinearExecutionKind::baseline;
        QuantizedLinearExecutionKind output_linear = QuantizedLinearExecutionKind::baseline;
        AttentionExecutionKind attention = AttentionExecutionKind::baseline;
        bool first = true;
        for (const auto& item : items) {
            auto* sequence = dynamic_cast<CudaSequence*>(item.sequence);
            if (!sequence) return Status::invalid_argument("CUDA decode batch contains a non-CUDA sequence");
            if (first) {
                block_linear = sequence->decode_block_linear();
                output_linear = sequence->decode_output_linear();
                attention = sequence->decode_attention();
                first = false;
            } else if (sequence->decode_block_linear() != block_linear ||
                       sequence->decode_output_linear() != output_linear ||
                       sequence->decode_attention() != attention) {
                return Status::invalid_argument("CUDA decode batch requires identical execution tactics");
            }
            tokens.push_back(item.token);
            caches.push_back(&sequence->cache());
        }
        return executor_->step_greedy_batch(tokens, caches, block_linear, output_linear, attention);
    }

    [[nodiscard]] std::uint64_t resident_device_bytes() const noexcept override {
        const auto stats = executor_->stats();
        return stats.resident_model_bytes + stats.workspace_bytes + stats.kv_pool_allocated_bytes;
    }

    [[nodiscard]] std::uint64_t prepared_artifact_device_bytes() const noexcept override {
        return executor_->stats().prepared_linear_bytes;
    }

    [[nodiscard]] std::uint64_t estimate_sequence_device_bytes(
        const ExecutionPlan& plan, std::uint64_t max_tokens) const noexcept override {
        const auto page_tokens = plan.kv.page_tokens.value_or(0U);
        if (page_tokens == 0U || max_tokens == 0U) return 0U;
        const auto& config = model_->config();
        const auto head_dimension = config.embedding_size / config.attention_head_count;
        const std::uint64_t kv_width = static_cast<std::uint64_t>(config.kv_head_count) * head_dimension;
        const std::uint64_t bytes_per_token = static_cast<std::uint64_t>(config.layer_count) * kv_width *
                                              2U * sizeof(float);
        const std::uint64_t pages = (max_tokens + page_tokens - 1U) / page_tokens;
        if (pages > std::numeric_limits<std::uint64_t>::max() /
                        (static_cast<std::uint64_t>(page_tokens) * bytes_per_token)) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        const std::uint64_t page_bytes = static_cast<std::uint64_t>(page_tokens) * bytes_per_token;
        const std::uint64_t max_pages = (config.context_length + page_tokens - 1U) / page_tokens;
        if (max_pages > std::numeric_limits<std::uint64_t>::max() / (2U * sizeof(float*))) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        const std::uint64_t page_table_bytes = max_pages * 2U * sizeof(float*);
        const std::uint64_t kv_bytes = pages * page_bytes;
        if (kv_bytes > std::numeric_limits<std::uint64_t>::max() - page_table_bytes) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        return kv_bytes + page_table_bytes;
    }

    [[nodiscard]] std::optional<std::uint64_t> sequence_capacity_bytes() const override {
        auto devices = cuda_devices();
        if (!devices) return std::nullopt;
        const auto it = std::find_if(devices.value().begin(), devices.value().end(), [&](const DeviceInfo& device) {
            return device.kind == DeviceKind::cuda && device.ordinal == device_ordinal_;
        });
        if (it == devices.value().end()) return std::nullopt;
        const auto pool = executor_->stats().kv_pool_allocated_bytes;
        if (it->free_memory_bytes > std::numeric_limits<std::uint64_t>::max() - pool) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        return it->free_memory_bytes + pool;
    }

    [[nodiscard]] std::optional<std::uint64_t> free_device_bytes() const override {
        auto devices = cuda_devices();
        if (!devices) return std::nullopt;
        const auto it = std::find_if(devices.value().begin(), devices.value().end(), [&](const DeviceInfo& device) {
            return device.kind == DeviceKind::cuda && device.ordinal == device_ordinal_;
        });
        if (it == devices.value().end()) return std::nullopt;
        return it->free_memory_bytes;
    }

    [[nodiscard]] std::uint64_t kv_pool_allocated_bytes() const noexcept override {
        return executor_->stats().kv_pool_allocated_bytes;
    }
    [[nodiscard]] std::uint64_t kv_pool_free_bytes() const noexcept override {
        return executor_->stats().kv_pool_free_bytes;
    }

private:
    std::shared_ptr<const ModelDefinition> model_;
    std::unique_ptr<CudaExecutor> executor_;
    int device_ordinal_{0};
    BackendCapabilities capabilities_{};
};

} // namespace

Result<std::unique_ptr<PreparedModel>> prepare_reference_model(
    std::shared_ptr<const ModelDefinition> model) {
    auto executor = ReferenceExecutor::create(model);
    if (!executor) return executor.status();
    return std::unique_ptr<PreparedModel>(
        std::make_unique<ReferencePreparedModel>(std::move(model), std::move(executor).value()));
}

Result<std::unique_ptr<PreparedModel>> prepare_cuda_model(
    std::shared_ptr<const ModelDefinition> model, int device_ordinal) {
    auto executor = CudaExecutor::create(model, device_ordinal);
    if (!executor) return executor.status();
    return std::unique_ptr<PreparedModel>(
        std::make_unique<CudaPreparedModel>(std::move(model), std::move(executor).value(), device_ordinal));
}

} // namespace air::runtime_detail

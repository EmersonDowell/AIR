#pragma once

#include "air/execution.hpp"
#include "air/model.hpp"
#include "air/result.hpp"
#include "air/types.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace air::runtime_detail {

struct SequenceResources {
    std::uint64_t committed_tokens{0};
    std::uint64_t committed_kv_bytes{0};
    std::uint64_t resident_device_bytes{0};
};

class SequenceCheckpoint {
public:
    virtual ~SequenceCheckpoint() = default;
    [[nodiscard]] virtual BackendKind backend() const noexcept = 0;
    [[nodiscard]] virtual std::uint64_t tokens() const noexcept = 0;
    [[nodiscard]] virtual SequenceResources resources() const noexcept = 0;
    [[nodiscard]] virtual Result<std::unique_ptr<SequenceCheckpoint>> clone() const = 0;
};

class SequenceState {
public:
    virtual ~SequenceState() = default;
    [[nodiscard]] virtual BackendKind backend() const noexcept = 0;
    [[nodiscard]] virtual Result<std::vector<float>> prefill(std::span<const TokenId> tokens) = 0;
    [[nodiscard]] virtual Status prefill_discard(std::span<const TokenId> tokens) {
        auto result = prefill(tokens);
        return result ? Status::ok() : result.status();
    }
    [[nodiscard]] virtual Result<TokenId> prefill_greedy(std::span<const TokenId>) {
        return Status::unsupported("backend does not provide device greedy prefill selection");
    }
    // Exact target-token log probabilities preserve full-vocabulary normalization
    // while allowing backends to avoid materializing full logits on the host.
    [[nodiscard]] virtual Result<std::vector<float>> prefill_target_logprobs(
        std::span<const TokenId>, std::span<const TokenId>) {
        return Status::unsupported("backend does not provide target-logprob prefill");
    }
    [[nodiscard]] virtual Result<std::vector<float>> decode(TokenId token) = 0;
    [[nodiscard]] virtual Result<std::vector<float>> decode_target_logprobs(
        TokenId, std::span<const TokenId>) {
        return Status::unsupported("backend does not provide target-logprob decode");
    }
    [[nodiscard]] virtual Result<TokenId> decode_greedy(TokenId) {
        return Status::unsupported("backend does not provide device greedy decode selection");
    }
    [[nodiscard]] virtual SequenceResources resources() const noexcept = 0;
    [[nodiscard]] virtual Result<std::unique_ptr<SequenceCheckpoint>> checkpoint() const = 0;
};


struct GreedyDecodeBatchItem {
    SequenceState* sequence{nullptr};
    TokenId token{0};
};


enum class PrefillBatchOutput : std::uint8_t {
    discard = 0,
    greedy,
    logits,
};

struct PrefillBatchItem {
    SequenceState* sequence{nullptr};
    std::span<const TokenId> tokens;
    PrefillBatchOutput output{PrefillBatchOutput::discard};
};

struct PrefillBatchItemResult {
    std::optional<TokenId> greedy_token;
    std::vector<float> logits;
};

struct PrefillBatchExecution {
    std::vector<PrefillBatchItemResult> items;
    std::uint32_t physical_batches{0};
    std::uint64_t physical_sequence_participations{0};
    std::uint64_t physical_tokens{0};
    std::uint32_t max_sequences{0};
};

// PreparedModel is the execution-owned representation of a canonical
// ModelDefinition. ModelDefinition remains the source of truth. A prepared
// backend may own device residency, packed weights, workspaces, page pools,
// or other derived execution state without becoming a second model definition.
class PreparedModel {
public:
    virtual ~PreparedModel() = default;
    [[nodiscard]] virtual BackendKind backend() const noexcept = 0;
    [[nodiscard]] virtual const BackendCapabilities& capabilities() const noexcept = 0;
    [[nodiscard]] virtual const ModelDefinition& model() const noexcept = 0;
    [[nodiscard]] virtual Result<std::unique_ptr<SequenceState>> create_sequence(
        const ExecutionPlan& plan) = 0;
    [[nodiscard]] virtual Result<std::unique_ptr<SequenceState>> restore_sequence(
        const ExecutionPlan& plan, const SequenceCheckpoint& checkpoint) = 0;
    // Optional backend-global prepared artifacts are resources, not model truth.
    // Admission may forecast their incremental device cost before materializing
    // them. The caller may trim artifacts only while no sequence is active.
    [[nodiscard]] virtual std::uint64_t estimate_plan_preparation_device_bytes(
        const ExecutionPlan&) const noexcept { return 0U; }
    [[nodiscard]] virtual Status trim_plan_artifacts(const ExecutionPlan&) { return Status::ok(); }
    // Materialize backend-global derived state only after the prospective plan
    // has passed capacity admission. Canonical ModelDefinition remains unchanged.
    [[nodiscard]] virtual Status prepare_plan(const ExecutionPlan&) { return Status::ok(); }
    // Peak temporary device bytes required to atomically checkpoint and
    // restore sequence_count active sequences under a new same-backend plan.
    // This excludes persistent plan artifacts and existing sequence state.
    [[nodiscard]] virtual std::uint64_t estimate_transition_temporary_device_bytes(
        const ExecutionPlan&, std::size_t) const noexcept { return 0U; }
    [[nodiscard]] virtual Result<PrefillBatchExecution> prefill_batch(
        std::span<const PrefillBatchItem>) {
        return Status::unsupported("prepared backend does not provide native multi-sequence prefill");
    }
    [[nodiscard]] virtual Result<std::vector<TokenId>> decode_greedy_batch(
        std::span<const GreedyDecodeBatchItem>) {
        return Status::unsupported("prepared backend does not provide native multi-sequence decode");
    }
    [[nodiscard]] virtual std::uint64_t resident_device_bytes() const noexcept = 0;
    [[nodiscard]] virtual std::uint64_t prepared_artifact_device_bytes() const noexcept { return 0U; }

    // Logical admission reservation for a sequence whose maximum committed
    // length is max_tokens. This is not necessarily eagerly allocated memory.
    [[nodiscard]] virtual std::uint64_t estimate_sequence_device_bytes(
        const ExecutionPlan& plan, std::uint64_t max_tokens) const noexcept = 0;

    // Total bytes currently usable for sequence-owned backend state. For a
    // page pool this includes reusable pages already owned by the pool plus
    // presently free device memory.
    [[nodiscard]] virtual std::optional<std::uint64_t> sequence_capacity_bytes() const = 0;
    [[nodiscard]] virtual std::optional<std::uint64_t> free_device_bytes() const = 0;

    [[nodiscard]] virtual std::uint64_t kv_pool_allocated_bytes() const noexcept = 0;
    [[nodiscard]] virtual std::uint64_t kv_pool_free_bytes() const noexcept = 0;
};

[[nodiscard]] Result<std::unique_ptr<PreparedModel>> prepare_reference_model(
    std::shared_ptr<const ModelDefinition> model);
[[nodiscard]] Result<std::unique_ptr<PreparedModel>> prepare_cuda_model(
    std::shared_ptr<const ModelDefinition> model, int device_ordinal);

} // namespace air::runtime_detail

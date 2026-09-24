#pragma once

#include "air/execution.hpp"
#include "runtime/backend.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace air::runtime_detail {

struct CapacityDecision {
    bool admit{false};
    std::uint64_t reservation_bytes{0};
    std::string reason;
};

struct TransitionCapacityDecision {
    bool admit{false};
    std::uint64_t plan_preparation_bytes{0};
    std::uint64_t temporary_device_bytes{0};
    std::string reason;
};

class CapacityScheduler final {
public:
    explicit CapacityScheduler(std::uint32_t max_active_requests)
        : max_active_requests_(max_active_requests) {}

    [[nodiscard]] CapacityDecision evaluate(std::size_t active_requests,
                                            std::uint64_t active_reserved_bytes,
                                            const PreparedModel& backend,
                                            const ExecutionPlan& plan,
                                            std::uint64_t requested_tokens,
                                            std::uint64_t plan_preparation_bytes = 0U) const;

    // Admission for an in-place execution-regime transition. This does not
    // reserve another sequence; existing sequence reservations remain valid.
    // It accounts for incremental prepared artifacts plus temporary migration
    // allocations required while old and restored sequence states coexist.
    [[nodiscard]] TransitionCapacityDecision evaluate_transition(
        const PreparedModel& backend,
        const ExecutionPlan& target_plan,
        std::uint64_t temporary_device_bytes) const;

private:
    std::uint32_t max_active_requests_{1};
};

enum class WorkPhase {
    prefill = 0,
    decode,
};

struct SchedulableSequence {
    std::size_t slot{0};
    WorkPhase phase{WorkPhase::prefill};
    // Maximum tokens this sequence may consume in this worker cycle.
    // Decode candidates use 1. Prefill candidates are bounded by remaining
    // prompt tokens, per-sequence quantum, and backend physical width.
    std::uint32_t max_tokens{1};
};

struct ScheduledSlice {
    std::size_t slot{0};
    WorkPhase phase{WorkPhase::prefill};
    std::uint32_t token_count{0};
};

struct ScheduledBatch {
    std::vector<ScheduledSlice> slices;
    std::uint32_t token_count{0};
    std::uint32_t decode_tokens{0};
    std::uint32_t prefill_tokens{0};
    std::uint32_t prefill_sequences{0};
};


// Physical execution compatibility identity. Evidence/strategy identifiers are
// deliberately excluded: different evidence records may lower to identical
// device work and must not fragment a physical batch for that reason alone.
struct ExecutionRegimeKey {
    BackendKind backend{BackendKind::reference};
    QuantizedLinearExecutionKind prefill_block{QuantizedLinearExecutionKind::baseline};
    QuantizedLinearExecutionKind decode_block{QuantizedLinearExecutionKind::baseline};
    QuantizedLinearExecutionKind decode_output{QuantizedLinearExecutionKind::baseline};
    AttentionExecutionKind prefill_attention{AttentionExecutionKind::baseline};
    AttentionExecutionKind decode_attention{AttentionExecutionKind::baseline};
    std::optional<std::uint32_t> kv_page_tokens;
    std::uint32_t prefill_quantum_tokens{0};

    [[nodiscard]] friend bool operator==(const ExecutionRegimeKey&,
                                         const ExecutionRegimeKey&) = default;
};

[[nodiscard]] inline ExecutionRegimeKey execution_regime_key(
    const ExecutionPlan& plan) noexcept {
    return ExecutionRegimeKey{
        plan.backend,
        plan.linear.prefill_block,
        plan.linear.decode_block,
        plan.linear.decode_output,
        plan.attention.prefill,
        plan.attention.decode,
        plan.kv.page_tokens,
        plan.scheduling.prefill_quantum_tokens,
    };
}

enum class RegimeAdmissionAction {
    compatible = 0,
    converge_to_fallback,
    use_pinned_fallback,
};

[[nodiscard]] inline RegimeAdmissionAction classify_regime_admission(
    const std::optional<ExecutionRegimeKey>& active_regime,
    const ExecutionRegimeKey& requested_regime,
    bool fallback_pinned) noexcept {
    if (fallback_pinned) return RegimeAdmissionAction::use_pinned_fallback;
    if (active_regime && *active_regime == requested_regime) {
        return RegimeAdmissionAction::compatible;
    }
    return RegimeAdmissionAction::converge_to_fallback;
}

struct RegimeRequestShape {
    std::uint64_t prompt_tokens{0};
    std::uint64_t max_output_tokens{0};
};

[[nodiscard]] inline bool same_regime_request_shape(
    const RegimeRequestShape& a,
    const RegimeRequestShape& b) noexcept {
    return a.prompt_tokens == b.prompt_tokens &&
           a.max_output_tokens == b.max_output_tokens;
}

[[nodiscard]] inline bool homogeneous_regime_shape_accepts(
    std::span<const RegimeRequestShape> active_shapes,
    const RegimeRequestShape& incoming) noexcept {
    if (active_shapes.empty()) return false;
    for (const auto& shape : active_shapes) {
        if (!same_regime_request_shape(shape, incoming)) return false;
    }
    return true;
}

// Return an exact group profile only when every participant has the same
// request-shape evidence dimensions. Heterogeneous groups remain unqualified;
// this helper never extrapolates a manifest region.
[[nodiscard]] std::optional<RequestProfile> homogeneous_regime_profile(
    std::span<const RegimeRequestShape> requests) noexcept;

class MicrobatchScheduler final {
public:
    // Decode-first ordering protects already-streaming requests from long
    // prompt starvation. Phase-local round robin prevents low-budget cycles
    // from repeatedly serving the same early slot.
    [[nodiscard]] std::vector<SchedulableSequence> order(
        std::span<const SchedulableSequence> candidates);

    // Own one worker-cycle token budget and return execution-ready slices.
    // This is the only micro-scheduling decision; ScheduledBatch is data.
    [[nodiscard]] ScheduledBatch schedule(
        std::span<const SchedulableSequence> candidates,
        std::uint32_t token_budget);

private:
    std::size_t decode_cursor_{0};
    std::size_t prefill_cursor_{0};
};

} // namespace air::runtime_detail

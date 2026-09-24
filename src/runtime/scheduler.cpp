#include "runtime/scheduler.hpp"

#include <algorithm>
#include <limits>

namespace air::runtime_detail {

CapacityDecision CapacityScheduler::evaluate(std::size_t active_requests,
                                             std::uint64_t active_reserved_bytes,
                                             const PreparedModel& backend,
                                             const ExecutionPlan& plan,
                                             std::uint64_t requested_tokens,
                                             std::uint64_t plan_preparation_bytes) const {
    if (active_requests >= max_active_requests_) {
        return CapacityDecision{false, 0U, "max-active-requests"};
    }

    const auto reservation = backend.estimate_sequence_device_bytes(plan, requested_tokens);

    // Prepared artifacts consume genuinely free device memory; unlike sequence
    // KV reservations they cannot be satisfied from already-owned free KV pages.
    if (plan_preparation_bytes != 0U) {
        const auto free = backend.free_device_bytes();
        if (free && plan_preparation_bytes > *free) {
            return CapacityDecision{false, reservation, "device-capacity-plan-artifact"};
        }
    }

    const auto capacity = backend.sequence_capacity_bytes();
    if (capacity) {
        if (plan_preparation_bytes > *capacity) {
            return CapacityDecision{false, reservation, "device-capacity-plan-artifact"};
        }
        const auto capacity_after_prepare = *capacity - plan_preparation_bytes;
        if (reservation > capacity_after_prepare ||
            active_reserved_bytes > capacity_after_prepare - reservation) {
            return CapacityDecision{false, reservation, "device-capacity"};
        }
    }
    return CapacityDecision{true, reservation, "admitted"};
}

TransitionCapacityDecision CapacityScheduler::evaluate_transition(
    const PreparedModel& backend,
    const ExecutionPlan& target_plan,
    std::uint64_t temporary_device_bytes) const {
    const auto preparation = backend.estimate_plan_preparation_device_bytes(target_plan);
    if (preparation > std::numeric_limits<std::uint64_t>::max() - temporary_device_bytes) {
        return TransitionCapacityDecision{false, preparation, temporary_device_bytes,
                                          "device-capacity-transition-overflow"};
    }
    const auto required = preparation + temporary_device_bytes;
    if (const auto free = backend.free_device_bytes(); free && required > *free) {
        return TransitionCapacityDecision{false, preparation, temporary_device_bytes,
                                          preparation != 0U
                                              ? "device-capacity-transition-plan-artifact"
                                              : "device-capacity-transition-temporary"};
    }
    return TransitionCapacityDecision{true, preparation, temporary_device_bytes, "admitted"};
}

std::vector<SchedulableSequence> MicrobatchScheduler::order(
    std::span<const SchedulableSequence> candidates) {
    std::vector<SchedulableSequence> decode;
    std::vector<SchedulableSequence> prefill;
    decode.reserve(candidates.size());
    prefill.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        (candidate.phase == WorkPhase::decode ? decode : prefill).push_back(candidate);
    }
    const auto rotate_phase = [](auto& phase, std::size_t& cursor) {
        if (phase.empty()) { cursor = 0U; return; }
        cursor %= phase.size();
        std::rotate(phase.begin(), phase.begin() + static_cast<std::ptrdiff_t>(cursor), phase.end());
        cursor = (cursor + 1U) % phase.size();
    };
    rotate_phase(decode, decode_cursor_);
    rotate_phase(prefill, prefill_cursor_);

    std::vector<SchedulableSequence> ordered;
    ordered.reserve(candidates.size());
    ordered.insert(ordered.end(), decode.begin(), decode.end());
    ordered.insert(ordered.end(), prefill.begin(), prefill.end());
    return ordered;
}



std::optional<RequestProfile> homogeneous_regime_profile(
    std::span<const RegimeRequestShape> requests) noexcept {
    if (requests.empty() ||
        requests.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        return std::nullopt;
    }
    const auto first = requests.front();
    for (const auto& request : requests.subspan(1U)) {
        if (request.prompt_tokens != first.prompt_tokens ||
            request.max_output_tokens != first.max_output_tokens) {
            return std::nullopt;
        }
    }
    return RequestProfile{
        first.prompt_tokens,
        first.max_output_tokens,
        static_cast<std::uint32_t>(requests.size()),
    };
}


ScheduledBatch MicrobatchScheduler::schedule(
    std::span<const SchedulableSequence> candidates,
    std::uint32_t token_budget) {
    ScheduledBatch batch;
    if (token_budget == 0U || candidates.empty()) return batch;

    const auto ordered = order(candidates);
    batch.slices.reserve(ordered.size());
    auto remaining_budget = token_budget;
    for (const auto& candidate : ordered) {
        if (remaining_budget == 0U) break;
        if (candidate.max_tokens == 0U) continue;

        const auto tokens = candidate.phase == WorkPhase::decode
            ? 1U
            : std::min(candidate.max_tokens, remaining_budget);
        if (tokens == 0U || tokens > remaining_budget) continue;

        batch.slices.push_back(ScheduledSlice{candidate.slot, candidate.phase, tokens});
        batch.token_count += tokens;
        remaining_budget -= tokens;
        if (candidate.phase == WorkPhase::decode) {
            ++batch.decode_tokens;
        } else {
            batch.prefill_tokens += tokens;
            ++batch.prefill_sequences;
        }
    }
    return batch;
}

} // namespace air::runtime_detail

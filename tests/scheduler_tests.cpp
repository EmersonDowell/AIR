#include "runtime/scheduler.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

class FakePreparedModel final : public air::runtime_detail::PreparedModel {
public:
    FakePreparedModel(std::uint64_t bytes_per_token,
                      std::optional<std::uint64_t> capacity,
                      std::uint64_t preparation_bytes = 0U)
        : bytes_per_token_(bytes_per_token), capacity_(capacity),
          preparation_bytes_(preparation_bytes) {
        capabilities_.backend = air::BackendKind::cuda;
        capabilities_.prefill_execution = air::PrefillExecutionKind::native_batch;
        capabilities_.kv_storage = air::KvStorageKind::paged;
        capabilities_.max_prefill_batch_width = 128U;
        capabilities_.max_decode_batch_width = 1U;
    }

    [[nodiscard]] air::BackendKind backend() const noexcept override {
        return air::BackendKind::cuda;
    }
    [[nodiscard]] const air::BackendCapabilities& capabilities() const noexcept override {
        return capabilities_;
    }
    [[nodiscard]] const air::ModelDefinition& model() const noexcept override { return model_; }

    [[nodiscard]] air::Result<std::unique_ptr<air::runtime_detail::SequenceState>>
    create_sequence(const air::ExecutionPlan&) override {
        return air::Status::unsupported("fake scheduler backend has no sequence");
    }
    [[nodiscard]] air::Result<std::unique_ptr<air::runtime_detail::SequenceState>>
    restore_sequence(const air::ExecutionPlan&,
                     const air::runtime_detail::SequenceCheckpoint&) override {
        return air::Status::unsupported("fake scheduler backend has no sequence");
    }

    [[nodiscard]] std::uint64_t estimate_plan_preparation_device_bytes(
        const air::ExecutionPlan&) const noexcept override { return preparation_bytes_; }
    [[nodiscard]] std::uint64_t resident_device_bytes() const noexcept override { return 0U; }
    [[nodiscard]] std::uint64_t estimate_sequence_device_bytes(
        const air::ExecutionPlan&, std::uint64_t max_tokens) const noexcept override {
        return max_tokens * bytes_per_token_;
    }
    [[nodiscard]] std::optional<std::uint64_t> sequence_capacity_bytes() const override {
        return capacity_;
    }
    [[nodiscard]] std::optional<std::uint64_t> free_device_bytes() const override {
        return capacity_;
    }
    [[nodiscard]] std::uint64_t kv_pool_allocated_bytes() const noexcept override { return 0U; }
    [[nodiscard]] std::uint64_t kv_pool_free_bytes() const noexcept override { return 0U; }

private:
    std::uint64_t bytes_per_token_{0};
    std::optional<std::uint64_t> capacity_;
    std::uint64_t preparation_bytes_{0};
    air::BackendCapabilities capabilities_{};
    air::ModelDefinition model_{};
};

void test_capacity_reservations() {
    FakePreparedModel backend(100U, 1000U);
    air::ExecutionPlan plan;
    plan.backend = air::BackendKind::cuda;
    plan.kv.page_tokens = 16U;

    air::runtime_detail::CapacityScheduler scheduler(4U);

    auto decision = scheduler.evaluate(0U, 0U, backend, plan, 4U);
    check(decision.admit, "first request should fit");
    check(decision.reservation_bytes == 400U, "reservation should use backend estimate");

    decision = scheduler.evaluate(1U, 400U, backend, plan, 6U);
    check(decision.admit, "reservation that exactly fills capacity should fit");
    check(decision.reservation_bytes == 600U, "second reservation should be 600 bytes");

    decision = scheduler.evaluate(1U, 401U, backend, plan, 6U);
    check(!decision.admit && decision.reason == "device-capacity",
          "logical future reservations must prevent overcommit");

    decision = scheduler.evaluate(0U, 0U, backend, plan, 11U);
    check(!decision.admit && decision.reason == "device-capacity" && decision.reservation_bytes == 1100U,
          "single request larger than device capacity must fail admission before sequence allocation");

    decision = scheduler.evaluate(4U, 0U, backend, plan, 1U);
    check(!decision.admit && decision.reason == "max-active-requests",
          "active-request ceiling must remain an independent admission guard");
}


void test_plan_preparation_capacity_is_reserved_before_materialization() {
    FakePreparedModel backend(100U, 1000U);
    air::ExecutionPlan plan;
    plan.backend = air::BackendKind::cuda;
    plan.kv.page_tokens = 16U;
    air::runtime_detail::CapacityScheduler scheduler(4U);

    auto decision = scheduler.evaluate(0U, 0U, backend, plan, 4U, 500U);
    check(decision.admit,
          "plan artifact plus sequence reservation that fits device capacity should admit");

    decision = scheduler.evaluate(0U, 0U, backend, plan, 6U, 500U);
    check(!decision.admit && decision.reason == "device-capacity",
          "prospective plan artifacts must reduce sequence admission capacity before allocation");

    decision = scheduler.evaluate(0U, 0U, backend, plan, 1U, 1100U);
    check(!decision.admit && decision.reason == "device-capacity-plan-artifact",
          "a plan artifact larger than genuinely free device memory must be rejected before preparation");
}


void test_transition_capacity_accounts_artifact_and_atomic_migration() {
    air::ExecutionPlan plan;
    plan.backend = air::BackendKind::cuda;
    plan.kv.page_tokens = 16U;
    air::runtime_detail::CapacityScheduler scheduler(4U);

    FakePreparedModel no_prepare(100U, 1000U);
    auto decision = scheduler.evaluate_transition(no_prepare, plan, 900U);
    check(decision.admit && decision.temporary_device_bytes == 900U,
          "transition temporary state fitting free device memory must admit");

    decision = scheduler.evaluate_transition(no_prepare, plan, 1001U);
    check(!decision.admit && decision.reason == "device-capacity-transition-temporary",
          "transition temporary state must be admitted against actual free device memory");

    FakePreparedModel with_prepare(100U, 1000U, 300U);
    decision = scheduler.evaluate_transition(with_prepare, plan, 700U);
    check(decision.admit && decision.plan_preparation_bytes == 300U &&
          decision.temporary_device_bytes == 700U,
          "plan artifact plus atomic migration temporary bytes that exactly fit must admit");

    decision = scheduler.evaluate_transition(with_prepare, plan, 701U);
    check(!decision.admit && decision.reason == "device-capacity-transition-plan-artifact",
          "transition admission must account for prepared artifacts plus temporary migration bytes");
}

void test_unbounded_reference_capacity() {
    FakePreparedModel backend(1000000U, std::nullopt);
    air::ExecutionPlan plan;
    air::runtime_detail::CapacityScheduler scheduler(2U);
    const auto decision = scheduler.evaluate(0U, 0U, backend, plan, 1000000U);
    check(decision.admit, "backend without a device capacity should be request-count bounded only");
}

void test_decode_first_microbatch_order() {
    const std::vector<air::runtime_detail::SchedulableSequence> candidates{
        {0U, air::runtime_detail::WorkPhase::prefill},
        {1U, air::runtime_detail::WorkPhase::decode},
        {2U, air::runtime_detail::WorkPhase::prefill},
        {3U, air::runtime_detail::WorkPhase::decode},
    };
    air::runtime_detail::MicrobatchScheduler scheduler;
    const auto ordered = scheduler.order(candidates);
    check(ordered.size() == candidates.size(), "microbatch scheduler must preserve candidates");
    if (ordered.size() == 4U) {
        check(ordered[0].slot == 1U && ordered[1].slot == 3U,
              "decode work must run before prefill work");
        check(ordered[2].slot == 0U && ordered[3].slot == 2U,
              "first pass must preserve phase-local input order before round-robin rotation");
    }
}


void test_round_robin_fairness_under_tiny_budget() {
    const std::vector<air::runtime_detail::SchedulableSequence> candidates{
        {0U, air::runtime_detail::WorkPhase::decode},
        {1U, air::runtime_detail::WorkPhase::decode},
        {2U, air::runtime_detail::WorkPhase::decode},
    };
    air::runtime_detail::MicrobatchScheduler scheduler;
    std::vector<std::size_t> first;
    for (int cycle = 0; cycle < 6; ++cycle) {
        const auto ordered = scheduler.order(candidates);
        if (!ordered.empty()) first.push_back(ordered.front().slot);
    }
    check(first == std::vector<std::size_t>({0U, 1U, 2U, 0U, 1U, 2U}),
          "decode sequences must round-robin when a cycle can serve only one slot");
}


void test_scheduled_batch_decode_priority_and_budget() {
    const std::vector<air::runtime_detail::SchedulableSequence> candidates{
        {0U, air::runtime_detail::WorkPhase::prefill, 4U},
        {1U, air::runtime_detail::WorkPhase::decode, 1U},
        {2U, air::runtime_detail::WorkPhase::prefill, 4U},
        {3U, air::runtime_detail::WorkPhase::decode, 1U},
    };
    air::runtime_detail::MicrobatchScheduler scheduler;
    const auto batch = scheduler.schedule(candidates, 5U);
    check(batch.token_count == 5U, "ScheduledBatch must not exceed the worker-cycle token budget");
    check(batch.decode_tokens == 2U, "decode work must consume protected budget first");
    check(batch.prefill_tokens == 3U, "prefill receives only the remaining token budget");
    check(batch.slices.size() == 3U &&
          batch.slices[0].slot == 1U && batch.slices[1].slot == 3U &&
          batch.slices[2].phase == air::runtime_detail::WorkPhase::prefill,
          "ScheduledBatch must preserve decode-first ordering");
}

void test_scheduled_batch_spans_prefill_sequences() {
    const std::vector<air::runtime_detail::SchedulableSequence> candidates{
        {0U, air::runtime_detail::WorkPhase::prefill, 4U},
        {1U, air::runtime_detail::WorkPhase::prefill, 4U},
        {2U, air::runtime_detail::WorkPhase::prefill, 4U},
    };
    air::runtime_detail::MicrobatchScheduler scheduler;
    const auto batch = scheduler.schedule(candidates, 9U);
    check(batch.prefill_sequences == 3U,
          "one ScheduledBatch must represent prefill slices from multiple sequences");
    check(batch.prefill_tokens == 9U && batch.token_count == 9U,
          "ScheduledBatch must own exact cross-sequence token allocation");
    check(batch.slices.size() == 3U &&
          batch.slices[0].token_count == 4U &&
          batch.slices[1].token_count == 4U &&
          batch.slices[2].token_count == 1U,
          "per-sequence token limits must be respected while filling the cycle budget");
}

void test_scheduled_batch_zero_limit_does_not_consume_budget() {
    const std::vector<air::runtime_detail::SchedulableSequence> candidates{
        {0U, air::runtime_detail::WorkPhase::prefill, 0U},
        {1U, air::runtime_detail::WorkPhase::prefill, 3U},
    };
    air::runtime_detail::MicrobatchScheduler scheduler;
    const auto batch = scheduler.schedule(candidates, 3U);
    check(batch.slices.size() == 1U && batch.slices.front().slot == 1U,
          "non-runnable/cancelled/done candidates represented by zero limit must not consume budget");
    check(batch.token_count == 3U, "zero-limit candidates must not reduce usable token budget");
}

void test_scheduled_batch_round_robin_fairness() {
    const std::vector<air::runtime_detail::SchedulableSequence> candidates{
        {0U, air::runtime_detail::WorkPhase::prefill, 1U},
        {1U, air::runtime_detail::WorkPhase::prefill, 1U},
        {2U, air::runtime_detail::WorkPhase::prefill, 1U},
    };
    air::runtime_detail::MicrobatchScheduler scheduler;
    std::vector<std::size_t> first;
    for (int cycle = 0; cycle < 6; ++cycle) {
        const auto batch = scheduler.schedule(candidates, 1U);
        if (!batch.slices.empty()) first.push_back(batch.slices.front().slot);
    }
    check(first == std::vector<std::size_t>({0U, 1U, 2U, 0U, 1U, 2U}),
          "ScheduledBatch must preserve phase-local round-robin fairness under tiny budgets");
}


void test_execution_regime_identity_ignores_strategy_label() {
    air::ExecutionPlan a;
    a.backend = air::BackendKind::cuda;
    a.strategy_id = "qualified-a";
    a.scheduling.prefill_quantum_tokens = 32U;
    a.kv.page_tokens = 16U;
    a.linear.prefill_block = air::QuantizedLinearExecutionKind::batch_reuse8;
    a.linear.decode_block = air::QuantizedLinearExecutionKind::batch_reuse8;
    a.linear.decode_output = air::QuantizedLinearExecutionKind::batch_reuse8;
    a.attention.prefill = air::AttentionExecutionKind::online_softmax;
    a.attention.decode = air::AttentionExecutionKind::baseline;

    auto b = a;
    b.strategy_id = "same-physical-plan-different-evidence";
    check(air::runtime_detail::execution_regime_key(a) ==
              air::runtime_detail::execution_regime_key(b),
          "strategy/evidence identity must not fragment the same physical regime");

    b.linear.decode_block = air::QuantizedLinearExecutionKind::dense_f32_cublas;
    check(!(air::runtime_detail::execution_regime_key(a) ==
            air::runtime_detail::execution_regime_key(b)),
          "a physical tactic change must change ExecutionRegime identity");
}

void test_regime_admission_lifecycle_policy() {
    air::ExecutionPlan dense;
    dense.backend = air::BackendKind::cuda;
    dense.strategy_id = "dense";
    dense.kv.page_tokens = 16U;
    dense.scheduling.prefill_quantum_tokens = 32U;
    dense.linear.prefill_block = air::QuantizedLinearExecutionKind::dense_f32_cublas;
    dense.linear.decode_block = air::QuantizedLinearExecutionKind::dense_f32_cublas;
    dense.linear.decode_output = air::QuantizedLinearExecutionKind::batch_reuse8;
    dense.attention.prefill = air::AttentionExecutionKind::online_softmax;

    auto same = dense;
    same.strategy_id = "different-evidence-same-physical";
    auto fallback = dense;
    fallback.strategy_id = "fallback";
    fallback.linear.prefill_block = air::QuantizedLinearExecutionKind::batch_reuse8;
    fallback.linear.decode_block = air::QuantizedLinearExecutionKind::batch_reuse8;

    const auto dense_key = air::runtime_detail::execution_regime_key(dense);
    const auto same_key = air::runtime_detail::execution_regime_key(same);
    const auto fallback_key = air::runtime_detail::execution_regime_key(fallback);

    check(air::runtime_detail::classify_regime_admission(
              dense_key, same_key, false) ==
              air::runtime_detail::RegimeAdmissionAction::compatible,
          "physically compatible arrivals must not force a regime change");

    check(air::runtime_detail::classify_regime_admission(
              dense_key, fallback_key, false) ==
              air::runtime_detail::RegimeAdmissionAction::converge_to_fallback,
          "the first incompatible arrival must request one fallback convergence");

    check(air::runtime_detail::classify_regime_admission(
              fallback_key, dense_key, true) ==
              air::runtime_detail::RegimeAdmissionAction::use_pinned_fallback,
          "once an epoch is deconverged, later arrivals must not cause tactic thrash");
}

void test_homogeneous_shape_arrival_classification() {
    const std::vector<air::runtime_detail::RegimeRequestShape> active{
        {160U, 32U}, {160U, 32U}, {160U, 32U}, {160U, 32U},
        {160U, 32U}, {160U, 32U},
    };
    check(air::runtime_detail::homogeneous_regime_shape_accepts(
              active, {160U, 32U}),
          "same-shape arrivals must remain eligible for later exact group convergence");
    check(!air::runtime_detail::homogeneous_regime_shape_accepts(
              active, {54U, 32U}),
          "different prompt shape must be treated as a genuinely incompatible arrival");
    check(!air::runtime_detail::homogeneous_regime_shape_accepts(
              active, {160U, 64U}),
          "different output horizon must be treated as a genuinely incompatible arrival");

    auto heterogeneous = active;
    heterogeneous.back().prompt_tokens = 54U;
    check(!air::runtime_detail::homogeneous_regime_shape_accepts(
              heterogeneous, {160U, 32U}),
          "an already heterogeneous live group must never manufacture shape compatibility");
}

void test_homogeneous_regime_profile_is_exact_only() {
    const std::vector<air::runtime_detail::RegimeRequestShape> same(8U, {160U, 32U});
    const auto profile = air::runtime_detail::homogeneous_regime_profile(same);
    check(profile && profile->prompt_tokens == 160U &&
          profile->max_output_tokens == 32U && profile->active_sequences == 8U,
          "homogeneous c8 group yields exact planner profile");

    auto mixed = same;
    mixed[0].prompt_tokens = 1042U;
    check(!air::runtime_detail::homogeneous_regime_profile(mixed),
          "heterogeneous prompt shapes must not manufacture one qualified regime profile");
}

} // namespace

int main() {
    test_capacity_reservations();
    test_plan_preparation_capacity_is_reserved_before_materialization();
    test_unbounded_reference_capacity();
    test_decode_first_microbatch_order();
    test_round_robin_fairness_under_tiny_budget();
    test_scheduled_batch_decode_priority_and_budget();
    test_scheduled_batch_spans_prefill_sequences();
    test_scheduled_batch_zero_limit_does_not_consume_budget();
    test_scheduled_batch_round_robin_fairness();
    test_execution_regime_identity_ignores_strategy_label();
    test_regime_admission_lifecycle_policy();
    test_homogeneous_shape_arrival_classification();
    test_homogeneous_regime_profile_is_exact_only();
    test_transition_capacity_accounts_artifact_and_atomic_migration();
    if (failures != 0) {
        std::cerr << failures << " scheduler test(s) failed\n";
        return 1;
    }
    std::cout << "scheduler tests passed\n";
    return 0;
}

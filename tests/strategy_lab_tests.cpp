#include "air/manifest.hpp"
#include "air/version.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

namespace {
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}

air::ExecutionPlan reuse_plan() {
    air::ExecutionPlan p;
    p.backend = air::BackendKind::cuda;
    p.strategy_id = "reuse8";
    p.scheduling.prefill_quantum_tokens = 32;
    p.kv.page_tokens = 16;
    p.linear.prefill_block = air::QuantizedLinearExecutionKind::batch_reuse8;
    p.linear.decode_block = air::QuantizedLinearExecutionKind::batch_reuse8;
    p.linear.decode_output = air::QuantizedLinearExecutionKind::batch_reuse8;
    p.attention.prefill = air::AttentionExecutionKind::online_softmax;
    return p;
}

air::ExecutionPlan dense_plan() {
    auto p = reuse_plan();
    p.strategy_id = "dense";
    p.linear.prefill_block = air::QuantizedLinearExecutionKind::dense_f32_cublas;
    p.linear.decode_block = air::QuantizedLinearExecutionKind::dense_f32_cublas;
    return p;
}

air::QualifiedStrategy reuse_strategy() {
    air::QualifiedStrategy s;
    s.workload = air::WorkloadClass::medium;
    s.region = air::WorkloadRegion{0, 2048, 1, 8};
    s.strategy_id = "reuse8";
    s.plan = reuse_plan();
    s.strict_qualified = true;
    s.samples = 5;
    s.mean_prefill_tokens_per_second = 380;
    s.prefill_tokens_per_second_confidence_half_width = 10;
    s.mean_decode_tokens_per_second = 90;
    s.decode_tokens_per_second_confidence_half_width = 3;
    s.aggregate_generated_tokens_per_second = 272;
    s.aggregate_generated_tokens_per_second_confidence_half_width = 4;
    s.p50_ttft_ms = 800;
    s.p50_ttft_confidence_half_width_ms = 20;
    s.p50_total_ms = 1200;
    s.p50_total_confidence_half_width_ms = 30;
    s.prepared_artifact_bytes = 0;
    s.preparation_measured = true;
    s.eviction_measured = true;
    s.evidence_id = "reuse-evidence";
    return s;
}

air::QualifiedStrategy dense_strategy() {
    auto s = reuse_strategy();
    s.strategy_id = "dense";
    s.plan = dense_plan();
    s.mean_prefill_tokens_per_second = 740;
    s.prefill_tokens_per_second_confidence_half_width = 20;
    s.mean_decode_tokens_per_second = 103;
    s.decode_tokens_per_second_confidence_half_width = 3;
    s.aggregate_generated_tokens_per_second = 311;
    s.aggregate_generated_tokens_per_second_confidence_half_width = 5;
    s.p50_ttft_ms = 430;
    s.p50_ttft_confidence_half_width_ms = 15;
    s.p50_total_ms = 700;
    s.p50_total_confidence_half_width_ms = 20;
    s.prepared_artifact_bytes = 1431306240ULL;
    s.preparation_measured = true;
    s.preparation_ms_mean = 1000;
    s.preparation_ms_confidence_half_width = 100;
    s.eviction_measured = true;
    s.eviction_ms_mean = 40;
    s.eviction_ms_confidence_half_width = 5;
    s.evidence_id = "dense-evidence";
    return s;
}

air::ExecutionManifest manifest() {
    air::ExecutionManifest m;
    m.schema_version = air::execution_manifest_schema_version;
    m.air_version = air::version_string();
    m.model_digest = "model";
    m.hardware_digest = "hardware";
    m.manifest_id = "strategy-test";
    m.strategies = {reuse_strategy(), dense_strategy()};
    return m;
}

air::PlanningDecision decide(air::ExecutionManifest m, air::StrategyLabConfig cfg,
                             air::RequestProfile request, air::RuntimeSnapshot runtime = {}) {
    air::ModelDefinition model;
    air::StrategyLabPlanner planner(std::move(m), reuse_plan(), cfg);
    return planner.decide(air::PlanningInput{model, request, runtime});
}

bool disposition(const air::PlanningDecision& d, const std::string& id, const std::string& value) {
    for (const auto& c : d.candidates) if (c.strategy_id == id) return c.disposition == value;
    return false;
}

} // namespace

int main() {
    air::StrategyLabConfig throughput;
    throughput.objective = air::StrategyObjective::maximum_throughput;

    // Cold, short work cannot amortize one second of preparation.
    auto short_cold = decide(manifest(), throughput, {64, 8, 1}, {8ULL << 30, 0, 0, "reuse8", 0.0});
    check(short_cold.plan.strategy_id == "reuse8", "short cold workload retains reuse8 when dense cannot amortize preparation");

    // Long horizon crosses the conservative break-even boundary.
    auto long_cfg = throughput;
    long_cfg.expected_horizon_tokens = 10000;
    auto long_cold = decide(manifest(), long_cfg, {1024, 128, 1}, {8ULL << 30, 0, 0, "reuse8", 0.0});
    check(long_cold.plan.strategy_id == "dense", "long workload selects dense after break-even");
    check(long_cold.estimated_transition_ms > 0.0, "cold dense selection exposes preparation transition cost");

    // Hot dense state naturally supplies hysteresis without timers.
    auto hot = decide(manifest(), throughput, {1024, 128, 1},
                      {8ULL << 30, 0, 1431306240ULL, "dense", 0.0});
    check(hot.plan.strategy_id == "dense" && hot.prepared_state_hot && hot.estimated_transition_ms == 0.0,
          "hot dense state is sticky because repeated work has zero transition cost");

    // Explicit user budget can make dense infeasible even when device free memory is ample.
    auto budget = throughput;
    budget.prepared_memory_budget_bytes = 512ULL << 20;
    auto constrained = decide(manifest(), budget, {1024, 128, 1}, {8ULL << 30, 0, 0, "reuse8", 0.0});
    check(constrained.plan.strategy_id == "reuse8" && disposition(constrained, "dense", "rejected:memory-infeasible"),
          "prepared-state budget rejects dense and selects reuse8");

    // Unqualified tactics are never considered product candidates.
    auto unqualified_manifest = manifest();
    unqualified_manifest.strategies[1].strict_qualified = false;
    auto unqualified = decide(std::move(unqualified_manifest), long_cfg, {1024, 128, 1}, {8ULL << 30, 0, 0, "reuse8", 0.0});
    check(unqualified.plan.strategy_id == "reuse8" && disposition(unqualified, "dense", "rejected:not-strict-qualified"),
          "non-strict-qualified dense tactic cannot be selected");

    // Higher residency is not justified when confidence bands overlap.
    auto overlap_manifest = manifest();
    overlap_manifest.strategies[1].mean_prefill_tokens_per_second = 430;
    overlap_manifest.strategies[1].prefill_tokens_per_second_confidence_half_width = 60;
    auto overlap = decide(std::move(overlap_manifest), long_cfg, {1024, 128, 1}, {8ULL << 30, 0, 0, "reuse8", 0.0});
    check(overlap.plan.strategy_id == "reuse8" && disposition(overlap, "dense", "rejected:confidence-overlap-with-lower-memory"),
          "confidence overlap conservatively favors lower residency");

    // Unknown preparation cost is not measured zero.
    auto no_prep_manifest = manifest();
    no_prep_manifest.strategies[1].preparation_measured = false;
    auto no_prep = decide(std::move(no_prep_manifest), long_cfg, {1024, 128, 1}, {8ULL << 30, 0, 0, "reuse8", 0.0});
    check(disposition(no_prep, "dense", "rejected:missing-preparation-evidence"),
          "cold dense candidate with unknown preparation cost is rejected");

    // Unknown eviction cost is not zero when a resident dense artifact would be dropped.
    auto no_evict_manifest = manifest();
    no_evict_manifest.strategies[1].eviction_measured = false;
    air::StrategyLabConfig min_vram;
    min_vram.objective = air::StrategyObjective::minimum_vram;
    auto no_evict = decide(std::move(no_evict_manifest), min_vram, {64, 8, 1},
                           {8ULL << 30, 0, 1431306240ULL, "dense", 0.0});
    check(disposition(no_evict, "reuse8", "rejected:missing-eviction-evidence") && no_evict.plan.strategy_id == "dense",
          "planner will not pretend eviction is free when its cost is unknown");

    // Minimum-VRAM is lexicographic when transition evidence exists.
    auto min = decide(manifest(), min_vram, {1024, 128, 1}, {8ULL << 30, 0, 0, "reuse8", 0.0});
    check(min.plan.strategy_id == "reuse8", "minimum-vram objective prefers zero-prepared-residency tactic");

    // Operation scope survives strategy selection.
    check(long_cold.plan.linear.decode_output == air::QuantizedLinearExecutionKind::batch_reuse8,
          "dense transformer-block strategy never leaks into decode_output");

    // Manifest freshness is independently tied to AIR/model/hardware identity.
    air::ModelDefinition default_model;
    auto stale = manifest();
    stale.air_version = "0.0.0";
    const auto validated = air::validate_manifest(stale, default_model);
    check(!validated.manifest && validated.status == "stale:air-version", "stale AIR evidence is rejected");

    if (failures) { std::cerr << failures << " strategy lab test(s) failed\n"; return 1; }
    std::cout << "strategy lab tests passed\n";
    return 0;
}

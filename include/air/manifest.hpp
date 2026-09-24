#pragma once

#include "air/execution.hpp"
#include "air/model.hpp"
#include "air/result.hpp"
#include "air/runtime.hpp"

#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace air {

inline constexpr std::uint32_t execution_manifest_schema_version = 10U;

enum class WorkloadClass {
    small = 0,
    medium,
    large,
    concurrent,
};

[[nodiscard]] const char* to_string(WorkloadClass workload) noexcept;
[[nodiscard]] WorkloadClass classify_workload(const RequestProfile& request) noexcept;

enum class StrategyObjective {
    interactive = 0,
    balanced,
    maximum_throughput,
    minimum_vram,
};

[[nodiscard]] const char* to_string(StrategyObjective objective) noexcept;
[[nodiscard]] Result<StrategyObjective> strategy_objective_from_string(std::string_view value);

struct StrategyLabConfig {
    StrategyObjective objective{StrategyObjective::balanced};
    // Zero means use the current request prompt+output horizon.
    std::uint64_t expected_horizon_tokens{0};
    // Zero means no additional user-imposed prepared-state cap. Device capacity
    // remains authoritative regardless of this value.
    std::uint64_t prepared_memory_budget_bytes{0};
    // Qualification-only escape hatch used by air-strategy-probe to measure an
    // eviction cost that cannot exist before the first measured transition.
    // General server/benchmark CLIs do not expose this flag.
    bool allow_unmeasured_eviction_for_qualification{false};
};

struct WorkloadRegion {
    std::uint64_t min_prompt_tokens{0};
    // Zero means unbounded.
    std::uint64_t max_prompt_tokens{0};
    std::uint32_t min_active_sequences{1};
    // Zero means unbounded.
    std::uint32_t max_active_sequences{0};

    [[nodiscard]] bool contains(const RequestProfile& request) const noexcept;
};

struct QualifiedStrategy {
    WorkloadClass workload{WorkloadClass::small};
    WorkloadRegion region{};
    std::string strategy_id;
    ExecutionPlan plan;

    // Product selection requires strict qualification. Research-only numerical
    // classifications must never set this true.
    bool strict_qualified{false};

    // Legacy qualification evidence retained for backward-compatible tooling.
    // Strategy Lab does not select directly from this opaque score.
    double score{0.0};
    double score_stddev{0.0};
    std::string selection_reason;
    double selection_margin{0.0};
    double selection_confidence_half_width{0.0};

    std::uint32_t samples{0};
    double p50_ttft_ms{0.0};
    double p50_ttft_stddev_ms{0.0};
    double p50_ttft_confidence_half_width_ms{0.0};
    double p50_total_ms{0.0};
    double p50_total_stddev_ms{0.0};
    double p50_total_confidence_half_width_ms{0.0};
    double mean_prefill_tokens_per_second{0.0};
    double prefill_tokens_per_second_stddev{0.0};
    double prefill_tokens_per_second_confidence_half_width{0.0};
    double mean_decode_tokens_per_second{0.0};
    double decode_tokens_per_second_stddev{0.0};
    double decode_tokens_per_second_confidence_half_width{0.0};
    double aggregate_generated_tokens_per_second{0.0};
    double aggregate_generated_tokens_per_second_confidence_half_width{0.0};

    std::uint64_t peak_kv_bytes{0};
    std::uint64_t peak_device_bytes{0};
    std::uint64_t prepared_artifact_bytes{0};

    bool preparation_measured{false};
    double preparation_ms_mean{0.0};
    double preparation_ms_stddev{0.0};
    double preparation_ms_confidence_half_width{0.0};
    bool eviction_measured{false};
    double eviction_ms_mean{0.0};
    double eviction_ms_stddev{0.0};
    double eviction_ms_confidence_half_width{0.0};

    std::uint64_t evidence_seed{0};
    std::string evidence_id;
};

struct ExecutionManifest {
    std::uint32_t schema_version{execution_manifest_schema_version};
    std::string air_version;
    std::string model_digest;
    std::string hardware_digest;
    std::string manifest_id;
    std::vector<QualifiedStrategy> strategies;

    [[nodiscard]] std::vector<const QualifiedStrategy*> candidates(const RequestProfile& request) const;
    [[nodiscard]] const QualifiedStrategy* find_strategy(std::string_view strategy_id) const noexcept;
};

struct ManifestLoadResult {
    std::optional<ExecutionManifest> manifest;
    std::string status{"not-found"};
};

struct StrategyCandidateDecision {
    std::string strategy_id;
    std::string disposition{"rejected:unknown"};
    bool eligible{false};
    bool memory_feasible{false};
    bool prepared_state_hot{false};
    double estimated_transition_ms{0.0};
    double estimated_horizon_ms{0.0};
    double estimated_break_even_tokens{0.0};
    double conservative_prefill_tokens_per_second{0.0};
    double conservative_decode_tokens_per_second{0.0};
    double conservative_aggregate_tokens_per_second{0.0};
    std::uint64_t prepared_artifact_bytes{0};
};

struct StrategyDecision {
    ExecutionPlan plan;
    StrategyObjective objective{StrategyObjective::balanced};
    std::string reason{"fallback-static"};
    std::uint32_t eligible_candidates{0};
    bool prepared_state_hot{false};
    double estimated_transition_ms{0.0};
    double estimated_break_even_tokens{0.0};
    std::vector<StrategyCandidateDecision> candidates;
};

[[nodiscard]] std::string model_digest(const ModelDefinition& model);
[[nodiscard]] std::string hardware_digest();
[[nodiscard]] std::filesystem::path default_manifest_path(const ModelDefinition& model);
[[nodiscard]] Status save_manifest(const ExecutionManifest& manifest, const std::filesystem::path& path);
[[nodiscard]] Result<ExecutionManifest> load_manifest(const std::filesystem::path& path);
[[nodiscard]] ManifestLoadResult validate_manifest(const ExecutionManifest& manifest,
                                                   const ModelDefinition& model);

class StrategyLabPlanner final : public Planner {
public:
    StrategyLabPlanner(ExecutionManifest manifest, ExecutionPlan fallback,
                       StrategyLabConfig config = {});
    [[nodiscard]] PlanningDecision decide(const PlanningInput& input) const override;
    [[nodiscard]] const ExecutionManifest& manifest() const noexcept { return manifest_; }
    [[nodiscard]] const StrategyLabConfig& config() const noexcept { return config_; }
private:
    ExecutionManifest manifest_;
    ExecutionPlan fallback_;
    StrategyLabConfig config_;
};

// Source compatibility for pre-Strategy-Lab callers. The implementation is
// transition-aware StrategyLabPlanner; there is no separate legacy planner.
using ManifestPlanner = StrategyLabPlanner;

} // namespace air

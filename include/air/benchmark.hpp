#pragma once

#include "air/serving.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace air {

inline constexpr std::string_view benchmark_report_schema = "air.benchmark.v11";

struct BenchmarkConfig {
    InferenceRequest request;
    std::uint32_t warmup_runs{1};
    std::uint32_t measured_runs{5};
    std::uint32_t concurrency{1};
};

struct BenchmarkSummary {
    std::uint64_t prompt_tokens{0};
    std::uint64_t generated_tokens{0};
    std::uint64_t prefix_reused_tokens{0};
    std::uint64_t peak_kv_bytes{0};
    std::uint64_t peak_device_bytes{0};
    std::uint64_t prepared_artifact_bytes{0};
    std::uint64_t native_decode_batches{0};
    std::uint64_t native_decode_sequences{0};
    std::uint64_t physical_prefill_batches{0};
    std::uint64_t physical_prefill_sequence_participations{0};
    std::uint64_t physical_prefill_tokens{0};
    std::uint32_t physical_prefill_max_sequences{0};
    double physical_prefill_average_sequences{0.0};
    double wall_ms{0.0};
    double aggregate_generated_tokens_per_second{0.0};
    double requests_per_second{0.0};
    double mean_plan_preparation_ms{0.0};
    double mean_plan_eviction_ms{0.0};
    double mean_prefill_tokens_per_second{0.0};
    double mean_decode_tokens_per_second{0.0};
    double p50_ttft_ms{0.0};
    double p95_ttft_ms{0.0};
    double p50_total_ms{0.0};
    double p95_total_ms{0.0};
};

struct BenchmarkReport {
    std::string air_version;
    std::string model_id;
    std::string model_architecture;
    std::string backend;
    std::string planner_mode{"static"};
    std::string manifest_id;
    std::string strategy_id{"static"};
    std::string strategy_objective{"static"};
    std::string strategy_decision_reason{"static"};
    std::string prefill_block_linear_tactic{"baseline"};
    std::string decode_block_linear_tactic{"baseline"};
    std::string decode_output_linear_tactic{"baseline"};
    std::string prefill_attention_tactic{"baseline"};
    std::string decode_attention_tactic{"baseline"};
    std::uint32_t planned_prefill_quantum_tokens{0};
    std::uint32_t planned_kv_page_tokens{0};
    std::uint64_t started_unix_ms{0};
    BenchmarkConfig config;
    BenchmarkSummary summary;
    std::vector<RequestMetrics> runs;
    std::vector<std::vector<TokenId>> output_tokens;

    [[nodiscard]] std::string to_json() const;
};

[[nodiscard]] Result<BenchmarkReport> run_benchmark(InferenceService& service,
                                                    const BenchmarkConfig& config);

} // namespace air

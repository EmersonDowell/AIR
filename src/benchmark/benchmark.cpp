#include "air/benchmark.hpp"

#include "air/version.hpp"

#include "json.hpp"
#include "../statistics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string_view>

namespace air {
namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t unix_ms_now() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

double mean_positive(const std::vector<double>& values) {
    double sum = 0.0;
    std::size_t count = 0;
    for (const double value : values) {
        if (value > 0.0 && std::isfinite(value)) {
            sum += value;
            ++count;
        }
    }
    return count == 0U ? 0.0 : sum / static_cast<double>(count);
}


void write_metrics(std::ostringstream& out, const RequestMetrics& metrics) {
    out << "{\"request_id\":" << metrics.request_id
        << ",\"sequence_id\":" << metrics.sequence_id
        << ",\"backend\":\"" << detail::json_escape(metrics.backend) << "\""
        << ",\"planner_mode\":\"" << detail::json_escape(metrics.planner_mode) << "\""
        << ",\"strategy_id\":\"" << detail::json_escape(metrics.strategy_id) << "\""
        << ",\"prompt_tokens\":" << metrics.prompt_tokens
        << ",\"generated_tokens\":" << metrics.generated_tokens
        << ",\"prefix_reused_tokens\":" << metrics.prefix_reused_tokens
        << ",\"kv_bytes\":" << metrics.kv_bytes
        << ",\"plan_preparation_bytes\":" << metrics.plan_preparation_bytes
        << ",\"plan_preparation_ms\":" << metrics.plan_preparation_ms
        << ",\"plan_eviction_ms\":" << metrics.plan_eviction_ms
        << ",\"strategy_objective\":\"" << detail::json_escape(metrics.strategy_objective) << "\""
        << ",\"strategy_decision_reason\":\"" << detail::json_escape(metrics.strategy_decision_reason) << "\""
        << ",\"strategy_eligible_candidates\":" << metrics.strategy_eligible_candidates
        << ",\"strategy_prepared_state_hot\":" << (metrics.strategy_prepared_state_hot ? "true" : "false")
        << ",\"strategy_estimated_transition_ms\":" << metrics.strategy_estimated_transition_ms
        << ",\"strategy_estimated_break_even_tokens\":" << metrics.strategy_estimated_break_even_tokens
        << ",\"strategy_candidates\":[";
    for (std::size_t i = 0; i < metrics.strategy_candidates.size(); ++i) {
        if (i) out << ',';
        const auto& c = metrics.strategy_candidates[i];
        out << "{\"strategy_id\":\"" << detail::json_escape(c.strategy_id) << "\""
            << ",\"disposition\":\"" << detail::json_escape(c.disposition) << "\""
            << ",\"eligible\":" << (c.eligible ? "true" : "false")
            << ",\"memory_feasible\":" << (c.memory_feasible ? "true" : "false")
            << ",\"prepared_state_hot\":" << (c.prepared_state_hot ? "true" : "false")
            << ",\"estimated_transition_ms\":" << c.estimated_transition_ms
            << ",\"estimated_horizon_ms\":" << c.estimated_horizon_ms
            << ",\"estimated_break_even_tokens\":" << c.estimated_break_even_tokens
            << ",\"prepared_artifact_bytes\":" << c.prepared_artifact_bytes << '}';
    }
    out << ']'
        << ",\"queue_ms\":" << metrics.queue_ms
        << ",\"prefill_ms\":" << metrics.prefill_ms
        << ",\"ttft_ms\":" << metrics.ttft_ms
        << ",\"decode_ms\":" << metrics.decode_ms
        << ",\"total_ms\":" << metrics.total_ms
        << ",\"prefill_tokens_per_second\":" << metrics.prefill_tokens_per_second()
        << ",\"decode_tokens_per_second\":" << metrics.decode_tokens_per_second()
        << '}';
}

} // namespace

Result<BenchmarkReport> run_benchmark(InferenceService& service, const BenchmarkConfig& config) {
    if (config.measured_runs == 0U) return Status::invalid_argument("benchmark requires at least one measured run");
    if (config.concurrency == 0U) return Status::invalid_argument("benchmark concurrency must be non-zero");
    if (config.request.prompt.empty() && config.request.messages.empty()) {
        return Status::invalid_argument("benchmark requires a prompt or chat messages");
    }

    for (std::uint32_t i = 0; i < config.warmup_runs; ++i) {
        auto warmup = service.generate(config.request);
        if (!warmup) return warmup.status();
    }

    BenchmarkReport report;
    report.air_version = version_string();
    report.model_id = service.model().fingerprint().model_id;
    report.model_architecture = service.model().config().architecture;
    report.backend = service.backend_name();
    {
        const auto initial = service.snapshot();
        report.planner_mode = initial.planner_mode;
        report.manifest_id = initial.manifest_id;
        report.strategy_id = initial.strategy_id;
        report.strategy_objective = initial.strategy_objective;
        report.strategy_decision_reason = initial.strategy_decision_reason;
        report.prefill_block_linear_tactic = initial.planned_prefill_block_linear_tactic;
        report.decode_block_linear_tactic = initial.planned_decode_block_linear_tactic;
        report.decode_output_linear_tactic = initial.planned_decode_output_linear_tactic;
        report.prefill_attention_tactic = initial.planned_prefill_attention_tactic;
        report.decode_attention_tactic = initial.planned_decode_attention_tactic;
        report.planned_prefill_quantum_tokens = initial.planned_prefill_quantum_tokens;
        report.planned_kv_page_tokens = initial.planned_kv_page_tokens;
    }
    report.started_unix_ms = unix_ms_now();
    report.config = config;
    report.runs.reserve(config.measured_runs);
    report.output_tokens.reserve(config.measured_runs);
    const auto initial_snapshot = service.snapshot();
    const auto initial_native_decode_batches = initial_snapshot.native_decode_batches;
    const auto initial_native_decode_sequences = initial_snapshot.native_decode_sequences;
    const auto initial_physical_prefill_batches =
        initial_snapshot.physical_prefill_batches;
    const auto initial_physical_prefill_sequence_participations =
        initial_snapshot.physical_prefill_sequence_participations;
    const auto initial_physical_prefill_tokens =
        initial_snapshot.physical_prefill_tokens;

    const auto wall_start = Clock::now();
    std::uint32_t launched = 0;
    while (launched < config.measured_runs) {
        const auto batch = std::min(config.concurrency, config.measured_runs - launched);
        std::vector<InferenceRequest> cohort(batch, config.request);
        auto responses = service.generate_cohort(cohort);
        if (!responses) return responses.status();
        if (responses.value().size() != batch) {
            return Status::internal_error("benchmark cohort returned the wrong response count");
        }
        for (auto& response : responses.value()) {
            report.runs.push_back(response.metrics);
            report.output_tokens.push_back(std::move(response.tokens));
        }
        launched += batch;
    }
    const auto wall_end = Clock::now();
    report.summary.wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    std::vector<double> ttft;
    std::vector<double> total;
    std::vector<double> prefill_tps;
    std::vector<double> decode_tps;
    std::vector<double> preparation_ms;
    std::vector<double> eviction_ms;
    ttft.reserve(report.runs.size());
    total.reserve(report.runs.size());
    prefill_tps.reserve(report.runs.size());
    decode_tps.reserve(report.runs.size());
    preparation_ms.reserve(report.runs.size());
    eviction_ms.reserve(report.runs.size());

    for (const auto& run : report.runs) {
        report.summary.prompt_tokens += run.prompt_tokens;
        report.summary.generated_tokens += run.generated_tokens;
        report.summary.prefix_reused_tokens += run.prefix_reused_tokens;
        ttft.push_back(run.ttft_ms);
        total.push_back(run.total_ms);
        prefill_tps.push_back(run.prefill_tokens_per_second());
        decode_tps.push_back(run.decode_tokens_per_second());
        preparation_ms.push_back(run.plan_preparation_ms);
        eviction_ms.push_back(run.plan_eviction_ms);
    }

    if (report.summary.wall_ms > 0.0) {
        const auto seconds = report.summary.wall_ms / 1000.0;
        report.summary.aggregate_generated_tokens_per_second =
            static_cast<double>(report.summary.generated_tokens) / seconds;
        report.summary.requests_per_second = static_cast<double>(report.runs.size()) / seconds;
    }
    report.summary.mean_prefill_tokens_per_second = mean_positive(prefill_tps);
    report.summary.mean_decode_tokens_per_second = mean_positive(decode_tps);
    report.summary.mean_plan_preparation_ms = mean_positive(preparation_ms);
    report.summary.mean_plan_eviction_ms = mean_positive(eviction_ms);
    report.summary.p50_ttft_ms = detail::percentile_linear(ttft, 0.50);
    report.summary.p95_ttft_ms = detail::percentile_linear(ttft, 0.95);
    report.summary.p50_total_ms = detail::percentile_linear(total, 0.50);
    report.summary.p95_total_ms = detail::percentile_linear(total, 0.95);
    const auto snapshot = service.snapshot();
    report.planner_mode = snapshot.planner_mode;
    report.manifest_id = snapshot.manifest_id;
    report.strategy_id = snapshot.strategy_id;
    report.strategy_objective = snapshot.strategy_objective;
    report.strategy_decision_reason = snapshot.strategy_decision_reason;
    report.prefill_block_linear_tactic = snapshot.planned_prefill_block_linear_tactic;
    report.decode_block_linear_tactic = snapshot.planned_decode_block_linear_tactic;
    report.decode_output_linear_tactic = snapshot.planned_decode_output_linear_tactic;
    report.prefill_attention_tactic = snapshot.planned_prefill_attention_tactic;
    report.decode_attention_tactic = snapshot.planned_decode_attention_tactic;
    report.planned_prefill_quantum_tokens = snapshot.planned_prefill_quantum_tokens;
    report.planned_kv_page_tokens = snapshot.planned_kv_page_tokens;
    report.summary.peak_kv_bytes = snapshot.peak_kv_bytes;
    report.summary.peak_device_bytes = snapshot.peak_device_bytes;
    report.summary.prepared_artifact_bytes = snapshot.current_prepared_artifact_bytes;
    report.summary.native_decode_batches = snapshot.native_decode_batches - initial_native_decode_batches;
    report.summary.native_decode_sequences = snapshot.native_decode_sequences - initial_native_decode_sequences;
    report.summary.physical_prefill_batches =
        snapshot.physical_prefill_batches - initial_physical_prefill_batches;
    report.summary.physical_prefill_sequence_participations =
        snapshot.physical_prefill_sequence_participations -
        initial_physical_prefill_sequence_participations;
    report.summary.physical_prefill_tokens =
        snapshot.physical_prefill_tokens - initial_physical_prefill_tokens;
    report.summary.physical_prefill_max_sequences =
        snapshot.physical_prefill_max_sequences;
    if (report.summary.physical_prefill_batches != 0U) {
        report.summary.physical_prefill_average_sequences =
            static_cast<double>(
                report.summary.physical_prefill_sequence_participations) /
            static_cast<double>(report.summary.physical_prefill_batches);
    }
    return report;
}

std::string BenchmarkReport::to_json() const {
    std::ostringstream out;
    out << std::setprecision(10);
    out << "{\n"
        << "  \"schema\": \"" << benchmark_report_schema << "\",\n"
        << "  \"air_version\": \"" << detail::json_escape(air_version) << "\",\n"
        << "  \"model_id\": \"" << detail::json_escape(model_id) << "\",\n"
        << "  \"model_architecture\": \"" << detail::json_escape(model_architecture) << "\",\n"
        << "  \"backend\": \"" << detail::json_escape(backend) << "\",\n"
        << "  \"planner_mode\": \"" << detail::json_escape(planner_mode) << "\",\n"
        << "  \"manifest_id\": \"" << detail::json_escape(manifest_id) << "\",\n"
        << "  \"strategy_id\": \"" << detail::json_escape(strategy_id) << "\",\n"
        << "  \"prefill_block_linear_tactic\": \"" << detail::json_escape(prefill_block_linear_tactic) << "\",\n"
        << "  \"decode_block_linear_tactic\": \"" << detail::json_escape(decode_block_linear_tactic) << "\",\n"
        << "  \"decode_output_linear_tactic\": \"" << detail::json_escape(decode_output_linear_tactic) << "\",\n"
        << "  \"prefill_attention_tactic\": \"" << detail::json_escape(prefill_attention_tactic) << "\",\n"
        << "  \"decode_attention_tactic\": \"" << detail::json_escape(decode_attention_tactic) << "\",\n"
        << "  \"planned_prefill_quantum_tokens\": " << planned_prefill_quantum_tokens << ",\n"
        << "  \"planned_kv_page_tokens\": " << planned_kv_page_tokens << ",\n"
        << "  \"started_unix_ms\": " << started_unix_ms << ",\n"
        << "  \"config\": {\"warmup_runs\":" << config.warmup_runs
        << ",\"measured_runs\":" << config.measured_runs
        << ",\"concurrency\":" << config.concurrency
        << ",\"max_new_tokens\":" << config.request.generation.max_new_tokens
        << ",\"temperature\":" << config.request.generation.sampling.temperature
        << ",\"top_p\":" << config.request.generation.sampling.top_p
        << ",\"top_k\":" << config.request.generation.sampling.top_k << "},\n"
        << "  \"summary\": {\n"
        << "    \"prompt_tokens\": " << summary.prompt_tokens << ",\n"
        << "    \"generated_tokens\": " << summary.generated_tokens << ",\n"
        << "    \"prefix_reused_tokens\": " << summary.prefix_reused_tokens << ",\n"
        << "    \"peak_kv_bytes\": " << summary.peak_kv_bytes << ",\n"
        << "    \"peak_device_bytes\": " << summary.peak_device_bytes << ",\n"
        << "    \"prepared_artifact_bytes\": " << summary.prepared_artifact_bytes << ",\n"
        << "    \"native_decode_batches\": " << summary.native_decode_batches << ",\n"
        << "    \"native_decode_sequences\": " << summary.native_decode_sequences << ",\n"
        << "    \"physical_prefill_batches\": " << summary.physical_prefill_batches << ",\n"
        << "    \"physical_prefill_sequence_participations\": " << summary.physical_prefill_sequence_participations << ",\n"
        << "    \"physical_prefill_tokens\": " << summary.physical_prefill_tokens << ",\n"
        << "    \"physical_prefill_max_sequences\": " << summary.physical_prefill_max_sequences << ",\n"
        << "    \"physical_prefill_average_sequences\": " << summary.physical_prefill_average_sequences << ",\n"
        << "    \"wall_ms\": " << summary.wall_ms << ",\n"
        << "    \"aggregate_generated_tokens_per_second\": " << summary.aggregate_generated_tokens_per_second << ",\n"
        << "    \"requests_per_second\": " << summary.requests_per_second << ",\n"
        << "    \"mean_prefill_tokens_per_second\": " << summary.mean_prefill_tokens_per_second << ",\n"
        << "    \"mean_decode_tokens_per_second\": " << summary.mean_decode_tokens_per_second << ",\n"
        << "    \"p50_ttft_ms\": " << summary.p50_ttft_ms << ",\n"
        << "    \"p95_ttft_ms\": " << summary.p95_ttft_ms << ",\n"
        << "    \"p50_total_ms\": " << summary.p50_total_ms << ",\n"
        << "    \"p95_total_ms\": " << summary.p95_total_ms << "\n"
        << "  },\n"
        << "  \"runs\": [\n";
    for (std::size_t i = 0; i < runs.size(); ++i) {
        out << "    ";
        write_metrics(out, runs[i]);
        out << (i + 1U == runs.size() ? "\n" : ",\n");
    }
    out << "  ],\n"
        << "  \"output_tokens\": [\n";
    for (std::size_t i = 0; i < output_tokens.size(); ++i) {
        out << "    [";
        for (std::size_t j = 0; j < output_tokens[i].size(); ++j) {
            if (j != 0U) out << ',';
            out << output_tokens[i][j];
        }
        out << ']' << (i + 1U == output_tokens.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    return out.str();
}

} // namespace air

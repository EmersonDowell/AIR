#include "air/manifest.hpp"

#include "air/version.hpp"

#include <boost/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include <sys/utsname.h>

namespace air {
namespace {
namespace json = boost::json;

class Fnv1a64 {
public:
    void bytes(const void* data, std::size_t size) noexcept {
        const auto* p = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < size; ++i) { value_ ^= p[i]; value_ *= 1099511628211ULL; }
    }
    void text(std::string_view value) noexcept { bytes(value.data(), value.size()); separator(); }
    template <class T> void scalar(const T& value) noexcept { bytes(&value, sizeof(value)); separator(); }
    [[nodiscard]] std::string digest() const {
        std::ostringstream out;
        out << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << value_;
        return out.str();
    }
private:
    void separator() noexcept { constexpr unsigned char marker = 0xffU; bytes(&marker, 1U); }
    std::uint64_t value_{14695981039346656037ULL};
};

void hash_optional(Fnv1a64& hash, const std::optional<TokenId>& id) {
    const bool present = id.has_value();
    hash.scalar(present);
    if (id) hash.scalar(*id);
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream out; out << in.rdbuf(); return out.str();
}

std::string stable_cpu_identity() {
    std::ifstream in("/proc/cpuinfo");
    if (!in) return {};
    std::ostringstream out;
    std::string line;
    std::size_t processors = 0;
    while (std::getline(in, line)) {
        if (line.rfind("processor", 0) == 0) { ++processors; continue; }
        const bool stable = line.rfind("vendor_id", 0) == 0 ||
                            line.rfind("cpu family", 0) == 0 ||
                            line.rfind("model\t", 0) == 0 ||
                            line.rfind("model name", 0) == 0 ||
                            line.rfind("stepping", 0) == 0 ||
                            line.rfind("flags", 0) == 0 ||
                            line.rfind("Features", 0) == 0;
        if (stable) out << line << '\n';
    }
    out << "processors=" << processors << '\n';
    out << "online=" << read_file("/sys/devices/system/cpu/online");
    return out.str();
}

std::optional<WorkloadClass> workload_from_string(std::string_view value) {
    if (value == "small") return WorkloadClass::small;
    if (value == "medium") return WorkloadClass::medium;
    if (value == "large") return WorkloadClass::large;
    if (value == "concurrent") return WorkloadClass::concurrent;
    return std::nullopt;
}

bool bool_field(const json::object& obj, const char* key, bool fallback = false) {
    const auto* v = obj.if_contains(key);
    return v && v->is_bool() ? v->as_bool() : fallback;
}

double double_field(const json::object& obj, const char* key, double fallback = 0.0) {
    const auto* v = obj.if_contains(key);
    return v && v->is_number() ? v->to_number<double>() : fallback;
}

std::uint64_t u64_field(const json::object& obj, const char* key, std::uint64_t fallback = 0U) {
    const auto* v = obj.if_contains(key);
    return v && v->is_number() ? v->to_number<std::uint64_t>() : fallback;
}

std::uint32_t u32_field(const json::object& obj, const char* key, std::uint32_t fallback = 0U) {
    const auto raw = u64_field(obj, key, fallback);
    return raw > std::numeric_limits<std::uint32_t>::max() ? fallback : static_cast<std::uint32_t>(raw);
}

std::string string_field(const json::object& obj, const char* key, std::string fallback = {}) {
    const auto* v = obj.if_contains(key);
    return v && v->is_string() ? std::string(v->as_string().c_str()) : std::move(fallback);
}

bool finite_nonnegative(double value) noexcept {
    return std::isfinite(value) && value >= 0.0;
}

Status validate_strategy_semantics(const QualifiedStrategy& s) {
    const auto invalid_stat = [&](double value, const char* name) -> Status {
        if (!finite_nonnegative(value)) {
            return Status::data_error(std::string("manifest strategy has invalid nonnegative statistic: ") + name);
        }
        return Status::ok();
    };

    const std::pair<double, const char*> stats[] = {
        {s.p50_ttft_ms, "p50_ttft_ms"},
        {s.p50_ttft_stddev_ms, "p50_ttft_stddev_ms"},
        {s.p50_ttft_confidence_half_width_ms, "p50_ttft_confidence_half_width_ms"},
        {s.p50_total_ms, "p50_total_ms"},
        {s.p50_total_stddev_ms, "p50_total_stddev_ms"},
        {s.p50_total_confidence_half_width_ms, "p50_total_confidence_half_width_ms"},
        {s.mean_prefill_tokens_per_second, "mean_prefill_tokens_per_second"},
        {s.prefill_tokens_per_second_stddev, "prefill_tokens_per_second_stddev"},
        {s.prefill_tokens_per_second_confidence_half_width, "prefill_tokens_per_second_confidence_half_width"},
        {s.mean_decode_tokens_per_second, "mean_decode_tokens_per_second"},
        {s.decode_tokens_per_second_stddev, "decode_tokens_per_second_stddev"},
        {s.decode_tokens_per_second_confidence_half_width, "decode_tokens_per_second_confidence_half_width"},
        {s.aggregate_generated_tokens_per_second, "aggregate_generated_tokens_per_second"},
        {s.aggregate_generated_tokens_per_second_confidence_half_width, "aggregate_generated_tokens_per_second_confidence_half_width"},
        {s.preparation_ms_mean, "preparation_ms_mean"},
        {s.preparation_ms_stddev, "preparation_ms_stddev"},
        {s.preparation_ms_confidence_half_width, "preparation_ms_confidence_half_width"},
        {s.eviction_ms_mean, "eviction_ms_mean"},
        {s.eviction_ms_stddev, "eviction_ms_stddev"},
        {s.eviction_ms_confidence_half_width, "eviction_ms_confidence_half_width"},
    };
    for (const auto& [value, name] : stats) {
        auto status = invalid_stat(value, name);
        if (!status.is_ok()) return status;
    }

    if (s.strict_qualified) {
        if (s.samples == 0U) return Status::data_error("strict-qualified manifest strategy has zero samples");
        if (s.evidence_id.empty()) return Status::data_error("strict-qualified manifest strategy is missing evidence_id");
        const bool has_latency = s.p50_ttft_ms > 0.0 || s.p50_total_ms > 0.0;
        const bool has_throughput = s.mean_prefill_tokens_per_second > 0.0 ||
                                    s.mean_decode_tokens_per_second > 0.0 ||
                                    s.aggregate_generated_tokens_per_second > 0.0;
        if (!has_latency || !has_throughput) {
            return Status::data_error("strict-qualified manifest strategy is missing required performance evidence");
        }
    }
    if (s.prepared_artifact_bytes > 0U && !s.preparation_measured) {
        return Status::data_error("manifest strategy with optional prepared state is missing measured preparation evidence");
    }
    return Status::ok();
}

Status validate_manifest_semantics(const ExecutionManifest& manifest) {
    if (manifest.air_version.empty() || manifest.model_digest.empty() || manifest.hardware_digest.empty() || manifest.manifest_id.empty()) {
        return Status::data_error("execution manifest identity fields must be non-empty");
    }
    std::unordered_set<std::string> strategy_ids;
    std::unordered_set<std::string> evidence_ids;
    for (const auto& strategy : manifest.strategies) {
        auto status = validate_strategy_semantics(strategy);
        if (!status.is_ok()) return status;
        if (!strategy_ids.insert(strategy.strategy_id).second) {
            return Status::data_error("execution manifest contains duplicate strategy_id: " + strategy.strategy_id);
        }
        if (!strategy.evidence_id.empty() && !evidence_ids.insert(strategy.evidence_id).second) {
            return Status::data_error("execution manifest contains duplicate evidence_id: " + strategy.evidence_id);
        }
    }
    return Status::ok();
}

json::object strategy_json(const QualifiedStrategy& s) {
    json::object out;
    out["workload"] = to_string(s.workload);
    out["strategy_id"] = s.strategy_id;
    out["strict_qualified"] = s.strict_qualified;
    out["region_min_prompt_tokens"] = s.region.min_prompt_tokens;
    out["region_max_prompt_tokens"] = s.region.max_prompt_tokens;
    out["region_min_active_sequences"] = s.region.min_active_sequences;
    out["region_max_active_sequences"] = s.region.max_active_sequences;
    out["backend"] = to_string(s.plan.backend);
    out["prefill_quantum_tokens"] = s.plan.scheduling.prefill_quantum_tokens;
    out["prefill_block_quantized_linear"] = to_string(s.plan.linear.prefill_block);
    out["decode_block_quantized_linear"] = to_string(s.plan.linear.decode_block);
    out["decode_output_quantized_linear"] = to_string(s.plan.linear.decode_output);
    out["prefill_attention"] = to_string(s.plan.attention.prefill);
    out["decode_attention"] = to_string(s.plan.attention.decode);
    if (s.plan.kv.page_tokens) out["kv_page_tokens"] = *s.plan.kv.page_tokens;
    else out["kv_page_tokens"] = nullptr;
    out["samples"] = s.samples;
    out["p50_ttft_ms"] = s.p50_ttft_ms;
    out["p50_ttft_confidence_half_width_ms"] = s.p50_ttft_confidence_half_width_ms;
    out["p50_total_ms"] = s.p50_total_ms;
    out["p50_total_confidence_half_width_ms"] = s.p50_total_confidence_half_width_ms;
    out["mean_prefill_tokens_per_second"] = s.mean_prefill_tokens_per_second;
    out["prefill_tokens_per_second_confidence_half_width"] = s.prefill_tokens_per_second_confidence_half_width;
    out["mean_decode_tokens_per_second"] = s.mean_decode_tokens_per_second;
    out["decode_tokens_per_second_confidence_half_width"] = s.decode_tokens_per_second_confidence_half_width;
    out["aggregate_generated_tokens_per_second"] = s.aggregate_generated_tokens_per_second;
    out["aggregate_generated_tokens_per_second_confidence_half_width"] = s.aggregate_generated_tokens_per_second_confidence_half_width;
    out["peak_kv_bytes"] = s.peak_kv_bytes;
    out["peak_device_bytes"] = s.peak_device_bytes;
    out["prepared_artifact_bytes"] = s.prepared_artifact_bytes;
    out["preparation_measured"] = s.preparation_measured;
    out["preparation_ms_mean"] = s.preparation_ms_mean;
    out["preparation_ms_stddev"] = s.preparation_ms_stddev;
    out["preparation_ms_confidence_half_width"] = s.preparation_ms_confidence_half_width;
    out["eviction_measured"] = s.eviction_measured;
    out["eviction_ms_mean"] = s.eviction_ms_mean;
    out["eviction_ms_stddev"] = s.eviction_ms_stddev;
    out["eviction_ms_confidence_half_width"] = s.eviction_ms_confidence_half_width;
    out["evidence_seed"] = s.evidence_seed;
    out["evidence_id"] = s.evidence_id;
    return out;
}

Result<QualifiedStrategy> strategy_from_json(const json::value& value) {
    if (!value.is_object()) return Status::data_error("manifest strategy must be an object");
    const auto& obj = value.as_object();
    const auto workload_name = string_field(obj, "workload");
    const auto strategy_id = string_field(obj, "strategy_id");
    const auto backend_name = string_field(obj, "backend");
    if (workload_name.empty() || strategy_id.empty() || backend_name.empty()) {
        return Status::data_error("manifest strategy is missing required string fields");
    }
    const auto workload = workload_from_string(workload_name);
    if (!workload) return Status::data_error("manifest strategy has unknown workload");
    const auto backend = backend_kind_from_string(backend_name);
    if (!backend) return backend.status();

    QualifiedStrategy out;
    out.workload = *workload;
    out.strategy_id = strategy_id;
    out.plan.strategy_id = strategy_id;
    out.plan.backend = backend.value();
    out.strict_qualified = bool_field(obj, "strict_qualified");
    out.region.min_prompt_tokens = u64_field(obj, "region_min_prompt_tokens");
    out.region.max_prompt_tokens = u64_field(obj, "region_max_prompt_tokens");
    out.region.min_active_sequences = u32_field(obj, "region_min_active_sequences", 1U);
    out.region.max_active_sequences = u32_field(obj, "region_max_active_sequences");

    const auto parse_linear = [&](const char* key) -> Result<QuantizedLinearExecutionKind> {
        const auto name = string_field(obj, key);
        if (name.empty()) return Status::data_error(std::string("manifest strategy missing tactic: ") + key);
        return quantized_linear_execution_kind_from_string(name);
    };
    auto prefill = parse_linear("prefill_block_quantized_linear"); if (!prefill) return prefill.status();
    auto decode = parse_linear("decode_block_quantized_linear"); if (!decode) return decode.status();
    auto output = parse_linear("decode_output_quantized_linear"); if (!output) return output.status();
    out.plan.linear.prefill_block = prefill.value();
    out.plan.linear.decode_block = decode.value();
    out.plan.linear.decode_output = output.value();

    const auto parse_attention = [&](const char* key) -> Result<AttentionExecutionKind> {
        const auto name = string_field(obj, key);
        if (name.empty()) return Status::data_error(std::string("manifest strategy missing tactic: ") + key);
        return attention_execution_kind_from_string(name);
    };
    auto pa = parse_attention("prefill_attention"); if (!pa) return pa.status();
    auto da = parse_attention("decode_attention"); if (!da) return da.status();
    out.plan.attention.prefill = pa.value();
    out.plan.attention.decode = da.value();
    out.plan.scheduling.prefill_quantum_tokens = u32_field(obj, "prefill_quantum_tokens");
    const auto* kv = obj.if_contains("kv_page_tokens");
    if (kv && kv->is_number()) {
        const auto page = kv->to_number<std::uint64_t>();
        if (page > 0U && page <= std::numeric_limits<std::uint32_t>::max()) out.plan.kv.page_tokens = static_cast<std::uint32_t>(page);
    }

    out.samples = u32_field(obj, "samples");
    out.p50_ttft_ms = double_field(obj, "p50_ttft_ms");
    out.p50_ttft_confidence_half_width_ms = double_field(obj, "p50_ttft_confidence_half_width_ms");
    out.p50_total_ms = double_field(obj, "p50_total_ms");
    out.p50_total_confidence_half_width_ms = double_field(obj, "p50_total_confidence_half_width_ms");
    out.mean_prefill_tokens_per_second = double_field(obj, "mean_prefill_tokens_per_second");
    out.prefill_tokens_per_second_confidence_half_width = double_field(obj, "prefill_tokens_per_second_confidence_half_width");
    out.mean_decode_tokens_per_second = double_field(obj, "mean_decode_tokens_per_second");
    out.decode_tokens_per_second_confidence_half_width = double_field(obj, "decode_tokens_per_second_confidence_half_width");
    out.aggregate_generated_tokens_per_second = double_field(obj, "aggregate_generated_tokens_per_second");
    out.aggregate_generated_tokens_per_second_confidence_half_width = double_field(obj, "aggregate_generated_tokens_per_second_confidence_half_width");
    out.peak_kv_bytes = u64_field(obj, "peak_kv_bytes");
    out.peak_device_bytes = u64_field(obj, "peak_device_bytes");
    out.prepared_artifact_bytes = u64_field(obj, "prepared_artifact_bytes");
    out.preparation_measured = bool_field(obj, "preparation_measured");
    out.preparation_ms_mean = double_field(obj, "preparation_ms_mean");
    out.preparation_ms_stddev = double_field(obj, "preparation_ms_stddev");
    out.preparation_ms_confidence_half_width = double_field(obj, "preparation_ms_confidence_half_width");
    out.eviction_measured = bool_field(obj, "eviction_measured");
    out.eviction_ms_mean = double_field(obj, "eviction_ms_mean");
    out.eviction_ms_stddev = double_field(obj, "eviction_ms_stddev");
    out.eviction_ms_confidence_half_width = double_field(obj, "eviction_ms_confidence_half_width");
    out.evidence_seed = u64_field(obj, "evidence_seed");
    out.evidence_id = string_field(obj, "evidence_id");

    if (out.plan.scheduling.prefill_quantum_tokens == 0U) return Status::data_error("manifest strategy has zero prefill scheduling quantum");
    if (out.region.min_active_sequences == 0U) return Status::data_error("manifest strategy region has zero minimum active sequence count");
    if (out.region.max_prompt_tokens && out.region.max_prompt_tokens < out.region.min_prompt_tokens) return Status::data_error("manifest strategy prompt region is inverted");
    if (out.region.max_active_sequences && out.region.max_active_sequences < out.region.min_active_sequences) return Status::data_error("manifest strategy concurrency region is inverted");
    auto semantic_status = validate_strategy_semantics(out);
    if (!semantic_status.is_ok()) return semantic_status;
    return out;
}

struct EstimatedCost {
    double transition_ms{0.0};
    double horizon_ms{std::numeric_limits<double>::infinity()};
    double break_even_tokens{0.0};
    double prefill_tps{0.0};
    double decode_tps{0.0};
    double aggregate_tps{0.0};
};

double positive_lower(double mean, double half_width) noexcept {
    return std::max(0.0, mean - std::max(0.0, half_width));
}

double positive_upper(double mean, double half_width) noexcept {
    return std::max(0.0, mean + std::max(0.0, half_width));
}

bool performance_present(const QualifiedStrategy& s, const RequestProfile& request) noexcept {
    if (request.active_sequences > 1U) return s.aggregate_generated_tokens_per_second > 0.0;
    return s.mean_prefill_tokens_per_second > 0.0 &&
           (request.max_output_tokens == 0U || s.mean_decode_tokens_per_second > 0.0);
}

EstimatedCost estimate_cost(const QualifiedStrategy& candidate,
                            const QualifiedStrategy* incumbent,
                            const RequestProfile& request,
                            const RuntimeSnapshot& runtime,
                            const StrategyLabConfig& config) {
    EstimatedCost out;
    const bool candidate_hot = candidate.prepared_artifact_bytes > 0U &&
        runtime.prepared_artifact_bytes >= candidate.prepared_artifact_bytes;
    if (!candidate_hot && candidate.prepared_artifact_bytes > 0U) {
        out.transition_ms += positive_upper(candidate.preparation_ms_mean,
                                            candidate.preparation_ms_confidence_half_width);
    }
    if (runtime.prepared_artifact_bytes > 0U && incumbent &&
        incumbent->strategy_id != candidate.strategy_id &&
        candidate.prepared_artifact_bytes < runtime.prepared_artifact_bytes) {
        out.transition_ms += positive_upper(incumbent->eviction_ms_mean,
                                            incumbent->eviction_ms_confidence_half_width);
    }

    out.prefill_tps = positive_lower(candidate.mean_prefill_tokens_per_second,
                                     candidate.prefill_tokens_per_second_confidence_half_width);
    out.decode_tps = positive_lower(candidate.mean_decode_tokens_per_second,
                                    candidate.decode_tokens_per_second_confidence_half_width);
    out.aggregate_tps = positive_lower(candidate.aggregate_generated_tokens_per_second,
                                       candidate.aggregate_generated_tokens_per_second_confidence_half_width);

    const auto horizon = config.expected_horizon_tokens > 0U
        ? config.expected_horizon_tokens
        : request.prompt_tokens + request.max_output_tokens * std::max<std::uint32_t>(1U, request.active_sequences);
    double compute_ms = 0.0;
    if (request.active_sequences > 1U && out.aggregate_tps > 0.0) {
        const auto output = config.expected_horizon_tokens > 0U ? horizon :
            request.max_output_tokens * static_cast<std::uint64_t>(request.active_sequences);
        compute_ms += static_cast<double>(output) * 1000.0 / out.aggregate_tps;
        if (request.prompt_tokens > 0U && out.prefill_tps > 0.0) {
            compute_ms += static_cast<double>(request.prompt_tokens) * 1000.0 / out.prefill_tps;
        }
    } else {
        if (request.prompt_tokens > 0U && out.prefill_tps > 0.0) {
            compute_ms += static_cast<double>(request.prompt_tokens) * 1000.0 / out.prefill_tps;
        }
        if (request.max_output_tokens > 0U && out.decode_tps > 0.0) {
            compute_ms += static_cast<double>(request.max_output_tokens) * 1000.0 / out.decode_tps;
        }
        if (config.expected_horizon_tokens > request.prompt_tokens + request.max_output_tokens && out.decode_tps > 0.0) {
            compute_ms += static_cast<double>(config.expected_horizon_tokens - request.prompt_tokens - request.max_output_tokens) * 1000.0 / out.decode_tps;
        }
    }
    out.horizon_ms = out.transition_ms + compute_ms;
    return out;
}

} // namespace

const char* to_string(WorkloadClass workload) noexcept {
    switch (workload) {
    case WorkloadClass::small: return "small";
    case WorkloadClass::medium: return "medium";
    case WorkloadClass::large: return "large";
    case WorkloadClass::concurrent: return "concurrent";
    }
    return "unknown";
}

WorkloadClass classify_workload(const RequestProfile& request) noexcept {
    if (request.active_sequences > 1U) return WorkloadClass::concurrent;
    if (request.prompt_tokens <= 256U) return WorkloadClass::small;
    if (request.prompt_tokens <= 2048U) return WorkloadClass::medium;
    return WorkloadClass::large;
}

const char* to_string(StrategyObjective objective) noexcept {
    switch (objective) {
    case StrategyObjective::interactive: return "interactive";
    case StrategyObjective::balanced: return "balanced";
    case StrategyObjective::maximum_throughput: return "maximum-throughput";
    case StrategyObjective::minimum_vram: return "minimum-vram";
    }
    return "unknown";
}

Result<StrategyObjective> strategy_objective_from_string(std::string_view value) {
    if (value == "interactive") return StrategyObjective::interactive;
    if (value == "balanced") return StrategyObjective::balanced;
    if (value == "maximum-throughput" || value == "max-throughput") return StrategyObjective::maximum_throughput;
    if (value == "minimum-vram" || value == "min-vram") return StrategyObjective::minimum_vram;
    return Status::data_error("unknown strategy objective: " + std::string(value));
}

bool WorkloadRegion::contains(const RequestProfile& request) const noexcept {
    if (request.prompt_tokens < min_prompt_tokens) return false;
    if (max_prompt_tokens && request.prompt_tokens > max_prompt_tokens) return false;
    if (request.active_sequences < min_active_sequences) return false;
    if (max_active_sequences && request.active_sequences > max_active_sequences) return false;
    return true;
}

std::vector<const QualifiedStrategy*> ExecutionManifest::candidates(const RequestProfile& request) const {
    std::vector<const QualifiedStrategy*> out;
    for (const auto& strategy : strategies) if (strategy.region.contains(request)) out.push_back(&strategy);
    return out;
}

const QualifiedStrategy* ExecutionManifest::find_strategy(std::string_view id) const noexcept {
    for (const auto& strategy : strategies) if (strategy.strategy_id == id) return &strategy;
    return nullptr;
}

std::string model_digest(const ModelDefinition& model) {
    Fnv1a64 hash;
    const auto& fp = model.fingerprint();
    const auto& c = model.config();
    const auto& t = model.tokenizer();
    hash.text(fp.format); hash.text(fp.architecture); hash.text(fp.model_id);
    hash.text(c.architecture); hash.scalar(c.layer_count); hash.scalar(c.embedding_size);
    hash.scalar(c.feed_forward_size); hash.scalar(c.attention_head_count); hash.scalar(c.kv_head_count);
    hash.scalar(c.attention_sliding_window); hash.scalar(c.rope_dimension_count); hash.scalar(c.context_length);
    hash.scalar(c.vocabulary_size); hash.scalar(c.rope_frequency_base); hash.text(c.rope_scaling_type);
    hash.scalar(c.rope_scaling_factor); hash.scalar(c.rope_scale_linear); hash.scalar(c.rms_norm_epsilon);
    hash.text(t.model); hash.text(t.pre_tokenizer); hash.text(t.chat_template);
    hash.scalar(t.add_bos); hash.scalar(t.add_eos);
    for (const auto& token : t.vocabulary) hash.text(token);
    for (const auto value : t.scores) hash.scalar(value);
    for (const auto value : t.token_types) hash.scalar(value);
    for (const auto& merge : t.merges) hash.text(merge);
    hash_optional(hash, t.special_ids.bos); hash_optional(hash, t.special_ids.eos);
    hash_optional(hash, t.special_ids.unknown); hash_optional(hash, t.special_ids.padding);
    hash_optional(hash, t.special_ids.eot); hash_optional(hash, t.special_ids.eom);
    for (const auto& tensor : model.tensors()) {
        hash.text(tensor.name); hash.scalar(tensor.format_type); hash.scalar(tensor.byte_offset);
        hash.scalar(tensor.byte_size); hash.scalar(tensor.byte_size_exact);
        for (const auto dimension : tensor.shape.dimensions) hash.scalar(dimension);
    }
    if (const auto& storage = model.storage()) {
        hash.scalar(storage->size_bytes());
        std::error_code timestamp_error;
        const auto timestamp = std::filesystem::last_write_time(storage->path(), timestamp_error);
        if (!timestamp_error) { const auto ticks = timestamp.time_since_epoch().count(); hash.scalar(ticks); }
        constexpr std::uint64_t sample = 65536U;
        const auto first_size = std::min(sample, storage->size_bytes());
        if (auto view = storage->view(0U, first_size); view) hash.bytes(view.value().data(), view.value().size());
        if (storage->size_bytes() > first_size) {
            const auto last_size = std::min(sample, storage->size_bytes());
            if (auto view = storage->view(storage->size_bytes() - last_size, last_size); view) hash.bytes(view.value().data(), view.value().size());
        }
    }
    return hash.digest();
}

std::string hardware_digest() {
    Fnv1a64 hash;
    struct utsname name {};
    if (uname(&name) == 0) { hash.text(name.sysname); hash.text(name.release); hash.text(name.machine); }
    hash.text(stable_cpu_identity());
    hash.text(read_file("/proc/driver/nvidia/version"));
    const std::filesystem::path gpu_root{"/proc/driver/nvidia/gpus"};
    std::error_code error;
    if (std::filesystem::exists(gpu_root, error)) {
        std::vector<std::filesystem::path> paths;
        for (const auto& entry : std::filesystem::directory_iterator(gpu_root, error)) { if (error) break; paths.push_back(entry.path()); }
        std::sort(paths.begin(), paths.end());
        for (const auto& path : paths) { hash.text(path.filename().string()); hash.text(read_file(path / "information")); }
    }
    return hash.digest();
}

std::filesystem::path default_manifest_path(const ModelDefinition& model) {
    const char* home = std::getenv("HOME");
    const std::filesystem::path root = home && *home ? std::filesystem::path(home) : std::filesystem::temp_directory_path();
    std::string name = model_digest(model);
    for (char& ch : name) if (ch == ':' || ch == '/') ch = '_';
    return root / ".cache" / "air" / "manifests" / (name + ".json");
}

Status save_manifest(const ExecutionManifest& manifest, const std::filesystem::path& path) {
    if (manifest.schema_version != execution_manifest_schema_version) return Status::invalid_argument("AIR only writes the current execution manifest schema");
    auto semantic_status = validate_manifest_semantics(manifest);
    if (!semantic_status.is_ok()) return semantic_status;
    json::object root;
    root["schema_version"] = manifest.schema_version;
    root["air_version"] = manifest.air_version;
    root["model_digest"] = manifest.model_digest;
    root["hardware_digest"] = manifest.hardware_digest;
    root["manifest_id"] = manifest.manifest_id;
    json::array strategies;
    for (const auto& s : manifest.strategies) strategies.push_back(strategy_json(s));
    root["strategies"] = std::move(strategies);
    std::error_code error;
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path(), error); if (error) return Status::io_error("unable to create manifest directory: " + error.message()); }
    const auto temporary = path.string() + ".tmp";
    { std::ofstream out(temporary, std::ios::trunc); if (!out) return Status::io_error("unable to open manifest for writing"); out << json::serialize(root) << '\n'; if (!out) return Status::io_error("unable to write manifest"); }
    std::filesystem::rename(temporary, path, error);
    if (error) { std::filesystem::remove(temporary); return Status::io_error("unable to commit manifest: " + error.message()); }
    return Status::ok();
}

Result<ExecutionManifest> load_manifest(const std::filesystem::path& path) {
    std::ifstream in(path); if (!in) return Status::io_error("unable to open execution manifest");
    std::ostringstream buffer; buffer << in.rdbuf();
    boost::system::error_code error;
    auto parsed = json::parse(buffer.str(), error);
    if (error || !parsed.is_object()) return Status::data_error("execution manifest is not valid JSON");
    const auto& obj = parsed.as_object();
    ExecutionManifest out;
    const auto* schema = obj.if_contains("schema_version");
    const auto* version = obj.if_contains("air_version");
    const auto* model = obj.if_contains("model_digest");
    const auto* hardware = obj.if_contains("hardware_digest");
    const auto* id = obj.if_contains("manifest_id");
    const auto* strategies = obj.if_contains("strategies");
    if (!schema || !schema->is_number() || !version || !version->is_string() || !model || !model->is_string() || !hardware || !hardware->is_string() || !id || !id->is_string() || !strategies || !strategies->is_array()) return Status::data_error("execution manifest is missing required fields");
    out.schema_version = schema->to_number<std::uint32_t>();
    if (out.schema_version != execution_manifest_schema_version) return Status::unsupported("execution manifest schema is not supported by this AIR release");
    out.air_version = version->as_string().c_str(); out.model_digest = model->as_string().c_str(); out.hardware_digest = hardware->as_string().c_str(); out.manifest_id = id->as_string().c_str();
    for (const auto& value : strategies->as_array()) { auto s = strategy_from_json(value); if (!s) return s.status(); out.strategies.push_back(std::move(s).value()); }
    auto semantic_status = validate_manifest_semantics(out);
    if (!semantic_status.is_ok()) return semantic_status;
    return out;
}

ManifestLoadResult validate_manifest(const ExecutionManifest& manifest, const ModelDefinition& model) {
    ManifestLoadResult result;
    if (manifest.schema_version != execution_manifest_schema_version) { result.status = "stale:schema-version"; return result; }
    if (manifest.air_version != version_string()) { result.status = "stale:air-version"; return result; }
    if (manifest.model_digest != model_digest(model)) { result.status = "stale:model-fingerprint"; return result; }
    if (manifest.hardware_digest != hardware_digest()) { result.status = "stale:hardware-fingerprint"; return result; }
    result.manifest = manifest; result.status = "loaded"; return result;
}

StrategyLabPlanner::StrategyLabPlanner(ExecutionManifest manifest, ExecutionPlan fallback, StrategyLabConfig config)
    : manifest_(std::move(manifest)), fallback_(std::move(fallback)), config_(config) {}

PlanningDecision StrategyLabPlanner::decide(const PlanningInput& input) const {
    PlanningDecision result;
    result.plan = fallback_;
    result.objective = to_string(config_.objective);
    result.reason = "fallback:no-eligible-strategy";

    const auto* incumbent = manifest_.find_strategy(input.runtime.current_strategy_id);
    const auto candidates = manifest_.candidates(input.request);
    std::vector<std::pair<const QualifiedStrategy*, EstimatedCost>> eligible;
    std::uint64_t lowest_prepared = std::numeric_limits<std::uint64_t>::max();
    for (const auto* c : candidates) if (c->strict_qualified) lowest_prepared = std::min(lowest_prepared, c->prepared_artifact_bytes);

    for (const auto* c : candidates) {
        PlanningCandidateTrace trace;
        trace.strategy_id = c->strategy_id;
        trace.prepared_artifact_bytes = c->prepared_artifact_bytes;
        trace.prepared_state_hot = c->prepared_artifact_bytes > 0U && input.runtime.prepared_artifact_bytes >= c->prepared_artifact_bytes;
        trace.memory_feasible = true;
        if (!c->strict_qualified) { trace.disposition = "rejected:not-strict-qualified"; result.candidates.push_back(std::move(trace)); continue; }
        if (!performance_present(*c, input.request)) { trace.disposition = "rejected:missing-performance-evidence"; result.candidates.push_back(std::move(trace)); continue; }
        const auto incremental = trace.prepared_state_hot ? 0U : c->prepared_artifact_bytes;
        if ((config_.prepared_memory_budget_bytes > 0U && c->prepared_artifact_bytes > config_.prepared_memory_budget_bytes) ||
            (input.runtime.free_device_memory_bytes > 0U && incremental > input.runtime.free_device_memory_bytes)) {
            trace.memory_feasible = false; trace.disposition = "rejected:memory-infeasible"; result.candidates.push_back(std::move(trace)); continue;
        }
        if (!trace.prepared_state_hot && c->prepared_artifact_bytes > 0U && !c->preparation_measured) {
            trace.disposition = "rejected:missing-preparation-evidence"; result.candidates.push_back(std::move(trace)); continue;
        }
        if (input.runtime.prepared_artifact_bytes > 0U && incumbent && incumbent->strategy_id != c->strategy_id &&
            c->prepared_artifact_bytes < input.runtime.prepared_artifact_bytes && !incumbent->eviction_measured &&
            !config_.allow_unmeasured_eviction_for_qualification) {
            trace.disposition = "rejected:missing-eviction-evidence"; result.candidates.push_back(std::move(trace)); continue;
        }

        // Higher-memory candidates must demonstrate statistical separation from a
        // lower-memory qualified alternative for the same region before AIR pays
        // the residency/transition cost.
        bool overlap = false;
        if (c->prepared_artifact_bytes > lowest_prepared && config_.objective != StrategyObjective::minimum_vram) {
            for (const auto* low : candidates) {
                if (!low->strict_qualified || low->prepared_artifact_bytes >= c->prepared_artifact_bytes || !performance_present(*low, input.request)) continue;
                if (config_.objective == StrategyObjective::interactive) {
                    const auto c_upper = positive_upper(c->p50_total_ms, c->p50_total_confidence_half_width_ms);
                    const auto low_lower = std::max(0.0, low->p50_total_ms - low->p50_total_confidence_half_width_ms);
                    if (c_upper >= low_lower) overlap = true;
                } else if (input.request.active_sequences > 1U) {
                    const auto c_lower = positive_lower(c->aggregate_generated_tokens_per_second, c->aggregate_generated_tokens_per_second_confidence_half_width);
                    const auto low_upper = positive_upper(low->aggregate_generated_tokens_per_second, low->aggregate_generated_tokens_per_second_confidence_half_width);
                    if (c_lower <= low_upper) overlap = true;
                } else {
                    const auto c_lower = positive_lower(c->mean_prefill_tokens_per_second, c->prefill_tokens_per_second_confidence_half_width);
                    const auto low_upper = positive_upper(low->mean_prefill_tokens_per_second, low->prefill_tokens_per_second_confidence_half_width);
                    if (c_lower <= low_upper) overlap = true;
                }
                if (overlap) break;
            }
        }
        if (overlap) { trace.disposition = "rejected:confidence-overlap-with-lower-memory"; result.candidates.push_back(std::move(trace)); continue; }

        auto cost = estimate_cost(*c, incumbent, input.request, input.runtime, config_);
        trace.estimated_transition_ms = cost.transition_ms;
        trace.estimated_horizon_ms = cost.horizon_ms;
        trace.disposition = "eligible";
        trace.eligible = true;
        eligible.push_back({c, cost});
        result.candidates.push_back(std::move(trace));
    }

    result.eligible_candidates = static_cast<std::uint32_t>(eligible.size());
    if (eligible.empty()) return result;

    // Populate candidate-local break-even estimates against the lowest-residency
    // eligible alternative. This is diagnostic only; selection still uses the
    // full conservative horizon cost above.
    const auto trace_low = std::min_element(eligible.begin(), eligible.end(), [](const auto& a, const auto& b) {
        return a.first->prepared_artifact_bytes < b.first->prepared_artifact_bytes;
    });
    if (trace_low != eligible.end()) {
        for (auto& trace : result.candidates) {
            if (!trace.eligible || trace.strategy_id == trace_low->first->strategy_id || trace.estimated_transition_ms <= 0.0) continue;
            const auto it = std::find_if(eligible.begin(), eligible.end(), [&](const auto& e) { return e.first->strategy_id == trace.strategy_id; });
            if (it == eligible.end()) continue;
            const double candidate_tps = input.request.active_sequences > 1U ? it->second.aggregate_tps : it->second.prefill_tps;
            const double base_tps = input.request.active_sequences > 1U
                ? positive_upper(trace_low->first->aggregate_generated_tokens_per_second, trace_low->first->aggregate_generated_tokens_per_second_confidence_half_width)
                : positive_upper(trace_low->first->mean_prefill_tokens_per_second, trace_low->first->prefill_tokens_per_second_confidence_half_width);
            if (candidate_tps > base_tps && base_tps > 0.0) {
                const double saved_ms_per_token = 1000.0 / base_tps - 1000.0 / candidate_tps;
                trace.estimated_break_even_tokens = trace.estimated_transition_ms / saved_ms_per_token;
            }
        }
    }

    auto chosen = eligible.begin();
    if (config_.objective == StrategyObjective::minimum_vram) {
        chosen = std::min_element(eligible.begin(), eligible.end(), [](const auto& a, const auto& b) {
            if (a.first->prepared_artifact_bytes != b.first->prepared_artifact_bytes) return a.first->prepared_artifact_bytes < b.first->prepared_artifact_bytes;
            return a.second.horizon_ms < b.second.horizon_ms;
        });
    } else if (config_.objective == StrategyObjective::interactive) {
        chosen = std::min_element(eligible.begin(), eligible.end(), [](const auto& a, const auto& b) {
            const auto aa = a.second.transition_ms + positive_upper(a.first->p50_ttft_ms, a.first->p50_ttft_confidence_half_width_ms) + positive_upper(a.first->p50_total_ms, a.first->p50_total_confidence_half_width_ms);
            const auto bb = b.second.transition_ms + positive_upper(b.first->p50_ttft_ms, b.first->p50_ttft_confidence_half_width_ms) + positive_upper(b.first->p50_total_ms, b.first->p50_total_confidence_half_width_ms);
            if (aa != bb) return aa < bb;
            return a.first->prepared_artifact_bytes < b.first->prepared_artifact_bytes;
        });
    } else {
        chosen = std::min_element(eligible.begin(), eligible.end(), [](const auto& a, const auto& b) {
            if (a.second.horizon_ms != b.second.horizon_ms) return a.second.horizon_ms < b.second.horizon_ms;
            return a.first->prepared_artifact_bytes < b.first->prepared_artifact_bytes;
        });
    }

    result.plan = chosen->first->plan;
    result.prepared_state_hot = chosen->first->prepared_artifact_bytes > 0U && input.runtime.prepared_artifact_bytes >= chosen->first->prepared_artifact_bytes;
    result.estimated_transition_ms = chosen->second.transition_ms;
    result.reason = result.prepared_state_hot ? "selected:qualified-hot-state" : "selected:qualified-break-even";

    // Compute an explainable conservative break-even against the lowest-memory
    // eligible strategy using candidate lower-bound throughput and baseline upper.
    const auto low = std::min_element(eligible.begin(), eligible.end(), [](const auto& a, const auto& b) { return a.first->prepared_artifact_bytes < b.first->prepared_artifact_bytes; });
    if (low != eligible.end() && low->first != chosen->first && chosen->second.transition_ms > 0.0) {
        double candidate_tps = input.request.active_sequences > 1U ? chosen->second.aggregate_tps : chosen->second.prefill_tps;
        double base_tps = input.request.active_sequences > 1U
            ? positive_upper(low->first->aggregate_generated_tokens_per_second, low->first->aggregate_generated_tokens_per_second_confidence_half_width)
            : positive_upper(low->first->mean_prefill_tokens_per_second, low->first->prefill_tokens_per_second_confidence_half_width);
        if (candidate_tps > base_tps && base_tps > 0.0) {
            const double ms_saved_per_token = 1000.0 / base_tps - 1000.0 / candidate_tps;
            result.estimated_break_even_tokens = chosen->second.transition_ms / ms_saved_per_token;
        }
    }
    return result;
}

} // namespace air

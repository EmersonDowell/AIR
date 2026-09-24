#include "air/qualification.hpp"

#include "air/benchmark.hpp"
#include "air/cuda.hpp"
#include "air/format.hpp"
#include "air/version.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <string_view>

namespace air {
namespace {

struct CandidateSpec {
    BackendPreference preference{BackendPreference::reference};
    BackendKind backend{BackendKind::reference};
    std::uint32_t quantum{32U};
    std::optional<std::uint32_t> kv_page;
    std::string strategy_id;
    bool canonical_default{false};
};

struct Sample {
    double ttft_ms{0.0};
    double total_ms{0.0};
    double prefill_tokens_per_second{0.0};
    double decode_tokens_per_second{0.0};
    std::uint64_t peak_kv_bytes{0};
    std::uint64_t peak_device_bytes{0};
};

struct CandidateMeasurement {
    CandidateSpec spec;
    std::vector<Sample> samples;
    std::vector<double> scores;
    QualifiedStrategy strategy;
};

struct MeanStddev {
    double mean{0.0};
    double stddev{0.0};
};

MeanStddev mean_stddev(const std::vector<double>& values) {
    if (values.empty()) return {};
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) /
                        static_cast<double>(values.size());
    if (values.size() < 2U) return {mean, 0.0};
    double sum = 0.0;
    for (const double value : values) {
        const double delta = value - mean;
        sum += delta * delta;
    }
    return {mean, std::sqrt(sum / static_cast<double>(values.size() - 1U))};
}


double student_t_95_critical(std::size_t samples) noexcept {
    // Two-sided 95% critical values for the small paired samples used by AIR.
    // Beyond 30 samples the normal approximation is adequate for qualification.
    if (samples <= 1U) return 0.0;
    static constexpr std::array<double, 30> values{{
        0.0,       // unused: df 0
        12.706,    // df 1
        4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
        2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093,
        2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045
    }};
    const std::size_t df = samples - 1U;
    return df < values.size() ? values[df] : 1.96;
}

double positive_or_zero(double value) noexcept {
    return std::isfinite(value) && value > 0.0 ? value : 0.0;
}

double benefit(double value, double maximum) noexcept {
    return maximum > 0.0 ? positive_or_zero(value) / maximum : 1.0;
}

double inverse_cost(double value, double minimum_positive) noexcept {
    if (minimum_positive <= 0.0) return 1.0;
    const auto v = positive_or_zero(value);
    return v > 0.0 ? minimum_positive / v : 0.0;
}

std::uint64_t unix_ms_now() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

std::uint64_t qualification_seed(const ModelDefinition& model, WorkloadClass workload) {
    constexpr std::uint64_t offset = 14695981039346656037ULL;
    constexpr std::uint64_t prime = 1099511628211ULL;
    std::uint64_t value = offset;
    auto add = [&](std::string_view text) {
        for (const char ch : text) {
            const auto byte = static_cast<unsigned char>(ch);
            value ^= byte;
            value *= prime;
        }
        value ^= 0xffU;
        value *= prime;
    };
    add(model_digest(model));
    add(hardware_digest());
    add(to_string(workload));
    return value;
}


void deterministic_shuffle(std::vector<std::size_t>& order, std::mt19937_64& rng) {
    // std::shuffle is not required to produce the same permutation across C++
    // library implementations. Use a small explicit Fisher-Yates shuffle so a
    // persisted evidence seed reproduces candidate order across installations.
    for (std::size_t i = order.size(); i > 1U; --i) {
        const std::uint64_t bound = static_cast<std::uint64_t>(i);
        const std::uint64_t cutoff = std::numeric_limits<std::uint64_t>::max() -
            (std::numeric_limits<std::uint64_t>::max() % bound);
        std::uint64_t value = 0U;
        do { value = rng(); } while (value >= cutoff);
        const auto j = static_cast<std::size_t>(value % bound);
        std::swap(order[i - 1U], order[j]);
    }
}

SchedulerConfig scheduler_for(const CandidateSpec& candidate) {
    SchedulerConfig scheduler;
    scheduler.prefill_quantum_tokens = candidate.quantum;
    if (candidate.kv_page) {
        if (candidate.backend == BackendKind::cuda) scheduler.cuda_kv_page_tokens = *candidate.kv_page;
        else scheduler.reference_kv_page_tokens = *candidate.kv_page;
    }
    scheduler.prefix_cache_entries = 0U;
    return scheduler;
}

Result<Sample> measure_once(const std::shared_ptr<ModelDefinition>& model,
                            const QualificationConfig& config,
                            const CandidateSpec& candidate) {
    auto service = InferenceService::create(
        model, candidate.preference, config.cuda_device, scheduler_for(candidate));
    if (!service) return service.status();

    BenchmarkConfig benchmark;
    benchmark.request.prompt = config.prompt;
    benchmark.request.generation.max_new_tokens = config.max_new_tokens;
    benchmark.request.generation.sampling.temperature = 0.0;
    benchmark.warmup_runs = 1U;
    benchmark.concurrency = config.workload == WorkloadClass::concurrent ? 2U : 1U;
    // A concurrent evidence round must actually overlap at least two requests.
    // measured_runs=1 would silently collapse run_benchmark back to concurrency 1.
    benchmark.measured_runs = benchmark.concurrency;
    auto report = run_benchmark(*service.value(), benchmark);
    if (!report) return report.status();

    const auto& summary = report.value().summary;
    Sample sample;
    sample.ttft_ms = summary.p50_ttft_ms;
    sample.total_ms = summary.p50_total_ms;
    sample.prefill_tokens_per_second = summary.mean_prefill_tokens_per_second;
    sample.decode_tokens_per_second = summary.mean_decode_tokens_per_second;
    sample.peak_kv_bytes = summary.peak_kv_bytes;
    sample.peak_device_bytes = summary.peak_device_bytes;
    return sample;
}

void score_round(std::vector<CandidateMeasurement>& measured, std::size_t round) {
    double max_decode = 0.0;
    double max_prefill = 0.0;
    double min_ttft = std::numeric_limits<double>::infinity();
    double min_total = std::numeric_limits<double>::infinity();
    std::uint64_t min_kv = std::numeric_limits<std::uint64_t>::max();

    for (const auto& candidate : measured) {
        const auto& sample = candidate.samples.at(round);
        max_decode = std::max(max_decode, positive_or_zero(sample.decode_tokens_per_second));
        max_prefill = std::max(max_prefill, positive_or_zero(sample.prefill_tokens_per_second));
        if (sample.ttft_ms > 0.0) min_ttft = std::min(min_ttft, sample.ttft_ms);
        if (sample.total_ms > 0.0) min_total = std::min(min_total, sample.total_ms);
        if (sample.peak_kv_bytes > 0U) min_kv = std::min(min_kv, sample.peak_kv_bytes);
    }
    if (!std::isfinite(min_ttft)) min_ttft = 0.0;
    if (!std::isfinite(min_total)) min_total = 0.0;
    if (min_kv == std::numeric_limits<std::uint64_t>::max()) min_kv = 0U;

    for (auto& candidate : measured) {
        const auto& sample = candidate.samples.at(round);
        const double memory_score = min_kv > 0U && sample.peak_kv_bytes > 0U
            ? static_cast<double>(min_kv) / static_cast<double>(sample.peak_kv_bytes) : 1.0;
        candidate.scores.push_back(
            0.30 * benefit(sample.decode_tokens_per_second, max_decode) +
            0.15 * benefit(sample.prefill_tokens_per_second, max_prefill) +
            0.25 * inverse_cost(sample.ttft_ms, min_ttft) +
            0.15 * inverse_cost(sample.total_ms, min_total) +
            0.15 * memory_score);
    }
}

void summarize(CandidateMeasurement& candidate, WorkloadClass workload, std::uint64_t seed) {
    QualifiedStrategy strategy;
    strategy.workload = workload;
    strategy.strategy_id = candidate.spec.strategy_id;
    strategy.plan.backend = candidate.spec.backend;
    strategy.plan.scheduling.prefill_quantum_tokens = candidate.spec.quantum;
    strategy.plan.kv.page_tokens = candidate.spec.kv_page;
    strategy.plan.strategy_id = strategy.strategy_id;
    strategy.samples = static_cast<std::uint32_t>(candidate.samples.size());
    strategy.evidence_seed = seed;

    std::vector<double> ttft, total, prefill, decode;
    ttft.reserve(candidate.samples.size());
    total.reserve(candidate.samples.size());
    prefill.reserve(candidate.samples.size());
    decode.reserve(candidate.samples.size());
    for (const auto& sample : candidate.samples) {
        ttft.push_back(sample.ttft_ms);
        total.push_back(sample.total_ms);
        prefill.push_back(sample.prefill_tokens_per_second);
        decode.push_back(sample.decode_tokens_per_second);
        strategy.peak_kv_bytes = std::max(strategy.peak_kv_bytes, sample.peak_kv_bytes);
        strategy.peak_device_bytes = std::max(strategy.peak_device_bytes, sample.peak_device_bytes);
    }

    const auto score_stats = mean_stddev(candidate.scores);
    const auto ttft_stats = mean_stddev(ttft);
    const auto total_stats = mean_stddev(total);
    const auto prefill_stats = mean_stddev(prefill);
    const auto decode_stats = mean_stddev(decode);
    strategy.score = score_stats.mean;
    strategy.score_stddev = score_stats.stddev;
    strategy.p50_ttft_ms = ttft_stats.mean;
    strategy.p50_ttft_stddev_ms = ttft_stats.stddev;
    strategy.p50_total_ms = total_stats.mean;
    strategy.p50_total_stddev_ms = total_stats.stddev;
    strategy.mean_prefill_tokens_per_second = prefill_stats.mean;
    strategy.prefill_tokens_per_second_stddev = prefill_stats.stddev;
    strategy.mean_decode_tokens_per_second = decode_stats.mean;
    strategy.decode_tokens_per_second_stddev = decode_stats.stddev;
    candidate.strategy = std::move(strategy);
}

} // namespace

Result<QualificationResult> qualify_model(const std::filesystem::path& model_path,
                                          const QualificationConfig& config) {
    if (config.prompt.empty()) return Status::invalid_argument("qualification prompt is empty");
    if (config.samples == 0U) return Status::invalid_argument("qualification samples must be non-zero");

    GgufFormat format;
    auto loaded = format.load(model_path);
    if (!loaded) return loaded.status();
    auto model = std::make_shared<ModelDefinition>(std::move(loaded).value());

    const std::array<std::uint32_t, 3> qualification_quanta{{16U, 32U, 128U}};
    constexpr std::uint32_t reference_page_tokens = 32U;
    constexpr std::uint32_t cuda_page_tokens = 16U;

    std::vector<CandidateSpec> specs;
    const auto add_reference = [&] {
        // Keep physical page geometry fixed while comparing scheduler quanta.
        // This isolates one causal variable instead of confounding page size and
        // scheduling policy in the same candidate label.
        for (const auto quantum : qualification_quanta) {
            CandidateSpec spec;
            spec.preference = BackendPreference::reference;
            spec.backend = BackendKind::reference;
            spec.quantum = quantum;
            spec.kv_page = reference_page_tokens;
            spec.strategy_id = std::string(to_string(config.workload)) + "-reference-q" +
                               std::to_string(quantum) + "-k" + std::to_string(reference_page_tokens);
            spec.canonical_default = quantum == 32U;
            specs.push_back(std::move(spec));
        }
    };
    const auto add_cuda = [&] {
        // CUDA native prefill makes scheduler quantum a real execution-facing
        // choice for every workload class: it controls how much prompt work is
        // submitted before yielding, up to the backend's native width. Keep KV
        // page geometry fixed so qualification measures that variable cleanly.
        for (const auto quantum : qualification_quanta) {
            CandidateSpec spec;
            spec.preference = BackendPreference::cuda;
            spec.backend = BackendKind::cuda;
            spec.quantum = quantum;
            spec.kv_page = cuda_page_tokens;
            spec.strategy_id = std::string(to_string(config.workload)) + "-cuda-q" +
                               std::to_string(quantum) + "-k" + std::to_string(cuda_page_tokens);
            spec.canonical_default = quantum == 32U;
            specs.push_back(std::move(spec));
        }
    };

    if (config.backend_mode == QualificationBackendMode::automatic) {
        add_cuda();
    } else {
        if (config.backend_mode == QualificationBackendMode::reference_only ||
            config.backend_mode == QualificationBackendMode::all) add_reference();
        if (config.backend_mode == QualificationBackendMode::cuda_only ||
            config.backend_mode == QualificationBackendMode::all) add_cuda();
    }

    // Preflight candidates once so every retained candidate has exactly the same
    // number of paired rounds. Automatic mode falls back to reference only when
    // CUDA is unavailable; explicit modes fail if none of their requested paths exist.
    std::vector<CandidateMeasurement> measured;
    for (const auto& spec : specs) {
        auto service = InferenceService::create(model, spec.preference, config.cuda_device, scheduler_for(spec));
        if (!service) {
            if (spec.backend == BackendKind::cuda) continue;
            return service.status();
        }
        measured.push_back(CandidateMeasurement{spec, {}, {}, {}});
    }
    if (config.backend_mode == QualificationBackendMode::automatic && measured.empty()) {
        specs.clear();
        add_reference();
        for (const auto& spec : specs) {
            auto service = InferenceService::create(model, spec.preference, config.cuda_device, scheduler_for(spec));
            if (!service) return service.status();
            measured.push_back(CandidateMeasurement{spec, {}, {}, {}});
        }
    }
    if (measured.empty()) return Status::unsupported("no requested qualification backend is available");

    // When both backend families are explicitly compared, CUDA is the canonical
    // default if available because automatic runtime policy prefers CUDA. Only one
    // candidate is allowed to carry the default role used by conservative selection.
    const bool have_cuda = std::any_of(measured.begin(), measured.end(), [](const auto& item) {
        return item.spec.backend == BackendKind::cuda;
    });
    for (auto& item : measured) {
        if (have_cuda && item.spec.backend != BackendKind::cuda) item.spec.canonical_default = false;
    }

    const std::uint64_t seed = config.seed == 0U ? qualification_seed(*model, config.workload) : config.seed;
    std::mt19937_64 rng(seed);
    std::vector<std::size_t> order(measured.size());
    std::iota(order.begin(), order.end(), 0U);

    // Interleave one sample per candidate per round and randomize candidate order.
    // This turns thermal/power/time drift into approximately balanced noise rather
    // than a systematic advantage for the first candidate in a fixed-order sweep.
    for (std::uint32_t round = 0; round < config.samples; ++round) {
        deterministic_shuffle(order, rng);
        for (const auto index : order) {
            auto sample = measure_once(model, config, measured[index].spec);
            if (!sample) return sample.status();
            measured[index].samples.push_back(std::move(sample).value());
        }
        score_round(measured, round);
    }
    for (auto& candidate : measured) summarize(candidate, config.workload, seed);

    const auto best_it = std::max_element(measured.begin(), measured.end(), [](const auto& a, const auto& b) {
        return a.strategy.score < b.strategy.score;
    });
    auto default_it = std::find_if(measured.begin(), measured.end(), [](const auto& item) {
        return item.spec.canonical_default;
    });
    if (default_it == measured.end()) default_it = measured.begin();

    auto selected_it = best_it;
    std::string selection_reason;
    double selection_margin = 0.0;
    double selection_half_width = 0.0;
    if (measured.size() == 1U) {
        selection_reason = "single-candidate";
    } else if (best_it == default_it) {
        selection_reason = "canonical-default-is-best";
    } else {
        std::vector<double> differences;
        differences.reserve(config.samples);
        for (std::size_t i = 0; i < best_it->scores.size(); ++i) {
            differences.push_back(best_it->scores[i] - default_it->scores[i]);
        }
        const auto diff = mean_stddev(differences);
        selection_margin = diff.mean;
        selection_half_width = differences.size() > 1U
            ? student_t_95_critical(differences.size()) * diff.stddev /
                  std::sqrt(static_cast<double>(differences.size()))
            : 0.0;
        constexpr double minimum_meaningful_score_margin = 0.02;
        if (differences.size() < 3U) {
            selected_it = default_it;
            selection_reason = "insufficient-samples-use-canonical-default";
        } else {
            const bool statistically_separated = diff.mean >= minimum_meaningful_score_margin &&
                diff.mean - selection_half_width > 0.0;
            if (statistically_separated) {
                selection_reason = "statistically-separated-best";
            } else {
                selected_it = default_it;
                selection_reason = "indistinguishable-use-canonical-default";
            }
        }
    }

    selected_it->strategy.selection_reason = selection_reason;
    selected_it->strategy.selection_margin = selection_margin;
    selected_it->strategy.selection_confidence_half_width = selection_half_width;

    ExecutionManifest manifest;
    const auto output = config.output.empty() ? default_manifest_path(*model) : config.output;
    if (std::filesystem::exists(output)) {
        auto existing = load_manifest(output);
        if (existing) {
            auto valid = validate_manifest(existing.value(), *model);
            if (valid.manifest) manifest = std::move(*valid.manifest);
        }
    }
    manifest.schema_version = execution_manifest_schema_version;
    manifest.air_version = version_string();
    manifest.model_digest = model_digest(*model);
    manifest.hardware_digest = hardware_digest();
    manifest.manifest_id = "manifest:" + manifest.model_digest.substr(manifest.model_digest.find(':') + 1U) +
                           ":" + std::to_string(unix_ms_now());
    manifest.strategies.erase(std::remove_if(manifest.strategies.begin(), manifest.strategies.end(),
        [&](const QualifiedStrategy& strategy) { return strategy.workload == config.workload; }),
        manifest.strategies.end());
    manifest.strategies.push_back(selected_it->strategy);
    const auto saved = save_manifest(manifest, output);
    if (!saved) return saved;

    QualificationResult result;
    result.manifest = manifest;
    result.selected = selected_it->strategy;
    result.selection_reason = selection_reason;
    result.selection_margin = selection_margin;
    result.selection_confidence_half_width = selection_half_width;
    result.evidence_seed = seed;
    for (auto& item : measured) result.candidates.push_back(std::move(item.strategy));
    return result;
}

} // namespace air

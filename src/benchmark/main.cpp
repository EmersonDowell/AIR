#include "air/benchmark.hpp"
#include "air/serving.hpp"
#include "air/version.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

namespace {

struct Options {
    Options() { scheduler.prefix_cache_entries = 0; }
    std::filesystem::path model;
    std::string prompt;
    std::filesystem::path prompt_file;
    std::filesystem::path output;
    air::BackendPreference backend{air::BackendPreference::automatic};
    int device{0};
    std::uint32_t tokens{128};
    std::uint32_t warmup{1};
    std::uint32_t runs{5};
    std::uint32_t concurrency{1};
    air::SchedulerConfig scheduler{};
    air::ManifestConfig manifest{};
    air::ExecutionConfig execution{};
};

void usage() {
    std::cout << "AIR Benchmark " << air::version_string() << "\n\n"
              << "Usage:\n"
              << "  air-bench -m model.gguf --prompt TEXT [options]\n"
              << "  air-bench -m model.gguf --prompt-file FILE [options]\n\n"
              << "Options:\n"
              << "  --backend auto|reference|cuda\n"
              << "  --device N\n"
              << "  --tokens N          generated tokens per request, default 128\n"
              << "  --warmup N          default 1\n"
              << "  --runs N            default 5\n"
              << "  --concurrency N     default 1\n"
              << "  --prefix-cache N    exact-prefix cache entries; capability-gated (reference today)\n"
              << "  --token-budget N\n"
              << "  --prefill-quantum N scheduler yield quantum\n"
              << "  --cuda-prefill-block-linear baseline|reuse4|reuse8|dense-f32-cublas\n"
              << "  --cuda-decode-block-linear baseline|reuse8|dense-f32-cublas\n"
              << "  --cuda-decode-output-linear baseline|reuse8\n"
              << "  --cuda-prefill-attention baseline|online-softmax\n"
              << "  --kv-page-tokens N set both backend KV page sizes\n"
              << "  --reference-kv-page-tokens N\n"
              << "  --cuda-kv-page-tokens N\n"
              << "  --manifest PATH     execution manifest path\n"
              << "  --no-manifest       disable adaptive manifest\n"
              << "  --require-manifest  fail if a valid manifest is unavailable\n"
              << "  --strategy-objective interactive|balanced|maximum-throughput|minimum-vram\n"
              << "  --strategy-horizon-tokens N\n"
              << "  --strategy-prepared-memory-budget-bytes N\n"
              << "  --output FILE       JSON report path\n";
}

std::optional<std::uint32_t> u32(std::string_view value) {
    try {
        const auto parsed = std::stoul(std::string(value));
        if (parsed > std::numeric_limits<std::uint32_t>::max()) return std::nullopt;
        return static_cast<std::uint32_t>(parsed);
    } catch (...) { return std::nullopt; }
}

std::optional<Options> parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto next = [&]() -> std::optional<std::string> {
            if (i + 1 >= argc) return std::nullopt;
            return std::string(argv[++i]);
        };
        if (arg == "--help" || arg == "-h") { usage(); std::exit(0); }
        if (arg == "--version") { std::cout << air::version_string() << '\n'; std::exit(0); }
        if (arg == "-m" || arg == "--model") { auto v=next(); if(!v)return{}; options.model=*v; }
        else if (arg == "--prompt") { auto v=next(); if(!v)return{}; options.prompt=*v; }
        else if (arg == "--prompt-file") { auto v=next(); if(!v)return{}; options.prompt_file=*v; }
        else if (arg == "--output") { auto v=next(); if(!v)return{}; options.output=*v; }
        else if (arg == "--manifest") { auto v=next(); if(!v)return{}; options.manifest.path=*v; }
        else if (arg == "--no-manifest") { options.manifest.enabled=false; }
        else if (arg == "--require-manifest") { options.manifest.require=true; }
        else if (arg == "--backend") {
            auto v=next(); if(!v)return{};
            if(*v=="auto") options.backend=air::BackendPreference::automatic;
            else if(*v=="reference") options.backend=air::BackendPreference::reference;
            else if(*v=="cuda") options.backend=air::BackendPreference::cuda;
            else return {};
        } else if (arg == "--device") { auto v=next(); if(!v)return{}; try{options.device=std::stoi(*v);}catch(...){return{};} }
        else if (arg == "--cuda-prefill-block-linear") {
            auto v=next(); if(!v)return{};
            auto tactic=air::quantized_linear_execution_kind_from_string(*v); if(!tactic)return{};
            options.execution.cuda_prefill_block_linear=tactic.value();
        }
        else if (arg == "--cuda-decode-block-linear") {
            auto v=next(); if(!v)return{};
            auto tactic=air::quantized_linear_execution_kind_from_string(*v); if(!tactic)return{};
            options.execution.cuda_decode_block_linear=tactic.value();
        }
        else if (arg == "--cuda-decode-output-linear") {
            auto v=next(); if(!v)return{};
            auto tactic=air::quantized_linear_execution_kind_from_string(*v); if(!tactic)return{};
            options.execution.cuda_decode_output_linear=tactic.value();
        }
        else if (arg == "--cuda-prefill-attention") {
            auto v=next(); if(!v)return{};
            auto tactic=air::attention_execution_kind_from_string(*v); if(!tactic)return{};
            options.execution.cuda_prefill_attention=tactic.value();
        }
        else if (arg == "--strategy-objective") {
            auto v=next(); if(!v)return{};
            auto objective=air::strategy_objective_from_string(*v); if(!objective)return{};
            options.manifest.strategy.objective=objective.value();
        }
        else if (arg == "--strategy-horizon-tokens") {
            auto v=next(); if(!v)return{};
            try { options.manifest.strategy.expected_horizon_tokens=std::stoull(*v); } catch(...) { return {}; }
        }
        else if (arg == "--strategy-prepared-memory-budget-bytes") {
            auto v=next(); if(!v)return{};
            try { options.manifest.strategy.prepared_memory_budget_bytes=std::stoull(*v); } catch(...) { return {}; }
        }
        else if (arg == "--tokens" || arg == "--warmup" || arg == "--runs" || arg == "--concurrency" ||
                 arg == "--prefix-cache" || arg == "--token-budget" || arg == "--prefill-quantum" || arg == "--kv-page-tokens" || arg == "--reference-kv-page-tokens" || arg == "--cuda-kv-page-tokens") {
            auto v=next(); if(!v)return{}; auto n=u32(*v); if(!n)return{};
            if(arg=="--tokens") options.tokens=*n;
            else if(arg=="--warmup") options.warmup=*n;
            else if(arg=="--runs") options.runs=*n;
            else if(arg=="--concurrency") options.concurrency=*n;
            else if(arg=="--prefix-cache") options.scheduler.prefix_cache_entries=*n;
            else if(arg=="--token-budget") options.scheduler.token_budget_per_cycle=*n;
            else if(arg=="--prefill-quantum") options.scheduler.prefill_quantum_tokens=*n;
            else if(arg=="--reference-kv-page-tokens") options.scheduler.reference_kv_page_tokens=*n;
            else if(arg=="--cuda-kv-page-tokens") options.scheduler.cuda_kv_page_tokens=*n;
            else { options.scheduler.reference_kv_page_tokens=*n; options.scheduler.cuda_kv_page_tokens=*n; }
        } else return {};
    }
    if (options.model.empty()) return {};
    if (!options.prompt.empty() && !options.prompt_file.empty()) return {};
    if (options.prompt.empty() && options.prompt_file.empty()) return {};
    if (options.runs == 0U || options.concurrency == 0U) return {};
    options.scheduler.max_active_requests = std::max(options.scheduler.max_active_requests, options.concurrency);
    return options;
}

std::optional<std::string> read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) return std::nullopt;
    std::ostringstream out; out << input.rdbuf(); return out.str();
}

} // namespace

int main(int argc, char** argv) {
    auto options = parse(argc, argv);
    if (!options) { usage(); return 2; }
    if (!options->prompt_file.empty()) {
        auto text = read_text(options->prompt_file);
        if (!text) { std::cerr << "unable to read prompt file\n"; return 2; }
        options->prompt = std::move(*text);
    }

    auto service = air::InferenceService::create(options->model, options->backend,
                                                 options->device, options->scheduler, {}, options->manifest, options->execution);
    if (!service) { std::cerr << "failed to initialize benchmark: " << service.status().message() << '\n'; return 3; }

    air::BenchmarkConfig config;
    config.request.prompt = options->prompt;
    config.request.generation.max_new_tokens = options->tokens;
    config.request.generation.sampling.temperature = 0.0;
    config.warmup_runs = options->warmup;
    config.measured_runs = options->runs;
    config.concurrency = options->concurrency;

    auto report = air::run_benchmark(*service.value(), config);
    if (!report) { std::cerr << "benchmark failed: " << report.status().message() << '\n'; return 4; }
    const auto& summary = report.value().summary;
    std::cout << "AIR benchmark\n"
              << "Backend:          " << report.value().backend << '\n'
              << "Planner:          " << report.value().planner_mode << '\n'
              << "Strategy:         " << report.value().strategy_id << '\n'
              << "Runs:             " << report.value().runs.size() << '\n'
              << "Concurrency:      " << config.concurrency << '\n'
              << "Wall:             " << summary.wall_ms << " ms\n"
              << "TTFT P50/P95:     " << summary.p50_ttft_ms << " / " << summary.p95_ttft_ms << " ms\n"
              << "Total P50/P95:    " << summary.p50_total_ms << " / " << summary.p95_total_ms << " ms\n"
              << "Prefill mean:     " << summary.mean_prefill_tokens_per_second << " tok/s\n"
              << "Decode mean:      " << summary.mean_decode_tokens_per_second << " tok/s\n"
              << "Aggregate output: " << summary.aggregate_generated_tokens_per_second << " tok/s\n"
              << "Requests/s:       " << summary.requests_per_second << '\n'
              << "Prefix reused:    " << summary.prefix_reused_tokens << " tokens\n"
              << "Peak KV:          " << summary.peak_kv_bytes << " bytes\n"
              << "Peak device:      " << summary.peak_device_bytes << " bytes\n";

    const auto json = report.value().to_json();
    if (options->output.empty()) {
        std::cout << "\n" << json;
    } else {
        std::ofstream output(options->output, std::ios::trunc);
        if (!output) { std::cerr << "unable to write benchmark report\n"; return 5; }
        output << json;
        std::cout << "Report:           " << options->output << '\n';
    }
    return 0;
}

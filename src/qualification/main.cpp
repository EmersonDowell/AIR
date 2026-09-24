#include "air/qualification.hpp"
#include "air/version.hpp"

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

namespace {
struct Options {
    std::filesystem::path model;
    std::string prompt{"Hello"};
    air::WorkloadClass workload{air::WorkloadClass::small};
    std::uint32_t tokens{32};
    std::uint32_t runs{3};
    int device{0};
    air::QualificationBackendMode backend_mode{air::QualificationBackendMode::automatic};
    std::uint64_t seed{0};
    std::filesystem::path output;
};

void usage() {
    std::cout << "AIR Qualification " << air::version_string() << "\n\n"
              << "Usage: air-qualify -m model.gguf [options]\n\n"
              << "  --prompt TEXT\n"
              << "  --workload small|medium|large|concurrent\n"
              << "  --tokens N          default 32\n"
              << "  --runs N            measured samples/candidate, default 3\n"
              << "  --device N\n"
              << "  --reference-only\n"
              << "  --cuda-only\n"
              << "  --all-backends      explicit reference + CUDA\n"
              << "  --seed N            reproduce randomized candidate order\n"
              << "  --output PATH       default ~/.cache/air/manifests/<model>.json\n";
}

std::optional<Options> parse(int argc, char** argv) {
    Options out;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::optional<std::string> {
            if (i + 1 >= argc) return std::nullopt;
            return std::string(argv[++i]);
        };
        if (arg == "-h" || arg == "--help") { usage(); std::exit(0); }
        if (arg == "--version") { std::cout << air::version_string() << '\n'; std::exit(0); }
        if (arg == "-m" || arg == "--model") { auto v=next(); if(!v)return{}; out.model=*v; }
        else if (arg == "--prompt") { auto v=next(); if(!v)return{}; out.prompt=*v; }
        else if (arg == "--output") { auto v=next(); if(!v)return{}; out.output=*v; }
        else if (arg == "--workload") {
            auto v=next(); if(!v)return{};
            if(*v=="small") out.workload=air::WorkloadClass::small;
            else if(*v=="medium") out.workload=air::WorkloadClass::medium;
            else if(*v=="large") out.workload=air::WorkloadClass::large;
            else if(*v=="concurrent") out.workload=air::WorkloadClass::concurrent;
            else return {};
        } else if (arg == "--tokens" || arg == "--runs") {
            auto v=next(); if(!v)return{}; try { auto n=std::stoul(*v); if(n==0U || n>UINT32_MAX)return{}; if(arg=="--tokens")out.tokens=static_cast<std::uint32_t>(n); else out.runs=static_cast<std::uint32_t>(n); } catch(...) { return {}; }
        } else if (arg == "--device") { auto v=next(); if(!v)return{}; try{out.device=std::stoi(*v);}catch(...){return{};} }
        else if (arg == "--seed") { auto v=next(); if(!v)return{}; try{out.seed=std::stoull(*v);}catch(...){return{};} }
        else if (arg == "--reference-only") { out.backend_mode=air::QualificationBackendMode::reference_only; }
        else if (arg == "--cuda-only") { out.backend_mode=air::QualificationBackendMode::cuda_only; }
        else if (arg == "--all-backends") { out.backend_mode=air::QualificationBackendMode::all; }
        else return {};
    }
    if (out.model.empty()) return {};
    return out;
}
}

int main(int argc, char** argv) {
    auto options = parse(argc, argv);
    if (!options) { usage(); return 2; }
    air::QualificationConfig config;
    config.prompt = options->prompt;
    config.workload = options->workload;
    config.max_new_tokens = options->tokens;
    config.samples = options->runs;
    config.cuda_device = options->device;
    config.backend_mode = options->backend_mode;
    config.seed = options->seed;
    config.output = options->output;
    auto result = air::qualify_model(options->model, config);
    if (!result) { std::cerr << "qualification failed: " << result.status().message() << '\n'; return 3; }
    for (const auto& candidate : result.value().candidates) {
        std::cout << candidate.strategy_id
                  << " score=" << candidate.score << "+/-" << candidate.score_stddev
                  << " ttft=" << candidate.p50_ttft_ms << "+/-" << candidate.p50_ttft_stddev_ms
                  << " total=" << candidate.p50_total_ms << "+/-" << candidate.p50_total_stddev_ms
                  << " prefill=" << candidate.mean_prefill_tokens_per_second << "+/-"
                  << candidate.prefill_tokens_per_second_stddev
                  << " decode=" << candidate.mean_decode_tokens_per_second << "+/-"
                  << candidate.decode_tokens_per_second_stddev
                  << " kv=" << candidate.peak_kv_bytes << '\n';
    }
    std::cout << "Selected: " << result.value().selected.strategy_id << '\n'
              << "Selection: " << result.value().selection_reason
              << " margin=" << result.value().selection_margin
              << " 95%half=" << result.value().selection_confidence_half_width << '\n'
              << "Evidence seed: " << result.value().evidence_seed << '\n'
              << "Manifest: " << (options->output.empty() ? "default cache" : options->output.string()) << '\n';
    return 0;
}

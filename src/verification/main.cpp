#include "air/cuda.hpp"
#include "air/format.hpp"
#include "air/reference.hpp"
#include "air/tokenizer.hpp"
#include "air/verification.hpp"
#include "air/version.hpp"

#include <boost/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Options {
    std::filesystem::path model;
    std::string prompt;
    std::vector<air::TokenId> exact_prompt_tokens;
    std::uint32_t generate{32};
    std::size_t top_k{8};
    std::uint32_t trace_from{0};
    std::uint32_t trace_count{0};
    int device{0};
    double atol{-1.0};
    air::QuantizedLinearExecutionKind cuda_prefill_block_linear{air::QuantizedLinearExecutionKind::baseline};
    air::QuantizedLinearExecutionKind cuda_decode_block_linear{air::QuantizedLinearExecutionKind::baseline};
    air::QuantizedLinearExecutionKind cuda_decode_output_linear{air::QuantizedLinearExecutionKind::baseline};
    air::AttentionExecutionKind cuda_prefill_attention{air::AttentionExecutionKind::baseline};
    air::AttentionExecutionKind cuda_decode_attention{air::AttentionExecutionKind::baseline};
    std::optional<std::filesystem::path> output;
};

struct StageResult {
    std::uint64_t position{0};
    std::int32_t layer{-1};
    air::VerificationStage stage{air::VerificationStage::embedding};
    air::VerificationComparison comparison;
};

struct DecisionResult {
    std::uint32_t index{0};
    air::TokenId reference_top{0};
    air::TokenId cuda_top{0};
    double reference_margin{0.0};
    double cuda_margin{0.0};
    air::VerificationComparison logits;
    std::vector<air::RankedToken> reference_top_k;
    std::vector<air::RankedToken> cuda_top_k;
};

void usage() {
    std::cout << "AIR Verify " << air::version_string() << "\n"
              << "Usage:\n"
              << "  air-verify -m model.gguf (--prompt TEXT | --tokens CSV) [options]\n\n"
              << "Options:\n"
              << "  --tokens CSV    exact numeric prompt token IDs; bypass tokenizer\n"
              << "  --generate N    teacher-forced greedy decisions after the prompt, default 32\n"
              << "  --top-k N       persist top-N logits per decision, default 8\n"
              << "  --trace-from N  decision whose consumed token starts stage tracing, default 0\n"
              << "  --trace-count N number of decision transitions with full stage snapshots, default 0\n"
              << "  --device N      CUDA device ordinal, default 0\n"
              << "  --atol X        optional absolute-error gate for every compared vector\n"
              << "  --cuda-prefill-block-linear KIND  baseline|reuse4|reuse8|dense-f32-cublas\n"
              << "  --cuda-decode-block-linear KIND   baseline|reuse8|dense-f32-cublas\n"
              << "  --cuda-decode-output-linear KIND  baseline|reuse8\n"
              << "  --cuda-prefill-attention KIND baseline|online-softmax\n"
              << "  --cuda-decode-attention KIND baseline\n"
              << "  --output FILE   write air.verification.v1 JSON evidence\n";
}

air::Result<Options> parse(int argc, char** argv) {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        return air::Status::invalid_state("help");
    }
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto need = [&](const char* name) -> air::Result<std::string> {
            if (i + 1 >= argc) return air::Status::invalid_argument(std::string(name) + " requires a value");
            ++i;
            return std::string(argv[i]);
        };
        try {
            if (arg == "-m" || arg == "--model") {
                auto value = need(arg.c_str()); if (!value) return value.status();
                options.model = value.value();
            } else if (arg == "--prompt") {
                auto value = need("--prompt"); if (!value) return value.status();
                options.prompt = value.value();
            } else if (arg == "--tokens") {
                auto value = need("--tokens"); if (!value) return value.status();
                std::size_t start = 0U;
                while (start <= value.value().size()) {
                    const auto comma = value.value().find(',', start);
                    const auto piece = value.value().substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                    if (piece.empty()) return air::Status::invalid_argument("--tokens contains an empty token id");
                    const auto parsed = std::stoll(piece);
                    if (parsed < 0LL || parsed > static_cast<long long>(std::numeric_limits<air::TokenId>::max())) {
                        return air::Status::invalid_argument("--tokens contains an out-of-range token id");
                    }
                    options.exact_prompt_tokens.push_back(static_cast<air::TokenId>(parsed));
                    if (comma == std::string::npos) break;
                    start = comma + 1U;
                }
            } else if (arg == "--generate") {
                auto value = need("--generate"); if (!value) return value.status();
                const auto parsed = std::stoul(value.value());
                if (parsed == 0UL || parsed > 4096UL) return air::Status::invalid_argument("--generate must be in [1,4096]");
                options.generate = static_cast<std::uint32_t>(parsed);
            } else if (arg == "--top-k") {
                auto value = need("--top-k"); if (!value) return value.status();
                const auto parsed = std::stoul(value.value());
                if (parsed == 0UL || parsed > 128UL) return air::Status::invalid_argument("--top-k must be in [1,128]");
                options.top_k = static_cast<std::size_t>(parsed);
            } else if (arg == "--trace-from") {
                auto value = need("--trace-from"); if (!value) return value.status();
                const auto parsed = std::stoul(value.value());
                if (parsed > 4096UL) return air::Status::invalid_argument("--trace-from must be in [0,4096]");
                options.trace_from = static_cast<std::uint32_t>(parsed);
            } else if (arg == "--trace-count") {
                auto value = need("--trace-count"); if (!value) return value.status();
                const auto parsed = std::stoul(value.value());
                if (parsed > 4096UL) return air::Status::invalid_argument("--trace-count must be in [0,4096]");
                options.trace_count = static_cast<std::uint32_t>(parsed);
            } else if (arg == "--device") {
                auto value = need("--device"); if (!value) return value.status();
                const auto parsed = std::stol(value.value());
                if (parsed < 0L || parsed > 1024L) return air::Status::invalid_argument("--device is outside supported range");
                options.device = static_cast<int>(parsed);
            } else if (arg == "--atol") {
                auto value = need("--atol"); if (!value) return value.status();
                options.atol = std::stod(value.value());
                if (!(options.atol >= 0.0) || !std::isfinite(options.atol)) {
                    return air::Status::invalid_argument("--atol requires a finite non-negative value");
                }
            } else if (arg == "--cuda-prefill-block-linear") {
                auto value = need("--cuda-prefill-block-linear"); if (!value) return value.status();
                auto tactic = air::quantized_linear_execution_kind_from_string(value.value());
                if (!tactic) return tactic.status();
                options.cuda_prefill_block_linear = tactic.value();
            } else if (arg == "--cuda-decode-block-linear") {
                auto value = need("--cuda-decode-block-linear"); if (!value) return value.status();
                auto tactic = air::quantized_linear_execution_kind_from_string(value.value());
                if (!tactic) return tactic.status();
                options.cuda_decode_block_linear = tactic.value();
            } else if (arg == "--cuda-decode-output-linear") {
                auto value = need("--cuda-decode-output-linear"); if (!value) return value.status();
                auto tactic = air::quantized_linear_execution_kind_from_string(value.value());
                if (!tactic) return tactic.status();
                options.cuda_decode_output_linear = tactic.value();
            } else if (arg == "--cuda-prefill-attention") {
                auto value = need("--cuda-prefill-attention"); if (!value) return value.status();
                auto tactic = air::attention_execution_kind_from_string(value.value());
                if (!tactic) return tactic.status();
                options.cuda_prefill_attention = tactic.value();
            } else if (arg == "--cuda-decode-attention") {
                auto value = need("--cuda-decode-attention"); if (!value) return value.status();
                auto tactic = air::attention_execution_kind_from_string(value.value());
                if (!tactic) return tactic.status();
                options.cuda_decode_attention = tactic.value();
            } else if (arg == "--output") {
                auto value = need("--output"); if (!value) return value.status();
                options.output = std::filesystem::path(value.value());
            } else {
                return air::Status::invalid_argument("unknown option: " + arg);
            }
        } catch (const std::exception&) {
            return air::Status::invalid_argument("invalid value for " + arg);
        }
    }
    if (options.model.empty()) return air::Status::invalid_argument("model path is required");
    const bool have_text = !options.prompt.empty();
    const bool have_tokens = !options.exact_prompt_tokens.empty();
    if (have_text == have_tokens) {
        return air::Status::invalid_argument("exactly one of --prompt or --tokens is required");
    }
    return options;
}

std::pair<air::TokenId, double> top_and_margin(std::span<const float> logits) {
    const auto ranked = air::top_k_logits(logits, 2U);
    if (ranked.empty()) return {-1, 0.0};
    const double margin = ranked.size() < 2U ? std::numeric_limits<double>::infinity() :
        static_cast<double>(ranked[0].logit) - static_cast<double>(ranked[1].logit);
    return {ranked[0].token, margin};
}

air::Result<std::vector<StageResult>> compare_traces(const air::VerificationTrace& reference,
                                                      const air::VerificationTrace& candidate) {
    const auto& left = reference.snapshots();
    const auto& right = candidate.snapshots();
    if (left.size() != right.size()) {
        return air::Status::internal_error("reference/CUDA verification trace lengths differ");
    }
    std::vector<StageResult> results;
    results.reserve(left.size());
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (left[i].position != right[i].position || left[i].layer != right[i].layer ||
            left[i].stage != right[i].stage) {
            return air::Status::internal_error("reference/CUDA verification stage ordering differs");
        }
        auto comparison = air::compare_vectors(left[i].values, right[i].values);
        if (!comparison) return comparison.status();
        results.push_back(StageResult{left[i].position, left[i].layer, left[i].stage,
                                      comparison.value()});
    }
    return results;
}

boost::json::object comparison_json(const air::VerificationComparison& value) {
    boost::json::object out;
    out["elements"] = value.elements;
    out["max_abs_error"] = value.max_abs_error;
    out["mean_abs_error"] = value.mean_abs_error;
    out["rms_error"] = value.rms_error;
    out["max_error_index"] = value.max_error_index;
    out["reference_at_max"] = value.reference_at_max;
    out["candidate_at_max"] = value.candidate_at_max;
    out["finite"] = value.finite;
    return out;
}

boost::json::array ranked_json(const std::vector<air::RankedToken>& ranked) {
    boost::json::array out;
    for (const auto& item : ranked) {
        boost::json::object value;
        value["token"] = item.token;
        value["logit"] = item.logit;
        out.emplace_back(std::move(value));
    }
    return out;
}

int run(const Options& options) {
    air::GgufFormat format;
    auto loaded = format.load(options.model);
    if (!loaded) {
        std::cerr << "model load failed: " << loaded.status().message() << '\n';
        return 3;
    }
    auto model = std::make_shared<air::ModelDefinition>(std::move(loaded).value());
    std::vector<air::TokenId> prompt_tokens;
    if (!options.exact_prompt_tokens.empty()) {
        prompt_tokens = options.exact_prompt_tokens;
        for (const auto token : prompt_tokens) {
            if (token < 0 || static_cast<std::uint64_t>(token) >= model->config().vocabulary_size) {
                std::cerr << "exact prompt contains token outside model vocabulary: " << token << '\n';
                return 5;
            }
        }
    } else {
        auto tokenizer = air::create_tokenizer(model->tokenizer_handle());
        if (!tokenizer) {
            std::cerr << "tokenizer unavailable: " << tokenizer.status().message() << '\n';
            return 4;
        }
        auto encoded = tokenizer.value()->encode(options.prompt);
        if (!encoded || encoded.value().empty()) {
            std::cerr << "prompt tokenization failed: "
                      << (encoded ? "prompt produced zero tokens" : encoded.status().message()) << '\n';
            return 5;
        }
        prompt_tokens = std::move(encoded).value();
    }
    auto reference = air::ReferenceExecutor::create(model);
    if (!reference) {
        std::cerr << "reference executor unavailable: " << reference.status().message() << '\n';
        return 6;
    }
    auto cuda = air::CudaExecutor::create(model, options.device);
    if (!cuda) {
        std::cerr << "CUDA executor unavailable: " << cuda.status().message() << '\n';
        return 7;
    }

    const auto& config = model->config();
    const auto head_dimension = config.embedding_size / config.attention_head_count;
    air::ReferenceKvCache reference_cache(config.layer_count, config.kv_head_count,
                                          head_dimension, config.context_length, 16U);
    auto cuda_cache = cuda.value()->create_kv_cache(16U);
    if (!cuda_cache) {
        std::cerr << "CUDA KV unavailable: " << cuda_cache.status().message() << '\n';
        return 8;
    }

    auto reference_logits = reference.value()->prefill(prompt_tokens, reference_cache);
    if (!reference_logits) {
        std::cerr << "reference prefill failed: " << reference_logits.status().message() << '\n';
        return 9;
    }
    auto cuda_logits = cuda.value()->prefill(prompt_tokens, *cuda_cache.value(), options.cuda_prefill_block_linear, options.cuda_prefill_attention);
    if (!cuda_logits) {
        std::cerr << "CUDA prefill failed: " << cuda_logits.status().message() << '\n';
        return 10;
    }

    std::vector<DecisionResult> decisions;
    std::vector<StageResult> stages;
    std::vector<air::TokenId> forced_tokens;
    bool all_top1_match = true;
    bool all_finite = true;
    bool within_atol = true;
    double worst_stage_error = 0.0;
    std::optional<StageResult> worst_stage;

    for (std::uint32_t decision_index = 0; decision_index < options.generate; ++decision_index) {
        auto logits_comparison = air::compare_vectors(reference_logits.value(), cuda_logits.value());
        if (!logits_comparison) {
            std::cerr << "logit comparison failed: " << logits_comparison.status().message() << '\n';
            return 11;
        }
        const auto [reference_top, reference_margin] = top_and_margin(reference_logits.value());
        const auto [cuda_top, cuda_margin] = top_and_margin(cuda_logits.value());
        all_top1_match = all_top1_match && reference_top == cuda_top;
        all_finite = all_finite && logits_comparison.value().finite;
        if (options.atol >= 0.0) within_atol = within_atol && logits_comparison.value().max_abs_error <= options.atol;

        decisions.push_back(DecisionResult{
            decision_index,
            reference_top,
            cuda_top,
            reference_margin,
            cuda_margin,
            logits_comparison.value(),
            air::top_k_logits(reference_logits.value(), options.top_k),
            air::top_k_logits(cuda_logits.value(), options.top_k),
        });
        forced_tokens.push_back(reference_top);

        std::cout << "decision " << decision_index
                  << " ref/cuda=" << reference_top << '/' << cuda_top
                  << " margin=" << reference_margin
                  << " max_abs=" << logits_comparison.value().max_abs_error
                  << (reference_top == cuda_top ? " match" : " MISMATCH") << '\n';

        if (decision_index + 1U == options.generate) break;
        const auto trace_end = static_cast<std::uint64_t>(options.trace_from) + options.trace_count;
        const bool trace_transition = options.trace_count != 0U &&
            decision_index >= options.trace_from && static_cast<std::uint64_t>(decision_index) < trace_end;
        if (trace_transition) {
            air::VerificationTrace reference_trace;
            air::VerificationTrace cuda_trace;
            auto next_reference = reference.value()->step_verified(reference_top, reference_cache, reference_trace);
            if (!next_reference) {
                std::cerr << "reference verified step failed: " << next_reference.status().message() << '\n';
                return 12;
            }
            auto next_cuda = cuda.value()->step_verified(
                reference_top, *cuda_cache.value(), cuda_trace,
                options.cuda_decode_block_linear, options.cuda_decode_output_linear,
                options.cuda_decode_attention);
            if (!next_cuda) {
                std::cerr << "CUDA verified step failed: " << next_cuda.status().message() << '\n';
                return 13;
            }
            auto compared = compare_traces(reference_trace, cuda_trace);
            if (!compared) {
                std::cerr << "stage comparison failed: " << compared.status().message() << '\n';
                return 14;
            }
            for (const auto& stage : compared.value()) {
                all_finite = all_finite && stage.comparison.finite;
                if (options.atol >= 0.0) within_atol = within_atol && stage.comparison.max_abs_error <= options.atol;
                if (!worst_stage || stage.comparison.max_abs_error > worst_stage_error) {
                    worst_stage_error = stage.comparison.max_abs_error;
                    worst_stage = stage;
                }
                stages.push_back(stage);
            }
            reference_logits = std::move(next_reference).value();
            cuda_logits = std::move(next_cuda).value();
        } else {
            auto next_reference = reference.value()->step(reference_top, reference_cache);
            if (!next_reference) {
                std::cerr << "reference teacher-forced step failed: " << next_reference.status().message() << '\n';
                return 12;
            }
            auto next_cuda = cuda.value()->step(
                reference_top, *cuda_cache.value(),
                options.cuda_decode_block_linear, options.cuda_decode_output_linear,
                options.cuda_decode_attention);
            if (!next_cuda) {
                std::cerr << "CUDA teacher-forced step failed: " << next_cuda.status().message() << '\n';
                return 13;
            }
            reference_logits = std::move(next_reference).value();
            cuda_logits = std::move(next_cuda).value();
        }
    }

    boost::json::object root;
    root["schema"] = air::verification_report_schema;
    root["air_version"] = air::version_string();
    root["model_id"] = model->fingerprint().model_id;
    root["model_architecture"] = model->config().architecture;
    root["prompt"] = options.prompt;
    root["prompt_source"] = options.exact_prompt_tokens.empty() ? "text" : "tokens";
    root["device"] = options.device;
    root["top_k"] = options.top_k;
    root["trace_from"] = options.trace_from;
    root["trace_count"] = options.trace_count;
    root["requested_atol"] = options.atol;
    root["cuda_prefill_block_linear"] = air::to_string(options.cuda_prefill_block_linear);
    root["cuda_decode_block_linear"] = air::to_string(options.cuda_decode_block_linear);
    root["cuda_decode_output_linear"] = air::to_string(options.cuda_decode_output_linear);
    root["cuda_prefill_attention"] = air::to_string(options.cuda_prefill_attention);
    root["cuda_decode_attention"] = air::to_string(options.cuda_decode_attention);

    boost::json::array prompt_tokens_json;
    for (const auto token : prompt_tokens) prompt_tokens_json.emplace_back(token);
    root["prompt_tokens"] = std::move(prompt_tokens_json);
    boost::json::array teacher_tokens;
    for (const auto token : forced_tokens) teacher_tokens.emplace_back(token);
    root["teacher_forced_tokens"] = std::move(teacher_tokens);

    boost::json::array decision_json;
    for (const auto& decision : decisions) {
        boost::json::object item;
        item["index"] = decision.index;
        item["reference_top"] = decision.reference_top;
        item["cuda_top"] = decision.cuda_top;
        item["top1_match"] = decision.reference_top == decision.cuda_top;
        item["reference_top1_top2_margin"] = decision.reference_margin;
        item["cuda_top1_top2_margin"] = decision.cuda_margin;
        item["logits"] = comparison_json(decision.logits);
        item["reference_top_k"] = ranked_json(decision.reference_top_k);
        item["cuda_top_k"] = ranked_json(decision.cuda_top_k);
        decision_json.emplace_back(std::move(item));
    }
    root["decisions"] = std::move(decision_json);

    boost::json::array stage_json;
    for (const auto& stage : stages) {
        boost::json::object item;
        item["position"] = stage.position;
        item["layer"] = stage.layer;
        item["stage"] = air::to_string(stage.stage);
        item["comparison"] = comparison_json(stage.comparison);
        stage_json.emplace_back(std::move(item));
    }
    root["stage_comparisons"] = std::move(stage_json);

    boost::json::object summary;
    summary["decisions"] = decisions.size();
    summary["stage_comparisons"] = stages.size();
    summary["top1_parity"] = all_top1_match;
    summary["finite"] = all_finite;
    summary["within_requested_atol"] = options.atol < 0.0 ? true : within_atol;
    summary["worst_stage_max_abs_error"] = worst_stage_error;
    if (worst_stage) {
        summary["worst_stage_position"] = worst_stage->position;
        summary["worst_stage_layer"] = worst_stage->layer;
        summary["worst_stage"] = air::to_string(worst_stage->stage);
    }
    root["summary"] = std::move(summary);

    if (options.output) {
        std::ofstream out(*options.output, std::ios::trunc);
        if (!out) {
            std::cerr << "failed to open verification output: " << *options.output << '\n';
            return 15;
        }
        out << boost::json::serialize(root) << '\n';
        if (!out) {
            std::cerr << "failed to write verification output\n";
            return 15;
        }
        std::cout << "Report: " << *options.output << '\n';
    }

    std::cout << "\nVerification summary\n"
              << "decisions:          " << decisions.size() << '\n'
              << "stage comparisons:  " << stages.size() << '\n'
              << "top-1 parity:       " << (all_top1_match ? "yes" : "no") << '\n'
              << "finite:             " << (all_finite ? "yes" : "no") << '\n'
              << "worst stage error:  " << worst_stage_error << '\n';
    if (worst_stage) {
        std::cout << "worst stage:        position=" << worst_stage->position
                  << " layer=" << worst_stage->layer
                  << " stage=" << air::to_string(worst_stage->stage) << '\n';
    }
    if (options.atol >= 0.0) {
        std::cout << "atol gate:          " << options.atol << " -> "
                  << (within_atol ? "PASS" : "FAIL") << '\n';
    } else {
        std::cout << "atol gate:          not requested\n";
    }

    return all_top1_match && all_finite && (options.atol < 0.0 || within_atol) ? 0 : 9;
}

} // namespace

int main(int argc, char** argv) {
    auto options = parse(argc, argv);
    if (!options) {
        if (options.status().code() == air::ErrorCode::invalid_state && options.status().message() == "help") {
            usage();
            return 0;
        }
        std::cerr << options.status().message() << '\n';
        usage();
        return 2;
    }
    return run(options.value());
}

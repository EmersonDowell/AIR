#include "air/cuda.hpp"
#include "air/format.hpp"
#include "air/manifest.hpp"
#include "air/reference.hpp"
#include "air/tokenizer.hpp"
#include "air/version.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

void print_usage() {
    std::cout << "AIR CLI " << air::version_string() << "\n"
              << "Usage:\n"
              << "  air-cli --help\n"
              << "  air-cli --version\n"
              << "  air-cli inspect <model.gguf> [--tensors]\n"
              << "  air-cli fingerprint <model.gguf>\n"
              << "  air-cli tokenize <model.gguf> <text>\n"
              << "  air-cli reference <model.gguf> <text> [--tokens N]\n"
              << "  air-cli cuda-info\n"
              << "  air-cli cuda <model.gguf> <text> [--tokens N] [--device N]\n"
              << "  air-cli cuda-compare <model.gguf> <text> [--device N] [--atol X]\n";
}

std::string human_bytes(std::uint64_t bytes) {
    static constexpr const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    std::size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < std::size(units)) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(unit == 0 ? 0 : 2) << value << ' ' << units[unit];
    return out.str();
}

int inspect(const std::filesystem::path& path, bool list_tensors) {
    air::GgufFormat format;
    auto result = format.load(path);
    if (!result) {
        std::cerr << "failed to inspect model: " << result.status().message() << '\n';
        return 3;
    }

    const auto& model = result.value();
    const auto& config = model.config();
    const auto& tokenizer = model.tokenizer();

    std::map<std::string, std::size_t> type_counts;
    std::size_t inexact_tensors = 0;
    for (const auto& tensor : model.tensors()) {
        std::string type = air::to_string(tensor.type);
        if (tensor.type == air::DataType::unknown) type += "(ggml:" + std::to_string(tensor.format_type) + ")";
        ++type_counts[type];
        if (!tensor.byte_size_exact) ++inexact_tensors;
    }

    std::cout << "AIR Model Inspector\n\n"
              << "Format:             " << model.fingerprint().format << '\n'
              << "Name:               " << model.fingerprint().model_id << '\n'
              << "Architecture:       " << config.architecture << '\n'
              << "File:               " << path << '\n'
              << "Mapped size:        " << human_bytes(model.storage()->size_bytes()) << '\n'
              << "Layers:             " << config.layer_count << '\n'
              << "Embedding:          " << config.embedding_size << '\n'
              << "Feed-forward:       " << config.feed_forward_size << '\n'
              << "Attention heads:    " << config.attention_head_count << '\n'
              << "KV heads:           " << config.kv_head_count << '\n'
              << "Sliding window:     " << config.attention_sliding_window << '\n'
              << "Context:            " << config.context_length << '\n'
              << "RoPE dimensions:    " << config.rope_dimension_count << '\n'
              << "RoPE base:          " << config.rope_frequency_base << '\n'
              << "RMS epsilon:        " << config.rms_norm_epsilon << '\n'
              << "Vocabulary:         " << config.vocabulary_size << '\n'
              << "Tokenizer:          " << tokenizer.model << '\n'
              << "Pre-tokenizer:      " << (tokenizer.pre_tokenizer.empty() ? "<unspecified>" : tokenizer.pre_tokenizer) << '\n'
              << "BPE merges:         " << tokenizer.merges.size() << '\n'
              << "Tensors:            " << model.tensors().size() << '\n'
              << "Inexact encodings:  " << inexact_tensors << '\n';

    std::cout << "\nTensor types:\n";
    for (const auto& [type, count] : type_counts) {
        std::cout << "  " << std::left << std::setw(18) << type << count << '\n';
    }

    if (list_tensors) {
        std::cout << "\nTensor table:\n";
        for (const auto& tensor : model.tensors()) {
            std::cout << "  " << tensor.name << "  [";
            for (std::size_t i = 0; i < tensor.shape.dimensions.size(); ++i) {
                if (i != 0) std::cout << 'x';
                std::cout << tensor.shape.dimensions[i];
            }
            std::cout << "]  " << air::to_string(tensor.type);
            if (tensor.type == air::DataType::unknown) std::cout << "(ggml:" << tensor.format_type << ')';
            std::cout << "  " << human_bytes(tensor.byte_size);
            if (!tensor.byte_size_exact) std::cout << " storage-span";
            std::cout << '\n';
        }
    }
    return 0;
}


int fingerprint(const std::filesystem::path& path) {
    air::GgufFormat format;
    auto loaded = format.load(path);
    if (!loaded) {
        std::cerr << "failed to fingerprint model: " << loaded.status().message() << '\n';
        return 3;
    }
    std::cout << "{\n"
              << "  \"air_version\": \"" << air::version_string() << "\",\n"
              << "  \"model_digest\": \"" << air::model_digest(loaded.value()) << "\",\n"
              << "  \"hardware_digest\": \"" << air::hardware_digest() << "\"\n"
              << "}\n";
    return 0;
}

int tokenize(const std::filesystem::path& path, const std::string& text) {
    air::GgufFormat format;
    auto model_result = format.load(path);
    if (!model_result) {
        std::cerr << "failed to load model: " << model_result.status().message() << '\n';
        return 3;
    }
    auto tokenizer_result = air::create_tokenizer(model_result.value().tokenizer_handle());
    if (!tokenizer_result) {
        std::cerr << "tokenizer unavailable: " << tokenizer_result.status().message() << '\n';
        return 4;
    }
    auto tokens = tokenizer_result.value()->encode(text);
    if (!tokens) {
        std::cerr << "tokenization failed: " << tokens.status().message() << '\n';
        return 5;
    }
    std::cout << "tokens(" << tokens.value().size() << "):";
    for (const auto token : tokens.value()) std::cout << ' ' << token;
    std::cout << '\n';
    return 0;
}

struct PreparedPrompt {
    std::shared_ptr<air::ModelDefinition> model;
    std::unique_ptr<air::Tokenizer> tokenizer;
    std::vector<air::TokenId> tokens;
};

air::Result<PreparedPrompt> prepare_prompt(const std::filesystem::path& path, const std::string& text) {
    air::GgufFormat format;
    auto loaded = format.load(path);
    if (!loaded) return loaded.status();
    auto model = std::make_shared<air::ModelDefinition>(std::move(loaded).value());
    auto tokenizer = air::create_tokenizer(model->tokenizer_handle());
    if (!tokenizer) return tokenizer.status();
    auto prompt = tokenizer.value()->encode(text);
    if (!prompt) return prompt.status();
    if (prompt.value().empty()) return air::Status::invalid_argument("prompt produces no tokens");
    return PreparedPrompt{std::move(model), std::move(tokenizer).value(), std::move(prompt).value()};
}

int reference_generate(const std::filesystem::path& path, const std::string& text, std::uint32_t max_tokens) {
    auto prepared = prepare_prompt(path, text);
    if (!prepared) {
        std::cerr << "failed to prepare prompt: " << prepared.status().message() << '\n';
        return 3;
    }
    auto executor = air::ReferenceExecutor::create(prepared.value().model);
    if (!executor) {
        std::cerr << "reference executor unavailable: " << executor.status().message() << '\n';
        return 6;
    }
    air::GenerationConfig config;
    config.max_new_tokens = max_tokens;
    config.sampling.temperature = 0.0;
    auto generated = executor.value()->generate(prepared.value().tokens, config);
    if (!generated) {
        std::cerr << "reference generation failed: " << generated.status().message() << '\n';
        return 7;
    }
    auto decoded = prepared.value().tokenizer->decode(generated.value().tokens, false);
    if (!decoded) {
        std::cerr << "decode failed: " << decoded.status().message() << '\n';
        return 8;
    }
    std::cout << decoded.value() << '\n';
    std::cout << "generated tokens(" << generated.value().tokens.size() << "):";
    for (const auto token : generated.value().tokens) std::cout << ' ' << token;
    std::cout << '\n';
    return 0;
}


int cuda_info() {
    auto devices = air::cuda_devices();
    if (!devices) {
        std::cerr << "CUDA unavailable: " << devices.status().message() << '\n';
        return devices.status().code() == air::ErrorCode::unsupported ? 4 : 5;
    }
    std::cout << "AIR CUDA devices: " << devices.value().size() << '\n';
    for (const auto& device : devices.value()) {
        std::cout << "  [" << device.ordinal << "] " << device.name
                  << "  sm_" << device.compute_major << device.compute_minor
                  << "  free=" << human_bytes(device.free_memory_bytes)
                  << "  total=" << human_bytes(device.total_memory_bytes) << '\n';
    }
    return 0;
}

int cuda_generate(const std::filesystem::path& path, const std::string& text,
                  std::uint32_t max_tokens, int device_ordinal) {
    auto prepared = prepare_prompt(path, text);
    if (!prepared) {
        std::cerr << "failed to prepare prompt: " << prepared.status().message() << '\n';
        return 3;
    }
    auto executor = air::CudaExecutor::create(prepared.value().model, device_ordinal);
    if (!executor) {
        std::cerr << "CUDA executor unavailable: " << executor.status().message() << '\n';
        return 6;
    }
    air::GenerationConfig config;
    config.max_new_tokens = max_tokens;
    config.sampling.temperature = 0.0;
    auto generated = executor.value()->generate(prepared.value().tokens, config);
    if (!generated) {
        std::cerr << "CUDA generation failed: " << generated.status().message() << '\n';
        return 7;
    }
    auto decoded = prepared.value().tokenizer->decode(generated.value().tokens, false);
    if (!decoded) {
        std::cerr << "decode failed: " << decoded.status().message() << '\n';
        return 8;
    }
    const auto stats = executor.value()->stats();
    std::cout << decoded.value() << '\n';
    std::cout << "generated tokens(" << generated.value().tokens.size() << "):";
    for (const auto token : generated.value().tokens) std::cout << ' ' << token;
    std::cout << "\nCUDA device:       " << executor.value()->device_ordinal()
              << "\nResident model:    " << human_bytes(stats.resident_model_bytes)
              << "\nWorkspace:         " << human_bytes(stats.workspace_bytes)
              << "\nKV logical ctx:    " << human_bytes(stats.kv_logical_context_bytes)
              << "\nKV page:           " << human_bytes(stats.kv_page_bytes)
              << "\nKV pool allocated: " << human_bytes(stats.kv_pool_allocated_bytes)
              << "\nKV pool free:      " << human_bytes(stats.kv_pool_free_bytes)
              << "\nHost -> device:    " << human_bytes(stats.host_to_device_bytes)
              << "\nDevice -> host:    " << human_bytes(stats.device_to_host_bytes)
              << "\nF32 matvec calls:  " << stats.f32_matvec_calls
              << "\nSpecial matvec:    " << stats.specialized_matvec_calls
              << "\nF32 matmul calls:  " << stats.f32_matmul_calls
              << "\nSpecial matmul:    " << stats.specialized_matmul_calls
              << "\nFull logit reads:  " << stats.full_logit_readbacks
              << "\nGreedy token reads:" << stats.greedy_token_readbacks
              << "\nOutputless prefill:" << stats.outputless_prefill_chunks << '\n';
    return 0;
}


int cuda_compare(const std::filesystem::path& path, const std::string& text,
                 int device_ordinal, double absolute_tolerance) {
    auto prepared = prepare_prompt(path, text);
    if (!prepared) {
        std::cerr << "failed to prepare prompt: " << prepared.status().message() << '\n';
        return 3;
    }
    auto reference = air::ReferenceExecutor::create(prepared.value().model);
    if (!reference) {
        std::cerr << "reference executor unavailable: " << reference.status().message() << '\n';
        return 6;
    }
    auto cuda = air::CudaExecutor::create(prepared.value().model, device_ordinal);
    if (!cuda) {
        std::cerr << "CUDA executor unavailable: " << cuda.status().message() << '\n';
        return 6;
    }
    const auto& config = prepared.value().model->config();
    const auto head_dimension = config.embedding_size / config.attention_head_count;
    air::ReferenceKvCache reference_cache(config.layer_count, config.kv_head_count,
                                          head_dimension, config.context_length);
    auto cuda_cache = cuda.value()->create_kv_cache();
    if (!cuda_cache) {
        std::cerr << "CUDA KV cache unavailable: " << cuda_cache.status().message() << '\n';
        return 6;
    }
    auto cuda_greedy_cache = cuda.value()->create_kv_cache();
    if (!cuda_greedy_cache) {
        std::cerr << "CUDA greedy KV cache unavailable: " << cuda_greedy_cache.status().message() << '\n';
        return 6;
    }

    double global_max = 0.0;
    double total_abs = 0.0;
    std::uint64_t compared = 0U;
    bool all_top1_match = true;
    for (std::size_t position = 0; position < prepared.value().tokens.size(); ++position) {
        const auto token = prepared.value().tokens[position];
        auto reference_logits = reference.value()->step(token, reference_cache);
        if (!reference_logits) {
            std::cerr << "reference step failed at position " << position << ": "
                      << reference_logits.status().message() << '\n';
            return 7;
        }
        auto cuda_logits = cuda.value()->step(token, *cuda_cache.value());
        if (!cuda_logits) {
            std::cerr << "CUDA step failed at position " << position << ": "
                      << cuda_logits.status().message() << '\n';
            return 7;
        }
        if (reference_logits.value().size() != cuda_logits.value().size()) {
            std::cerr << "logit vector size mismatch at position " << position << '\n';
            return 7;
        }
        double position_max = 0.0;
        for (std::size_t i = 0; i < reference_logits.value().size(); ++i) {
            const double difference = std::abs(static_cast<double>(reference_logits.value()[i]) -
                                               static_cast<double>(cuda_logits.value()[i]));
            position_max = std::max(position_max, difference);
            global_max = std::max(global_max, difference);
            total_abs += difference;
            ++compared;
        }
        const auto reference_top = static_cast<air::TokenId>(std::distance(
            reference_logits.value().begin(), std::max_element(reference_logits.value().begin(), reference_logits.value().end())));
        const auto cuda_top = static_cast<air::TokenId>(std::distance(
            cuda_logits.value().begin(), std::max_element(cuda_logits.value().begin(), cuda_logits.value().end())));
        const bool top_match = reference_top == cuda_top;
        auto device_greedy = cuda.value()->step_greedy(token, *cuda_greedy_cache.value());
        if (!device_greedy) {
            std::cerr << "CUDA device-greedy step failed at position " << position << ": "
                      << device_greedy.status().message() << '\n';
            return 7;
        }
        const bool greedy_match = reference_top == device_greedy.value();
        all_top1_match = all_top1_match && top_match && greedy_match;
        std::cout << "position " << position
                  << " token=" << token
                  << " max_abs=" << position_max
                  << " top1=" << reference_top << '/' << cuda_top
                  << " greedy=" << device_greedy.value()
                  << (top_match && greedy_match ? " match" : " MISMATCH") << '\n';
    }

    const double mean_abs = compared == 0U ? 0.0 : total_abs / static_cast<double>(compared);
    std::cout << "\nCUDA/reference decode-step comparison\n"
              << "positions:       " << prepared.value().tokens.size() << '\n'
              << "logits compared: " << compared << '\n'
              << "max abs error:   " << global_max << '\n'
              << "mean abs error:  " << mean_abs << '\n'
              << "top-1 parity:    " << (all_top1_match ? "yes" : "no") << '\n';

    // A second, independent gate validates native multi-token CUDA prefill.
    // prefill path. The step-by-step comparison above cannot exercise that
    // implementation because it intentionally drives decode width one.
    air::ReferenceKvCache reference_prefill_cache(config.layer_count, config.kv_head_count,
                                                  head_dimension, config.context_length);
    auto cuda_prefill_cache = cuda.value()->create_kv_cache(16U);
    if (!cuda_prefill_cache) {
        std::cerr << "CUDA prefill KV cache unavailable: " << cuda_prefill_cache.status().message() << '\n';
        return 7;
    }
    auto reference_prefill = reference.value()->prefill(prepared.value().tokens, reference_prefill_cache);
    if (!reference_prefill) {
        std::cerr << "reference prefill failed: " << reference_prefill.status().message() << '\n';
        return 7;
    }
    auto cuda_prefill = cuda.value()->prefill(prepared.value().tokens, *cuda_prefill_cache.value());
    if (!cuda_prefill) {
        std::cerr << "CUDA native prefill failed: " << cuda_prefill.status().message() << '\n';
        return 7;
    }
    if (reference_prefill.value().size() != cuda_prefill.value().size()) {
        std::cerr << "native prefill logit vector size mismatch\n";
        return 7;
    }
    double prefill_max = 0.0;
    double prefill_total = 0.0;
    for (std::size_t i = 0; i < reference_prefill.value().size(); ++i) {
        const double difference = std::abs(static_cast<double>(reference_prefill.value()[i]) -
                                           static_cast<double>(cuda_prefill.value()[i]));
        prefill_max = std::max(prefill_max, difference);
        prefill_total += difference;
    }
    const double prefill_mean = reference_prefill.value().empty() ? 0.0 :
        prefill_total / static_cast<double>(reference_prefill.value().size());
    const auto reference_prefill_top = static_cast<air::TokenId>(std::distance(
        reference_prefill.value().begin(),
        std::max_element(reference_prefill.value().begin(), reference_prefill.value().end())));
    const auto cuda_prefill_top = static_cast<air::TokenId>(std::distance(
        cuda_prefill.value().begin(),
        std::max_element(cuda_prefill.value().begin(), cuda_prefill.value().end())));
    auto cuda_prefill_greedy_cache = cuda.value()->create_kv_cache(16U);
    if (!cuda_prefill_greedy_cache) {
        std::cerr << "CUDA greedy prefill KV cache unavailable: "
                  << cuda_prefill_greedy_cache.status().message() << '\n';
        return 7;
    }
    auto cuda_prefill_greedy = cuda.value()->prefill_greedy(
        prepared.value().tokens, *cuda_prefill_greedy_cache.value());
    if (!cuda_prefill_greedy) {
        std::cerr << "CUDA device-greedy native prefill failed: "
                  << cuda_prefill_greedy.status().message() << '\n';
        return 7;
    }
    const bool prefill_top_match = reference_prefill_top == cuda_prefill_top;
    const bool prefill_greedy_match = reference_prefill_top == cuda_prefill_greedy.value();
    std::cout << "\nCUDA/reference native-prefill comparison\n"
              << "tokens:           " << prepared.value().tokens.size() << '\n'
              << "logits compared:  " << reference_prefill.value().size() << '\n'
              << "max abs error:    " << prefill_max << '\n'
              << "mean abs error:   " << prefill_mean << '\n'
              << "top-1:            " << reference_prefill_top << '/' << cuda_prefill_top
              << " greedy=" << cuda_prefill_greedy.value()
              << (prefill_top_match && prefill_greedy_match ? " match" : " MISMATCH") << '\n';

    const double combined_max = std::max(global_max, prefill_max);
    const bool combined_top1 = all_top1_match && prefill_top_match && prefill_greedy_match;
    if (absolute_tolerance >= 0.0) {
        const bool within = combined_max <= absolute_tolerance;
        std::cout << "combined atol:    " << absolute_tolerance << " -> " << (within ? "PASS" : "FAIL") << '\n';
        return within && combined_top1 ? 0 : 9;
    }
    std::cout << "combined atol:    not requested\n";
    return combined_top1 ? 0 : 9;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 0;
    }

    const std::string command = argv[1];
    if (command == "--help" || command == "-h") {
        print_usage();
        return 0;
    }
    if (command == "--version") {
        std::cout << air::version_string() << '\n';
        return 0;
    }
    if (command == "inspect") {
        if (argc < 3 || argc > 4) {
            std::cerr << "inspect requires a model path and optional --tensors\n";
            return 2;
        }
        const bool list_tensors = argc == 4 && std::string(argv[3]) == "--tensors";
        if (argc == 4 && !list_tensors) {
            std::cerr << "unknown inspect option: " << argv[3] << '\n';
            return 2;
        }
        return inspect(argv[2], list_tensors);
    }
    if (command == "fingerprint") {
        if (argc != 3) {
            std::cerr << "fingerprint requires a model path\n";
            return 2;
        }
        return fingerprint(argv[2]);
    }
    if (command == "tokenize") {
        if (argc != 4) {
            std::cerr << "tokenize requires a model path and one quoted text argument\n";
            return 2;
        }
        return tokenize(argv[2], argv[3]);
    }
    if (command == "reference") {
        if (argc != 4 && argc != 6) {
            std::cerr << "reference requires a model path, quoted text, and optional --tokens N\n";
            return 2;
        }
        std::uint32_t max_tokens = 32;
        if (argc == 6) {
            if (std::string(argv[4]) != "--tokens") {
                std::cerr << "unknown reference option: " << argv[4] << '\n';
                return 2;
            }
            try {
                const auto parsed = std::stoul(argv[5]);
                if (parsed == 0 || parsed > 1000000UL) throw std::out_of_range("token count");
                max_tokens = static_cast<std::uint32_t>(parsed);
            } catch (const std::exception&) {
                std::cerr << "--tokens requires an integer in [1, 1000000]\n";
                return 2;
            }
        }
        return reference_generate(argv[2], argv[3], max_tokens);
    }

    if (command == "cuda-info") {
        if (argc != 2) {
            std::cerr << "cuda-info takes no arguments\n";
            return 2;
        }
        return cuda_info();
    }
    if (command == "cuda") {
        if (argc < 4) {
            std::cerr << "cuda requires a model path and quoted text\n";
            return 2;
        }
        std::uint32_t max_tokens = 32;
        int device_ordinal = 0;
        for (int i = 4; i < argc; i += 2) {
            if (i + 1 >= argc) {
                std::cerr << "CUDA options require a value\n";
                return 2;
            }
            const std::string option = argv[i];
            try {
                if (option == "--tokens") {
                    const auto parsed = std::stoul(argv[i + 1]);
                    if (parsed == 0 || parsed > 1000000UL) throw std::out_of_range("token count");
                    max_tokens = static_cast<std::uint32_t>(parsed);
                } else if (option == "--device") {
                    const auto parsed = std::stol(argv[i + 1]);
                    if (parsed < 0 || parsed > 1024) throw std::out_of_range("device ordinal");
                    device_ordinal = static_cast<int>(parsed);
                } else {
                    std::cerr << "unknown cuda option: " << option << '\n';
                    return 2;
                }
            } catch (const std::exception&) {
                std::cerr << "invalid value for " << option << '\n';
                return 2;
            }
        }
        return cuda_generate(argv[2], argv[3], max_tokens, device_ordinal);
    }

    if (command == "cuda-compare") {
        if (argc < 4) {
            std::cerr << "cuda-compare requires a model path and quoted text\n";
            return 2;
        }
        int device_ordinal = 0;
        double absolute_tolerance = -1.0;
        for (int i = 4; i < argc; i += 2) {
            if (i + 1 >= argc) {
                std::cerr << "cuda-compare options require a value\n";
                return 2;
            }
            const std::string option = argv[i];
            try {
                if (option == "--device") {
                    const auto parsed = std::stol(argv[i + 1]);
                    if (parsed < 0 || parsed > 1024) throw std::out_of_range("device ordinal");
                    device_ordinal = static_cast<int>(parsed);
                } else if (option == "--atol") {
                    absolute_tolerance = std::stod(argv[i + 1]);
                    if (!(absolute_tolerance >= 0.0) || !std::isfinite(absolute_tolerance)) {
                        throw std::out_of_range("absolute tolerance");
                    }
                } else {
                    std::cerr << "unknown cuda-compare option: " << option << '\n';
                    return 2;
                }
            } catch (const std::exception&) {
                std::cerr << "invalid value for " << option << '\n';
                return 2;
            }
        }
        return cuda_compare(argv[2], argv[3], device_ordinal, absolute_tolerance);
    }

    std::cerr << "unknown command: " << command << '\n';
    print_usage();
    return 2;
}

#pragma once

#include "air/result.hpp"
#include "air/types.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <span>
#include <vector>

namespace air {

struct SamplingConfig {
    double temperature{0.0};
    std::uint32_t top_k{0};
    double top_p{1.0};
    std::uint64_t seed{0};
};

struct GenerationConfig {
    std::uint32_t max_new_tokens{32};
    SamplingConfig sampling{};
    bool stop_on_eos{true};
};

struct GenerationResult {
    std::vector<TokenId> tokens;
    bool hit_eos{false};
};

struct GenerationTiming {
    std::uint64_t prompt_tokens{0};
    std::uint64_t generated_tokens{0};
    double prefill_ms{0.0};
    double ttft_ms{0.0};
    double decode_ms{0.0};
    double total_ms{0.0};

    [[nodiscard]] double prefill_tokens_per_second() const noexcept;
    [[nodiscard]] double decode_tokens_per_second() const noexcept;
};

struct GenerationTrace {
    GenerationResult result;
    GenerationTiming timing;
};

using PrefillFunction = std::function<Result<std::vector<float>>(std::span<const TokenId>)>;
using DecodeFunction = std::function<Result<std::vector<float>>(TokenId)>;
using TokenObserver = std::function<Status(TokenId, std::uint32_t)>;

class Sampler final {
public:
    explicit Sampler(SamplingConfig config);
    [[nodiscard]] Result<TokenId> sample(std::span<const float> logits);

private:
    SamplingConfig config_;
    std::mt19937_64 random_;
};

[[nodiscard]] Result<GenerationTrace> run_autoregressive(
    std::span<const TokenId> prompt,
    const GenerationConfig& config,
    std::optional<TokenId> eos_token,
    PrefillFunction prefill,
    DecodeFunction decode,
    TokenObserver observer = {});

} // namespace air

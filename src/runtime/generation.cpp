#include "air/generation.hpp"

#include <chrono>

namespace air {
namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

} // namespace

double GenerationTiming::prefill_tokens_per_second() const noexcept {
    if (prefill_ms <= 0.0) return 0.0;
    return static_cast<double>(prompt_tokens) * 1000.0 / prefill_ms;
}

double GenerationTiming::decode_tokens_per_second() const noexcept {
    if (generated_tokens <= 1U || decode_ms <= 0.0) return 0.0;
    return static_cast<double>(generated_tokens - 1U) * 1000.0 / decode_ms;
}

Result<GenerationTrace> run_autoregressive(
    std::span<const TokenId> prompt,
    const GenerationConfig& config,
    std::optional<TokenId> eos_token,
    PrefillFunction prefill,
    DecodeFunction decode,
    TokenObserver observer) {
    if (prompt.empty()) return Status::invalid_argument("generation requires a non-empty prompt");

    GenerationTrace trace;
    trace.timing.prompt_tokens = prompt.size();
    trace.result.tokens.reserve(config.max_new_tokens);
    if (config.max_new_tokens == 0U) return trace;

    const auto total_start = Clock::now();
    const auto prefill_start = total_start;
    auto logits = prefill(prompt);
    if (!logits) return logits.status();
    const auto prefill_end = Clock::now();
    trace.timing.prefill_ms = milliseconds(prefill_end - prefill_start);

    Sampler sampler(config.sampling);
    auto first_token_time = prefill_end;
    auto decode_start = prefill_end;

    for (std::uint32_t index = 0; index < config.max_new_tokens; ++index) {
        auto sampled = sampler.sample(logits.value());
        if (!sampled) return sampled.status();
        const TokenId token = sampled.value();
        if (index == 0U) first_token_time = Clock::now();

        trace.result.tokens.push_back(token);
        trace.timing.generated_tokens = trace.result.tokens.size();
        if (observer) {
            const auto observed = observer(token, index);
            if (!observed) return observed;
        }

        if (config.stop_on_eos && eos_token && token == *eos_token) {
            trace.result.hit_eos = true;
            break;
        }
        if (index + 1U == config.max_new_tokens) break;

        auto next = decode(token);
        if (!next) return next.status();
        logits = std::move(next);
    }

    const auto finish = Clock::now();
    trace.timing.ttft_ms = milliseconds(first_token_time - total_start);
    trace.timing.decode_ms = milliseconds(finish - decode_start);
    trace.timing.total_ms = milliseconds(finish - total_start);
    return trace;
}

} // namespace air

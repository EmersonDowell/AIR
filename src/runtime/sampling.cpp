#include "air/generation.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace air {

Sampler::Sampler(SamplingConfig config) : config_(config), random_(config.seed) {}

Result<TokenId> Sampler::sample(std::span<const float> logits) {
    if (logits.empty()) return Status::invalid_argument("cannot sample empty logits");
    if (config_.temperature <= 0.0) {
        const auto it = std::max_element(logits.begin(), logits.end());
        return static_cast<TokenId>(std::distance(logits.begin(), it));
    }
    if (!(config_.top_p > 0.0 && config_.top_p <= 1.0)) {
        return Status::invalid_argument("top_p must be in (0, 1]");
    }

    std::vector<std::pair<std::size_t, double>> candidates;
    candidates.reserve(logits.size());
    const double inverse_temperature = 1.0 / config_.temperature;
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const double weight = std::exp((static_cast<double>(logits[i]) - max_logit) * inverse_temperature);
        candidates.emplace_back(i, weight);
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
        return left.second > right.second;
    });
    if (config_.top_k != 0U && config_.top_k < candidates.size()) candidates.resize(config_.top_k);

    double total = 0.0;
    for (const auto& candidate : candidates) total += candidate.second;
    if (!(total > 0.0) || !std::isfinite(total)) {
        return Status::internal_error("sampling distribution is non-finite");
    }

    if (config_.top_p < 1.0) {
        double cumulative = 0.0;
        std::size_t keep = 0;
        for (; keep < candidates.size(); ++keep) {
            cumulative += candidates[keep].second / total;
            if (cumulative >= config_.top_p) {
                ++keep;
                break;
            }
        }
        candidates.resize(std::max<std::size_t>(1U, keep));
        total = 0.0;
        for (const auto& candidate : candidates) total += candidate.second;
    }

    std::uniform_real_distribution<double> distribution(0.0, total);
    const double selected = distribution(random_);
    double cumulative = 0.0;
    for (const auto& candidate : candidates) {
        cumulative += candidate.second;
        if (selected <= cumulative) return static_cast<TokenId>(candidate.first);
    }
    return static_cast<TokenId>(candidates.back().first);
}

} // namespace air

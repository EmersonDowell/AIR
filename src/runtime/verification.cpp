#include "air/verification.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace air {

const char* to_string(VerificationStage stage) noexcept {
    switch (stage) {
    case VerificationStage::embedding: return "embedding";
    case VerificationStage::attention_norm: return "attention_norm";
    case VerificationStage::q_projection: return "q_projection";
    case VerificationStage::k_projection: return "k_projection";
    case VerificationStage::v_projection: return "v_projection";
    case VerificationStage::q_rope: return "q_rope";
    case VerificationStage::k_rope: return "k_rope";
    case VerificationStage::attention: return "attention";
    case VerificationStage::attention_projection: return "attention_projection";
    case VerificationStage::attention_residual: return "attention_residual";
    case VerificationStage::ffn_norm: return "ffn_norm";
    case VerificationStage::ffn_gate: return "ffn_gate";
    case VerificationStage::ffn_up: return "ffn_up";
    case VerificationStage::ffn_activation: return "ffn_activation";
    case VerificationStage::ffn_down: return "ffn_down";
    case VerificationStage::ffn_residual: return "ffn_residual";
    case VerificationStage::final_norm: return "final_norm";
    case VerificationStage::logits: return "logits";
    }
    return "unknown";
}

void VerificationTrace::record(std::uint64_t position, std::int32_t layer,
                               VerificationStage stage, std::span<const float> values) {
    VerificationSnapshot snapshot;
    snapshot.position = position;
    snapshot.layer = layer;
    snapshot.stage = stage;
    snapshot.values.assign(values.begin(), values.end());
    snapshots_.push_back(std::move(snapshot));
}

Result<VerificationComparison> compare_vectors(std::span<const float> reference,
                                               std::span<const float> candidate) {
    if (reference.size() != candidate.size()) {
        return Status::invalid_argument("verification vectors have different lengths");
    }
    VerificationComparison result;
    result.elements = reference.size();
    double sum_abs = 0.0;
    double sum_squared = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double left = static_cast<double>(reference[i]);
        const double right = static_cast<double>(candidate[i]);
        if (!std::isfinite(left) || !std::isfinite(right)) result.finite = false;
        const double error = std::abs(left - right);
        sum_abs += error;
        sum_squared += error * error;
        if (error > result.max_abs_error || i == 0U) {
            result.max_abs_error = error;
            result.max_error_index = i;
            result.reference_at_max = reference[i];
            result.candidate_at_max = candidate[i];
        }
    }
    if (!reference.empty()) {
        const double count = static_cast<double>(reference.size());
        result.mean_abs_error = sum_abs / count;
        result.rms_error = std::sqrt(sum_squared / count);
    }
    return result;
}

std::vector<RankedToken> top_k_logits(std::span<const float> logits, std::size_t k) {
    k = std::min(k, logits.size());
    std::vector<std::size_t> indices(logits.size());
    std::iota(indices.begin(), indices.end(), std::size_t{0});
    auto better = [&logits](std::size_t left, std::size_t right) {
        if (logits[left] != logits[right]) return logits[left] > logits[right];
        return left < right;
    };
    if (k < indices.size()) {
        std::partial_sort(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(k),
                          indices.end(), better);
        indices.resize(k);
    } else {
        std::sort(indices.begin(), indices.end(), better);
    }
    std::vector<RankedToken> ranked;
    ranked.reserve(k);
    for (const auto index : indices) {
        ranked.push_back(RankedToken{static_cast<TokenId>(index), logits[index]});
    }
    return ranked;
}

} // namespace air

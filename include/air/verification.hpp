#pragma once

#include "air/result.hpp"
#include "air/types.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace air {

inline constexpr std::string_view verification_report_schema = "air.verification.v1";

enum class VerificationStage : std::uint8_t {
    embedding = 0,
    attention_norm,
    q_projection,
    k_projection,
    v_projection,
    q_rope,
    k_rope,
    attention,
    attention_projection,
    attention_residual,
    ffn_norm,
    ffn_gate,
    ffn_up,
    ffn_activation,
    ffn_down,
    ffn_residual,
    final_norm,
    logits,
};

[[nodiscard]] const char* to_string(VerificationStage stage) noexcept;

struct VerificationSnapshot {
    std::uint64_t position{0};
    std::int32_t layer{-1};
    VerificationStage stage{VerificationStage::embedding};
    std::vector<float> values;
};

class VerificationTrace final {
public:
    void record(std::uint64_t position, std::int32_t layer,
                VerificationStage stage, std::span<const float> values);
    void clear() noexcept { snapshots_.clear(); }

    [[nodiscard]] const std::vector<VerificationSnapshot>& snapshots() const noexcept {
        return snapshots_;
    }

private:
    std::vector<VerificationSnapshot> snapshots_;
};

struct VerificationComparison {
    std::size_t elements{0};
    double max_abs_error{0.0};
    double mean_abs_error{0.0};
    double rms_error{0.0};
    std::size_t max_error_index{0};
    float reference_at_max{0.0F};
    float candidate_at_max{0.0F};
    bool finite{true};
};

[[nodiscard]] Result<VerificationComparison> compare_vectors(
    std::span<const float> reference, std::span<const float> candidate);

struct RankedToken {
    TokenId token{0};
    float logit{0.0F};
};

[[nodiscard]] std::vector<RankedToken> top_k_logits(std::span<const float> logits,
                                                    std::size_t k);

} // namespace air

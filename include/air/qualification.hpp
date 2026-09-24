#pragma once

#include "air/manifest.hpp"
#include "air/serving.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace air {

enum class QualificationBackendMode { automatic = 0, reference_only, cuda_only, all };

struct QualificationConfig {
    std::string prompt;
    WorkloadClass workload{WorkloadClass::small};
    std::uint32_t max_new_tokens{32};
    std::uint32_t samples{3};
    int cuda_device{0};
    QualificationBackendMode backend_mode{QualificationBackendMode::automatic};
    // Zero derives a deterministic seed from model + hardware + workload.
    std::uint64_t seed{0};
    std::filesystem::path output;
};

struct QualificationResult {
    ExecutionManifest manifest;
    std::vector<QualifiedStrategy> candidates;
    QualifiedStrategy selected;
    std::string selection_reason;
    double selection_margin{0.0};
    double selection_confidence_half_width{0.0};
    std::uint64_t evidence_seed{0};
};

[[nodiscard]] Result<QualificationResult> qualify_model(
    const std::filesystem::path& model_path,
    const QualificationConfig& config);

} // namespace air

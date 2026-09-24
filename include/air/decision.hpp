#pragma once

#include "air/result.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace air {

enum class WorkloadKind {
    generation = 0,
    bounded_decision,
};

[[nodiscard]] const char* to_string(WorkloadKind kind) noexcept;

enum class DeterminismRequirement {
    required = 0,
    stochastic_allowed,
};

[[nodiscard]] const char* to_string(DeterminismRequirement value) noexcept;
[[nodiscard]] Result<DeterminismRequirement> determinism_requirement_from_string(
    std::string_view value);

enum class DecisionScoringPolicy {
    sequence_logprob_sum = 0,
    sequence_logprob_mean,
    // AUTO is not a heuristic escape hatch. It means a later production
    // planner may choose only from strictly qualified scoring policies.
    qualified_auto,
};

[[nodiscard]] const char* to_string(DecisionScoringPolicy policy) noexcept;
[[nodiscard]] Result<DecisionScoringPolicy> decision_scoring_policy_from_string(
    std::string_view value);

enum class DecisionOutputCardinality {
    exactly_one = 0,
    zero_or_one,
    one_or_more,
    zero_or_more,
};

[[nodiscard]] const char* to_string(DecisionOutputCardinality value) noexcept;
[[nodiscard]] Result<DecisionOutputCardinality> decision_output_cardinality_from_string(
    std::string_view value);

enum class DecisionScoreSemantics {
    // V1 deliberately has no "probability" score semantics. Prompt 2 only
    // established candidate-set-normalized scores, not calibration.
    candidate_set_normalized = 0,
};

[[nodiscard]] const char* to_string(DecisionScoreSemantics value) noexcept;

struct DecisionCandidate {
    // Stable semantic identity. Array position, display text, tokenizer IDs,
    // and model-facing representation are not semantic identity.
    std::string id;
    // Human-facing surface. Duplicate display strings are legal when IDs differ.
    std::string display_text;
    // Optional model-facing representation. If absent, an execution adapter may
    // use display_text. Changing this representation does not change semantic ID.
    std::optional<std::string> model_text;
};

struct DecisionRequest {
    // Architecture-neutral text input. Chat/API adapters may render richer
    // external forms into this field later; no Qwen template is part of V1.
    std::string input_text;
    std::vector<DecisionCandidate> candidates;
    DecisionScoringPolicy scoring_policy{DecisionScoringPolicy::qualified_auto};
    DecisionOutputCardinality output_cardinality{DecisionOutputCardinality::exactly_one};
    DeterminismRequirement determinism{DeterminismRequirement::required};
};

struct DecisionCandidateScore {
    std::string candidate_id;
    double normalized_score{0.0};
};

struct DecisionResult {
    std::vector<std::string> selected_candidate_ids;
    std::vector<DecisionCandidateScore> scores;
    // A concrete policy must be reported after execution. qualified_auto is a
    // request-side constraint and is never a resolved result policy.
    DecisionScoringPolicy applied_scoring_policy{DecisionScoringPolicy::sequence_logprob_sum};
    DecisionScoreSemantics score_semantics{DecisionScoreSemantics::candidate_set_normalized};
};

struct DecisionWorkloadDescriptor {
    std::uint32_t candidate_count{0};
    DecisionScoringPolicy scoring_policy{DecisionScoringPolicy::qualified_auto};
    DecisionOutputCardinality output_cardinality{DecisionOutputCardinality::exactly_one};
    bool requires_qualified_scoring_policy{true};
};

struct WorkloadDescriptor {
    WorkloadKind kind{WorkloadKind::bounded_decision};
    DeterminismRequirement determinism{DeterminismRequirement::required};
    // Present only for bounded_decision in V1. Generation remains on the
    // established InferenceRequest contract and is not migrated speculatively.
    std::optional<DecisionWorkloadDescriptor> decision;
};

[[nodiscard]] Status validate_decision_request(const DecisionRequest& request);
[[nodiscard]] Result<WorkloadDescriptor> describe_workload(const DecisionRequest& request);
[[nodiscard]] Status validate_decision_result(const DecisionRequest& request,
                                              const DecisionResult& result);

} // namespace air

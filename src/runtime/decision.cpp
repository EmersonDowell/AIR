#include "air/decision.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_set>

namespace air {
namespace {

[[nodiscard]] bool valid(DecisionScoringPolicy value) noexcept {
    switch (value) {
    case DecisionScoringPolicy::sequence_logprob_sum:
    case DecisionScoringPolicy::sequence_logprob_mean:
    case DecisionScoringPolicy::qualified_auto:
        return true;
    }
    return false;
}

[[nodiscard]] bool valid(DecisionOutputCardinality value) noexcept {
    switch (value) {
    case DecisionOutputCardinality::exactly_one:
    case DecisionOutputCardinality::zero_or_one:
    case DecisionOutputCardinality::one_or_more:
    case DecisionOutputCardinality::zero_or_more:
        return true;
    }
    return false;
}

[[nodiscard]] bool valid(DeterminismRequirement value) noexcept {
    switch (value) {
    case DeterminismRequirement::required:
    case DeterminismRequirement::stochastic_allowed:
        return true;
    }
    return false;
}

[[nodiscard]] bool cardinality_accepts(DecisionOutputCardinality rule,
                                       std::size_t selected) noexcept {
    switch (rule) {
    case DecisionOutputCardinality::exactly_one: return selected == 1U;
    case DecisionOutputCardinality::zero_or_one: return selected <= 1U;
    case DecisionOutputCardinality::one_or_more: return selected >= 1U;
    case DecisionOutputCardinality::zero_or_more: return true;
    }
    return false;
}

} // namespace

const char* to_string(WorkloadKind kind) noexcept {
    switch (kind) {
    case WorkloadKind::generation: return "generation";
    case WorkloadKind::bounded_decision: return "bounded-decision";
    }
    return "unknown";
}

const char* to_string(DeterminismRequirement value) noexcept {
    switch (value) {
    case DeterminismRequirement::required: return "required";
    case DeterminismRequirement::stochastic_allowed: return "stochastic-allowed";
    }
    return "unknown";
}

Result<DeterminismRequirement> determinism_requirement_from_string(std::string_view value) {
    if (value == "required" || value == "deterministic") {
        return DeterminismRequirement::required;
    }
    if (value == "stochastic-allowed" || value == "stochastic_allowed") {
        return DeterminismRequirement::stochastic_allowed;
    }
    return Status::invalid_argument("unknown determinism requirement: " + std::string(value));
}

const char* to_string(DecisionScoringPolicy policy) noexcept {
    switch (policy) {
    case DecisionScoringPolicy::sequence_logprob_sum: return "sequence-logprob-sum";
    case DecisionScoringPolicy::sequence_logprob_mean: return "sequence-logprob-mean";
    case DecisionScoringPolicy::qualified_auto: return "qualified-auto";
    }
    return "unknown";
}

Result<DecisionScoringPolicy> decision_scoring_policy_from_string(std::string_view value) {
    if (value == "sequence-logprob-sum" || value == "sequence_sum") {
        return DecisionScoringPolicy::sequence_logprob_sum;
    }
    if (value == "sequence-logprob-mean" || value == "sequence_mean") {
        return DecisionScoringPolicy::sequence_logprob_mean;
    }
    if (value == "qualified-auto" || value == "qualified_auto") {
        return DecisionScoringPolicy::qualified_auto;
    }
    return Status::invalid_argument("unknown decision scoring policy: " + std::string(value));
}

const char* to_string(DecisionOutputCardinality value) noexcept {
    switch (value) {
    case DecisionOutputCardinality::exactly_one: return "exactly-one";
    case DecisionOutputCardinality::zero_or_one: return "zero-or-one";
    case DecisionOutputCardinality::one_or_more: return "one-or-more";
    case DecisionOutputCardinality::zero_or_more: return "zero-or-more";
    }
    return "unknown";
}

Result<DecisionOutputCardinality> decision_output_cardinality_from_string(
    std::string_view value) {
    if (value == "exactly-one" || value == "exactly_one") {
        return DecisionOutputCardinality::exactly_one;
    }
    if (value == "zero-or-one" || value == "zero_or_one") {
        return DecisionOutputCardinality::zero_or_one;
    }
    if (value == "one-or-more" || value == "one_or_more") {
        return DecisionOutputCardinality::one_or_more;
    }
    if (value == "zero-or-more" || value == "zero_or_more") {
        return DecisionOutputCardinality::zero_or_more;
    }
    return Status::invalid_argument("unknown decision output cardinality: " +
                                    std::string(value));
}

const char* to_string(DecisionScoreSemantics value) noexcept {
    switch (value) {
    case DecisionScoreSemantics::candidate_set_normalized:
        return "candidate-set-normalized";
    }
    return "unknown";
}

Status validate_decision_request(const DecisionRequest& request) {
    if (request.input_text.empty()) {
        return Status::invalid_argument("decision input text must not be empty");
    }
    if (request.candidates.empty()) {
        return Status::invalid_argument("decision request requires at least one candidate");
    }
    if (request.candidates.size() >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        return Status::invalid_argument("decision candidate count exceeds uint32 range");
    }
    if (!valid(request.scoring_policy)) {
        return Status::invalid_argument("decision request has an invalid scoring policy");
    }
    if (!valid(request.output_cardinality)) {
        return Status::invalid_argument("decision request has an invalid output cardinality");
    }
    if (!valid(request.determinism)) {
        return Status::invalid_argument("decision request has an invalid determinism requirement");
    }

    std::unordered_set<std::string> ids;
    ids.reserve(request.candidates.size());
    for (const auto& candidate : request.candidates) {
        if (candidate.id.empty()) {
            return Status::invalid_argument("decision candidate semantic id must not be empty");
        }
        if (!ids.insert(candidate.id).second) {
            return Status::invalid_argument("decision candidate semantic ids must be unique");
        }
        if (candidate.display_text.empty()) {
            return Status::invalid_argument("decision candidate display text must not be empty");
        }
        if (candidate.model_text && candidate.model_text->empty()) {
            return Status::invalid_argument(
                "decision candidate model-facing representation must not be empty when present");
        }
    }
    return Status::ok();
}

Result<WorkloadDescriptor> describe_workload(const DecisionRequest& request) {
    const auto status = validate_decision_request(request);
    if (!status) return status;

    WorkloadDescriptor descriptor;
    descriptor.kind = WorkloadKind::bounded_decision;
    descriptor.determinism = request.determinism;
    descriptor.decision = DecisionWorkloadDescriptor{
        static_cast<std::uint32_t>(request.candidates.size()),
        request.scoring_policy,
        request.output_cardinality,
        request.scoring_policy == DecisionScoringPolicy::qualified_auto,
    };
    return descriptor;
}

Status validate_decision_result(const DecisionRequest& request,
                                const DecisionResult& result) {
    const auto request_status = validate_decision_request(request);
    if (!request_status) return request_status;

    if (!valid(result.applied_scoring_policy) ||
        result.applied_scoring_policy == DecisionScoringPolicy::qualified_auto) {
        return Status::invalid_argument(
            "decision result must report a concrete applied scoring policy");
    }
    if (request.scoring_policy != DecisionScoringPolicy::qualified_auto &&
        request.scoring_policy != result.applied_scoring_policy) {
        return Status::invalid_argument(
            "decision result scoring policy does not satisfy the explicit request");
    }
    if (result.score_semantics != DecisionScoreSemantics::candidate_set_normalized) {
        return Status::invalid_argument("unsupported decision result score semantics");
    }

    std::unordered_set<std::string> candidate_ids;
    candidate_ids.reserve(request.candidates.size());
    for (const auto& candidate : request.candidates) candidate_ids.insert(candidate.id);

    if (result.scores.size() != request.candidates.size()) {
        return Status::invalid_argument(
            "decision result must provide one normalized score per candidate");
    }
    std::unordered_set<std::string> scored_ids;
    scored_ids.reserve(result.scores.size());
    double sum = 0.0;
    for (const auto& score : result.scores) {
        if (!candidate_ids.contains(score.candidate_id)) {
            return Status::invalid_argument("decision result score references unknown candidate id");
        }
        if (!scored_ids.insert(score.candidate_id).second) {
            return Status::invalid_argument("decision result contains duplicate candidate scores");
        }
        if (!std::isfinite(score.normalized_score) ||
            score.normalized_score < 0.0 || score.normalized_score > 1.0) {
            return Status::invalid_argument(
                "decision normalized scores must be finite values in [0,1]");
        }
        sum += score.normalized_score;
    }
    if (std::abs(sum - 1.0) > 1e-6) {
        return Status::invalid_argument(
            "decision candidate-set-normalized scores must sum to one");
    }

    std::unordered_set<std::string> selected_ids;
    selected_ids.reserve(result.selected_candidate_ids.size());
    for (const auto& selected : result.selected_candidate_ids) {
        if (!candidate_ids.contains(selected)) {
            return Status::invalid_argument("decision result selected unknown candidate id");
        }
        if (!selected_ids.insert(selected).second) {
            return Status::invalid_argument("decision result selected a candidate more than once");
        }
    }
    if (!cardinality_accepts(request.output_cardinality,
                             result.selected_candidate_ids.size())) {
        return Status::invalid_argument(
            "decision result does not satisfy requested output cardinality");
    }
    return Status::ok();
}

} // namespace air

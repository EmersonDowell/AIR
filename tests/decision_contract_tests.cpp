#include "air/decision.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

air::DecisionRequest base_request() {
    air::DecisionRequest request;
    request.input_text = "Route this request.";
    request.candidates = {
        {"billing", "Billing", std::string("billing")},
        {"technical", "Technical support", std::string("technical support")},
    };
    request.scoring_policy = air::DecisionScoringPolicy::qualified_auto;
    request.output_cardinality = air::DecisionOutputCardinality::exactly_one;
    request.determinism = air::DeterminismRequirement::required;
    return request;
}

air::DecisionResult valid_result() {
    air::DecisionResult result;
    result.selected_candidate_ids = {"billing"};
    result.scores = {{"billing", 0.75}, {"technical", 0.25}};
    result.applied_scoring_policy = air::DecisionScoringPolicy::sequence_logprob_sum;
    return result;
}

template <typename T>
concept HasGenerationMember = requires(T value) { value.generation; };

template <typename T>
concept HasTokenIdMember = requires(T value) { value.token_id; };

template <typename T>
concept HasModelRevisionMember = requires(T value) { value.model_revision; };

template <typename T>
concept HasKvMember = requires(T value) { value.kv; };

void test_semantic_identity_and_representation() {
    auto request = base_request();
    check(air::validate_decision_request(request).is_ok(), "baseline decision request validates");

    auto reordered = request;
    std::swap(reordered.candidates[0], reordered.candidates[1]);
    check(air::validate_decision_request(reordered).is_ok(),
          "presentation order does not affect request validity");

    const auto a = air::describe_workload(request);
    const auto b = air::describe_workload(reordered);
    check(a && b && a.value().decision && b.value().decision &&
          a.value().decision->candidate_count == b.value().decision->candidate_count &&
          a.value().decision->scoring_policy == b.value().decision->scoring_policy &&
          a.value().decision->output_cardinality == b.value().decision->output_cardinality,
          "planner-visible workload shape does not encode candidate presentation order");

    auto representation_changed = request;
    representation_changed.candidates[0].display_text = "Account and billing";
    representation_changed.candidates[0].model_text = "account payment invoice";
    check(air::validate_decision_request(representation_changed).is_ok() &&
          representation_changed.candidates[0].id == request.candidates[0].id,
          "representation can change without changing semantic candidate id");

    auto duplicate_display = request;
    duplicate_display.candidates[1].display_text = duplicate_display.candidates[0].display_text;
    duplicate_display.candidates[1].model_text = duplicate_display.candidates[0].model_text;
    check(air::validate_decision_request(duplicate_display).is_ok(),
          "equal candidate surfaces are legal when semantic ids differ");

    auto duplicate_id = request;
    duplicate_id.candidates[1].id = duplicate_id.candidates[0].id;
    check(!air::validate_decision_request(duplicate_id).is_ok(),
          "duplicate semantic candidate ids fail closed");

    static_assert(!HasGenerationMember<air::DecisionRequest>);
    static_assert(!HasTokenIdMember<air::DecisionCandidate>);
    static_assert(!HasModelRevisionMember<air::DecisionRequest>);
    static_assert(!HasKvMember<air::DecisionRequest>);
}

void test_request_edges() {
    auto empty = base_request();
    empty.candidates.clear();
    check(!air::validate_decision_request(empty).is_ok(), "empty candidate set is invalid");

    auto one = base_request();
    one.candidates.resize(1);
    check(air::validate_decision_request(one).is_ok(),
          "one-candidate semantic contract is valid; usefulness is not contract validation");

    auto empty_input = base_request();
    empty_input.input_text.clear();
    check(!air::validate_decision_request(empty_input).is_ok(), "empty input is invalid");

    auto empty_model_text = base_request();
    empty_model_text.candidates[0].model_text = std::string{};
    check(!air::validate_decision_request(empty_model_text).is_ok(),
          "present but empty model-facing representation is invalid");

    auto multitoken = base_request();
    multitoken.candidates[0].display_text = "This is a long multi-token candidate surface.";
    multitoken.candidates[0].model_text = "long candidate continuation with several tokens";
    check(air::validate_decision_request(multitoken).is_ok(),
          "multi-token candidate surfaces are contract-valid");

    auto large = base_request();
    large.candidates.clear();
    large.candidates.reserve(4096);
    for (std::uint32_t i = 0; i < 4096U; ++i) {
        large.candidates.push_back({
            "candidate-" + std::to_string(i),
            "Candidate " + std::to_string(i),
            std::nullopt,
        });
    }
    check(air::validate_decision_request(large).is_ok(),
          "semantic contract has no arbitrary small candidate-count cap");

    check(!air::decision_scoring_policy_from_string("winner-takes-all"),
          "unknown scoring policy fails explicitly");
    check(!air::decision_output_cardinality_from_string("sometimes-many"),
          "unknown output cardinality fails explicitly");
    check(!air::determinism_requirement_from_string("maybe"),
          "unknown determinism requirement fails explicitly");
}

void test_scoring_policy_and_workload_descriptor() {
    auto request = base_request();
    auto descriptor = air::describe_workload(request);
    check(descriptor && descriptor.value().kind == air::WorkloadKind::bounded_decision &&
          descriptor.value().decision &&
          descriptor.value().decision->requires_qualified_scoring_policy,
          "qualified-auto is represented as a strict planner qualification requirement");

    request.scoring_policy = air::DecisionScoringPolicy::sequence_logprob_mean;
    descriptor = air::describe_workload(request);
    check(descriptor && descriptor.value().decision &&
          !descriptor.value().decision->requires_qualified_scoring_policy &&
          descriptor.value().decision->scoring_policy ==
              air::DecisionScoringPolicy::sequence_logprob_mean,
          "explicit scoring policy is preserved without pretending it is AUTO");

    request.determinism = air::DeterminismRequirement::stochastic_allowed;
    descriptor = air::describe_workload(request);
    check(descriptor && descriptor.value().determinism ==
          air::DeterminismRequirement::stochastic_allowed,
          "deterministic versus stochastic semantics are explicit workload data");
}

void test_result_semantics() {
    auto request = base_request();
    auto result = valid_result();
    check(air::validate_decision_result(request, result).is_ok(),
          "normalized candidate-set result validates");

    auto unnormalized = result;
    unnormalized.scores[0].normalized_score = 0.8;
    check(!air::validate_decision_result(request, unnormalized).is_ok(),
          "candidate-set-normalized scores must sum to one");

    auto duplicate_score = result;
    duplicate_score.scores[1].candidate_id = "billing";
    check(!air::validate_decision_result(request, duplicate_score).is_ok(),
          "each semantic candidate receives exactly one score");

    auto unknown_selected = result;
    unknown_selected.selected_candidate_ids = {"unknown"};
    check(!air::validate_decision_result(request, unknown_selected).is_ok(),
          "selected ids must reference semantic candidates");

    auto unresolved = result;
    unresolved.applied_scoring_policy = air::DecisionScoringPolicy::qualified_auto;
    check(!air::validate_decision_result(request, unresolved).is_ok(),
          "qualified-auto must resolve to a concrete result policy");

    auto explicit_request = request;
    explicit_request.scoring_policy = air::DecisionScoringPolicy::sequence_logprob_mean;
    check(!air::validate_decision_result(explicit_request, result).is_ok(),
          "result cannot silently change an explicit scoring policy");

    auto zero_or_one = request;
    zero_or_one.output_cardinality = air::DecisionOutputCardinality::zero_or_one;
    auto abstained = result;
    abstained.selected_candidate_ids.clear();
    check(air::validate_decision_result(zero_or_one, abstained).is_ok(),
          "zero-or-one expresses abstention without inventing a fake candidate");

    auto one_or_more = request;
    one_or_more.output_cardinality = air::DecisionOutputCardinality::one_or_more;
    auto multiple = result;
    multiple.selected_candidate_ids = {"billing", "technical"};
    check(air::validate_decision_result(one_or_more, multiple).is_ok(),
          "one-or-more expresses multi-valid output semantics");

    one_or_more.output_cardinality = air::DecisionOutputCardinality::exactly_one;
    check(!air::validate_decision_result(one_or_more, multiple).is_ok(),
          "exactly-one rejects multiple selections");

    auto zero_or_more = request;
    zero_or_more.output_cardinality = air::DecisionOutputCardinality::zero_or_more;
    auto none = result;
    none.selected_candidate_ids.clear();
    check(air::validate_decision_result(zero_or_more, none).is_ok(),
          "zero-or-more expresses both abstention and multiple-valid semantics");

    check(std::string(air::to_string(air::DecisionScoreSemantics::candidate_set_normalized)) ==
          "candidate-set-normalized",
          "V1 score semantics are explicitly normalized, not calibrated probability");
}

} // namespace

int main() {
    test_semantic_identity_and_representation();
    test_request_edges();
    test_scoring_policy_and_workload_descriptor();
    test_result_semantics();

    if (failures != 0) {
        std::cerr << failures << " decision contract test(s) failed\n";
        return 1;
    }
    std::cout << "decision contract tests passed\n";
    return 0;
}

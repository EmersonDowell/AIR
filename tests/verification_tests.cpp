#include "air/verification.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

void test_vector_comparison() {
    const std::vector<float> left = {1.0F, -2.0F, 4.0F};
    const std::vector<float> right = {1.5F, -2.0F, 3.0F};
    auto compared = air::compare_vectors(left, right);
    check(compared && compared.value().elements == 3U, "comparison reports vector size");
    if (compared) {
        check(std::fabs(compared.value().max_abs_error - 1.0) < 1.0e-12,
              "comparison reports maximum absolute error");
        check(compared.value().max_error_index == 2U,
              "comparison reports deterministic maximum-error index");
        check(compared.value().finite, "finite vectors report finite comparison");
    }

    const std::vector<float> nonfinite = {1.0F, std::numeric_limits<float>::infinity()};
    auto nonfinite_result = air::compare_vectors(nonfinite, nonfinite);
    check(nonfinite_result && !nonfinite_result.value().finite,
          "comparison reports non-finite input even when values are identical");

    auto mismatch = air::compare_vectors(left, std::span<const float>(right).first(2));
    check(!mismatch && mismatch.status().code() == air::ErrorCode::invalid_argument,
          "comparison rejects mismatched vector lengths");
}

void test_top_k() {
    const std::vector<float> logits = {3.0F, 9.0F, 9.0F, -1.0F, 5.0F};
    const auto ranked = air::top_k_logits(logits, 4U);
    check(ranked.size() == 4U, "top-k returns requested bounded count");
    if (ranked.size() == 4U) {
        check(ranked[0].token == 1 && ranked[1].token == 2,
              "top-k breaks equal logits by lower token id");
        check(ranked[2].token == 4 && ranked[3].token == 0,
              "top-k preserves descending logit order");
    }
    check(air::top_k_logits(logits, 0U).empty(), "top-k zero request is empty");
}

void test_trace_storage() {
    air::VerificationTrace trace;
    const std::vector<float> values = {1.0F, 2.0F, 3.0F};
    trace.record(7U, 2, air::VerificationStage::q_projection, values);
    check(trace.snapshots().size() == 1U, "trace stores one snapshot");
    if (!trace.snapshots().empty()) {
        const auto& snapshot = trace.snapshots().front();
        check(snapshot.position == 7U && snapshot.layer == 2 &&
              snapshot.stage == air::VerificationStage::q_projection,
              "trace preserves snapshot identity");
        check(snapshot.values == values, "trace owns a stable copy of observed values");
    }
    trace.clear();
    check(trace.snapshots().empty(), "trace clear removes captured evidence");
}

} // namespace

int main() {
    test_vector_comparison();
    test_top_k();
    test_trace_storage();
    if (failures != 0) {
        std::cerr << failures << " verification test(s) failed\n";
        return 1;
    }
    std::cout << "all AIR verification tests passed\n";
    return 0;
}

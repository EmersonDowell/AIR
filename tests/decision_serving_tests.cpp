#include "air/serving.hpp"
#include "air/storage.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

class ModelBuilder {
public:
    void add_f32(std::string name, std::vector<std::uint64_t> shape, std::span<const float> values) {
        std::vector<std::byte> raw;
        raw.reserve(values.size() * 4U);
        for (const float value : values) {
            const auto bits = std::bit_cast<std::uint32_t>(value);
            for (unsigned shift = 0; shift < 32U; shift += 8U) {
                raw.push_back(static_cast<std::byte>((bits >> shift) & 0xffU));
            }
        }
        air::TensorDescriptor descriptor;
        descriptor.name = std::move(name);
        descriptor.type = air::DataType::f32;
        descriptor.format_type = 0;
        descriptor.shape.dimensions = std::move(shape);
        descriptor.byte_offset = bytes_.size();
        descriptor.byte_size = raw.size();
        descriptor.byte_size_exact = true;
        tensors_.push_back(std::move(descriptor));
        bytes_.insert(bytes_.end(), raw.begin(), raw.end());
    }

    std::shared_ptr<air::ModelDefinition> finish() {
        const auto path = std::filesystem::temp_directory_path() / "air-serving-fixture.bin";
        {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(bytes_.data()), static_cast<std::streamsize>(bytes_.size()));
        }
        auto storage = air::ModelStorage::map_read_only(path);
        if (!storage) throw std::runtime_error(storage.status().message());
        std::filesystem::remove(path);

        constexpr std::uint32_t embedding = 4;
        air::ModelConfig config;
        config.architecture = "qwen2";
        config.layer_count = 1;
        config.embedding_size = embedding;
        config.feed_forward_size = 6;
        config.attention_head_count = 2;
        config.kv_head_count = 1;
        config.rope_dimension_count = 2;
        config.context_length = 4096;
        config.vocabulary_size = 4;
        config.rope_frequency_base = 10000.0;
        config.rms_norm_epsilon = 1.0e-5;

        air::TokenizerDefinition tokenizer;
        tokenizer.model = "gpt2";
        tokenizer.pre_tokenizer = "gpt2";
        tokenizer.vocabulary = {"a", "b", "c", "d"};
        tokenizer.token_types = {1, 1, 1, 1};
        tokenizer.special_ids.eos = 3;

        auto model = std::make_shared<air::ModelDefinition>(
            air::ModelFingerprint{"test", "qwen2", "serving-fixture"},
            std::move(config), std::move(tokenizer), std::move(tensors_), std::move(storage).value());
        const auto valid = model->validate();
        if (!valid) throw std::runtime_error(valid.message());
        return model;
    }

private:
    std::vector<std::byte> bytes_;
    std::vector<air::TensorDescriptor> tensors_;
};

std::shared_ptr<air::ModelDefinition> tiny_model() {
    constexpr std::uint32_t embedding = 4;
    constexpr std::uint32_t vocab = 4;
    constexpr std::uint32_t ffn = 6;
    const std::vector<float> identity = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    const std::vector<float> ones(embedding, 1.0F);
    const std::vector<float> zero44(embedding * embedding, 0.0F);
    const std::vector<float> zero42(embedding * 2U, 0.0F);
    const std::vector<float> zero46(embedding * ffn, 0.0F);
    const std::vector<float> zero64(ffn * embedding, 0.0F);

    ModelBuilder builder;
    builder.add_f32("token_embd.weight", {embedding, vocab}, identity);
    builder.add_f32("output_norm.weight", {embedding}, ones);
    builder.add_f32("blk.0.attn_norm.weight", {embedding}, ones);
    builder.add_f32("blk.0.attn_q.weight", {embedding, embedding}, zero44);
    builder.add_f32("blk.0.attn_k.weight", {embedding, 2}, zero42);
    builder.add_f32("blk.0.attn_v.weight", {embedding, 2}, zero42);
    builder.add_f32("blk.0.attn_output.weight", {embedding, embedding}, zero44);
    builder.add_f32("blk.0.ffn_norm.weight", {embedding}, ones);
    builder.add_f32("blk.0.ffn_gate.weight", {embedding, ffn}, zero46);
    builder.add_f32("blk.0.ffn_up.weight", {embedding, ffn}, zero46);
    builder.add_f32("blk.0.ffn_down.weight", {ffn, embedding}, zero64);
    return builder.finish();
}


air::DecisionRequest base_decision() {
    air::DecisionRequest request;
    request.input_text = "a";
    request.candidates = {
        {"left", "AA", std::string("aa")},
        {"right", "BB", std::string("bb")},
    };
    request.scoring_policy = air::DecisionScoringPolicy::sequence_logprob_sum;
    request.output_cardinality = air::DecisionOutputCardinality::exactly_one;
    request.determinism = air::DeterminismRequirement::required;
    return request;
}

void test_reference_decision_through_one_service() {
    air::SchedulerConfig scheduler;
    scheduler.max_active_requests = 4;
    scheduler.max_queued_requests = 16;
    scheduler.token_budget_per_cycle = 2;
    scheduler.prefill_quantum_tokens = 1;
    scheduler.reference_kv_page_tokens = 2;

    auto service_result = air::InferenceService::create(
        tiny_model(), air::BackendPreference::reference, 0, scheduler);
    check(service_result.is_ok(), "reference service creates for Decision V1");
    if (!service_result) return;
    auto& service = service_result.value();

    auto request = base_decision();
    auto decision = service->decide(request);
    check(decision.is_ok(), "Decision V1 completes through InferenceService");
    if (decision) {
        const auto& response = decision.value();
        check(response.decision.selected_candidate_ids.size() == 1,
              "Decision V1 returns exactly one semantic candidate");
        check(response.decision.scores.size() == 2,
              "Decision V1 returns one normalized score per candidate");
        double sum = 0.0;
        for (const auto& score : response.decision.scores) sum += score.normalized_score;
        check(std::abs(sum - 1.0) < 1e-6,
              "Decision V1 candidate-set-normalized scores sum to one");
        check(!response.calibrated && !response.abstention_qualified,
              "Decision V1 does not claim calibration or abstention reliability");
        check(response.metrics.workload == "decision",
              "Decision response reports decision workload identity");
        check(response.candidate_tokens_scored >= 4,
              "multi-token candidates are actually scored");
        check(response.branch_count == 2,
              "multi-token candidates branch from one shared prefix");
    }

    request.scoring_policy = air::DecisionScoringPolicy::sequence_logprob_mean;
    auto mean = service->decide(request);
    check(mean.is_ok() &&
          mean.value().decision.applied_scoring_policy ==
              air::DecisionScoringPolicy::sequence_logprob_mean,
          "explicit mean scoring policy is preserved");

    request.scoring_policy = air::DecisionScoringPolicy::qualified_auto;
    auto automatic = service->decide(request);
    check(!automatic &&
          automatic.status().code() == air::ErrorCode::unsupported,
          "unqualified AUTO scorer fails closed in production Decision V1");

    request = base_decision();
    request.output_cardinality = air::DecisionOutputCardinality::zero_or_one;
    auto abstention = service->decide(request);
    check(!abstention &&
          abstention.status().code() == air::ErrorCode::unsupported,
          "uncalibrated abstention execution fails closed");

    air::CancellationSource cancelled;
    cancelled.cancel();
    auto cancelled_result =
        service->decide(base_decision(), cancelled.token());
    check(!cancelled_result &&
          cancelled_result.status().code() == air::ErrorCode::cancelled,
          "decision cancellation before submission is deterministic");

    const auto events = service->recent_events(64);
    bool queued = false;
    bool completed = false;
    for (const auto& event : events) {
        queued = queued || event.type == "decision_queued";
        completed = completed || event.type == "decision_complete";
    }
    check(queued && completed,
          "Decision work is observable in the existing runtime event surface");

    const auto snapshot = service->snapshot();
    check(snapshot.completed_decisions >= 2,
          "service snapshot counts completed decisions");
}

void test_mixed_generate_decision_fairness() {
    air::SchedulerConfig scheduler;
    scheduler.max_active_requests = 4;
    scheduler.max_queued_requests = 32;
    scheduler.token_budget_per_cycle = 2;
    scheduler.prefill_quantum_tokens = 1;
    scheduler.reference_kv_page_tokens = 2;

    auto service_result = air::InferenceService::create(
        tiny_model(), air::BackendPreference::reference, 0, scheduler);
    check(service_result.is_ok(), "mixed-workload reference service creates");
    if (!service_result) return;
    auto& service = service_result.value();

    air::InferenceRequest generation;
    generation.prompt = "a";
    generation.generation.max_new_tokens = 24;
    generation.generation.sampling.temperature = 0.0;

    auto decision_request = base_decision();

    auto generation_future = std::async(
        std::launch::async, [&] { return service->generate(generation); });
    auto decision_future = std::async(
        std::launch::async, [&] { return service->decide(decision_request); });

    auto generated = generation_future.get();
    auto decided = decision_future.get();
    check(generated.is_ok() && decided.is_ok(),
          "mixed Generate + Decision complete through one service worker");

    const auto events = service->recent_events(128);
    bool generation_queued = false;
    bool decision_queued = false;
    bool plan_selected = false;
    for (const auto& event : events) {
        generation_queued = generation_queued || event.type == "request_queued";
        decision_queued = decision_queued || event.type == "decision_queued";
        plan_selected = plan_selected || event.type == "plan_selected";
    }
    check(generation_queued && decision_queued && plan_selected,
          "mixed workloads share the existing queue/planner observability");
}


void test_queued_and_branch_cancellation() {
    air::SchedulerConfig scheduler;
    scheduler.max_active_requests = 1;
    scheduler.max_queued_requests = 8;
    scheduler.token_budget_per_cycle = 1;
    scheduler.prefill_quantum_tokens = 1;
    scheduler.reference_kv_page_tokens = 2;

    auto service_result = air::InferenceService::create(
        tiny_model(), air::BackendPreference::reference, 0, scheduler);
    check(service_result.is_ok(), "cancellation service creates");
    if (!service_result) return;
    auto& service = service_result.value();

    air::InferenceRequest long_generation;
    long_generation.prompt = "a";
    long_generation.generation.max_new_tokens = 512;
    long_generation.generation.sampling.temperature = 0.0;

    auto blocker = std::async(
        std::launch::async, [&] { return service->generate(long_generation); });

    for (int i = 0; i < 500; ++i) {
        if (service->snapshot().active_requests >= 1U) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    air::CancellationSource queued_cancel;
    auto queued_future = std::async(
        std::launch::async, [&] {
            return service->decide(base_decision(), queued_cancel.token());
        });

    for (int i = 0; i < 500; ++i) {
        if (service->snapshot().queued_requests >= 1U) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    queued_cancel.cancel();
    auto queued_result = queued_future.get();
    check(!queued_result &&
          queued_result.status().code() == air::ErrorCode::cancelled,
          "queued Decision work observes cancellation through the shared scheduler");

    auto blocker_result = blocker.get();
    check(blocker_result.is_ok(), "blocking generation completes after queued cancellation");

    air::DecisionRequest long_decision;
    long_decision.input_text = "a";
    std::string candidate_a(256, 'a');
    std::string candidate_b(256, 'b');
    long_decision.candidates = {
        {"long-a", "long-a", candidate_a},
        {"long-b", "long-b", candidate_b},
    };
    long_decision.scoring_policy =
        air::DecisionScoringPolicy::sequence_logprob_sum;
    long_decision.output_cardinality =
        air::DecisionOutputCardinality::exactly_one;

    air::CancellationSource branch_cancel;
    auto branch_future = std::async(
        std::launch::async, [&] {
            return service->decide(long_decision, branch_cancel.token());
        });

    bool observed_active = false;
    for (int i = 0; i < 500; ++i) {
        const auto snapshot = service->snapshot();
        if (snapshot.active_requests >= 1U) {
            observed_active = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check(observed_active, "long Decision work becomes active before branch cancellation");
    branch_cancel.cancel();
    auto branch_result = branch_future.get();
    check(!branch_result &&
          branch_result.status().code() == air::ErrorCode::cancelled,
          "active Decision branch observes cancellation without leaking work");

    const auto snapshot = service->snapshot();
    check(snapshot.active_requests == 0U && snapshot.queued_requests == 0U,
          "cancellation leaves no queued or active work behind");
}


void test_shutdown_cancels_active_decision() {
    air::SchedulerConfig scheduler;
    scheduler.max_active_requests = 1;
    scheduler.max_queued_requests = 8;
    scheduler.token_budget_per_cycle = 1;
    scheduler.prefill_quantum_tokens = 1;
    scheduler.reference_kv_page_tokens = 2;

    auto service_result = air::InferenceService::create(
        tiny_model(), air::BackendPreference::reference, 0, scheduler);
    check(service_result.is_ok(), "Decision shutdown service creates");
    if (!service_result) return;
    auto& service = service_result.value();

    air::DecisionRequest request;
    request.input_text = "a";
    request.candidates = {
        {"long-a", "long-a", std::string(1024, 'a')},
        {"long-b", "long-b", std::string(1024, 'b')},
    };
    request.scoring_policy =
        air::DecisionScoringPolicy::sequence_logprob_sum;
    request.output_cardinality =
        air::DecisionOutputCardinality::exactly_one;

    auto future = std::async(
        std::launch::async, [&] { return service->decide(request); });

    bool active = false;
    for (int i = 0; i < 1000; ++i) {
        if (service->snapshot().active_requests >= 1U) {
            active = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check(active, "long Decision becomes active before service shutdown");

    service->shutdown();
    auto result = future.get();
    check(!result && result.status().code() == air::ErrorCode::cancelled,
          "service shutdown cancels active Decision work");

    const auto snapshot = service->snapshot();
    check(snapshot.active_requests == 0U &&
          snapshot.queued_requests == 0U &&
          snapshot.admission_reserved_bytes == 0U &&
          snapshot.current_kv_bytes == 0U,
          "Decision shutdown leaves no live scheduler or sequence resources");
}


} // namespace

int main() {
    test_reference_decision_through_one_service();
    test_mixed_generate_decision_fairness();
    test_queued_and_branch_cancellation();
    test_shutdown_cancels_active_decision();

    if (failures != 0) {
        std::cerr << failures << " decision serving test(s) failed\\n";
        return 1;
    }
    std::cout << "decision serving tests passed\\n";
    return 0;
}

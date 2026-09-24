#include "runtime/sequence_state_store.hpp"

#include "air/execution.hpp"
#include "air/model.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
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

class FakeCheckpoint final : public air::runtime_detail::SequenceCheckpoint {
public:
    FakeCheckpoint(air::BackendKind backend,
                   std::uint64_t tokens,
                   std::uint64_t checkpoint_bytes,
                   std::uint64_t device_bytes,
                   int payload,
                   bool fail_clone = false)
        : backend_(backend),
          resources_{tokens, checkpoint_bytes, device_bytes},
          payload_(payload),
          fail_clone_(fail_clone) {}

    [[nodiscard]] air::BackendKind backend() const noexcept override { return backend_; }
    [[nodiscard]] std::uint64_t tokens() const noexcept override {
        return resources_.committed_tokens;
    }
    [[nodiscard]] air::runtime_detail::SequenceResources resources() const noexcept override {
        return resources_;
    }
    [[nodiscard]] air::Result<std::unique_ptr<air::runtime_detail::SequenceCheckpoint>>
    clone() const override {
        if (fail_clone_) return air::Status::internal_error("synthetic checkpoint clone failure");
        return std::unique_ptr<air::runtime_detail::SequenceCheckpoint>(
            std::make_unique<FakeCheckpoint>(*this));
    }

    [[nodiscard]] int payload() const noexcept { return payload_; }
    void set_payload(int value) noexcept { payload_ = value; }

private:
    air::BackendKind backend_{air::BackendKind::reference};
    air::runtime_detail::SequenceResources resources_{};
    int payload_{0};
    bool fail_clone_{false};
};

air::runtime_detail::SequenceStateCompatibility identity(std::string model = "model-A") {
    return air::runtime_detail::SequenceStateCompatibility{
        std::move(model),
        "tokenizer-A",
        "position-A",
        "none",
        air::BackendKind::reference,
        "reference-sequence-checkpoint-v1",
        16U,
    };
}

void test_longest_exact_prefix_and_identity() {
    air::runtime_detail::SequenceStateStore store(4);
    const std::vector<air::TokenId> p2{1, 2};
    const std::vector<air::TokenId> p4{1, 2, 3, 4};
    const std::vector<float> logits2{0.1F, 0.2F};
    const std::vector<float> logits4{0.3F, 0.4F};

    check(store.insert(p2, identity(), std::make_unique<FakeCheckpoint>(
        air::BackendKind::reference, 2U, 200U, 0U, 2), logits2).is_ok(),
        "short exact prefix stores");
    check(store.insert(p4, identity(), std::make_unique<FakeCheckpoint>(
        air::BackendKind::reference, 4U, 400U, 0U, 4), logits4).is_ok(),
        "long exact prefix stores");

    const std::vector<air::TokenId> prompt{1, 2, 3, 4, 5};
    auto match = store.longest(prompt, identity());
    check(match && match->tokens == 4U, "longest compatible token prefix wins");
    check(match && match->logits == logits4, "stored logits follow the longest prefix");

    auto incompatible = identity("model-B");
    check(!store.longest(prompt, incompatible),
          "model revision mismatch invalidates exact prefix state");

    auto tok = identity();
    tok.tokenizer_identity = "tokenizer-B";
    check(!store.longest(prompt, tok),
          "tokenizer identity mismatch invalidates exact prefix state");

    auto pos = identity();
    pos.positional_semantics = "position-B";
    check(!store.longest(prompt, pos),
          "positional semantics mismatch invalidates exact prefix state");

    auto adapter = identity();
    adapter.adapter_identity = "adapter-B";
    check(!store.longest(prompt, adapter),
          "adapter identity mismatch invalidates exact prefix state");

    auto format = identity();
    format.state_format = "future-recurrent-state-v1";
    check(!store.longest(prompt, format),
          "backend state-format mismatch invalidates exact prefix state");

    auto page = identity();
    page.page_tokens = 32U;
    check(!store.longest(prompt, page),
          "physical state page geometry mismatch invalidates exact prefix state");
}

void test_branch_clone_isolation_and_eviction_lifetime() {
    air::runtime_detail::SequenceStateStore store(1);
    const std::vector<air::TokenId> p{7, 8, 9};
    const std::vector<float> logits{1.0F};

    check(store.insert(p, identity(), std::make_unique<FakeCheckpoint>(
        air::BackendKind::reference, 3U, 300U, 0U, 11), logits).is_ok(),
        "branch source stores");

    auto first = store.longest(p, identity());
    check(first.has_value(), "branch clone is returned");
    auto* mutable_clone =
        dynamic_cast<FakeCheckpoint*>(first ? first->checkpoint.get() : nullptr);
    check(mutable_clone && mutable_clone->payload() == 11,
          "cloned branch carries checkpoint payload");
    if (mutable_clone) mutable_clone->set_payload(99);

    auto second = store.longest(p, identity());
    const auto* second_clone =
        dynamic_cast<const FakeCheckpoint*>(second ? second->checkpoint.get() : nullptr);
    check(second_clone && second_clone->payload() == 11,
          "branch mutation does not corrupt authoritative stored checkpoint");

    const std::vector<air::TokenId> other{42};
    check(store.insert(other, identity(), std::make_unique<FakeCheckpoint>(
        air::BackendKind::reference, 1U, 100U, 0U, 42), logits).is_ok(),
        "capacity pressure evicts old entry");

    check(!store.longest(p, identity()), "evicted parent is no longer discoverable");
    check(mutable_clone && mutable_clone->payload() == 99,
          "previously returned child clone survives parent eviction");

    const auto metrics = store.metrics();
    check(metrics.evictions == 1U && metrics.entries == 1U,
          "eviction and entry accounting are explicit");
}

void test_resource_policy_and_validation() {
    air::runtime_detail::SequenceStateStore store(2);
    const std::vector<air::TokenId> p{1, 2, 3};
    const std::vector<float> logits{0.1F, 0.2F, 0.3F};

    auto device_identity = identity();
    device_identity.backend = air::BackendKind::cuda;
    device_identity.state_format = "cuda-sequence-checkpoint-v1";

    const auto device_status = store.insert(
        p,
        device_identity,
        std::make_unique<FakeCheckpoint>(
            air::BackendKind::cuda, 3U, 300U, 4096U, 1),
        logits);
    check(!device_status.is_ok() &&
          device_status.code() == air::ErrorCode::unsupported,
          "persistent device-resident state is rejected until resource admission exists");
    check(store.metrics().rejected_device_residency == 1U &&
          store.metrics().retained_device_bytes == 0U,
          "device-state rejection is measured and retains zero device bytes");

    const std::vector<air::TokenId> empty;
    check(!store.insert(empty, identity(), std::make_unique<FakeCheckpoint>(
        air::BackendKind::reference, 0U, 0U, 0U, 0), logits).is_ok(),
        "zero-length persistent prefix fails closed");

    check(!store.insert(p, identity(), std::make_unique<FakeCheckpoint>(
        air::BackendKind::reference, 2U, 200U, 0U, 0), logits).is_ok(),
        "checkpoint/prefix token mismatch fails closed");

    auto wrong_backend = identity();
    check(!store.insert(p, wrong_backend, std::make_unique<FakeCheckpoint>(
        air::BackendKind::cuda, 3U, 300U, 0U, 0), logits).is_ok(),
        "checkpoint/backend identity mismatch fails closed");

    check(store.insert(p, identity(), std::make_unique<FakeCheckpoint>(
        air::BackendKind::reference, 3U, 300U, 0U, 3), logits).is_ok(),
        "host-resident checkpoint stores");
    auto m = store.metrics();
    check(m.retained_checkpoint_bytes == 300U &&
          m.retained_device_bytes == 0U &&
          m.retained_metadata_bytes >=
              p.size() * sizeof(air::TokenId) + logits.size() * sizeof(float),
          "retained state and metadata bytes are explicitly accounted");

    store.clear();
    m = store.metrics();
    check(m.entries == 0U && m.retained_checkpoint_bytes == 0U &&
          m.retained_metadata_bytes == 0U && m.retained_device_bytes == 0U,
          "clear deterministically releases retained-state accounting");
}

air::ModelDefinition make_model(std::string chat_template, double rope_base) {
    air::ModelFingerprint fp{"gguf", "qwen2", "state-test-model"};
    air::ModelConfig config;
    config.architecture = "qwen2";
    config.layer_count = 2U;
    config.embedding_size = 8U;
    config.feed_forward_size = 16U;
    config.attention_head_count = 2U;
    config.kv_head_count = 1U;
    config.rope_dimension_count = 4U;
    config.context_length = 128U;
    config.vocabulary_size = 2U;
    config.rope_frequency_base = rope_base;
    config.rope_scaling_type = "none";
    config.rope_scaling_factor = 1.0;
    config.rope_scale_linear = 1.0;

    air::TokenizerDefinition tokenizer;
    tokenizer.model = "gpt2";
    tokenizer.pre_tokenizer = "default";
    tokenizer.chat_template = std::move(chat_template);
    tokenizer.vocabulary = {"a", "b"};
    return air::ModelDefinition(std::move(fp), std::move(config),
                                std::move(tokenizer), {});
}

void test_compatibility_identity_derivation() {
    auto model = make_model("template-A", 10000.0);
    air::ExecutionPlan plan;
    plan.backend = air::BackendKind::reference;
    plan.kv.page_tokens = 16U;

    const auto a =
        air::runtime_detail::make_sequence_state_compatibility(model, plan);
    const auto same =
        air::runtime_detail::make_sequence_state_compatibility(model, plan);
    check(a == same, "compatibility derivation is deterministic");

    auto tokenizer_changed = make_model("template-B", 10000.0);
    const auto b =
        air::runtime_detail::make_sequence_state_compatibility(tokenizer_changed, plan);
    check(a.tokenizer_identity != b.tokenizer_identity &&
          a.model_revision != b.model_revision,
          "tokenizer changes invalidate tokenizer and model revision identity");

    auto position_changed = make_model("template-A", 500000.0);
    const auto c =
        air::runtime_detail::make_sequence_state_compatibility(position_changed, plan);
    check(a.positional_semantics != c.positional_semantics &&
          a.model_revision != c.model_revision,
          "positional-semantics changes invalidate reusable state");

    const auto adapter =
        air::runtime_detail::make_sequence_state_compatibility(model, plan, "adapter-v2");
    check(adapter.adapter_identity != a.adapter_identity,
          "adapter state participates in compatibility identity");

    auto cuda_plan = plan;
    cuda_plan.backend = air::BackendKind::cuda;
    const auto cuda =
        air::runtime_detail::make_sequence_state_compatibility(model, cuda_plan);
    check(cuda.backend == air::BackendKind::cuda &&
          cuda.state_format != a.state_format,
          "backend state format is explicit compatibility identity");
}

void test_clone_failure_is_fail_closed() {
    air::runtime_detail::SequenceStateStore store(2);
    const std::vector<air::TokenId> p{9, 9};
    const std::vector<float> logits{0.1F};
    check(store.insert(p, identity(), std::make_unique<FakeCheckpoint>(
        air::BackendKind::reference, 2U, 200U, 0U, 1, true), logits).is_ok(),
        "synthetic failing checkpoint stores");
    check(!store.longest(p, identity()),
        "checkpoint clone failure fails closed instead of returning mutable/partial state");
    const auto metrics = store.metrics();
    check(metrics.clone_failures == 1U && metrics.misses == 1U,
        "clone failure is observable and counted as a miss");
}

void test_disabled_store() {
    air::runtime_detail::SequenceStateStore store(0);
    const std::vector<air::TokenId> p{1};
    const std::vector<float> logits{0.5F};
    check(store.insert(p, identity(), std::make_unique<FakeCheckpoint>(
        air::BackendKind::reference, 1U, 100U, 0U, 1), logits).is_ok(),
        "disabled store preserves no-op compatibility");
    check(!store.longest(p, identity()), "disabled store never returns reusable state");
    check(store.metrics().entries == 0U, "disabled store retains no hidden mutable truth");
}

} // namespace

int main() {
    test_longest_exact_prefix_and_identity();
    test_branch_clone_isolation_and_eviction_lifetime();
    test_resource_policy_and_validation();
    test_compatibility_identity_derivation();
    test_clone_failure_is_fail_closed();
    test_disabled_store();

    if (failures != 0) {
        std::cerr << failures << " sequence state store test(s) failed\n";
        return 1;
    }
    std::cout << "sequence state store tests passed\n";
    return 0;
}

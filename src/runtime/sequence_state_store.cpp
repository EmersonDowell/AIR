#include "runtime/sequence_state_store.hpp"

#include "air/manifest.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace air::runtime_detail {
namespace {

class Fnv1a64 final {
public:
    void bytes(const void* data, std::size_t size) noexcept {
        const auto* p = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < size; ++i) {
            value_ ^= p[i];
            value_ *= 1099511628211ULL;
        }
    }

    void text(std::string_view value) noexcept {
        bytes(value.data(), value.size());
        separator();
    }

    template <class T>
    void scalar(const T& value) noexcept {
        bytes(&value, sizeof(value));
        separator();
    }

    [[nodiscard]] std::string digest() const {
        std::ostringstream out;
        out << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << value_;
        return out.str();
    }

private:
    void separator() noexcept {
        constexpr unsigned char marker = 0xffU;
        bytes(&marker, 1U);
    }

    std::uint64_t value_{14695981039346656037ULL};
};

void hash_optional(Fnv1a64& hash, const std::optional<TokenId>& id) {
    const bool present = id.has_value();
    hash.scalar(present);
    if (id) hash.scalar(*id);
}

[[nodiscard]] std::string tokenizer_digest(const TokenizerDefinition& tokenizer) {
    Fnv1a64 hash;
    hash.text(tokenizer.model);
    hash.text(tokenizer.pre_tokenizer);
    hash.text(tokenizer.chat_template);
    hash.scalar(tokenizer.add_bos);
    hash.scalar(tokenizer.add_eos);
    for (const auto& token : tokenizer.vocabulary) hash.text(token);
    for (const auto score : tokenizer.scores) hash.scalar(score);
    for (const auto type : tokenizer.token_types) hash.scalar(type);
    for (const auto& merge : tokenizer.merges) hash.text(merge);
    hash_optional(hash, tokenizer.special_ids.bos);
    hash_optional(hash, tokenizer.special_ids.eos);
    hash_optional(hash, tokenizer.special_ids.unknown);
    hash_optional(hash, tokenizer.special_ids.padding);
    hash_optional(hash, tokenizer.special_ids.eot);
    hash_optional(hash, tokenizer.special_ids.eom);
    return hash.digest();
}

[[nodiscard]] std::string positional_digest(const ModelConfig& config) {
    Fnv1a64 hash;
    hash.text(config.architecture);
    hash.scalar(config.context_length);
    hash.scalar(config.attention_sliding_window);
    hash.scalar(config.rope_dimension_count);
    hash.scalar(config.rope_frequency_base);
    hash.text(config.rope_scaling_type);
    hash.scalar(config.rope_scaling_factor);
    hash.scalar(config.rope_scale_linear);
    return hash.digest();
}

[[nodiscard]] std::string state_format(BackendKind backend) {
    switch (backend) {
    case BackendKind::reference:
        return "reference-sequence-checkpoint-v1";
    case BackendKind::cuda:
        return "cuda-sequence-checkpoint-v1";
    }
    return "unknown-sequence-checkpoint";
}

[[nodiscard]] bool valid_identity(const SequenceStateCompatibility& identity) noexcept {
    return !identity.model_revision.empty() &&
           !identity.tokenizer_identity.empty() &&
           !identity.positional_semantics.empty() &&
           !identity.adapter_identity.empty() &&
           !identity.state_format.empty();
}

} // namespace

SequenceStateCompatibility make_sequence_state_compatibility(
    const ModelDefinition& model,
    const ExecutionPlan& plan,
    std::string_view adapter_identity) {
    return SequenceStateCompatibility{
        model_digest(model),
        tokenizer_digest(model.tokenizer()),
        positional_digest(model.config()),
        std::string(adapter_identity),
        plan.backend,
        state_format(plan.backend),
        plan.kv.page_tokens,
    };
}

std::uint64_t SequenceStateStore::metadata_bytes(const Entry& entry) noexcept {
    const auto token_bytes = static_cast<std::uint64_t>(entry.tokens.size()) *
                             static_cast<std::uint64_t>(sizeof(TokenId));
    const auto logit_bytes = static_cast<std::uint64_t>(entry.logits.size()) *
                             static_cast<std::uint64_t>(sizeof(float));
    if (token_bytes > std::numeric_limits<std::uint64_t>::max() - logit_bytes) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return token_bytes + logit_bytes;
}

void SequenceStateStore::subtract_accounting(const Entry& entry) noexcept {
    metrics_.retained_checkpoint_bytes -=
        std::min(metrics_.retained_checkpoint_bytes, entry.resources.committed_kv_bytes);
    metrics_.retained_device_bytes -=
        std::min(metrics_.retained_device_bytes, entry.resources.resident_device_bytes);
    const auto bytes = metadata_bytes(entry);
    metrics_.retained_metadata_bytes -=
        std::min(metrics_.retained_metadata_bytes, bytes);
}

void SequenceStateStore::add_accounting(const Entry& entry) noexcept {
    metrics_.retained_checkpoint_bytes += entry.resources.committed_kv_bytes;
    metrics_.retained_device_bytes += entry.resources.resident_device_bytes;
    metrics_.retained_metadata_bytes += metadata_bytes(entry);
}

std::optional<SequenceStateStore::Match> SequenceStateStore::longest(
    std::span<const TokenId> prompt,
    const SequenceStateCompatibility& compatibility) {
    ++metrics_.lookups;
    if (capacity_entries_ == 0U || prompt.empty() || !valid_identity(compatibility)) {
        ++metrics_.misses;
        return std::nullopt;
    }

    Entry* best = nullptr;
    for (auto& entry : entries_) {
        if (entry.compatibility != compatibility) continue;
        if (entry.tokens.size() > prompt.size()) continue;
        if (!std::equal(entry.tokens.begin(), entry.tokens.end(), prompt.begin())) continue;
        if (!best || entry.tokens.size() > best->tokens.size()) best = &entry;
    }

    if (!best || !best->checkpoint) {
        ++metrics_.misses;
        return std::nullopt;
    }

    auto cloned = best->checkpoint->clone();
    if (!cloned) {
        ++metrics_.clone_failures;
        ++metrics_.misses;
        return std::nullopt;
    }

    best->age = ++age_;
    ++metrics_.hits;
    return Match{
        best->checkpoint->tokens(),
        std::move(cloned).value(),
        best->logits,
        best->resources,
    };
}

Status SequenceStateStore::insert(
    std::span<const TokenId> prefix,
    SequenceStateCompatibility compatibility,
    std::unique_ptr<SequenceCheckpoint> checkpoint,
    std::span<const float> logits) {
    if (capacity_entries_ == 0U) return Status::ok();
    if (prefix.empty()) {
        return Status::invalid_argument("sequence state store cannot persist an empty prefix");
    }
    if (!valid_identity(compatibility)) {
        return Status::invalid_argument("sequence state compatibility identity is incomplete");
    }
    if (!checkpoint) {
        return Status::invalid_argument("sequence state store checkpoint is null");
    }
    if (checkpoint->backend() != compatibility.backend) {
        return Status::invalid_argument(
            "sequence state checkpoint backend does not match compatibility identity");
    }
    if (checkpoint->tokens() != prefix.size()) {
        return Status::invalid_argument(
            "sequence state checkpoint token count does not match stored prefix");
    }

    const auto resources = checkpoint->resources();
    if (resources.committed_tokens != checkpoint->tokens()) {
        return Status::invalid_argument(
            "sequence state checkpoint resource token count is inconsistent");
    }
    if (resources.resident_device_bytes != 0U) {
        ++metrics_.rejected_device_residency;
        return Status::unsupported(
            "persistent device-resident sequence state requires CapacityScheduler accounting");
    }

    for (auto& entry : entries_) {
        if (entry.compatibility == compatibility &&
            entry.tokens.size() == prefix.size() &&
            std::equal(entry.tokens.begin(), entry.tokens.end(), prefix.begin())) {
            subtract_accounting(entry);
            entry.checkpoint = std::move(checkpoint);
            entry.logits.assign(logits.begin(), logits.end());
            entry.resources = resources;
            entry.age = ++age_;
            add_accounting(entry);
            ++metrics_.replacements;
            metrics_.entries = entries_.size();
            return Status::ok();
        }
    }

    if (entries_.size() >= capacity_entries_) {
        auto oldest = std::min_element(
            entries_.begin(), entries_.end(),
            [](const Entry& a, const Entry& b) { return a.age < b.age; });
        if (oldest != entries_.end()) {
            subtract_accounting(*oldest);
            entries_.erase(oldest);
            ++metrics_.evictions;
        }
    }

    Entry entry{
        std::move(compatibility),
        std::vector<TokenId>(prefix.begin(), prefix.end()),
        std::move(checkpoint),
        std::vector<float>(logits.begin(), logits.end()),
        resources,
        ++age_,
    };
    add_accounting(entry);
    entries_.push_back(std::move(entry));
    ++metrics_.inserts;
    metrics_.entries = entries_.size();
    return Status::ok();
}

void SequenceStateStore::clear() noexcept {
    entries_.clear();
    metrics_.entries = 0U;
    metrics_.retained_checkpoint_bytes = 0U;
    metrics_.retained_metadata_bytes = 0U;
    metrics_.retained_device_bytes = 0U;
}

SequenceStateStoreMetrics SequenceStateStore::metrics() const noexcept {
    auto out = metrics_;
    out.capacity_entries = capacity_entries_;
    out.entries = entries_.size();
    return out;
}

} // namespace air::runtime_detail

#pragma once

#include "air/execution.hpp"
#include "air/model.hpp"
#include "air/status.hpp"
#include "air/types.hpp"
#include "runtime/backend.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace air::runtime_detail {

// Compatibility identity for reusable execution state.
//
// Semantic identity and execution-state identity are deliberately separate.
// Candidate IDs, user text, and other semantic labels do not belong here.
// This identity describes when backend sequence state may be restored safely.
struct SequenceStateCompatibility {
    // Full canonical model/revision digest. Includes current model/tokenizer
    // definition and backing artifact evidence used by AIR's manifest logic.
    std::string model_revision;

    // Explicit tokenizer identity remains separate even though model_revision
    // also contains tokenizer evidence. This makes invalidation intent visible.
    std::string tokenizer_identity;

    // Positional/state-evolution semantics such as RoPE/sliding-window rules.
    std::string positional_semantics;

    // Current AIR has no adapter stack. "none" is explicit rather than hidden.
    // Future adapter/LoRA state must participate here before reuse is allowed.
    std::string adapter_identity{"none"};

    BackendKind backend{BackendKind::reference};

    // Backend state representation version, not a public neural architecture.
    // Transformer KV is one implementation, not the AIR state abstraction.
    std::string state_format;

    // Physical page geometry is restore compatibility data when present.
    std::optional<std::uint32_t> page_tokens;

    [[nodiscard]] friend bool operator==(const SequenceStateCompatibility&,
                                         const SequenceStateCompatibility&) = default;
};

[[nodiscard]] SequenceStateCompatibility make_sequence_state_compatibility(
    const ModelDefinition& model,
    const ExecutionPlan& plan,
    std::string_view adapter_identity = "none");

struct SequenceStateStoreMetrics {
    std::size_t capacity_entries{0};
    std::size_t entries{0};

    std::uint64_t retained_checkpoint_bytes{0};
    std::uint64_t retained_metadata_bytes{0};
    std::uint64_t retained_device_bytes{0};

    std::uint64_t lookups{0};
    std::uint64_t hits{0};
    std::uint64_t misses{0};
    std::uint64_t inserts{0};
    std::uint64_t replacements{0};
    std::uint64_t evictions{0};
    std::uint64_t clone_failures{0};
    std::uint64_t rejected_device_residency{0};
};

// Single authoritative registry for persistent, exact reusable sequence state.
//
// R1 deliberately accepts only checkpoints with zero device-resident bytes.
// CapacityScheduler remains the sole device resource/admission authority, so
// persistent CUDA state cannot become an unaccounted second owner by accident.
class SequenceStateStore final {
public:
    explicit SequenceStateStore(std::size_t capacity_entries)
        : capacity_entries_(capacity_entries) {}

    struct Match {
        std::uint64_t tokens{0};
        std::unique_ptr<SequenceCheckpoint> checkpoint;
        std::vector<float> logits;
        SequenceResources resources{};
    };

    [[nodiscard]] std::optional<Match> longest(
        std::span<const TokenId> prompt,
        const SequenceStateCompatibility& compatibility);

    [[nodiscard]] Status insert(
        std::span<const TokenId> prefix,
        SequenceStateCompatibility compatibility,
        std::unique_ptr<SequenceCheckpoint> checkpoint,
        std::span<const float> logits);

    void clear() noexcept;

    [[nodiscard]] SequenceStateStoreMetrics metrics() const noexcept;
    [[nodiscard]] std::size_t capacity_entries() const noexcept {
        return capacity_entries_;
    }

private:
    struct Entry {
        SequenceStateCompatibility compatibility;
        std::vector<TokenId> tokens;
        std::unique_ptr<SequenceCheckpoint> checkpoint;
        std::vector<float> logits;
        SequenceResources resources{};
        std::uint64_t age{0};
    };

    [[nodiscard]] static std::uint64_t metadata_bytes(const Entry& entry) noexcept;
    void subtract_accounting(const Entry& entry) noexcept;
    void add_accounting(const Entry& entry) noexcept;

    std::size_t capacity_entries_{0};
    std::uint64_t age_{0};
    std::vector<Entry> entries_;
    SequenceStateStoreMetrics metrics_{};
};

} // namespace air::runtime_detail

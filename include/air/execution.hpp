#pragma once

#include "air/result.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace air {

enum class BackendKind {
    reference = 0,
    cuda,
};

[[nodiscard]] const char* to_string(BackendKind backend) noexcept;
[[nodiscard]] Result<BackendKind> backend_kind_from_string(std::string_view value);

enum class PrefillExecutionKind {
    serial = 0,
    native_batch,
};

[[nodiscard]] const char* to_string(PrefillExecutionKind kind) noexcept;

enum class KvStorageKind {
    contiguous = 0,
    paged,
};

[[nodiscard]] const char* to_string(KvStorageKind kind) noexcept;

enum class QuantizedLinearExecutionKind {
    baseline = 0,
    batch_reuse4,
    batch_reuse8,
    q5q8_dp4a_hybrid,
    dense_f32_cublas,
};

[[nodiscard]] const char* to_string(QuantizedLinearExecutionKind kind) noexcept;
[[nodiscard]] Result<QuantizedLinearExecutionKind> quantized_linear_execution_kind_from_string(
    std::string_view value);

enum class AttentionExecutionKind {
    baseline = 0,
    online_softmax,
};

[[nodiscard]] const char* to_string(AttentionExecutionKind kind) noexcept;
[[nodiscard]] Result<AttentionExecutionKind> attention_execution_kind_from_string(
    std::string_view value);

struct BackendCapabilities {
    BackendKind backend{BackendKind::reference};
    PrefillExecutionKind prefill_execution{PrefillExecutionKind::serial};
    KvStorageKind kv_storage{KvStorageKind::contiguous};
    std::uint32_t max_prefill_batch_width{1};
    std::uint32_t max_decode_batch_width{1};
    bool sequence_checkpointing{false};
    bool exact_prefix_reuse{false};
    // True only when the backend can implement temperature<=0 token selection
    // without materializing the full vocabulary logits on the host. The host
    // Sampler remains the semantic oracle for all stochastic policies.
    bool device_greedy_selection{false};
    // Linear tactics are capability-scoped by operation. Transformer-block
    // matrices and the terminal vocabulary projection are not assumed to share
    // the same valid tactic set. This prevents a broad phase-level tactic from
    // silently leaking into an operation it was never prepared to execute.
    std::vector<QuantizedLinearExecutionKind> prefill_block_quantized_linear{QuantizedLinearExecutionKind::baseline};
    std::vector<QuantizedLinearExecutionKind> decode_block_quantized_linear{QuantizedLinearExecutionKind::baseline};
    std::vector<QuantizedLinearExecutionKind> decode_output_quantized_linear{QuantizedLinearExecutionKind::baseline};
    std::vector<AttentionExecutionKind> prefill_attention{AttentionExecutionKind::baseline};
    std::vector<AttentionExecutionKind> decode_attention{AttentionExecutionKind::baseline};
};

struct RequestProfile {
    std::uint64_t prompt_tokens{0};
    std::uint64_t max_output_tokens{0};
    std::uint32_t active_sequences{1};
};

struct RuntimeSnapshot {
    std::uint64_t free_device_memory_bytes{0};
    std::uint64_t resident_kv_bytes{0};
    std::uint64_t prepared_artifact_bytes{0};
    std::string current_strategy_id;
    double device_utilization{0.0};
};

// SchedulingPolicy contains scheduler behavior, not backend-kernel geometry.
// prefill_quantum_tokens is the maximum prompt work a sequence may consume
// before yielding back to the micro-scheduler. It does not imply native
// multi-token GPU prefill.
struct SchedulingPolicy {
    std::uint32_t prefill_quantum_tokens{32};
};

// KvPolicy describes physical KV geometry only. page_tokens must be present
// only for backends whose KV storage is physically paged.
struct KvPolicy {
    std::optional<std::uint32_t> page_tokens;
};

struct LinearPolicy {
    // Transformer-block linear work during native prefill.
    QuantizedLinearExecutionKind prefill_block{QuantizedLinearExecutionKind::baseline};
    // Transformer-block linear work during decode.
    QuantizedLinearExecutionKind decode_block{QuantizedLinearExecutionKind::baseline};
    // Terminal vocabulary projection for native multi-sequence decode. Keeping
    // this explicit avoids hidden tensor-name fallback when a block tactic has
    // a narrower preparation scope (for example dense-f32-cublas).
    QuantizedLinearExecutionKind decode_output{QuantizedLinearExecutionKind::baseline};
};

struct AttentionPolicy {
    AttentionExecutionKind prefill{AttentionExecutionKind::baseline};
    AttentionExecutionKind decode{AttentionExecutionKind::baseline};
};

struct ExecutionPlan {
    BackendKind backend{BackendKind::reference};
    std::string strategy_id{"static"};
    SchedulingPolicy scheduling{};
    KvPolicy kv{};
    LinearPolicy linear{};
    AttentionPolicy attention{};
};

[[nodiscard]] Status validate_execution_plan(const ExecutionPlan& plan,
                                             const BackendCapabilities& capabilities);


} // namespace air

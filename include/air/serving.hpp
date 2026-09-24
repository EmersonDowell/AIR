#pragma once

#include "air/generation.hpp"
#include "air/decision.hpp"
#include "air/execution.hpp"
#include "air/model.hpp"
#include "air/manifest.hpp"
#include "air/result.hpp"
#include "air/types.hpp"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace air {

enum class BackendPreference {
    automatic = 0,
    reference,
    cuda,
};

[[nodiscard]] const char* to_string(BackendPreference backend) noexcept;

struct ManifestConfig {
    bool enabled{true};
    bool require{false};
    std::filesystem::path path;
    StrategyLabConfig strategy{};
};

struct ExecutionConfig {
    QuantizedLinearExecutionKind cuda_prefill_block_linear{QuantizedLinearExecutionKind::baseline};
    QuantizedLinearExecutionKind cuda_decode_block_linear{QuantizedLinearExecutionKind::baseline};
    QuantizedLinearExecutionKind cuda_decode_output_linear{QuantizedLinearExecutionKind::baseline};
    AttentionExecutionKind cuda_prefill_attention{AttentionExecutionKind::baseline};
    AttentionExecutionKind cuda_decode_attention{AttentionExecutionKind::baseline};
};

struct SchedulerConfig {
    std::uint32_t max_active_requests{8};
    // Bounded submission queue. Rejection is explicit backpressure, not hidden buffering.
    std::uint32_t max_queued_requests{256};
    std::uint32_t token_budget_per_cycle{256};
    // Scheduler yield quantum. This does not imply native backend prefill batching.
    std::uint32_t prefill_quantum_tokens{32};
    // Physical page sizes. Both reference and CUDA use paged KV;
    // page geometry is backend-owned state, not scheduler batch geometry.
    std::uint32_t reference_kv_page_tokens{32};
    std::uint32_t cuda_kv_page_tokens{16};
    std::uint32_t prefix_cache_entries{32};
    std::uint32_t latency_window{256};
    // Maximum decoded stream events buffered per request. Delivery happens on
    // the caller thread so a slow client cannot block the inference scheduler.
    std::uint32_t stream_queue_capacity{256};
};

struct ChatMessage {
    std::string role;
    std::string content;
};

struct InferenceRequest {
    std::string prompt;
    std::vector<ChatMessage> messages;
    GenerationConfig generation{};
};

struct RequestMetrics {
    RequestId request_id{0};
    SequenceId sequence_id{0};
    std::string backend;
    std::string workload{"generation"};
    std::string planner_mode{"static"};
    std::string strategy_id{"static"};
    std::uint64_t prompt_tokens{0};
    std::uint64_t generated_tokens{0};
    std::uint64_t prefix_reused_tokens{0};
    std::uint64_t kv_bytes{0};
    std::uint64_t plan_preparation_bytes{0};
    std::string strategy_objective{"static"};
    std::string strategy_decision_reason{"static"};
    std::uint32_t strategy_eligible_candidates{0};
    bool strategy_prepared_state_hot{false};
    double strategy_estimated_transition_ms{0.0};
    double strategy_estimated_break_even_tokens{0.0};
    std::vector<PlanningCandidateTrace> strategy_candidates;
    double queue_ms{0.0};
    double plan_preparation_ms{0.0};
    double plan_eviction_ms{0.0};
    double prefill_ms{0.0};
    double ttft_ms{0.0};
    double decode_ms{0.0};
    double total_ms{0.0};

    [[nodiscard]] double prefill_tokens_per_second() const noexcept;
    [[nodiscard]] double decode_tokens_per_second() const noexcept;
};

struct InferenceResponse {
    std::string text;
    std::vector<TokenId> tokens;
    bool hit_eos{false};
    RequestMetrics metrics;
};

struct DecisionResponse {
    DecisionResult decision;
    RequestMetrics metrics;
    std::uint64_t candidate_tokens_scored{0};
    std::uint64_t branch_count{0};
    // Prompt 2 established candidate-set normalization, not calibration.
    bool calibrated{false};
    bool abstention_qualified{false};
};

struct RuntimeEvent {
    std::uint64_t sequence{0};
    std::uint64_t unix_ms{0};
    std::string type;
    RequestId request_id{0};
    std::string detail;
};

struct ServiceSnapshot {
    std::string backend;
    std::string planner_mode{"static"};
    std::string manifest_status{"disabled"};
    std::string manifest_id;
    std::string strategy_id{"static"};
    std::string strategy_objective{"static"};
    std::string strategy_decision_reason{"static"};
    std::uint32_t strategy_eligible_candidates{0};
    bool strategy_prepared_state_hot{false};
    double strategy_estimated_transition_ms{0.0};
    double strategy_estimated_break_even_tokens{0.0};
    std::vector<PlanningCandidateTrace> strategy_candidates;
    std::string selected_backend;
    std::uint32_t planned_prefill_quantum_tokens{0};
    std::uint32_t planned_kv_page_tokens{0};
    std::string planned_prefill_block_linear_tactic{"baseline"};
    std::string planned_decode_block_linear_tactic{"baseline"};
    std::string planned_decode_output_linear_tactic{"baseline"};
    std::string planned_prefill_attention_tactic{"baseline"};
    std::string planned_decode_attention_tactic{"baseline"};
    std::uint64_t queued_requests{0};
    std::uint64_t active_requests{0};
    std::uint64_t rejected_overload_requests{0};
    std::uint64_t completed_decisions{0};
    std::uint32_t max_queued_requests{0};
    std::uint64_t completed_requests{0};
    std::uint64_t failed_requests{0};
    std::uint64_t cancelled_requests{0};
    std::uint64_t stream_delivery_failures{0};
    std::uint64_t native_decode_batches{0};
    std::uint64_t native_decode_sequences{0};
    std::uint64_t physical_prefill_batches{0};
    std::uint64_t physical_prefill_sequence_participations{0};
    std::uint64_t physical_prefill_tokens{0};
    std::uint32_t physical_prefill_max_sequences{0};
    std::uint64_t total_prompt_tokens{0};
    std::uint64_t total_generated_tokens{0};
    std::uint64_t total_prefix_reused_tokens{0};
    std::uint64_t peak_kv_bytes{0};
    std::uint64_t peak_device_bytes{0};
    std::uint64_t current_kv_bytes{0};
    std::uint64_t current_device_bytes{0};
    std::uint64_t current_prepared_artifact_bytes{0};
    double aggregate_generated_tokens_per_second{0.0};
    double p50_total_ms{0.0};
    double p95_total_ms{0.0};
    std::uint32_t max_active_requests{0};
    std::uint32_t token_budget_per_cycle{0};
    std::uint32_t prefill_quantum_tokens{0};
    std::uint32_t reference_kv_page_tokens{0};
    std::uint32_t cuda_kv_page_tokens{0};
    std::uint32_t max_prefill_batch_width{1};
    std::uint32_t max_decode_batch_width{1};
    std::uint32_t stream_queue_capacity{0};
    std::string prefill_execution{"serial"};
    std::string kv_storage{"contiguous"};
    std::uint64_t admission_reserved_bytes{0};
    std::uint64_t admission_capacity_bytes{0};
    std::uint64_t kv_pool_allocated_bytes{0};
    std::uint64_t kv_pool_free_bytes{0};
    bool physical_paged_kv{false};
    bool sequence_checkpointing{false};
    bool prefix_cache_enabled{false};
    bool device_greedy_selection{false};
    // Prompt-4 sequence-state authority observability. Additive fields preserve
    // existing runtime JSON/API compatibility while exposing retained-state truth.
    std::uint64_t sequence_state_store_capacity_entries{0};
    std::uint64_t sequence_state_store_entries{0};
    std::uint64_t sequence_state_store_hits{0};
    std::uint64_t sequence_state_store_misses{0};
    std::uint64_t sequence_state_store_evictions{0};
    std::uint64_t sequence_state_store_clone_failures{0};
    std::uint64_t sequence_state_store_rejected_device_residency{0};
    std::uint64_t sequence_state_store_retained_checkpoint_bytes{0};
    std::uint64_t sequence_state_store_retained_metadata_bytes{0};
    std::uint64_t sequence_state_store_retained_device_bytes{0};
};

using StreamCallback = std::function<Status(std::string_view text_delta, TokenId token)>;

class CancellationToken final {
public:
    CancellationToken() = default;
    [[nodiscard]] bool is_cancelled() const noexcept {
        return state_ && state_->load(std::memory_order_acquire);
    }
    [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(state_); }

private:
    explicit CancellationToken(std::shared_ptr<std::atomic_bool> state) : state_(std::move(state)) {}
    std::shared_ptr<std::atomic_bool> state_;
    friend class CancellationSource;
};

class CancellationSource final {
public:
    CancellationSource() : state_(std::make_shared<std::atomic_bool>(false)) {}
    [[nodiscard]] CancellationToken token() const noexcept { return CancellationToken(state_); }
    void cancel() noexcept { state_->store(true, std::memory_order_release); }
    [[nodiscard]] bool is_cancelled() const noexcept { return state_->load(std::memory_order_acquire); }

private:
    std::shared_ptr<std::atomic_bool> state_;
};

class InferenceService final {
public:
    ~InferenceService();
    InferenceService(InferenceService&&) noexcept;
    InferenceService& operator=(InferenceService&&) noexcept;
    InferenceService(const InferenceService&) = delete;
    InferenceService& operator=(const InferenceService&) = delete;

    [[nodiscard]] static Result<std::unique_ptr<InferenceService>> create(
        const std::filesystem::path& model_path,
        BackendPreference backend = BackendPreference::automatic,
        int cuda_device = 0,
        SchedulerConfig scheduler = {},
        std::filesystem::path event_log_path = {},
        ManifestConfig manifest = {},
        ExecutionConfig execution = {});
    [[nodiscard]] static Result<std::unique_ptr<InferenceService>> create(
        std::shared_ptr<ModelDefinition> model,
        BackendPreference backend = BackendPreference::automatic,
        int cuda_device = 0,
        SchedulerConfig scheduler = {},
        std::filesystem::path event_log_path = {},
        ManifestConfig manifest = {},
        ExecutionConfig execution = {});

    [[nodiscard]] Result<InferenceResponse> generate(const InferenceRequest& request,
                                                     StreamCallback stream = {},
                                                     CancellationToken cancellation = {});
    // Bounded semantic decisions use the same submission queue, CapacityScheduler,
    // MicroBatchScheduler, PreparedModel and backend SequenceState architecture.
    [[nodiscard]] Result<DecisionResponse> decide(const DecisionRequest& request,
                                                  CancellationToken cancellation = {});
    // Atomically submit a caller-known production cohort before waking the existing
    // scheduler. This is not a second scheduler or execution path: every request
    // still passes through normal planning, capacity admission, PreparedModel
    // preparation, MicroBatchScheduler ordering, and the same backend executor.
    [[nodiscard]] Result<std::vector<InferenceResponse>> generate_cohort(
        std::span<const InferenceRequest> requests);
    [[nodiscard]] Result<std::string> render_chat(const std::vector<ChatMessage>& messages) const;
    [[nodiscard]] ServiceSnapshot snapshot() const;
    [[nodiscard]] std::vector<RuntimeEvent> recent_events(std::size_t limit = 64) const;
    [[nodiscard]] const ModelDefinition& model() const noexcept;
    [[nodiscard]] std::string backend_name() const;
    void shutdown() noexcept;

private:
    struct Impl;
    explicit InferenceService(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace air

#include "air/serving.hpp"

#include "air/format.hpp"
#include "air/manifest.hpp"
#include "air/runtime.hpp"
#include "air/tokenizer.hpp"

#include "runtime/backend.hpp"
#include "runtime/scheduler.hpp"
#include "runtime/sequence_state_store.hpp"

#include "json.hpp"
#include "../statistics.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>
#include <utility>

namespace air {
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] double ms(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

[[nodiscard]] std::uint64_t unix_ms_now() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}


[[nodiscard]] std::string drain_utf8(std::string& pending, bool flush) {
    std::string out;
    std::size_t i = 0;
    const auto replacement = [&out] { out += "\xEF\xBF\xBD"; };
    while (i < pending.size()) {
        const auto lead = static_cast<unsigned char>(pending[i]);
        if (lead < 0x80U) {
            out.push_back(pending[i++]);
            continue;
        }
        std::size_t width = 0;
        std::uint32_t minimum = 0;
        std::uint32_t codepoint = 0;
        if ((lead & 0xE0U) == 0xC0U) { width = 2; minimum = 0x80U; codepoint = lead & 0x1FU; }
        else if ((lead & 0xF0U) == 0xE0U) { width = 3; minimum = 0x800U; codepoint = lead & 0x0FU; }
        else if ((lead & 0xF8U) == 0xF0U) { width = 4; minimum = 0x10000U; codepoint = lead & 0x07U; }
        else {
            replacement();
            ++i;
            continue;
        }
        if (pending.size() - i < width) {
            if (!flush) break;
            replacement();
            ++i;
            continue;
        }
        bool valid = true;
        for (std::size_t j = 1; j < width; ++j) {
            const auto ch = static_cast<unsigned char>(pending[i + j]);
            if ((ch & 0xC0U) != 0x80U) { valid = false; break; }
            codepoint = (codepoint << 6U) | (ch & 0x3FU);
        }
        if (!valid || codepoint < minimum || codepoint > 0x10FFFFU ||
            (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) {
            replacement();
            ++i;
            continue;
        }
        out.append(pending, i, width);
        i += width;
    }
    pending.erase(0, i);
    if (flush && !pending.empty()) {
        replacement();
        pending.clear();
    }
    return out;
}

[[nodiscard]] std::string sanitize_utf8(std::string value) {
    return drain_utf8(value, true);
}

struct WorkItem {
    enum class Kind : std::uint8_t {
        generation = 0,
        decision,
    };

    RequestId request_id{0};
    SequenceId sequence_id{0};
    Kind kind{Kind::generation};

    InferenceRequest request;
    DecisionRequest decision_request;

    std::vector<TokenId> prompt_tokens;
    std::vector<std::vector<TokenId>> decision_candidate_tokens;

    struct StreamEvent {
        std::string delta;
        TokenId token{0};
    };
    bool stream_enabled{false};
    std::mutex stream_mutex;
    std::condition_variable stream_wake;
    std::deque<StreamEvent> stream_events;
    bool stream_closed{false};

    CancellationToken cancellation;
    std::atomic_bool service_cancelled{false};

    std::promise<Result<InferenceResponse>> promise;
    std::promise<Result<DecisionResponse>> decision_promise;

    Clock::time_point submitted_at{};
    ExecutionPlan plan{};
    PlanningDecision planning_decision{};

    std::unique_ptr<runtime_detail::SequenceState> session;
    bool plan_ready{false};
    std::unique_ptr<Sampler> sampler;
    std::vector<float> logits;
    std::vector<TokenId> generated;
    std::string stream_pending_bytes;

    // Decision-scoring state. The shared prompt prefix is checkpointed once.
    // Candidate branches are restored from that checkpoint and advanced one
    // token per scheduler decode slice so Generate and Decision share the same
    // fairness/backpressure mechanism.
    std::unique_ptr<runtime_detail::SequenceCheckpoint> decision_prefix_checkpoint;
    std::unique_ptr<runtime_detail::SequenceState> decision_branch;
    std::vector<double> decision_score_sums;
    std::vector<double> decision_raw_scores;
    std::vector<float> decision_first_logprobs;
    std::size_t decision_candidate_index{0};
    std::size_t decision_token_position{0};
    std::uint64_t decision_candidate_tokens_scored{0};
    std::uint64_t decision_branch_count{0};
    DecisionResult decision_result{};

    std::size_t prompt_position{0};
    std::uint64_t prefix_reused_tokens{0};
    std::uint64_t admission_reservation_bytes{0};
    std::uint64_t plan_preparation_bytes{0};
    double plan_preparation_ms{0.0};
    double plan_eviction_ms{0.0};
    bool hit_eos{false};
    bool done{false};
    bool failed{false};
    Status failure{};
    Clock::time_point admitted_at{};
    Clock::time_point prefill_started_at{};
    Clock::time_point prefill_finished_at{};
    Clock::time_point first_token_at{};
    Clock::time_point finished_at{};
    double prefill_compute_ms{0.0};
    double decode_compute_ms{0.0};

    [[nodiscard]] bool is_decision() const noexcept {
        return kind == Kind::decision;
    }

    [[nodiscard]] std::uint64_t max_output_tokens() const noexcept {
        if (!is_decision()) return request.generation.max_new_tokens;
        std::size_t maximum = 0U;
        for (const auto& candidate : decision_candidate_tokens) {
            maximum = std::max(maximum, candidate.size());
        }
        return static_cast<std::uint64_t>(maximum);
    }

    [[nodiscard]] double temperature() const noexcept {
        return is_decision() ? 0.0 : request.generation.sampling.temperature;
    }

    [[nodiscard]] bool cancellation_requested() const noexcept {
        return service_cancelled.load(std::memory_order_acquire) || cancellation.is_cancelled();
    }
};

} // namespace

struct InferenceService::Impl {
    std::shared_ptr<ModelDefinition> model;
    std::unique_ptr<Tokenizer> tokenizer;
    BackendPreference requested_backend{BackendPreference::automatic};
    std::string backend{"reference"};
    std::unique_ptr<Runtime> runtime;
    std::string planner_mode{"static"};
    std::string manifest_status{"disabled"};
    std::string manifest_id;
    ExecutionPlan last_plan{};
    PlanningDecision last_decision{};
    SchedulerConfig config{};
    ExecutionConfig execution{};
    int cuda_device{0};
    std::unique_ptr<runtime_detail::PreparedModel> reference;
    std::unique_ptr<runtime_detail::PreparedModel> cuda;
    runtime_detail::SequenceStateStore sequence_state_store{0};
    runtime_detail::CapacityScheduler capacity_scheduler{1};
    runtime_detail::MicrobatchScheduler microbatch_scheduler{};
    // Prompt-2 regime state. Physical compatibility is always derived from
    // active WorkItem plans; this state only records busy-epoch policy.
    std::string last_regime_observation_signature;
    ExecutionPlan fallback_plan{};
    std::uint64_t regime_epoch_id{0};
    bool regime_epoch_open{false};
    bool regime_epoch_fallback_pinned{false};

    mutable std::mutex mutex;
    std::condition_variable wake;
    std::deque<std::shared_ptr<WorkItem>> queued;
    // Monotonic submission epoch. A ScheduledBatch is built against one
    // observed queue generation; if new work arrives while prefill slices are
    // executing, the worker yields at the next prefill-slice boundary so the
    // only scheduler can reconsider admission and batch geometry promptly.
    std::uint64_t queue_generation{0};
    std::vector<std::shared_ptr<WorkItem>> active;
    bool stopping{false};
    std::thread worker;
    std::atomic<RequestId> next_request{1};
    std::atomic<SequenceId> next_sequence{1};

    Clock::time_point service_started{Clock::now()};
    std::uint64_t completed{0};
    std::uint64_t completed_decisions{0};
    std::uint64_t rejected_overload{0};
    std::uint64_t failed{0};
    std::uint64_t cancelled{0};
    std::uint64_t stream_delivery_failures{0};
    std::uint64_t native_decode_batches{0};
    std::uint64_t native_decode_sequences{0};
    std::uint64_t physical_prefill_batches{0};
    std::uint64_t physical_prefill_sequence_participations{0};
    std::uint64_t physical_prefill_tokens{0};
    std::uint32_t physical_prefill_max_sequences{0};
    std::uint64_t prompt_tokens{0};
    std::uint64_t generated_tokens{0};
    std::uint64_t prefix_reused_tokens{0};
    std::uint64_t peak_kv_bytes{0};
    std::uint64_t peak_device_bytes{0};
    std::uint64_t active_admission_reserved_bytes{0};
    std::deque<double> recent_latency;

    std::deque<RuntimeEvent> events;
    std::uint64_t next_event{1};
    std::filesystem::path event_log_path;
    std::ofstream event_log;

    Impl(SchedulerConfig scheduler, ExecutionConfig execution_config)
        : config(scheduler),
          execution(execution_config),
          sequence_state_store(scheduler.prefix_cache_entries),
          capacity_scheduler(scheduler.max_active_requests) {}

    void event(std::string type, RequestId request_id, std::string detail) {
        RuntimeEvent record{next_event++, unix_ms_now(), std::move(type), request_id, std::move(detail)};
        {
            std::lock_guard lock(mutex);
            events.push_back(record);
            while (events.size() > 512U) events.pop_front();
            if (event_log.is_open()) {
                event_log << "{\"seq\":" << record.sequence
                          << ",\"unix_ms\":" << record.unix_ms
                          << ",\"type\":\"" << detail::json_escape(record.type)
                          << "\",\"request_id\":" << record.request_id
                          << ",\"detail\":\"" << detail::json_escape(record.detail) << "\"}\n";
                event_log.flush();
            }
        }
    }

    [[nodiscard]] runtime_detail::PreparedModel* prepared_for(BackendKind backend_kind) noexcept {
        return backend_kind == BackendKind::cuda ? cuda.get() : reference.get();
    }

    [[nodiscard]] const runtime_detail::PreparedModel* prepared_for(BackendKind backend_kind) const noexcept {
        return backend_kind == BackendKind::cuda ? cuda.get() : reference.get();
    }

    void apply_plan_defaults(ExecutionPlan& plan) const {
        if (plan.strategy_id.empty()) plan.strategy_id = "static";
        if (plan.scheduling.prefill_quantum_tokens == 0U) {
            plan.scheduling.prefill_quantum_tokens = config.prefill_quantum_tokens;
        }
        if (!plan.kv.page_tokens || *plan.kv.page_tokens == 0U) {
            plan.kv.page_tokens = plan.backend == BackendKind::cuda
                ? config.cuda_kv_page_tokens
                : config.reference_kv_page_tokens;
        }
    }

    [[nodiscard]] Status plan_item(WorkItem& item) {
        // Queued requests are replanned whenever they reach the admission head.
        // Their concurrency region and prepared-state economics may have changed
        // while they waited; caching a prior request-scoped plan would make
        // lifecycle policy depend on stale arrival timing.
        RequestProfile profile;
        profile.prompt_tokens = item.prompt_tokens.size();
        profile.max_output_tokens = item.max_output_tokens();
        if (profile.prompt_tokens > model->config().context_length ||
            profile.max_output_tokens > model->config().context_length - profile.prompt_tokens) {
            return Status::invalid_argument("request prompt plus maximum output exceeds model context length");
        }
        RuntimeSnapshot runtime_snapshot;
        {
            std::lock_guard lock(mutex);
            profile.active_sequences = static_cast<std::uint32_t>(
                std::max<std::size_t>(1U, active.size() + queued.size()));
            runtime_snapshot.resident_kv_bytes = peak_kv_bytes;
        }
        if (cuda) {
            if (const auto free = cuda->free_device_bytes()) runtime_snapshot.free_device_memory_bytes = *free;
            runtime_snapshot.prepared_artifact_bytes = cuda->prepared_artifact_device_bytes();
        }
        {
            std::lock_guard lock(mutex);
            runtime_snapshot.current_strategy_id = last_plan.strategy_id;
        }

        if (item.is_decision()) {
            // V1 Decision execution is qualified only on the carried static/fallback
            // physical profile. Strategy Lab has no Decision-specific training data yet.
            item.planning_decision = {};
            item.planning_decision.plan = fallback_plan;
            item.planning_decision.objective = "decision-v1-qualified-static";
            item.planning_decision.reason = "decision-v1-no-adaptive-auto-plan";
            item.plan = fallback_plan;
        } else {
            item.planning_decision = runtime->decide(profile, runtime_snapshot);
            item.plan = item.planning_decision.plan;
        }
        apply_plan_defaults(item.plan);
        auto* prepared = prepared_for(item.plan.backend);
        if (!prepared) {
            return Status::unsupported(std::string("planned ") + to_string(item.plan.backend) +
                                       " backend is unavailable");
        }
        const auto valid = validate_execution_plan(item.plan, prepared->capabilities());
        if (!valid) return valid;
        item.plan_ready = true;
        return Status::ok();
    }


    struct GroupRegimeCandidate {
        RequestProfile profile{};
        ExecutionPlan plan{};
        PlanningDecision decision{};
        std::vector<std::size_t> slots;
        bool safe_fallback{false};
        std::string transition_reason{"qualified-homogeneous-group"};
    };

    [[nodiscard]] std::optional<GroupRegimeCandidate> group_regime_candidate() {
        if (planner_mode != "adaptive") return std::nullopt;

        std::vector<runtime_detail::RegimeRequestShape> shapes;
        std::vector<std::size_t> slots;
        shapes.reserve(active.size());
        slots.reserve(active.size());
        for (std::size_t slot = 0; slot < active.size(); ++slot) {
            const auto& item = active[slot];
            if (!item || item->done || !item->session) continue;
            if (item->is_decision()) return std::nullopt;
            shapes.push_back(runtime_detail::RegimeRequestShape{
                static_cast<std::uint64_t>(item->prompt_tokens.size()),
                item->max_output_tokens(),
            });
            slots.push_back(slot);
        }
        if (shapes.empty()) return std::nullopt;

        const auto profile = runtime_detail::homogeneous_regime_profile(shapes);
        if (!profile) {
            const auto signature = std::string("heterogeneous:") + std::to_string(shapes.size());
            if (signature != last_regime_observation_signature) {
                last_regime_observation_signature = signature;
                event("regime_candidate_skipped", 0,
                      "reason=heterogeneous-request-shape width=" +
                      std::to_string(shapes.size()));
            }
            return std::nullopt;
        }

        RuntimeSnapshot snapshot;
        if (cuda) {
            if (const auto free = cuda->free_device_bytes()) {
                snapshot.free_device_memory_bytes = *free;
            }
            snapshot.prepared_artifact_bytes = cuda->prepared_artifact_device_bytes();
        }
        {
            std::lock_guard lock(mutex);
            snapshot.current_strategy_id = last_plan.strategy_id;
            std::uint64_t resident = 0U;
            for (const auto& item : active) {
                if (item && item->session && !item->done) {
                    resident += item->session->resources().committed_kv_bytes;
                }
            }
            snapshot.resident_kv_bytes = resident;
        }

        auto decision = runtime->decide(*profile, snapshot);
        auto candidate_plan = decision.plan;
        apply_plan_defaults(candidate_plan);
        decision.plan = candidate_plan;
        const auto* prepared = prepared_for(candidate_plan.backend);
        if (!prepared) return std::nullopt;
        const auto valid = validate_execution_plan(candidate_plan, prepared->capabilities());
        if (!valid) return std::nullopt;

        const auto key = runtime_detail::execution_regime_key(candidate_plan);
        const auto signature =
            std::to_string(profile->prompt_tokens) + ":" +
            std::to_string(profile->max_output_tokens) + ":" +
            std::to_string(profile->active_sequences) + ":" +
            candidate_plan.strategy_id + ":" +
            std::to_string(static_cast<int>(key.backend)) + ":" +
            std::to_string(static_cast<int>(key.prefill_block)) + ":" +
            std::to_string(static_cast<int>(key.decode_block)) + ":" +
            std::to_string(static_cast<int>(key.decode_output)) + ":" +
            std::to_string(static_cast<int>(key.prefill_attention)) + ":" +
            std::to_string(static_cast<int>(key.decode_attention)) + ":" +
            std::to_string(key.prefill_quantum_tokens);
        if (signature != last_regime_observation_signature) {
            last_regime_observation_signature = signature;
            event(decision.eligible_candidates == 0U
                      ? "regime_candidate_none"
                      : "regime_candidate",
                  0,
                  "width=" + std::to_string(profile->active_sequences) +
                  " prompt=" + std::to_string(profile->prompt_tokens) +
                  " output=" + std::to_string(profile->max_output_tokens) +
                  " strategy=" + candidate_plan.strategy_id +
                  " eligible=" + std::to_string(decision.eligible_candidates) +
                  " reason=" + decision.reason +
                  " block=" + std::string(to_string(candidate_plan.linear.prefill_block)) +
                  " attention=" + std::string(to_string(candidate_plan.attention.prefill)) +
                  " prepared_hot=" + (decision.prepared_state_hot ? "true" : "false"));
        }
        return GroupRegimeCandidate{*profile, std::move(candidate_plan),
                                    std::move(decision), std::move(slots),
                                    false, "qualified-homogeneous-group"};
    }

    [[nodiscard]] std::optional<runtime_detail::ExecutionRegimeKey>
    active_execution_regime() const {
        std::optional<runtime_detail::ExecutionRegimeKey> key;
        for (const auto& item : active) {
            if (!item || item->done || !item->session) continue;
            const auto candidate = runtime_detail::execution_regime_key(item->plan);
            if (!key) key = candidate;
            else if (!(*key == candidate)) return std::nullopt;
        }
        return key;
    }

    [[nodiscard]] std::vector<runtime_detail::RegimeRequestShape>
    active_regime_request_shapes() const {
        std::vector<runtime_detail::RegimeRequestShape> shapes;
        shapes.reserve(active.size());
        for (const auto& item : active) {
            if (!item || item->done || !item->session) continue;
            shapes.push_back(runtime_detail::RegimeRequestShape{
                static_cast<std::uint64_t>(item->prompt_tokens.size()),
                item->max_output_tokens(),
            });
        }
        return shapes;
    }

    [[nodiscard]] std::optional<ExecutionPlan> active_execution_plan() const {
        std::optional<ExecutionPlan> plan;
        std::optional<runtime_detail::ExecutionRegimeKey> key;
        for (const auto& item : active) {
            if (!item || item->done || !item->session) continue;
            const auto candidate = runtime_detail::execution_regime_key(item->plan);
            if (!key) {
                key = candidate;
                plan = item->plan;
            } else if (!(*key == candidate)) {
                return std::nullopt;
            }
        }
        return plan;
    }

    [[nodiscard]] GroupRegimeCandidate fallback_regime_candidate(
        std::string reason) const {
        GroupRegimeCandidate candidate;
        candidate.plan = fallback_plan;
        candidate.decision.plan = fallback_plan;
        candidate.decision.objective = "busy-epoch-fallback";
        candidate.decision.reason = reason;
        candidate.decision.eligible_candidates = 0U;
        candidate.safe_fallback = true;
        candidate.transition_reason = std::move(reason);
        for (std::size_t slot = 0; slot < active.size(); ++slot) {
            if (active[slot] && !active[slot]->done && active[slot]->session) {
                candidate.slots.push_back(slot);
            }
        }
        return candidate;
    }

    [[nodiscard]] Status converge_group_regime(const GroupRegimeCandidate& candidate) {
        if (candidate.slots.empty() ||
            (!candidate.safe_fallback && candidate.decision.eligible_candidates == 0U)) {
            return Status::ok();
        }
        auto* prepared = prepared_for(candidate.plan.backend);
        if (!prepared) return Status::unsupported("candidate regime backend is unavailable");
        if (!prepared->capabilities().sequence_checkpointing) {
            event("regime_transition_skipped", 0, "reason=checkpointing-unavailable");
            return Status::ok();
        }

        const auto target_key = runtime_detail::execution_regime_key(candidate.plan);
        bool transition_needed = false;
        for (const auto slot : candidate.slots) {
            if (slot >= active.size() || !active[slot] || active[slot]->done || !active[slot]->session) {
                event("regime_transition_skipped", 0, "reason=group-changed-before-transition");
                return Status::ok();
            }
            const auto& item = *active[slot];
            if (item.plan.backend != candidate.plan.backend ||
                item.plan.kv.page_tokens != candidate.plan.kv.page_tokens) {
                event("regime_transition_skipped", 0, "reason=incompatible-backend-or-kv-geometry");
                return Status::ok();
            }
            if (item.temperature() > 0.0 || item.is_decision()) {
                event("regime_transition_skipped", 0, "reason=stochastic-request-not-qualified");
                return Status::ok();
            }
            const auto requested_tokens = static_cast<std::uint64_t>(item.prompt_tokens.size()) +
                                          item.max_output_tokens();
            const auto target_reservation =
                prepared->estimate_sequence_device_bytes(candidate.plan, requested_tokens);
            if (target_reservation != item.admission_reservation_bytes) {
                event("regime_transition_skipped", 0, "reason=sequence-reservation-change");
                return Status::ok();
            }
            transition_needed = transition_needed ||
                !(runtime_detail::execution_regime_key(item.plan) == target_key);
        }
        if (!transition_needed) return Status::ok();

        const auto temporary_bytes = prepared->estimate_transition_temporary_device_bytes(
            candidate.plan, candidate.slots.size());
        auto admission = capacity_scheduler.evaluate_transition(
            *prepared, candidate.plan, temporary_bytes);
        if (!admission.admit) {
            event("regime_transition_blocked", 0,
                  "reason=" + admission.reason +
                  " strategy=" + candidate.plan.strategy_id +
                  " prepare_bytes=" + std::to_string(admission.plan_preparation_bytes) +
                  " temporary_bytes=" + std::to_string(admission.temporary_device_bytes));
            return Status::ok();
        }

        const auto started = Clock::now();
        const auto prepared_status = prepared->prepare_plan(candidate.plan);
        if (!prepared_status) {
            event("regime_transition_failed", 0,
                  "stage=prepare strategy=" + candidate.plan.strategy_id +
                  " reason=" + prepared_status.message());
            return Status::ok();
        }
        admission = capacity_scheduler.evaluate_transition(*prepared, candidate.plan, temporary_bytes);
        if (!admission.admit) {
            event("regime_transition_blocked", 0,
                  "reason=capacity-changed-after-prepare strategy=" + candidate.plan.strategy_id +
                  " temporary_bytes=" + std::to_string(temporary_bytes));
            return Status::ok();
        }

        std::vector<std::unique_ptr<runtime_detail::SequenceCheckpoint>> checkpoints;
        checkpoints.reserve(candidate.slots.size());
        for (const auto slot : candidate.slots) {
            auto checkpoint = active[slot]->session->checkpoint();
            if (!checkpoint) {
                event("regime_transition_failed", 0,
                      "stage=checkpoint strategy=" + candidate.plan.strategy_id +
                      " reason=" + checkpoint.status().message());
                return Status::ok();
            }
            checkpoints.push_back(std::move(checkpoint).value());
        }

        std::vector<std::unique_ptr<runtime_detail::SequenceState>> restored;
        restored.reserve(candidate.slots.size());
        for (const auto& checkpoint : checkpoints) {
            auto sequence = prepared->restore_sequence(candidate.plan, *checkpoint);
            if (!sequence) {
                event("regime_transition_failed", 0,
                      "stage=restore strategy=" + candidate.plan.strategy_id +
                      " reason=" + sequence.status().message());
                return Status::ok();
            }
            restored.push_back(std::move(sequence).value());
        }

        // Commit only after every checkpoint and restore succeeded. Until this
        // point every WorkItem still owns its original session and plan.
        for (std::size_t index = 0; index < candidate.slots.size(); ++index) {
            auto& item = *active[candidate.slots[index]];
            item.session = std::move(restored[index]);
            item.plan = candidate.plan;
            item.planning_decision = candidate.decision;
        }
        {
            std::lock_guard lock(mutex);
            last_plan = candidate.plan;
            last_decision = candidate.decision;
        }
        event("regime_transition_committed", 0,
              "epoch=" + std::to_string(regime_epoch_id) +
              " width=" + std::to_string(candidate.slots.size()) +
              " strategy=" + candidate.plan.strategy_id +
              " reason=" + candidate.transition_reason +
              " block=" + std::string(to_string(candidate.plan.linear.prefill_block)) +
              " decode_block=" + std::string(to_string(candidate.plan.linear.decode_block)) +
              " attention=" + std::string(to_string(candidate.plan.attention.prefill)) +
              " prepare_bytes=" + std::to_string(admission.plan_preparation_bytes) +
              " temporary_bytes=" + std::to_string(temporary_bytes) +
              " elapsed_ms=" + std::to_string(ms(Clock::now() - started)));
        return Status::ok();
    }

    [[nodiscard]] Result<std::string> decode_generated(std::span<const TokenId> tokens) const {
        return tokenizer->decode(tokens, false);
    }

    void fail_item(WorkItem& item, Status status) {
        item.failed = true;
        item.done = true;
        item.failure = std::move(status);
        item.finished_at = Clock::now();
    }

    [[nodiscard]] Status queue_stream_event(WorkItem& item, std::string delta, TokenId token) {
        if (!item.stream_enabled || delta.empty()) return Status::ok();
        {
            std::lock_guard lock(item.stream_mutex);
            if (item.stream_events.size() >= config.stream_queue_capacity) {
                return Status::cancelled("stream consumer exceeded bounded delivery queue");
            }
            item.stream_events.push_back(WorkItem::StreamEvent{std::move(delta), token});
        }
        item.stream_wake.notify_one();
        return Status::ok();
    }

    [[nodiscard]] Status emit_token(WorkItem& item, TokenId token) {
        item.generated.push_back(token);
        if (item.generated.size() == 1U) item.first_token_at = Clock::now();
        if (!item.stream_enabled) return Status::ok();
        const TokenId one[] = {token};
        auto decoded = tokenizer->decode(one, false);
        if (!decoded) return decoded.status();
        item.stream_pending_bytes += decoded.value();
        return queue_stream_event(item, drain_utf8(item.stream_pending_bytes, false), token);
    }

    [[nodiscard]] Status accept_token(WorkItem& item, TokenId token) {
        const auto emitted = emit_token(item, token);
        if (!emitted) return emitted;
        if (item.request.generation.stop_on_eos && model->tokenizer().special_ids.eos &&
            token == *model->tokenizer().special_ids.eos) {
            item.hit_eos = true;
            item.done = true;
            item.finished_at = Clock::now();
        } else if (item.generated.size() >= item.request.generation.max_new_tokens) {
            item.done = true;
            item.finished_at = Clock::now();
        }
        return Status::ok();
    }

    [[nodiscard]] Status sample_next(WorkItem& item) {
        if (!item.sampler) item.sampler = std::make_unique<Sampler>(item.request.generation.sampling);
        auto sampled = item.sampler->sample(item.logits);
        if (!sampled) return sampled.status();
        return accept_token(item, sampled.value());
    }

    [[nodiscard]] Status complete_decision(WorkItem& item) {
        if (item.decision_score_sums.size() != item.decision_candidate_tokens.size() ||
            item.decision_candidate_tokens.size() != item.decision_request.candidates.size()) {
            return Status::internal_error("decision score/candidate cardinality mismatch");
        }

        item.decision_raw_scores.resize(item.decision_score_sums.size());
        for (std::size_t i = 0; i < item.decision_score_sums.size(); ++i) {
            const auto count = item.decision_candidate_tokens[i].size();
            if (count == 0U) return Status::internal_error("decision candidate tokenization is empty");
            const auto sum = item.decision_score_sums[i];
            item.decision_raw_scores[i] =
                item.decision_request.scoring_policy == DecisionScoringPolicy::sequence_logprob_mean
                    ? sum / static_cast<double>(count)
                    : sum;
        }

        const auto maximum = *std::max_element(
            item.decision_raw_scores.begin(), item.decision_raw_scores.end());
        double normalizer = 0.0;
        for (const auto score : item.decision_raw_scores) {
            normalizer += std::exp(score - maximum);
        }
        if (!std::isfinite(normalizer) || normalizer <= 0.0) {
            return Status::data_error("decision candidate normalization is non-finite");
        }

        std::size_t best = 0U;
        item.decision_result.scores.clear();
        item.decision_result.scores.reserve(item.decision_raw_scores.size());
        for (std::size_t i = 0; i < item.decision_raw_scores.size(); ++i) {
            if (item.decision_raw_scores[i] > item.decision_raw_scores[best]) best = i;
            item.decision_result.scores.push_back(DecisionCandidateScore{
                item.decision_request.candidates[i].id,
                std::exp(item.decision_raw_scores[i] - maximum) / normalizer,
            });
        }
        item.decision_result.selected_candidate_ids = {
            item.decision_request.candidates[best].id
        };
        item.decision_result.applied_scoring_policy =
            item.decision_request.scoring_policy;
        item.decision_result.score_semantics =
            DecisionScoreSemantics::candidate_set_normalized;

        const auto valid =
            validate_decision_result(item.decision_request, item.decision_result);
        if (!valid) return valid;

        item.done = true;
        item.finished_at = Clock::now();
        return Status::ok();
    }

    [[nodiscard]] Status prepare_next_decision_branch(WorkItem& item) {
        auto* prepared = prepared_for(item.plan.backend);
        if (!prepared) return Status::unsupported("planned backend is unavailable");
        if (!item.decision_prefix_checkpoint) {
            return Status::invalid_state("decision prefix checkpoint is missing");
        }

        item.session.reset();
        item.decision_branch.reset();
        while (item.decision_candidate_index < item.decision_candidate_tokens.size()) {
            const auto& candidate =
                item.decision_candidate_tokens[item.decision_candidate_index];
            if (candidate.empty()) {
                return Status::internal_error("decision candidate tokenization is empty");
            }
            if (candidate.size() == 1U) {
                ++item.decision_candidate_index;
                continue;
            }
            auto branch =
                prepared->restore_sequence(item.plan, *item.decision_prefix_checkpoint);
            if (!branch) return branch.status();
            item.session = std::move(branch).value();
            item.decision_token_position = 1U;
            ++item.decision_branch_count;
            return Status::ok();
        }
        return complete_decision(item);
    }

    [[nodiscard]] Status initialize_decision_after_prefill(
        WorkItem& item,
        std::vector<float> first_logprobs) {
        if (first_logprobs.size() != item.decision_candidate_tokens.size()) {
            return Status::internal_error(
                "decision prefill returned the wrong target-logprob count");
        }
        if (!item.session) {
            return Status::invalid_state("decision shared-prefix session is missing");
        }

        auto checkpoint = item.session->checkpoint();
        if (!checkpoint) return checkpoint.status();
        item.decision_prefix_checkpoint = std::move(checkpoint).value();
        item.session.reset();

        item.decision_first_logprobs = std::move(first_logprobs);
        item.decision_score_sums.assign(
            item.decision_first_logprobs.begin(), item.decision_first_logprobs.end());
        item.decision_candidate_tokens_scored =
            static_cast<std::uint64_t>(item.decision_candidate_tokens.size());
        item.decision_candidate_index = 0U;
        item.decision_token_position = 0U;
        return prepare_next_decision_branch(item);
    }

    [[nodiscard]] bool advance_decision(WorkItem& item, std::uint32_t& budget) {
        if (!item.is_decision() || item.done || budget == 0U) return false;
        if (item.cancellation_requested()) {
            fail_item(item, Status::cancelled("decision cancelled during candidate scoring"));
            return true;
        }
        if (!item.session ||
            item.decision_candidate_index >= item.decision_candidate_tokens.size()) {
            const auto prepared = prepare_next_decision_branch(item);
            if (!prepared) fail_item(item, prepared);
            return true;
        }

        const auto& candidate =
            item.decision_candidate_tokens[item.decision_candidate_index];
        if (item.decision_token_position == 0U ||
            item.decision_token_position >= candidate.size()) {
            fail_item(item, Status::internal_error("decision branch token position is invalid"));
            return true;
        }

        const auto previous = candidate[item.decision_token_position - 1U];
        const auto target = candidate[item.decision_token_position];
        const std::array<TokenId, 1> targets{target};
        const auto compute_start = Clock::now();
        auto logprob =
            item.session->decode_target_logprobs(previous, targets);
        item.decode_compute_ms += ms(Clock::now() - compute_start);
        if (!logprob) {
            fail_item(item, logprob.status());
            return true;
        }
        if (logprob.value().size() != 1U ||
            !std::isfinite(static_cast<double>(logprob.value().front()))) {
            fail_item(item, Status::data_error(
                "decision target-logprob decode returned invalid data"));
            return true;
        }
        if (item.cancellation_requested()) {
            fail_item(item, Status::cancelled("decision cancelled during candidate scoring"));
            return true;
        }

        item.decision_score_sums[item.decision_candidate_index] +=
            static_cast<double>(logprob.value().front());
        ++item.decision_candidate_tokens_scored;
        ++item.decision_token_position;
        --budget;

        if (item.decision_token_position >= candidate.size()) {
            ++item.decision_candidate_index;
            item.session.reset();
            const auto prepared = prepare_next_decision_branch(item);
            if (!prepared) fail_item(item, prepared);
        }
        return true;
    }

    [[nodiscard]] Status initialize_item(WorkItem& item) {
        if (item.cancellation_requested()) return Status::cancelled("request cancelled before admission");
        if (!item.plan_ready) return Status::invalid_state("request must be planned before admission");
        auto* prepared = prepared_for(item.plan.backend);
        if (!prepared) return Status::unsupported("planned backend is unavailable");

        item.admitted_at = Clock::now();
        item.prefill_started_at = item.admitted_at;
        auto session_result = prepared->create_sequence(item.plan);
        if (!session_result) return session_result.status();
        item.session = std::move(session_result).value();
        {
            std::lock_guard lock(mutex);
            last_plan = item.plan;
            last_decision = item.planning_decision;
        }
        event("plan_selected", item.request_id,
              "mode=" + planner_mode + " strategy=" + item.plan.strategy_id +
              " backend=" + std::string(to_string(item.plan.backend)) +
              " prefill_quantum=" + std::to_string(item.plan.scheduling.prefill_quantum_tokens) +
              " kv_page=" + (item.plan.kv.page_tokens ? std::to_string(*item.plan.kv.page_tokens) : "none") +
              " objective=" + item.planning_decision.objective +
              " reason=" + item.planning_decision.reason +
              " transition_ms=" + std::to_string(item.planning_decision.estimated_transition_ms) +
              " break_even_tokens=" + std::to_string(item.planning_decision.estimated_break_even_tokens));

        const auto& capabilities = prepared->capabilities();
        if (!item.is_decision() && capabilities.exact_prefix_reuse && capabilities.sequence_checkpointing) {
            const auto compatibility =
                runtime_detail::make_sequence_state_compatibility(prepared->model(), item.plan);
            auto match = sequence_state_store.longest(item.prompt_tokens, compatibility);
            if (match) {
                auto restored = prepared->restore_sequence(item.plan, *match->checkpoint);
                if (restored) {
                    item.prefix_reused_tokens = match->tokens;
                    item.prompt_position = static_cast<std::size_t>(match->tokens);
                    item.logits = std::move(match->logits);
                    item.session = std::move(restored).value();
                    event("prefix_hit", item.request_id,
                          "reused_tokens=" + std::to_string(item.prefix_reused_tokens));
                }
            }
        }
        if (!item.is_decision() && item.prompt_position == item.prompt_tokens.size() && !item.logits.empty()) {
            item.prefill_finished_at = Clock::now();
            if (item.request.generation.max_new_tokens == 0U) {
                item.done = true;
                item.finished_at = item.prefill_finished_at;
            } else {
                const auto sampled = sample_next(item);
                if (!sampled) return sampled;
            }
        }
        return Status::ok();
    }

    void maybe_store_prefix(WorkItem& item) {
        if (item.is_decision()) return;
        if (!item.session || item.prompt_position == 0U || item.logits.empty()) return;
        const auto* prepared = prepared_for(item.plan.backend);
        if (!prepared) return;
        const auto& capabilities = prepared->capabilities();
        if (!capabilities.exact_prefix_reuse || !capabilities.sequence_checkpointing) return;

        const bool page_boundary = item.plan.kv.page_tokens &&
            item.prompt_position % *item.plan.kv.page_tokens == 0U;
        const bool full_prompt = item.prompt_position == item.prompt_tokens.size();
        if (!page_boundary && !full_prompt) return;

        auto checkpoint = item.session->checkpoint();
        if (!checkpoint) return;
        const auto compatibility =
            runtime_detail::make_sequence_state_compatibility(prepared->model(), item.plan);
        const auto stored = sequence_state_store.insert(
            std::span<const TokenId>(item.prompt_tokens.data(), item.prompt_position),
            compatibility,
            std::move(checkpoint).value(),
            item.logits);
        if (!stored) {
            event("sequence_state_store_reject", item.request_id,
                  "code=" + std::to_string(static_cast<int>(stored.code())) +
                  " reason=" + stored.message());
        }
    }

    [[nodiscard]] bool advance_prefill(WorkItem& item, std::uint32_t& budget) {
        if (item.cancellation_requested()) {
            fail_item(item, Status::cancelled(
                item.is_decision() ? "decision cancelled during prefill"
                                   : "request cancelled during prefill"));
            return true;
        }
        if (item.prompt_position >= item.prompt_tokens.size() || budget == 0U) return false;

        const auto remaining = item.prompt_tokens.size() - item.prompt_position;
        std::size_t chunk = std::min<std::size_t>({
            remaining, item.plan.scheduling.prefill_quantum_tokens, budget});
        const auto* prepared = prepared_for(item.plan.backend);
        if (!prepared) {
            fail_item(item, Status::unsupported("planned backend is unavailable"));
            return true;
        }
        const auto& capabilities = prepared->capabilities();
        if (capabilities.prefill_execution == PrefillExecutionKind::native_batch) {
            chunk = std::min<std::size_t>(chunk, capabilities.max_prefill_batch_width);
        }

        const bool final_prompt_chunk =
            item.prompt_position + chunk == item.prompt_tokens.size();
        const bool output_token_needed =
            !item.is_decision() && final_prompt_chunk &&
            item.request.generation.max_new_tokens != 0U;
        const bool device_greedy =
            output_token_needed && capabilities.device_greedy_selection &&
            item.request.generation.sampling.temperature <= 0.0;
        const bool need_intermediate_logits =
            !item.is_decision() && capabilities.exact_prefix_reuse;

        const auto tokens = std::span<const TokenId>(
            item.prompt_tokens.data() +
                static_cast<std::ptrdiff_t>(item.prompt_position),
            chunk);

        const auto compute_start = Clock::now();
        Status execution_status = Status::ok();
        std::optional<TokenId> selected;
        std::optional<std::vector<float>> logits;
        std::optional<std::vector<float>> first_logprobs;

        if (item.is_decision() && final_prompt_chunk) {
            std::vector<TokenId> targets;
            targets.reserve(item.decision_candidate_tokens.size());
            for (const auto& candidate : item.decision_candidate_tokens) {
                if (candidate.empty()) {
                    execution_status =
                        Status::invalid_argument("decision candidate produces no tokens");
                    break;
                }
                targets.push_back(candidate.front());
            }
            if (execution_status) {
                auto result =
                    item.session->prefill_target_logprobs(tokens, targets);
                if (!result) execution_status = result.status();
                else first_logprobs = std::move(result).value();
            }
        } else if (device_greedy) {
            auto result = item.session->prefill_greedy(tokens);
            if (!result) execution_status = result.status();
            else selected = result.value();
        } else if ((!final_prompt_chunk || !output_token_needed) &&
                   !need_intermediate_logits) {
            execution_status = item.session->prefill_discard(tokens);
        } else {
            auto result = item.session->prefill(tokens);
            if (!result) execution_status = result.status();
            else logits = std::move(result).value();
        }

        item.prefill_compute_ms += ms(Clock::now() - compute_start);
        if (!execution_status) {
            fail_item(item, execution_status);
            return true;
        }
        if (item.cancellation_requested()) {
            fail_item(item, Status::cancelled(
                item.is_decision() ? "decision cancelled during prefill"
                                   : "request cancelled during prefill"));
            return true;
        }

        if (logits) item.logits = std::move(*logits);
        item.prompt_position += chunk;
        budget -= static_cast<std::uint32_t>(chunk);
        maybe_store_prefix(item);

        if (item.prompt_position == item.prompt_tokens.size()) {
            item.prefill_finished_at = Clock::now();

            if (item.is_decision()) {
                if (!first_logprobs) {
                    fail_item(item, Status::internal_error(
                        "decision final prefill omitted target log probabilities"));
                    return true;
                }
                const auto initialized =
                    initialize_decision_after_prefill(
                        item, std::move(*first_logprobs));
                if (!initialized) fail_item(item, initialized);
            } else if (item.request.generation.max_new_tokens == 0U) {
                item.done = true;
                item.finished_at = item.prefill_finished_at;
            } else if (selected) {
                const auto accepted = accept_token(item, *selected);
                if (!accepted) fail_item(item, accepted);
            } else {
                const auto sampled = sample_next(item);
                if (!sampled) fail_item(item, sampled);
            }
        }
        return true;
    }

    [[nodiscard]] bool try_advance_prefill_batch(
        std::span<WorkItem*> items,
        std::span<const std::uint32_t> token_counts) {
        if (items.size() < 2U || items.size() != token_counts.size()) return false;
        WorkItem& first = *items.front();
        auto* prepared = prepared_for(first.plan.backend);
        if (!prepared ||
            prepared->capabilities().prefill_execution != PrefillExecutionKind::native_batch ||
            prepared->capabilities().exact_prefix_reuse) {
            return false;
        }

        std::vector<runtime_detail::PrefillBatchItem> batch;
        batch.reserve(items.size());
        for (std::size_t i = 0; i < items.size(); ++i) {
            auto* item = items[i];
            if (!item || item->done || item->failed || !item->session ||
                item->is_decision() || item->cancellation_requested() ||
                item->prompt_position >= item->prompt_tokens.size() ||
                token_counts[i] == 0U ||
                item->temperature() > 0.0 ||
                item->plan.backend != first.plan.backend ||
                item->plan.linear.prefill_block != first.plan.linear.prefill_block ||
                item->plan.attention.prefill != first.plan.attention.prefill ||
                item->plan.kv.page_tokens != first.plan.kv.page_tokens) {
                return false;
            }

            const auto remaining =
                item->prompt_tokens.size() - item->prompt_position;
            if (token_counts[i] > remaining) return false;
            const bool final_prompt_chunk =
                item->prompt_position + token_counts[i] ==
                item->prompt_tokens.size();
            const bool output_token_needed =
                final_prompt_chunk &&
                item->request.generation.max_new_tokens != 0U;
            const auto output = output_token_needed
                ? runtime_detail::PrefillBatchOutput::greedy
                : runtime_detail::PrefillBatchOutput::discard;
            const auto tokens = std::span<const TokenId>(
                item->prompt_tokens.data() +
                    static_cast<std::ptrdiff_t>(item->prompt_position),
                token_counts[i]);
            batch.push_back(runtime_detail::PrefillBatchItem{
                item->session.get(), tokens, output});
        }

        const auto compute_start = Clock::now();
        auto executed = prepared->prefill_batch(batch);
        const double elapsed = ms(Clock::now() - compute_start);
        if (!executed) {
            if (executed.status().code() == ErrorCode::unsupported) {
                return false;
            }
            for (auto* item : items) fail_item(*item, executed.status());
            return true;
        }
        if (executed.value().items.size() != items.size()) {
            const auto failure = Status::internal_error(
                "native prefill batch returned the wrong result count");
            for (auto* item : items) fail_item(*item, failure);
            return true;
        }

        {
            std::lock_guard lock(mutex);
            physical_prefill_batches += executed.value().physical_batches;
            physical_prefill_sequence_participations +=
                executed.value().physical_sequence_participations;
            physical_prefill_tokens += executed.value().physical_tokens;
            physical_prefill_max_sequences = std::max(
                physical_prefill_max_sequences,
                executed.value().max_sequences);
        }
        event("prefill_batch", 0,
              "backend=" + std::string(to_string(first.plan.backend)) +
              " logical_sequences=" + std::to_string(items.size()) +
              " physical_batches=" +
                  std::to_string(executed.value().physical_batches) +
              " sequence_participations=" +
                  std::to_string(
                      executed.value().physical_sequence_participations) +
              " tokens=" +
                  std::to_string(executed.value().physical_tokens) +
              " max_sequences=" +
                  std::to_string(executed.value().max_sequences) +
              " block_linear=" +
                  std::string(to_string(first.plan.linear.prefill_block)) +
              " attention=" +
                  std::string(to_string(first.plan.attention.prefill)));

        for (std::size_t i = 0; i < items.size(); ++i) {
            auto& item = *items[i];
            item.prefill_compute_ms += elapsed;
            if (item.cancellation_requested()) {
                fail_item(item,
                          Status::cancelled(
                              "request cancelled during batched prefill"));
                continue;
            }

            item.prompt_position += token_counts[i];
            maybe_store_prefix(item);
            if (item.prompt_position != item.prompt_tokens.size()) continue;

            item.prefill_finished_at = Clock::now();
            if (item.request.generation.max_new_tokens == 0U) {
                item.done = true;
                item.finished_at = item.prefill_finished_at;
                continue;
            }

            const auto selected =
                executed.value().items[i].greedy_token;
            if (!selected) {
                fail_item(item, Status::internal_error(
                    "greedy native prefill batch omitted final token"));
                continue;
            }
            const auto accepted = accept_token(item, *selected);
            if (!accepted) fail_item(item, accepted);
        }
        return true;
    }

    [[nodiscard]] bool advance_decode(WorkItem& item, std::uint32_t& budget) {
        if (item.is_decision()) return advance_decision(item, budget);
        if (item.cancellation_requested()) {
            fail_item(item, Status::cancelled("request cancelled during decode"));
            return true;
        }
        if (item.done || item.prompt_position < item.prompt_tokens.size() || budget == 0U) return false;
        if (item.generated.empty()) return false;
        const auto* prepared = prepared_for(item.plan.backend);
        if (!prepared) {
            fail_item(item, Status::unsupported("planned backend is unavailable"));
            return true;
        }
        const auto previous = item.generated.back();
        const bool device_greedy = prepared->capabilities().device_greedy_selection &&
            item.request.generation.sampling.temperature <= 0.0;
        const auto compute_start = Clock::now();
        if (device_greedy) {
            auto token = item.session->decode_greedy(previous);
            item.decode_compute_ms += ms(Clock::now() - compute_start);
            if (!token) {
                fail_item(item, token.status());
                return true;
            }
            if (item.cancellation_requested()) {
                fail_item(item, Status::cancelled("request cancelled during decode"));
                return true;
            }
            --budget;
            const auto accepted = accept_token(item, token.value());
            if (!accepted) fail_item(item, accepted);
            return true;
        }

        auto logits = item.session->decode(previous);
        item.decode_compute_ms += ms(Clock::now() - compute_start);
        if (!logits) {
            fail_item(item, logits.status());
            return true;
        }
        if (item.cancellation_requested()) {
            fail_item(item, Status::cancelled("request cancelled during decode"));
            return true;
        }
        item.logits = std::move(logits).value();
        --budget;
        const auto sampled = sample_next(item);
        if (!sampled) fail_item(item, sampled);
        return true;
    }

    [[nodiscard]] bool advance_decode_batch(std::span<WorkItem*> items, std::uint32_t& budget) {
        if (items.size() < 2U || items.size() > budget) return false;
        WorkItem& first = *items.front();
        auto* prepared = prepared_for(first.plan.backend);
        if (!prepared || prepared->capabilities().max_decode_batch_width < items.size()) return false;

        std::vector<runtime_detail::GreedyDecodeBatchItem> batch;
        batch.reserve(items.size());
        for (auto* item : items) {
            if (!item || item->done || item->failed || item->is_decision() || !item->session || item->generated.empty() ||
                item->prompt_position < item->prompt_tokens.size()) return false;
            if (item->cancellation_requested()) return false;
            if (item->plan.backend != first.plan.backend ||
                item->plan.linear.decode_block != first.plan.linear.decode_block ||
                item->plan.linear.decode_output != first.plan.linear.decode_output ||
                item->plan.attention.decode != first.plan.attention.decode ||
                item->temperature() > 0.0) return false;
            batch.push_back(runtime_detail::GreedyDecodeBatchItem{item->session.get(), item->generated.back()});
        }

        const auto compute_start = Clock::now();
        auto selected = prepared->decode_greedy_batch(batch);
        const double elapsed = ms(Clock::now() - compute_start);
        if (!selected) {
            for (auto* item : items) fail_item(*item, selected.status());
            return true;
        }
        if (selected.value().size() != items.size()) {
            for (auto* item : items) {
                fail_item(*item, Status::internal_error("native decode batch returned the wrong token count"));
            }
            return true;
        }

        budget -= static_cast<std::uint32_t>(items.size());
        {
            std::lock_guard lock(mutex);
            ++native_decode_batches;
            native_decode_sequences += static_cast<std::uint64_t>(items.size());
        }
        event("decode_batch", 0, "backend=" + std::string(to_string(first.plan.backend)) +
              " width=" + std::to_string(items.size()) +
              " block_linear=" + std::string(to_string(first.plan.linear.decode_block)) +
              " output_linear=" + std::string(to_string(first.plan.linear.decode_output)));
        for (std::size_t index = 0; index < items.size(); ++index) {
            auto& item = *items[index];
            item.decode_compute_ms += elapsed;
            if (item.cancellation_requested()) {
                fail_item(item, Status::cancelled("request cancelled during native decode batch"));
                continue;
            }
            const auto accepted = accept_token(item, selected.value()[index]);
            if (!accepted) fail_item(item, accepted);
        }
        return true;
    }

    void finish_item(const std::shared_ptr<WorkItem>& item) {
        InferenceResponse generation_response;
        DecisionResponse decision_response;

        if (!item->failed && !item->is_decision()) {
            auto decoded = decode_generated(item->generated);
            if (!decoded) {
                item->failed = true;
                item->failure = decoded.status();
            } else {
                generation_response.text =
                    sanitize_utf8(std::move(decoded).value());
                if (item->stream_enabled &&
                    !item->stream_pending_bytes.empty()) {
                    const auto final_delta =
                        drain_utf8(item->stream_pending_bytes, true);
                    const auto stream_queued = queue_stream_event(
                        *item, final_delta, static_cast<TokenId>(-1));
                    if (!stream_queued) {
                        item->failed = true;
                        item->failure = stream_queued;
                    }
                }
            }
        }

        const auto finished =
            item->finished_at == Clock::time_point{}
                ? Clock::now()
                : item->finished_at;

        RequestMetrics metrics;
        metrics.request_id = item->request_id;
        metrics.sequence_id = item->sequence_id;
        metrics.backend = to_string(item->plan.backend);
        metrics.workload = item->is_decision() ? "decision" : "generation";
        metrics.planner_mode = planner_mode;
        metrics.strategy_id = item->plan.strategy_id;
        metrics.prompt_tokens = item->prompt_tokens.size();
        metrics.generated_tokens =
            item->is_decision() ? 0U : item->generated.size();
        metrics.prefix_reused_tokens = item->prefix_reused_tokens;

        if (item->session) {
            metrics.kv_bytes = item->session->resources().committed_kv_bytes;
        } else if (item->decision_branch) {
            metrics.kv_bytes =
                item->decision_branch->resources().committed_kv_bytes;
        } else if (item->decision_prefix_checkpoint) {
            metrics.kv_bytes =
                item->decision_prefix_checkpoint->resources().committed_kv_bytes;
        }

        metrics.plan_preparation_bytes = item->plan_preparation_bytes;
        metrics.strategy_objective = item->planning_decision.objective;
        metrics.strategy_decision_reason =
            item->planning_decision.reason;
        metrics.strategy_eligible_candidates =
            item->planning_decision.eligible_candidates;
        metrics.strategy_prepared_state_hot =
            item->planning_decision.prepared_state_hot;
        metrics.strategy_estimated_transition_ms =
            item->planning_decision.estimated_transition_ms;
        metrics.strategy_estimated_break_even_tokens =
            item->planning_decision.estimated_break_even_tokens;
        metrics.strategy_candidates = item->planning_decision.candidates;
        metrics.plan_preparation_ms = item->plan_preparation_ms;
        metrics.plan_eviction_ms = item->plan_eviction_ms;
        metrics.queue_ms = ms(item->admitted_at - item->submitted_at);
        if (item->prefill_finished_at != Clock::time_point{}) {
            metrics.prefill_ms = item->prefill_compute_ms;
        }
        if (item->is_decision()) {
            metrics.decode_ms = item->decode_compute_ms;
        } else if (item->first_token_at != Clock::time_point{}) {
            metrics.ttft_ms =
                ms(item->first_token_at - item->submitted_at);
            metrics.decode_ms = item->decode_compute_ms;
        }
        metrics.total_ms = ms(finished - item->submitted_at);

        if (!item->is_decision()) {
            generation_response.tokens = item->generated;
            generation_response.hit_eos = item->hit_eos;
            generation_response.metrics = metrics;
        } else {
            decision_response.decision = item->decision_result;
            decision_response.metrics = metrics;
            decision_response.candidate_tokens_scored =
                item->decision_candidate_tokens_scored;
            decision_response.branch_count =
                item->decision_branch_count;
            decision_response.calibrated = false;
            decision_response.abstention_qualified = false;
        }

        {
            std::lock_guard lock(mutex);
            if (item->failed) {
                if (item->failure.code() == ErrorCode::cancelled) {
                    ++cancelled;
                } else {
                    ++failed;
                }
            } else {
                ++completed;
                if (item->is_decision()) ++completed_decisions;
                prompt_tokens += metrics.prompt_tokens;
                generated_tokens += metrics.generated_tokens;
                prefix_reused_tokens += metrics.prefix_reused_tokens;
                peak_kv_bytes = std::max(peak_kv_bytes, metrics.kv_bytes);
                recent_latency.push_back(metrics.total_ms);
                while (recent_latency.size() > config.latency_window) {
                    recent_latency.pop_front();
                }
            }
        }

        // All live/branch state is released before the caller observes
        // completion. Backend pools may retain free pages by policy, but the
        // completed work item keeps no page references or admission ownership.
        item->session.reset();
        item->decision_branch.reset();
        item->decision_prefix_checkpoint.reset();
        item->sampler.reset();
        item->logits.clear();
        item->logits.shrink_to_fit();

        {
            std::lock_guard lock(item->stream_mutex);
            item->stream_closed = true;
        }
        item->stream_wake.notify_all();

        if (item->failed) {
            event(
                item->failure.code() == ErrorCode::cancelled
                    ? (item->is_decision()
                           ? "decision_cancelled"
                           : "request_cancelled")
                    : (item->is_decision()
                           ? "decision_failed"
                           : "request_failed"),
                item->request_id,
                item->failure.message());
            if (item->is_decision()) {
                item->decision_promise.set_value(item->failure);
            } else {
                item->promise.set_value(item->failure);
            }
            return;
        }

        if (item->is_decision()) {
            event(
                "decision_complete",
                item->request_id,
                "prompt=" + std::to_string(metrics.prompt_tokens) +
                    " candidates=" +
                    std::to_string(item->decision_request.candidates.size()) +
                    " candidate_tokens=" +
                    std::to_string(
                        item->decision_candidate_tokens_scored) +
                    " scoring=" +
                    std::string(to_string(
                        item->decision_request.scoring_policy)));
            item->decision_promise.set_value(
                std::move(decision_response));
        } else {
            event(
                "request_complete",
                item->request_id,
                "prompt=" + std::to_string(metrics.prompt_tokens) +
                    " generated=" +
                    std::to_string(metrics.generated_tokens));
            item->promise.set_value(
                std::move(generation_response));
        }
    }

    void run() {
        event("scheduler_start", 0, "backend=" + backend);
        while (true) {
            {
                std::unique_lock lock(mutex);
                wake.wait(lock, [this] { return stopping || !queued.empty() || !active.empty(); });
                if (stopping && queued.empty() && active.empty()) break;
            }

            std::uint64_t cycle_queue_generation = 0U;
            {
                std::lock_guard lock(mutex);
                cycle_queue_generation = queue_generation;
            }

            // Capacity scheduling owns admission. Planning happens before
            // sequence allocation so backend resource requirements can be
            // considered without partially admitting a request.
            while (true) {
                std::shared_ptr<WorkItem> item;
                std::size_t active_count = 0U;
                {
                    std::lock_guard lock(mutex);
                    if (queued.empty()) break;
                    item = queued.front();
                    active_count = active.size();
                }

                if (item->cancellation_requested()) {
                    {
                        std::lock_guard lock(mutex);
                        if (!queued.empty() && queued.front() == item) queued.pop_front();
                    }
                    item->admitted_at = Clock::now();
                    fail_item(*item, Status::cancelled("request cancelled while queued"));
                    finish_item(item);
                    continue;
                }

                const auto planned = plan_item(*item);
                if (!planned) {
                    {
                        std::lock_guard lock(mutex);
                        if (!queued.empty() && queued.front() == item) queued.pop_front();
                    }
                    item->admitted_at = Clock::now();
                    fail_item(*item, planned);
                    finish_item(item);
                    continue;
                }

                // A busy epoch has one physical execution regime. Compatible
                // arrivals join it directly. The first incompatible arrival
                // converges all live sessions to the configured safe fallback
                // regime, which remains pinned until the epoch drains. If that
                // convergence cannot be made atomically/resource-truthfully,
                // leave the request queued rather than fragmenting the epoch.
                if (planner_mode == "adaptive" && active_count != 0U) {
                    const auto fallback_key =
                        runtime_detail::execution_regime_key(fallback_plan);
                    auto current_key = active_execution_regime();
                    auto requested_key =
                        runtime_detail::execution_regime_key(item->plan);

                    const auto active_shapes = active_regime_request_shapes();
                    const runtime_detail::RegimeRequestShape incoming_shape{
                        static_cast<std::uint64_t>(item->prompt_tokens.size()),
                        item->max_output_tokens(),
                    };
                    const bool same_homogeneous_shape =
                        !regime_epoch_fallback_pinned &&
                        runtime_detail::homogeneous_regime_shape_accepts(
                            active_shapes, incoming_shape);
                    if (same_homogeneous_shape && current_key &&
                        !(*current_key == requested_key)) {
                        if (const auto current_plan = active_execution_plan()) {
                            event("regime_admission_shape_compatible",
                                  item->request_id,
                                  "epoch=" + std::to_string(regime_epoch_id) +
                                  " requested_strategy=" + item->plan.strategy_id +
                                  " current_strategy=" + current_plan->strategy_id +
                                  " prompt=" + std::to_string(incoming_shape.prompt_tokens) +
                                  " output=" + std::to_string(incoming_shape.max_output_tokens));
                            item->plan = *current_plan;
                            item->planning_decision.plan = item->plan;
                            item->planning_decision.reason =
                                "busy-epoch-shape-compatible-current-regime";
                            item->planning_decision.prepared_state_hot = false;
                            item->planning_decision.estimated_transition_ms = 0.0;
                            item->planning_decision.estimated_break_even_tokens = 0.0;
                            requested_key =
                                runtime_detail::execution_regime_key(item->plan);
                        }
                    }

                    const auto admission_action =
                        runtime_detail::classify_regime_admission(
                            current_key, requested_key, regime_epoch_fallback_pinned);

                    if (admission_action !=
                        runtime_detail::RegimeAdmissionAction::compatible) {
                        if (admission_action ==
                            runtime_detail::RegimeAdmissionAction::converge_to_fallback) {
                            auto fallback_candidate = fallback_regime_candidate(
                                "incompatible-arrival-fallback");
                            const auto converged =
                                converge_group_regime(fallback_candidate);
                            if (!converged) {
                                event("regime_admission_wait", item->request_id,
                                      "epoch=" + std::to_string(regime_epoch_id) +
                                      " reason=fallback-transition-internal-error detail=" +
                                      converged.message());
                                break;
                            }
                            current_key = active_execution_regime();
                            if (!current_key || !(*current_key == fallback_key)) {
                                event("regime_admission_wait", item->request_id,
                                      "epoch=" + std::to_string(regime_epoch_id) +
                                      " reason=fallback-transition-unavailable");
                                break;
                            }
                            regime_epoch_fallback_pinned = true;
                            event("regime_epoch_fallback_pinned", item->request_id,
                                  "epoch=" + std::to_string(regime_epoch_id) +
                                  " requested_strategy=" + item->plan.strategy_id +
                                  " fallback_block=" +
                                  std::string(to_string(fallback_plan.linear.prefill_block)) +
                                  " active=" + std::to_string(active_count));
                        }

                        item->plan = fallback_plan;
                        item->planning_decision.plan = fallback_plan;
                        item->planning_decision.reason =
                            "busy-epoch-fallback-pinned";
                        item->planning_decision.prepared_state_hot = false;
                        item->planning_decision.estimated_transition_ms = 0.0;
                        item->planning_decision.estimated_break_even_tokens = 0.0;
                    } else {
                        event("regime_admission_compatible", item->request_id,
                              "epoch=" + std::to_string(regime_epoch_id) +
                              " strategy=" + item->plan.strategy_id);
                    }
                }

                auto* prepared = prepared_for(item->plan.backend);
                if (!prepared) {
                    {
                        std::lock_guard lock(mutex);
                        if (!queued.empty() && queued.front() == item) queued.pop_front();
                    }
                    item->admitted_at = Clock::now();
                    fail_item(*item, Status::unsupported("planned backend is unavailable"));
                    finish_item(item);
                    continue;
                }

                // Optional backend-global prepared artifacts are part of capacity
                // admission. When the backend is idle, first reconcile away any
                // tactic artifacts the incoming plan does not require. This keeps
                // low-VRAM and high-throughput strategies genuinely reversible.
                if (active_count == 0U) {
                    const auto trim_started = Clock::now();
                    const auto trimmed = prepared->trim_plan_artifacts(item->plan);
                    item->plan_eviction_ms += ms(Clock::now() - trim_started);
                    if (!trimmed) {
                        {
                            std::lock_guard lock(mutex);
                            if (!queued.empty() && queued.front() == item) queued.pop_front();
                        }
                        item->admitted_at = Clock::now();
                        fail_item(*item, trimmed);
                        finish_item(item);
                        continue;
                    }
                }

                std::uint64_t active_reserved = 0U;
                {
                    std::lock_guard lock(mutex);
                    active_reserved = active_admission_reserved_bytes;
                }
                const std::uint64_t requested_tokens =
                    static_cast<std::uint64_t>(item->prompt_tokens.size()) +
                    item->max_output_tokens();
                const auto preparation_bytes =
                    prepared->estimate_plan_preparation_device_bytes(item->plan);
                item->plan_preparation_bytes = preparation_bytes;
                auto decision = capacity_scheduler.evaluate(
                    active_count, active_reserved, *prepared, item->plan,
                    requested_tokens, preparation_bytes);
                if (!decision.admit) {
                    if ((decision.reason == "device-capacity" ||
                         decision.reason == "device-capacity-plan-artifact") &&
                        active_count == 0U) {
                        {
                            std::lock_guard lock(mutex);
                            if (!queued.empty() && queued.front() == item) queued.pop_front();
                        }
                        item->admitted_at = Clock::now();
                        fail_item(*item, Status::invalid_state(
                            decision.reason == "device-capacity-plan-artifact"
                                ? "insufficient free device memory for execution-plan preparation"
                                : "insufficient free device memory for one sequence reservation"));
                        finish_item(item);
                        continue;
                    }
                    break;
                }

                const auto plan_prepare_started = Clock::now();
                const auto prepared_plan = prepared->prepare_plan(item->plan);
                item->plan_preparation_ms += ms(Clock::now() - plan_prepare_started);
                if (!prepared_plan) {
                    {
                        std::lock_guard lock(mutex);
                        if (!queued.empty() && queued.front() == item) queued.pop_front();
                    }
                    item->admitted_at = Clock::now();
                    fail_item(*item, prepared_plan);
                    finish_item(item);
                    continue;
                }

                // Re-check after materialization so external device-memory changes
                // between the forecast and allocation cannot create an overcommit.
                decision = capacity_scheduler.evaluate(
                    active_count, active_reserved, *prepared, item->plan, requested_tokens, 0U);
                if (!decision.admit) {
                    if (active_count == 0U) {
                        {
                            std::lock_guard lock(mutex);
                            if (!queued.empty() && queued.front() == item) queued.pop_front();
                        }
                        item->admitted_at = Clock::now();
                        fail_item(*item, Status::invalid_state(
                            "device capacity changed during execution-plan preparation"));
                        finish_item(item);
                        continue;
                    }
                    break;
                }

                std::size_t admitted_count = 0U;
                {
                    std::lock_guard lock(mutex);
                    if (queued.empty() || queued.front() != item) continue;
                    queued.pop_front();
                    item->admission_reservation_bytes = decision.reservation_bytes;
                    active_admission_reserved_bytes += decision.reservation_bytes;
                    active.push_back(item);
                    admitted_count = active.size();
                }
                if (!regime_epoch_open) {
                    regime_epoch_open = true;
                    regime_epoch_fallback_pinned = false;
                    ++regime_epoch_id;
                    event("regime_epoch_start", item->request_id,
                          "epoch=" + std::to_string(regime_epoch_id));
                }
                const auto initialized = initialize_item(*item);
                if (!initialized) fail_item(*item, initialized);
                else event("request_admitted", item->request_id,
                           "active=" + std::to_string(admitted_count) +
                           " epoch=" + std::to_string(regime_epoch_id));
            }

            bool progressed = false;
            for (auto& item : active) {
                if (!item->done && item->cancellation_requested()) {
                    fail_item(*item, Status::cancelled("request cancelled while active"));
                    progressed = true;
                }
            }

            if (auto regime = group_regime_candidate()) {
                if (regime_epoch_fallback_pinned) {
                    const auto fallback_key =
                        runtime_detail::execution_regime_key(fallback_plan);
                    const auto candidate_key =
                        runtime_detail::execution_regime_key(regime->plan);
                    if (regime->decision.eligible_candidates != 0U &&
                        !(candidate_key == fallback_key)) {
                        event("regime_reoptimize_shadow", 0,
                              "epoch=" + std::to_string(regime_epoch_id) +
                              " width=" + std::to_string(regime->slots.size()) +
                              " strategy=" + regime->plan.strategy_id +
                              " reason=fallback-pinned-until-epoch-drain");
                    }
                } else {
                    const auto converged = converge_group_regime(*regime);
                    if (!converged) {
                        event("regime_transition_failed", 0,
                              "stage=internal reason=" + converged.message());
                    }
                }
            }

            std::vector<runtime_detail::SchedulableSequence> candidates;
            candidates.reserve(active.size());
            for (std::size_t slot = 0; slot < active.size(); ++slot) {
                const auto& item = active[slot];
                if (item->done) continue;
                const bool is_prefill = item->prompt_position < item->prompt_tokens.size();
                std::uint32_t max_tokens = 1U;
                if (is_prefill) {
                    const auto remaining = item->prompt_tokens.size() - item->prompt_position;
                    std::size_t bounded = std::min<std::size_t>(
                        remaining, item->plan.scheduling.prefill_quantum_tokens);
                    if (const auto* prepared = prepared_for(item->plan.backend);
                        prepared && prepared->capabilities().prefill_execution ==
                            PrefillExecutionKind::native_batch) {
                        bounded = std::min<std::size_t>(
                            bounded, prepared->capabilities().max_prefill_batch_width);
                    }
                    max_tokens = static_cast<std::uint32_t>(
                        std::min<std::size_t>(bounded, std::numeric_limits<std::uint32_t>::max()));
                }
                candidates.push_back(runtime_detail::SchedulableSequence{
                    slot,
                    is_prefill ? runtime_detail::WorkPhase::prefill
                               : runtime_detail::WorkPhase::decode,
                    max_tokens});
            }

            const auto scheduled_batch =
                microbatch_scheduler.schedule(candidates, config.token_budget_per_cycle);
            bool yield_for_new_admission = false;
            for (std::size_t schedule_index = 0;
                 schedule_index < scheduled_batch.slices.size();) {
                const auto& slice = scheduled_batch.slices[schedule_index];
                auto& item = active[slice.slot];
                if (item->done) { ++schedule_index; continue; }

                if (slice.phase == runtime_detail::WorkPhase::prefill) {
                    std::vector<WorkItem*> prefill_items;
                    std::vector<std::uint32_t> prefill_counts;
                    std::size_t consumed = 0U;
                    prefill_items.reserve(
                        scheduled_batch.slices.size() - schedule_index);
                    prefill_counts.reserve(
                        scheduled_batch.slices.size() - schedule_index);

                    for (std::size_t cursor = schedule_index;
                         cursor < scheduled_batch.slices.size(); ++cursor) {
                        const auto& candidate = scheduled_batch.slices[cursor];
                        if (candidate.phase != runtime_detail::WorkPhase::prefill) {
                            break;
                        }
                        auto& other = active[candidate.slot];
                        if (other->done || other->failed || !other->session ||
                            other->is_decision() || other->cancellation_requested() ||
                            other->temperature() > 0.0 ||
                            other->plan.backend != item->plan.backend ||
                            other->plan.linear.prefill_block !=
                                item->plan.linear.prefill_block ||
                            other->plan.attention.prefill !=
                                item->plan.attention.prefill ||
                            other->plan.kv.page_tokens !=
                                item->plan.kv.page_tokens) {
                            break;
                        }
                        prefill_items.push_back(other.get());
                        prefill_counts.push_back(candidate.token_count);
                        ++consumed;
                    }

                    bool batched = false;
                    if (prefill_items.size() >= 2U) {
                        batched = try_advance_prefill_batch(
                            prefill_items, prefill_counts);
                    }
                    if (batched) {
                        progressed = true;
                        schedule_index += consumed;
                    } else {
                        auto slice_budget = slice.token_count;
                        progressed =
                            advance_prefill(*item, slice_budget) || progressed;
                        ++schedule_index;
                    }

                    // A physical multi-sequence prefill call is one scheduler
                    // execution boundary. If new work arrived while it ran, the
                    // next outer cycle re-enters normal admission immediately.
                    std::uint64_t observed_generation = cycle_queue_generation;
                    std::size_t queued_count = 0U;
                    {
                        std::lock_guard lock(mutex);
                        observed_generation = queue_generation;
                        queued_count = queued.size();
                    }
                    if (queued_count != 0U &&
                        observed_generation != cycle_queue_generation) {
                        event("scheduler_yield_for_admission", 0,
                              "queued=" + std::to_string(queued_count) +
                              " from_generation=" +
                                  std::to_string(cycle_queue_generation) +
                              " to_generation=" +
                                  std::to_string(observed_generation));
                        yield_for_new_admission = true;
                        break;
                    }
                    continue;
                }

                auto* prepared = prepared_for(item->plan.backend);
                const bool batch_candidate = prepared &&
                    !item->is_decision() &&
                    prepared->capabilities().max_decode_batch_width > 1U &&
                    item->temperature() <= 0.0 &&
                    (item->plan.linear.decode_block != QuantizedLinearExecutionKind::baseline ||
                     item->plan.linear.decode_output != QuantizedLinearExecutionKind::baseline);
                if (batch_candidate) {
                    const auto width_limit = std::min<std::size_t>(
                        prepared->capabilities().max_decode_batch_width,
                        scheduled_batch.slices.size() - schedule_index);
                    std::vector<WorkItem*> batch_items;
                    batch_items.reserve(width_limit);
                    std::size_t consumed = 0U;
                    for (std::size_t cursor = schedule_index;
                         cursor < scheduled_batch.slices.size() &&
                         batch_items.size() < width_limit; ++cursor) {
                        const auto& candidate = scheduled_batch.slices[cursor];
                        if (candidate.phase != runtime_detail::WorkPhase::decode) break;
                        auto& other = active[candidate.slot];
                        if (other->done || other->is_decision() || other->temperature() > 0.0 ||
                            other->plan.backend != item->plan.backend ||
                            other->plan.linear.decode_block != item->plan.linear.decode_block ||
                            other->plan.linear.decode_output != item->plan.linear.decode_output ||
                            other->plan.attention.decode != item->plan.attention.decode) break;
                        batch_items.push_back(other.get());
                        ++consumed;
                    }
                    if (batch_items.size() >= 2U) {
                        auto batch_budget = static_cast<std::uint32_t>(batch_items.size());
                        const bool batched = advance_decode_batch(batch_items, batch_budget);
                        progressed = batched || progressed;
                        if (batched) {
                            schedule_index += consumed;
                            continue;
                        }
                    }
                }

                std::uint32_t slice_budget = 1U;
                progressed = advance_decode(*item, slice_budget) || progressed;
                ++schedule_index;
            }

            if (yield_for_new_admission) {
                // Remaining slices were only data from the superseded scheduling
                // decision. They are intentionally discarded and will be rebuilt
                // by MicrobatchScheduler on the next cycle.
                progressed = true;
            }

            std::uint64_t active_kv_bytes = 0U;
            for (const auto& item : active) {
                std::uint64_t item_kv = 0U;
                if (item->session) {
                    item_kv = std::max(
                        item_kv, item->session->resources().committed_kv_bytes);
                }
                if (item->decision_prefix_checkpoint) {
                    item_kv = std::max(
                        item_kv,
                        item->decision_prefix_checkpoint->resources().committed_kv_bytes);
                }
                if (item->decision_branch) {
                    item_kv = std::max(
                        item_kv,
                        item->decision_branch->resources().committed_kv_bytes);
                }
                active_kv_bytes += item_kv;
            }
            {
                std::lock_guard lock(mutex);
                peak_kv_bytes = std::max(peak_kv_bytes, active_kv_bytes);
                if (cuda) {
                    peak_device_bytes = std::max(peak_device_bytes, cuda->resident_device_bytes());
                }
            }

            std::vector<std::shared_ptr<WorkItem>> finished;
            bool epoch_drained = false;
            {
                std::lock_guard lock(mutex);
                auto it = std::remove_if(active.begin(), active.end(), [this, &finished](const auto& item) {
                    if (!item->done) return false;
                    finished.push_back(item);
                    if (item->admission_reservation_bytes <= active_admission_reserved_bytes) {
                        active_admission_reserved_bytes -= item->admission_reservation_bytes;
                    } else {
                        active_admission_reserved_bytes = 0U;
                    }
                    item->admission_reservation_bytes = 0U;
                    return true;
                });
                active.erase(it, active.end());
                epoch_drained = regime_epoch_open && active.empty();
            }
            for (const auto& item : finished) finish_item(item);

            if (epoch_drained) {
                event("regime_epoch_end", 0,
                      "epoch=" + std::to_string(regime_epoch_id) +
                      " fallback_pinned=" +
                      (regime_epoch_fallback_pinned ? "true" : "false"));
                regime_epoch_open = false;
                regime_epoch_fallback_pinned = false;
            }

            if (!finished.empty()) wake.notify_one();
            if (!progressed && finished.empty()) std::this_thread::yield();
        }
        event("scheduler_stop", 0, "stopped");
    }

};

const char* to_string(BackendPreference backend) noexcept {
    switch (backend) {
    case BackendPreference::automatic: return "auto";
    case BackendPreference::reference: return "reference";
    case BackendPreference::cuda: return "cuda";
    }
    return "unknown";
}

double RequestMetrics::prefill_tokens_per_second() const noexcept {
    const auto effective = prompt_tokens > prefix_reused_tokens ? prompt_tokens - prefix_reused_tokens : 0U;
    if (prefill_ms <= 0.0) return 0.0;
    return static_cast<double>(effective) * 1000.0 / prefill_ms;
}

double RequestMetrics::decode_tokens_per_second() const noexcept {
    if (generated_tokens <= 1U || decode_ms <= 0.0) return 0.0;
    return static_cast<double>(generated_tokens - 1U) * 1000.0 / decode_ms;
}

InferenceService::InferenceService(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
InferenceService::~InferenceService() { shutdown(); }

void InferenceService::shutdown() noexcept {
    if (!impl_) return;
    {
        std::lock_guard lock(impl_->mutex);
        impl_->stopping = true;
        for (auto& item : impl_->queued) item->service_cancelled.store(true, std::memory_order_release);
        for (auto& item : impl_->active) item->service_cancelled.store(true, std::memory_order_release);
    }
    impl_->wake.notify_all();
    if (impl_->worker.joinable() && impl_->worker.get_id() != std::this_thread::get_id()) {
        impl_->worker.join();
    }
}
InferenceService::InferenceService(InferenceService&&) noexcept = default;
InferenceService& InferenceService::operator=(InferenceService&&) noexcept = default;

Result<std::unique_ptr<InferenceService>> InferenceService::create(
    const std::filesystem::path& model_path,
    BackendPreference backend,
    int cuda_device,
    SchedulerConfig scheduler,
    std::filesystem::path event_log_path,
    ManifestConfig manifest,
    ExecutionConfig execution) {
    GgufFormat format;
    auto loaded = format.load(model_path);
    if (!loaded) return loaded.status();
    auto model = std::make_shared<ModelDefinition>(std::move(loaded).value());
    return create(std::move(model), backend, cuda_device, scheduler,
                  std::move(event_log_path), std::move(manifest), execution);
}

Result<std::unique_ptr<InferenceService>> InferenceService::create(
    std::shared_ptr<ModelDefinition> model,
    BackendPreference backend,
    int cuda_device,
    SchedulerConfig scheduler,
    std::filesystem::path event_log_path,
    ManifestConfig manifest,
    ExecutionConfig execution) {
    if (!model) return Status::invalid_argument("inference service requires a model");
    const auto valid = model->validate();
    if (!valid) return valid;
    if (scheduler.max_active_requests == 0U || scheduler.token_budget_per_cycle == 0U ||
        scheduler.prefill_quantum_tokens == 0U || scheduler.reference_kv_page_tokens == 0U ||
        scheduler.cuda_kv_page_tokens == 0U || scheduler.stream_queue_capacity == 0U) {
        return Status::invalid_argument("scheduler limits must all be non-zero");
    }
    if (manifest.require && !manifest.enabled) {
        return Status::invalid_argument("required manifest cannot be disabled");
    }

    auto impl = std::make_unique<Impl>(scheduler, execution);
    impl->model = std::move(model);
    auto tokenizer = create_tokenizer(impl->model->tokenizer_handle());
    if (!tokenizer) return tokenizer.status();
    impl->tokenizer = std::move(tokenizer).value();
    impl->requested_backend = backend;
    impl->cuda_device = cuda_device;
    impl->event_log_path = std::move(event_log_path);
    if (!impl->event_log_path.empty()) {
        impl->event_log.open(impl->event_log_path, std::ios::app);
        if (!impl->event_log) return Status::io_error("unable to open runtime event log");
    }

    std::shared_ptr<const ModelDefinition> canonical = impl->model;
    if (backend != BackendPreference::cuda) {
        auto reference = runtime_detail::prepare_reference_model(canonical);
        if (!reference) return reference.status();
        impl->reference = std::move(reference).value();
    }
    if (backend != BackendPreference::reference) {
        auto cuda = runtime_detail::prepare_cuda_model(canonical, cuda_device);
        if (cuda) {
            impl->cuda = std::move(cuda).value();
            impl->peak_device_bytes = impl->cuda->resident_device_bytes();
        } else if (backend == BackendPreference::cuda) {
            return cuda.status();
        }
    }

    ExecutionPlan fallback;
    fallback.backend = backend == BackendPreference::cuda ? BackendKind::cuda :
                       (backend == BackendPreference::automatic && impl->cuda
                            ? BackendKind::cuda : BackendKind::reference);
    fallback.strategy_id = "static";
    fallback.scheduling.prefill_quantum_tokens = scheduler.prefill_quantum_tokens;
    fallback.kv.page_tokens = fallback.backend == BackendKind::cuda
        ? scheduler.cuda_kv_page_tokens
        : scheduler.reference_kv_page_tokens;
    if (fallback.backend == BackendKind::cuda) {
        fallback.linear.prefill_block = execution.cuda_prefill_block_linear;
        fallback.linear.decode_block = execution.cuda_decode_block_linear;
        fallback.linear.decode_output = execution.cuda_decode_output_linear;
        fallback.attention.prefill = execution.cuda_prefill_attention;
        fallback.attention.decode = execution.cuda_decode_attention;
    }
    impl->fallback_plan = fallback;

    std::unique_ptr<Planner> planner;
    if (backend == BackendPreference::automatic && manifest.enabled) {
        const auto path = manifest.path.empty() ? default_manifest_path(*impl->model) : manifest.path;
        if (std::filesystem::exists(path)) {
            auto loaded_manifest = load_manifest(path);
            if (!loaded_manifest) {
                impl->manifest_status = "invalid:" + loaded_manifest.status().message();
            } else {
                auto validated = validate_manifest(loaded_manifest.value(), *impl->model);
                impl->manifest_status = validated.status;
                if (validated.manifest) {
                    bool strategy_invalid = false;
                    std::string strategy_status;
                    for (const auto& strategy : validated.manifest->strategies) {
                        auto* prepared = impl->prepared_for(strategy.plan.backend);
                        if (!prepared) {
                            strategy_invalid = true;
                            strategy_status = strategy.plan.backend == BackendKind::cuda
                                ? "stale:qualified-cuda-unavailable"
                                : "stale:qualified-backend-unavailable";
                            break;
                        }
                        const auto plan_valid = validate_execution_plan(strategy.plan, prepared->capabilities());
                        if (!plan_valid) {
                            strategy_invalid = true;
                            strategy_status = "stale:strategy-capability-mismatch";
                            break;
                        }
                    }
                    if (strategy_invalid) {
                        impl->manifest_status = std::move(strategy_status);
                    } else {
                        impl->manifest_id = validated.manifest->manifest_id;
                        impl->planner_mode = "adaptive";
                        planner = std::make_unique<StrategyLabPlanner>(std::move(*validated.manifest), fallback, manifest.strategy);
                    }
                }
            }
        } else {
            impl->manifest_status = "not-found";
        }
        if (manifest.require && !planner) {
            return Status::invalid_state("required execution manifest unavailable: " + impl->manifest_status);
        }
    } else {
        impl->manifest_status = backend == BackendPreference::automatic ? "disabled" : "not-applicable-explicit-backend";
    }
    if (!planner) planner = std::make_unique<StaticPlanner>(fallback);
    impl->runtime = std::make_unique<Runtime>(impl->model, std::move(planner));
    impl->last_decision = impl->runtime->decide(RequestProfile{1U, 1U, 1U}, RuntimeSnapshot{});
    impl->last_plan = impl->last_decision.plan;
    impl->apply_plan_defaults(impl->last_plan);
    if (auto* prepared = impl->prepared_for(impl->last_plan.backend)) {
        const auto plan_valid = validate_execution_plan(impl->last_plan, prepared->capabilities());
        if (!plan_valid) return plan_valid;
    }
    impl->backend = impl->planner_mode == "adaptive" ? "adaptive" : to_string(fallback.backend);

    impl->worker = std::thread([raw = impl.get()] { raw->run(); });
    return std::unique_ptr<InferenceService>(new InferenceService(std::move(impl)));
}

Result<std::string> InferenceService::render_chat(const std::vector<ChatMessage>& messages) const {
    if (messages.empty()) return Status::invalid_argument("chat request requires at least one message");
    if (impl_->model->config().architecture != "qwen2") {
        return Status::unsupported("AIR chat rendering is currently defined only for qwen2");
    }
    std::string rendered;
    for (const auto& message : messages) {
        if (message.role != "system" && message.role != "user" && message.role != "assistant") {
            return Status::invalid_argument("unsupported chat role: " + message.role);
        }
        rendered += "<|im_start|>";
        rendered += message.role;
        rendered += '\n';
        rendered += message.content;
        rendered += "<|im_end|>\n";
    }
    rendered += "<|im_start|>assistant\n";
    return rendered;
}

Result<InferenceResponse> InferenceService::generate(const InferenceRequest& request, StreamCallback stream, CancellationToken cancellation) {
    if (cancellation.is_cancelled()) return Status::cancelled("request cancelled before submission");
    std::string prompt = request.prompt;
    if (!request.messages.empty()) {
        auto rendered = render_chat(request.messages);
        if (!rendered) return rendered.status();
        prompt = std::move(rendered).value();
    }
    if (prompt.empty()) return Status::invalid_argument("request prompt is empty");

    auto tokens = impl_->tokenizer->encode(prompt);
    if (!tokens) return tokens.status();
    if (tokens.value().empty()) return Status::invalid_argument("request prompt produces no tokens");
    if (tokens.value().size() + request.generation.max_new_tokens > impl_->model->config().context_length) {
        return Status::invalid_argument("prompt plus requested output exceeds model context length");
    }

    auto item = std::make_shared<WorkItem>();
    item->request_id = impl_->next_request.fetch_add(1);
    item->sequence_id = impl_->next_sequence.fetch_add(1);
    item->request = request;
    item->prompt_tokens = std::move(tokens).value();
    item->stream_enabled = static_cast<bool>(stream);
    item->cancellation = std::move(cancellation);
    item->submitted_at = Clock::now();
    auto future = item->promise.get_future();
    {
        std::lock_guard lock(impl_->mutex);
        if (impl_->stopping) return Status::invalid_state("inference service is stopping");
        if (impl_->config.max_queued_requests != 0U &&
            impl_->queued.size() >= impl_->config.max_queued_requests) {
            ++impl_->rejected_overload;
            return Status::invalid_state("inference submission queue is full");
        }
        impl_->queued.push_back(item);
        ++impl_->queue_generation;
    }
    impl_->event("request_queued", item->request_id,
                 "prompt_tokens=" + std::to_string(item->prompt_tokens.size()));
    impl_->wake.notify_one();

    std::optional<Status> stream_failure;
    if (stream) {
        while (true) {
            std::optional<WorkItem::StreamEvent> event;
            {
                std::unique_lock lock(item->stream_mutex);
                item->stream_wake.wait(lock, [&] {
                    return !item->stream_events.empty() || item->stream_closed;
                });
                if (!item->stream_events.empty()) {
                    event = std::move(item->stream_events.front());
                    item->stream_events.pop_front();
                } else if (item->stream_closed) {
                    break;
                }
            }
            if (event && !stream_failure) {
                const auto delivered = stream(event->delta, event->token);
                if (!delivered) {
                    stream_failure = delivered;
                    item->service_cancelled.store(true, std::memory_order_release);
                    {
                        std::lock_guard lock(item->stream_mutex);
                        item->stream_events.clear();
                    }
                    impl_->wake.notify_one();
                }
            }
        }
    }
    auto result = future.get();
    if (stream_failure) {
        {
            std::lock_guard lock(impl_->mutex);
            ++impl_->stream_delivery_failures;
        }
        impl_->event("stream_delivery_failed", item->request_id, stream_failure->message());
        return *stream_failure;
    }
    return result;
}

Result<DecisionResponse> InferenceService::decide(
    const DecisionRequest& request,
    CancellationToken cancellation) {
    if (cancellation.is_cancelled()) {
        return Status::cancelled("decision cancelled before submission");
    }

    const auto valid = validate_decision_request(request);
    if (!valid) return valid;

    // Prompt 2 did not qualify a workload-independent AUTO scorer.
    if (request.scoring_policy == DecisionScoringPolicy::qualified_auto) {
        return Status::unsupported(
            "Decision V1 requires explicit sequence-logprob-sum or sequence-logprob-mean");
    }

    // Score calibration/thresholding was not established, so production V1
    // executes only deterministic argmax selection. The broader semantic
    // cardinality contract remains available for later qualification.
    if (request.output_cardinality != DecisionOutputCardinality::exactly_one) {
        return Status::unsupported(
            "Decision V1 execution currently qualifies exactly-one output only");
    }

    auto prompt_tokens = impl_->tokenizer->encode(request.input_text);
    if (!prompt_tokens) return prompt_tokens.status();
    if (prompt_tokens.value().empty()) {
        return Status::invalid_argument("decision input produces no tokens");
    }

    TokenizeOptions candidate_options;
    candidate_options.add_bos = false;
    candidate_options.add_eos = false;

    std::vector<std::vector<TokenId>> candidate_tokens;
    candidate_tokens.reserve(request.candidates.size());
    std::size_t maximum_candidate_tokens = 0U;

    for (const auto& candidate : request.candidates) {
        const std::string_view surface =
            candidate.model_text
                ? std::string_view(*candidate.model_text)
                : std::string_view(candidate.display_text);
        auto encoded =
            impl_->tokenizer->encode(surface, candidate_options);
        if (!encoded) return encoded.status();
        if (encoded.value().empty()) {
            return Status::invalid_argument(
                "decision candidate model representation produces no tokens: " +
                candidate.id);
        }
        maximum_candidate_tokens =
            std::max(maximum_candidate_tokens, encoded.value().size());
        candidate_tokens.push_back(std::move(encoded).value());
    }

    if (prompt_tokens.value().size() + maximum_candidate_tokens >
        impl_->model->config().context_length) {
        return Status::invalid_argument(
            "decision input plus longest candidate exceeds model context length");
    }

    auto item = std::make_shared<WorkItem>();
    item->request_id = impl_->next_request.fetch_add(1);
    item->sequence_id = impl_->next_sequence.fetch_add(1);
    item->kind = WorkItem::Kind::decision;
    item->decision_request = request;
    item->prompt_tokens = std::move(prompt_tokens).value();
    item->decision_candidate_tokens = std::move(candidate_tokens);
    item->stream_enabled = false;
    item->cancellation = std::move(cancellation);
    item->submitted_at = Clock::now();

    auto future = item->decision_promise.get_future();
    {
        std::lock_guard lock(impl_->mutex);
        if (impl_->stopping) {
            return Status::invalid_state("inference service is stopping");
        }
        if (impl_->config.max_queued_requests != 0U &&
            impl_->queued.size() >= impl_->config.max_queued_requests) {
            ++impl_->rejected_overload;
            return Status::invalid_state("inference submission queue is full");
        }
        impl_->queued.push_back(item);
        ++impl_->queue_generation;
    }

    impl_->event(
        "decision_queued",
        item->request_id,
        "prompt_tokens=" + std::to_string(item->prompt_tokens.size()) +
            " candidates=" +
            std::to_string(item->decision_request.candidates.size()) +
            " scoring=" +
            std::string(to_string(item->decision_request.scoring_policy)));
    impl_->wake.notify_one();
    return future.get();
}

Result<std::vector<InferenceResponse>> InferenceService::generate_cohort(
    std::span<const InferenceRequest> requests) {
    if (requests.empty()) return Status::invalid_argument("inference cohort must not be empty");
    if (requests.size() > impl_->config.max_active_requests) {
        return Status::invalid_argument("inference cohort exceeds configured maximum active requests");
    }

    std::vector<std::shared_ptr<WorkItem>> items;
    std::vector<std::future<Result<InferenceResponse>>> futures;
    items.reserve(requests.size());
    futures.reserve(requests.size());

    // Complete every fallible caller-side transformation before publishing any
    // cohort member. A malformed request therefore cannot leave a partial cohort
    // in the production queue.
    for (const auto& request : requests) {
        std::string prompt = request.prompt;
        if (!request.messages.empty()) {
            auto rendered = render_chat(request.messages);
            if (!rendered) return rendered.status();
            prompt = std::move(rendered).value();
        }
        if (prompt.empty()) return Status::invalid_argument("request prompt is empty");

        auto tokens = impl_->tokenizer->encode(prompt);
        if (!tokens) return tokens.status();
        if (tokens.value().empty()) return Status::invalid_argument("request prompt produces no tokens");
        if (tokens.value().size() + request.generation.max_new_tokens > impl_->model->config().context_length) {
            return Status::invalid_argument("prompt plus requested output exceeds model context length");
        }

        auto item = std::make_shared<WorkItem>();
        item->request_id = impl_->next_request.fetch_add(1);
        item->sequence_id = impl_->next_sequence.fetch_add(1);
        item->request = request;
        item->prompt_tokens = std::move(tokens).value();
        item->stream_enabled = false;
        futures.push_back(item->promise.get_future());
        items.push_back(std::move(item));
    }

    const auto submitted_at = Clock::now();
    {
        std::lock_guard lock(impl_->mutex);
        if (impl_->stopping) return Status::invalid_state("inference service is stopping");
        if (impl_->config.max_queued_requests != 0U &&
            impl_->queued.size() + items.size() > impl_->config.max_queued_requests) {
            ++impl_->rejected_overload;
            return Status::invalid_state("inference cohort exceeds submission queue capacity");
        }
        for (const auto& item : items) {
            item->submitted_at = submitted_at;
            impl_->queued.push_back(item);
        }
        ++impl_->queue_generation;
    }
    for (const auto& item : items) {
        impl_->event("request_queued", item->request_id,
                     "prompt_tokens=" + std::to_string(item->prompt_tokens.size()) + " cohort=atomic");
    }
    // Publish the complete cohort with one wake. active+queued therefore remains
    // the cohort width as members move through the existing admission loop.
    impl_->wake.notify_one();

    std::vector<InferenceResponse> responses;
    responses.reserve(futures.size());
    std::optional<Status> first_failure;
    for (auto& future : futures) {
        auto result = future.get();
        if (!result) {
            if (!first_failure) first_failure = result.status();
            continue;
        }
        responses.push_back(std::move(result).value());
    }
    if (first_failure) return *first_failure;
    if (responses.size() != requests.size()) {
        return Status::internal_error("inference cohort completed with missing responses");
    }
    return responses;
}

ServiceSnapshot InferenceService::snapshot() const {
    std::lock_guard lock(impl_->mutex);
    ServiceSnapshot out;
    out.backend = impl_->backend;
    out.planner_mode = impl_->planner_mode;
    out.manifest_status = impl_->manifest_status;
    out.manifest_id = impl_->manifest_id;
    out.strategy_id = impl_->last_plan.strategy_id;
    out.strategy_objective = impl_->last_decision.objective;
    out.strategy_decision_reason = impl_->last_decision.reason;
    out.strategy_eligible_candidates = impl_->last_decision.eligible_candidates;
    out.strategy_prepared_state_hot = impl_->last_decision.prepared_state_hot;
    out.strategy_estimated_transition_ms = impl_->last_decision.estimated_transition_ms;
    out.strategy_estimated_break_even_tokens = impl_->last_decision.estimated_break_even_tokens;
    out.strategy_candidates = impl_->last_decision.candidates;
    out.selected_backend = to_string(impl_->last_plan.backend);
    out.planned_prefill_quantum_tokens = impl_->last_plan.scheduling.prefill_quantum_tokens;
    out.planned_kv_page_tokens = impl_->last_plan.kv.page_tokens.value_or(0U);
    out.planned_prefill_block_linear_tactic = to_string(impl_->last_plan.linear.prefill_block);
    out.planned_decode_block_linear_tactic = to_string(impl_->last_plan.linear.decode_block);
    out.planned_decode_output_linear_tactic = to_string(impl_->last_plan.linear.decode_output);
    out.planned_prefill_attention_tactic = to_string(impl_->last_plan.attention.prefill);
    out.planned_decode_attention_tactic = to_string(impl_->last_plan.attention.decode);
    out.queued_requests = impl_->queued.size();
    out.active_requests = impl_->active.size();
    out.rejected_overload_requests = impl_->rejected_overload;
    out.completed_decisions = impl_->completed_decisions;
    out.max_queued_requests = impl_->config.max_queued_requests;
    out.completed_requests = impl_->completed;
    out.failed_requests = impl_->failed;
    out.cancelled_requests = impl_->cancelled;
    out.stream_delivery_failures = impl_->stream_delivery_failures;
    out.native_decode_batches = impl_->native_decode_batches;
    out.native_decode_sequences = impl_->native_decode_sequences;
    out.physical_prefill_batches = impl_->physical_prefill_batches;
    out.physical_prefill_sequence_participations =
        impl_->physical_prefill_sequence_participations;
    out.physical_prefill_tokens = impl_->physical_prefill_tokens;
    out.physical_prefill_max_sequences =
        impl_->physical_prefill_max_sequences;
    out.total_prompt_tokens = impl_->prompt_tokens;
    out.total_generated_tokens = impl_->generated_tokens;
    out.total_prefix_reused_tokens = impl_->prefix_reused_tokens;
    out.peak_kv_bytes = impl_->peak_kv_bytes;
    out.peak_device_bytes = impl_->peak_device_bytes;
    for (const auto& item : impl_->active) {
        if (item->session) out.current_kv_bytes += item->session->resources().committed_kv_bytes;
    }
    if (impl_->cuda) {
        out.current_device_bytes = impl_->cuda->resident_device_bytes();
        out.current_prepared_artifact_bytes = impl_->cuda->prepared_artifact_device_bytes();
    }
    const auto elapsed = std::chrono::duration<double>(Clock::now() - impl_->service_started).count();
    if (elapsed > 0.0) out.aggregate_generated_tokens_per_second = static_cast<double>(impl_->generated_tokens) / elapsed;
    out.p50_total_ms = detail::percentile_linear(impl_->recent_latency, 0.50);
    out.p95_total_ms = detail::percentile_linear(impl_->recent_latency, 0.95);
    out.max_active_requests = impl_->config.max_active_requests;
    out.token_budget_per_cycle = impl_->config.token_budget_per_cycle;
    out.prefill_quantum_tokens = impl_->config.prefill_quantum_tokens;
    out.reference_kv_page_tokens = impl_->config.reference_kv_page_tokens;
    out.cuda_kv_page_tokens = impl_->config.cuda_kv_page_tokens;
    out.admission_reserved_bytes = impl_->active_admission_reserved_bytes;
    if (const auto* prepared = impl_->prepared_for(impl_->last_plan.backend)) {
        const auto& capabilities = prepared->capabilities();
        out.max_prefill_batch_width = capabilities.max_prefill_batch_width;
        out.max_decode_batch_width = capabilities.max_decode_batch_width;
        out.stream_queue_capacity = impl_->config.stream_queue_capacity;
        out.prefill_execution = to_string(capabilities.prefill_execution);
        out.kv_storage = to_string(capabilities.kv_storage);
        if (const auto capacity = prepared->sequence_capacity_bytes()) {
            out.admission_capacity_bytes = *capacity;
        }
        out.kv_pool_allocated_bytes = prepared->kv_pool_allocated_bytes();
        out.kv_pool_free_bytes = prepared->kv_pool_free_bytes();
        out.physical_paged_kv = capabilities.kv_storage == KvStorageKind::paged;
        out.sequence_checkpointing = capabilities.sequence_checkpointing;
        out.prefix_cache_enabled = capabilities.exact_prefix_reuse &&
            impl_->sequence_state_store.capacity_entries() != 0U;
        out.device_greedy_selection = capabilities.device_greedy_selection;
        const auto state_store = impl_->sequence_state_store.metrics();
        out.sequence_state_store_capacity_entries = state_store.capacity_entries;
        out.sequence_state_store_entries = state_store.entries;
        out.sequence_state_store_hits = state_store.hits;
        out.sequence_state_store_misses = state_store.misses;
        out.sequence_state_store_evictions = state_store.evictions;
        out.sequence_state_store_clone_failures = state_store.clone_failures;
        out.sequence_state_store_rejected_device_residency = state_store.rejected_device_residency;
        out.sequence_state_store_retained_checkpoint_bytes = state_store.retained_checkpoint_bytes;
        out.sequence_state_store_retained_metadata_bytes = state_store.retained_metadata_bytes;
        out.sequence_state_store_retained_device_bytes = state_store.retained_device_bytes;
    }
    return out;
}

std::vector<RuntimeEvent> InferenceService::recent_events(std::size_t limit) const {
    std::lock_guard lock(impl_->mutex);
    const auto count = std::min(limit, impl_->events.size());
    return std::vector<RuntimeEvent>(impl_->events.end() - static_cast<std::ptrdiff_t>(count), impl_->events.end());
}

const ModelDefinition& InferenceService::model() const noexcept { return *impl_->model; }
std::string InferenceService::backend_name() const { return impl_->backend; }

} // namespace air

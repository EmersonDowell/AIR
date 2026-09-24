#include "air/benchmark.hpp"
#include "air/manifest.hpp"
#include "air/version.hpp"
#include "air/reference.hpp"
#include "air/serving.hpp"
#include "air/storage.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
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

void append_token(air::ReferenceKvCache& cache, float marker) {
    const std::vector<float> key = {marker, marker + 0.25F};
    const std::vector<float> value = {marker + 1.0F, marker + 1.25F};
    check(cache.append_pending(0, key, value).is_ok(), "KV page stages token");
    check(cache.commit_token().is_ok(), "KV page commits token");
}

void test_paged_kv_fork_and_cow() {
    air::ReferenceKvCache cache(1, 1, 2, 16, 2);
    append_token(cache, 1.0F);
    append_token(cache, 2.0F);
    append_token(cache, 3.0F);
    append_token(cache, 4.0F);
    check(cache.page_count() == 2, "reference KV uses configured physical page size");

    auto forked_result = cache.fork(4);
    check(forked_result.is_ok(), "full-page KV prefix can fork");
    if (!forked_result) return;
    auto forked = std::move(forked_result).value();
    check(cache.shared_page_count() == 2 && forked.shared_page_count() == 2,
          "forked committed full pages are physically shared");

    append_token(forked, 9.0F);
    check(forked.size() == 5 && cache.size() == 4, "fork append advances only child sequence");
    auto original = cache.key(0, 3, 0);
    auto child = forked.key(0, 3, 0);
    check(original && child && original.value()[0] == child.value()[0],
          "shared committed prefix remains identical after child append");

    auto partial_result = cache.fork(3);
    check(partial_result.is_ok(), "partial-page prefix can fork");
    if (partial_result) {
        auto partial = std::move(partial_result).value();
        check(partial.size() == 3 && partial.page_count() == 2,
              "partial fork preserves exact committed token count");
        append_token(partial, 7.0F);
        auto original_tail = cache.key(0, 3, 0);
        auto changed_tail = partial.key(0, 3, 0);
        check(original_tail && changed_tail && original_tail.value()[0] != changed_tail.value()[0],
              "partial-page fork has an independent copy-on-write tail");
    }
}

void test_service_scheduler_prefix_and_streaming() {
    air::SchedulerConfig scheduler;
    scheduler.max_active_requests = 4;
    scheduler.token_budget_per_cycle = 4;
    scheduler.prefill_quantum_tokens = 1;
    scheduler.reference_kv_page_tokens = 2;
    scheduler.prefix_cache_entries = 16;

    auto service_result = air::InferenceService::create(
        tiny_model(), air::BackendPreference::reference, 0, scheduler);
    check(service_result.is_ok(), "reference inference service starts from canonical in-memory model");
    if (!service_result) return;
    auto& service = service_result.value();

    air::InferenceRequest request;
    request.prompt = "a";
    request.generation.max_new_tokens = 3;
    request.generation.sampling.temperature = 0.0;

    std::size_t streamed = 0;
    auto first = service->generate(request, [&](std::string_view, air::TokenId) {
        ++streamed;
        return air::Status::ok();
    });
    check(first && first.value().tokens == std::vector<air::TokenId>({0, 0, 0}),
          "scheduler preserves deterministic generation semantics");
    check(streamed == 3, "stream callback observes every generated token through scheduler");
    check(first && first.value().metrics.prefix_reused_tokens == 0,
          "first request does not report fictional prefix reuse");

    auto second = service->generate(request);
    check(second && second.value().metrics.prefix_reused_tokens == 1,
          "second exact prompt reuses committed reference KV prefix");

    air::InferenceRequest two = request;
    two.prompt = "aa";
    auto two_result = service->generate(two);
    check(two_result && two_result.value().metrics.prefix_reused_tokens == 1,
          "longer prompt resumes from shorter exact cached prefix");
    air::InferenceRequest three = request;
    three.prompt = "aaa";
    auto three_result = service->generate(three);
    check(three_result && three_result.value().metrics.prefix_reused_tokens == 2,
          "longest page-aligned exact token prefix is reused for further extension");

    std::vector<std::future<air::Result<air::InferenceResponse>>> futures;
    for (int i = 0; i < 4; ++i) {
        futures.push_back(std::async(std::launch::async, [&service, request] { return service->generate(request); }));
    }
    for (auto& future : futures) {
        auto response = future.get();
        check(response.is_ok(), "concurrent submitted request completes through one scheduler worker");
    }

    const auto snapshot = service->snapshot();
    check(snapshot.completed_requests == 8, "service accounts completed requests");
    check(snapshot.total_prefix_reused_tokens >= 8,
          "bounded exact prefix cache contributes reuse across repeated requests");
    check(snapshot.physical_paged_kv && snapshot.prefix_cache_enabled,
          "runtime capability snapshot distinguishes physical reference paging and prefix reuse");
    check(snapshot.kv_storage == "paged" && snapshot.prefill_execution == "serial" &&
          snapshot.sequence_checkpointing && snapshot.admission_reserved_bytes == 0U &&
          snapshot.kv_pool_allocated_bytes == 0U,
          "runtime snapshot is sourced from prepared reference backend capabilities");
    check(snapshot.token_budget_per_cycle == 4 && snapshot.prefill_quantum_tokens == 1,
          "scheduler policy is visible in runtime snapshot");
    const auto events = service->recent_events(128);
    check(std::any_of(events.begin(), events.end(), [](const air::RuntimeEvent& event) {
              return event.type == "prefix_hit";
          }), "runtime event log records prefix decisions");
}

void test_benchmark_uses_serving_pipeline() {
    air::SchedulerConfig scheduler;
    scheduler.max_active_requests = 3;
    scheduler.token_budget_per_cycle = 8;
    scheduler.prefill_quantum_tokens = 2;
    scheduler.reference_kv_page_tokens = 2;
    scheduler.prefix_cache_entries = 8;
    auto service_result = air::InferenceService::create(
        tiny_model(), air::BackendPreference::reference, 0, scheduler);
    check(service_result.is_ok(), "benchmark service starts");
    if (!service_result) return;

    air::BenchmarkConfig config;
    config.request.prompt = "a";
    config.request.generation.max_new_tokens = 2;
    config.warmup_runs = 0;
    config.measured_runs = 4;
    config.concurrency = 2;
    auto report = air::run_benchmark(*service_result.value(), config);
    check(report.is_ok(), "benchmark runs through InferenceService rather than a second executor path");
    if (!report) return;
    check(report.value().runs.size() == 4 && report.value().output_tokens.size() == 4 &&
              report.value().summary.generated_tokens == 8,
          "benchmark report accounts every measured request and preserves output tokens");
    check(report.value().summary.requests_per_second > 0.0 &&
          report.value().summary.aggregate_generated_tokens_per_second > 0.0,
          "benchmark computes aggregate throughput from measured wall time");
    check(service_result.value()->snapshot().max_decode_batch_width == 1U,
          "reference backend does not advertise CUDA native decode batching");
    const auto json = report.value().to_json();
    check(json.find("air.benchmark.v11") != std::string::npos &&
          json.find("\"runs\"") != std::string::npos &&
          json.find("\"output_tokens\"") != std::string::npos &&
          json.find("\"prefill_block_linear_tactic\": \"baseline\"") != std::string::npos,
          "benchmark emits versioned self-describing machine-readable JSON");
}


void test_cancellation_and_reclamation() {
    air::SchedulerConfig scheduler;
    scheduler.max_active_requests = 1;
    scheduler.token_budget_per_cycle = 1;
    scheduler.prefill_quantum_tokens = 1;
    scheduler.reference_kv_page_tokens = 2;
    scheduler.prefix_cache_entries = 0;
    scheduler.stream_queue_capacity = 4096;

    auto service_result = air::InferenceService::create(
        tiny_model(), air::BackendPreference::reference, 0, scheduler);
    check(service_result.is_ok(), "cancellation test service starts");
    if (!service_result) return;
    auto& service = service_result.value();

    air::InferenceRequest request;
    request.prompt = "a";
    request.generation.max_new_tokens = 3000;
    request.generation.sampling.temperature = 0.0;

    air::CancellationSource already_cancelled;
    already_cancelled.cancel();
    auto immediate = service->generate(request, {}, already_cancelled.token());
    check(!immediate && immediate.status().code() == air::ErrorCode::cancelled,
          "pre-cancelled request is rejected before queue admission");

    air::CancellationSource first_cancel;
    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    bool first_token_seen = false;
    bool release_first = false;
    auto first = std::async(std::launch::async, [&] {
        return service->generate(request,
            [&](std::string_view, air::TokenId) {
                std::unique_lock lock(gate_mutex);
                first_token_seen = true;
                gate_cv.notify_all();
                gate_cv.wait(lock, [&] { return release_first; });
                return air::Status::ok();
            }, first_cancel.token());
    });
    {
        std::unique_lock lock(gate_mutex);
        gate_cv.wait(lock, [&] { return first_token_seen; });
    }

    air::CancellationSource queued_cancel;
    auto queued = std::async(std::launch::async, [&] {
        return service->generate(request, {}, queued_cancel.token());
    });
    for (int attempt = 0; attempt < 100; ++attempt) {
        if (service->snapshot().queued_requests >= 1U) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check(service->snapshot().queued_requests >= 1U,
          "second request remains queued while active capacity is occupied");
    queued_cancel.cancel();
    first_cancel.cancel();
    {
        std::lock_guard lock(gate_mutex);
        release_first = true;
    }
    gate_cv.notify_all();

    auto first_result = first.get();
    auto queued_result = queued.get();
    check(!first_result && first_result.status().code() == air::ErrorCode::cancelled,
          "active request cancellation propagates as cancelled status");
    check(!queued_result && queued_result.status().code() == air::ErrorCode::cancelled,
          "queued request cancellation propagates without admission");

    const auto snapshot = service->snapshot();
    check(snapshot.active_requests == 0U && snapshot.queued_requests == 0U,
          "cancelled requests leave no active or queued work");
    check(snapshot.admission_reserved_bytes == 0U && snapshot.current_kv_bytes == 0U,
          "cancellation releases admission reservations and committed KV");
    check(snapshot.cancelled_requests == 2U && snapshot.failed_requests == 0U,
          "cancellation is accounted separately from runtime failure");
    const auto events = service->recent_events(128);
    check(std::count_if(events.begin(), events.end(), [](const air::RuntimeEvent& event) {
              return event.type == "request_cancelled";
          }) >= 2,
          "runtime event log exposes cancellation decisions");
}

void test_shutdown_cancels_work() {
    air::SchedulerConfig scheduler;
    scheduler.max_active_requests = 1;
    scheduler.token_budget_per_cycle = 1;
    scheduler.prefill_quantum_tokens = 1;
    scheduler.reference_kv_page_tokens = 2;
    scheduler.prefix_cache_entries = 0;
    scheduler.stream_queue_capacity = 4096;

    auto service_result = air::InferenceService::create(
        tiny_model(), air::BackendPreference::reference, 0, scheduler);
    check(service_result.is_ok(), "shutdown test service starts");
    if (!service_result) return;
    auto& service = service_result.value();

    air::InferenceRequest request;
    request.prompt = "a";
    request.generation.max_new_tokens = 3000;
    request.generation.sampling.temperature = 0.0;

    std::atomic_bool first_token{false};
    auto response = std::async(std::launch::async, [&] {
        return service->generate(request, [&](std::string_view, air::TokenId) {
            first_token.store(true, std::memory_order_release);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            return air::Status::ok();
        });
    });
    for (int attempt = 0; attempt < 200 && !first_token.load(std::memory_order_acquire); ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check(first_token.load(std::memory_order_acquire), "shutdown test reaches active generation");

    const auto started = std::chrono::steady_clock::now();
    service->shutdown();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();
    auto result = response.get();
    check(!result && result.status().code() == air::ErrorCode::cancelled,
          "service shutdown cancels active generation");
    check(elapsed < 1000, "service shutdown is bounded by an execution slice rather than full generation");

    const auto snapshot = service->snapshot();
    check(snapshot.active_requests == 0U && snapshot.queued_requests == 0U &&
          snapshot.admission_reserved_bytes == 0U && snapshot.current_kv_bytes == 0U,
          "shutdown leaves no live scheduler or sequence resources");
    auto rejected = service->generate(request);
    check(!rejected && rejected.status().code() == air::ErrorCode::invalid_state,
          "stopped service rejects new work explicitly");
}


void test_stream_delivery_isolated_and_bounded() {
    air::SchedulerConfig scheduler;
    scheduler.max_active_requests = 2;
    scheduler.token_budget_per_cycle = 8;
    scheduler.prefill_quantum_tokens = 4;
    scheduler.reference_kv_page_tokens = 2;
    scheduler.prefix_cache_entries = 0;
    scheduler.stream_queue_capacity = 64;

    auto service_result = air::InferenceService::create(
        tiny_model(), air::BackendPreference::reference, 0, scheduler);
    check(service_result.is_ok(), "stream isolation service starts");
    if (!service_result) return;
    auto& service = service_result.value();

    air::InferenceRequest slow_request;
    slow_request.prompt = "a";
    slow_request.generation.max_new_tokens = 8;
    slow_request.generation.sampling.temperature = 0.0;

    std::atomic_int callbacks{0};
    auto slow = std::async(std::launch::async, [&] {
        return service->generate(slow_request, [&](std::string_view, air::TokenId) {
            callbacks.fetch_add(1, std::memory_order_acq_rel);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            return air::Status::ok();
        });
    });
    for (int attempt = 0; attempt < 200 && callbacks.load(std::memory_order_acquire) == 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check(callbacks.load(std::memory_order_acquire) > 0,
          "slow stream consumer receives a token");

    air::InferenceRequest fast_request = slow_request;
    fast_request.generation.max_new_tokens = 2;
    const auto started = std::chrono::steady_clock::now();
    auto fast = service->generate(fast_request);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();
    check(fast.is_ok(), "non-stream request completes while another caller is slowly consuming stream output");
    check(elapsed < 250, "stream callback latency does not run on the inference scheduler thread");
    auto slow_result = slow.get();
    check(slow_result.is_ok(), "bounded slow stream drains successfully when consumer remains within capacity");

    air::SchedulerConfig bounded = scheduler;
    bounded.stream_queue_capacity = 1;
    auto bounded_service = air::InferenceService::create(
        tiny_model(), air::BackendPreference::reference, 0, bounded);
    check(bounded_service.is_ok(), "bounded stream service starts");
    if (!bounded_service) return;
    air::InferenceRequest burst = slow_request;
    burst.generation.max_new_tokens = 64;
    auto overflow = bounded_service.value()->generate(burst, [&](std::string_view, air::TokenId) {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        return air::Status::ok();
    });
    check(!overflow && overflow.status().code() == air::ErrorCode::cancelled,
          "stalled stream consumer is cancelled instead of blocking or growing an unbounded queue");
    const auto bounded_snapshot = bounded_service.value()->snapshot();
    check(bounded_snapshot.cancelled_requests == 1U && bounded_snapshot.failed_requests == 0U &&
          bounded_snapshot.active_requests == 0U && bounded_snapshot.current_kv_bytes == 0U,
          "stream backpressure cancellation releases scheduler and sequence resources");
}



void test_repeated_service_lifecycle() {
    air::InferenceRequest request;
    request.prompt = "a";
    request.generation.max_new_tokens = 2;
    request.generation.sampling.temperature = 0.0;
    for (int iteration = 0; iteration < 20; ++iteration) {
        air::SchedulerConfig scheduler;
        scheduler.max_active_requests = 2;
        scheduler.reference_kv_page_tokens = 2;
        scheduler.prefix_cache_entries = 0;
        auto service = air::InferenceService::create(
            tiny_model(), air::BackendPreference::reference, 0, scheduler);
        check(service.is_ok(), "repeated lifecycle service starts");
        if (!service) return;
        auto response = service.value()->generate(request);
        check(response.is_ok(), "repeated lifecycle request completes");
        const auto before_shutdown = service.value()->snapshot();
        check(before_shutdown.active_requests == 0U && before_shutdown.queued_requests == 0U &&
              before_shutdown.admission_reserved_bytes == 0U && before_shutdown.current_kv_bytes == 0U,
              "completed request returns logical resources before service shutdown");
        service.value()->shutdown();
        const auto stopped = service.value()->snapshot();
        check(stopped.active_requests == 0U && stopped.queued_requests == 0U,
              "repeated shutdown leaves scheduler idle");
    }
}

void test_atomic_cohort_preserves_exact_concurrency_region() {
    auto model = tiny_model();
    air::ExecutionManifest manifest;
    manifest.schema_version = air::execution_manifest_schema_version;
    manifest.air_version = air::version_string();
    manifest.model_digest = air::model_digest(*model);
    manifest.hardware_digest = air::hardware_digest();
    manifest.manifest_id = "manifest:cohort-serving-test";

    air::QualifiedStrategy strategy;
    strategy.workload = air::WorkloadClass::small;
    strategy.strategy_id = "exact-c4-reference";
    strategy.plan.backend = air::BackendKind::reference;
    strategy.plan.strategy_id = strategy.strategy_id;
    strategy.plan.scheduling.prefill_quantum_tokens = 1;
    strategy.plan.kv.page_tokens = 2;
    strategy.strict_qualified = true;
    strategy.region = air::WorkloadRegion{0, 256, 4, 4};
    strategy.samples = 3;
    strategy.p50_ttft_ms = 5.0;
    strategy.p50_total_ms = 10.0;
    strategy.mean_prefill_tokens_per_second = 100.0;
    strategy.mean_decode_tokens_per_second = 50.0;
    // Concurrent Strategy Lab requests require measured aggregate throughput
    // evidence. Leaving this at zero makes the fixture itself ineligible and
    // tests fallback behavior rather than atomic-cohort semantics.
    strategy.aggregate_generated_tokens_per_second = 80.0;
    strategy.aggregate_generated_tokens_per_second_confidence_half_width = 1.0;
    strategy.evidence_id = "cohort-serving-evidence";
    manifest.strategies.push_back(strategy);

    const auto path = std::filesystem::temp_directory_path() / "air-serving-cohort-manifest.json";
    check(air::save_manifest(manifest, path).is_ok(), "cohort serving manifest saves");

    air::SchedulerConfig scheduler;
    scheduler.max_active_requests = 4;
    scheduler.token_budget_per_cycle = 8;
    scheduler.prefill_quantum_tokens = 1;
    scheduler.reference_kv_page_tokens = 2;
    air::ManifestConfig manifest_config;
    manifest_config.path = path;
    manifest_config.require = true;

    auto service_result = air::InferenceService::create(
        model, air::BackendPreference::automatic, 0, scheduler, {}, manifest_config);
    check(service_result.is_ok(), "adaptive cohort service starts");
    if (service_result) {
        air::InferenceRequest request;
        request.prompt = "a";
        request.generation.max_new_tokens = 2;
        request.generation.sampling.temperature = 0.0;
        const std::vector<air::InferenceRequest> requests(4, request);
        auto responses = service_result.value()->generate_cohort(requests);
        check(responses && responses.value().size() == 4,
              "atomic cohort returns every response through the production service");
        if (responses) {
            for (const auto& response : responses.value()) {
                check(response.metrics.strategy_id == strategy.strategy_id,
                      "every atomic cohort member sees the exact concurrency-qualified strategy");
                check(response.metrics.planner_mode == "adaptive",
                      "atomic cohort stays on the adaptive production planner");
            }
        }
    }
    std::filesystem::remove(path);
}

void test_corrupt_manifest_fallback_and_require() {
    const auto path = std::filesystem::temp_directory_path() / "air-corrupt-manifest.json";
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << "{ this is not valid json";
    }
    air::ManifestConfig optional;
    optional.path = path;
    optional.enabled = true;
    optional.require = false;
    auto fallback = air::InferenceService::create(
        tiny_model(), air::BackendPreference::automatic, 0, {}, {}, optional);
    check(fallback.is_ok(), "corrupt optional manifest falls back to static execution");
    if (fallback) {
        const auto snapshot = fallback.value()->snapshot();
        check(snapshot.planner_mode == "static" && snapshot.manifest_status.rfind("invalid:", 0) == 0,
              "runtime reports corrupt optional manifest instead of silently claiming adaptive mode");
    }

    air::ManifestConfig required = optional;
    required.require = true;
    auto rejected = air::InferenceService::create(
        tiny_model(), air::BackendPreference::automatic, 0, {}, {}, required);
    check(!rejected && rejected.status().code() == air::ErrorCode::invalid_state,
          "corrupt required manifest prevents service startup");
    std::filesystem::remove(path);
}

void test_adaptive_manifest_is_consumed_by_service() {
    auto model = tiny_model();
    air::ExecutionManifest manifest;
    manifest.schema_version = air::execution_manifest_schema_version;
    manifest.air_version = air::version_string();
    manifest.model_digest = air::model_digest(*model);
    manifest.hardware_digest = air::hardware_digest();
    manifest.manifest_id = "manifest:serving-test";
    air::QualifiedStrategy strategy;
    strategy.workload = air::WorkloadClass::small;
    strategy.strategy_id = "small-reference-p1-k2";
    strategy.plan.backend = air::BackendKind::reference;
    strategy.plan.strategy_id = strategy.strategy_id;
    strategy.plan.scheduling.prefill_quantum_tokens = 1;
    strategy.plan.kv.page_tokens = 2;
    strategy.strict_qualified = true;
    strategy.region = air::WorkloadRegion{0, 256, 1, 1};
    strategy.samples = 3;
    strategy.p50_ttft_ms = 5.0;
    strategy.p50_total_ms = 10.0;
    strategy.mean_prefill_tokens_per_second = 100.0;
    strategy.mean_decode_tokens_per_second = 50.0;
    strategy.evidence_id = "serving-test-evidence";
    manifest.strategies.push_back(strategy);
    const auto path = std::filesystem::temp_directory_path() / "air-serving-manifest.json";
    check(air::save_manifest(manifest, path).is_ok(), "serving test manifest saves");

    air::SchedulerConfig scheduler;
    scheduler.prefix_cache_entries = 8;
    air::ManifestConfig manifest_config;
    manifest_config.path = path;
    manifest_config.require = true;
    auto service_result = air::InferenceService::create(
        model, air::BackendPreference::automatic, 0, scheduler, {}, manifest_config);
    check(service_result.is_ok(), "automatic service accepts fresh execution manifest");
    if (service_result) {
        air::InferenceRequest request;
        request.prompt = "a";
        request.generation.max_new_tokens = 2;
        request.generation.sampling.temperature = 0.0;
        auto response = service_result.value()->generate(request);
        check(response.is_ok(), "adaptive service executes request");
        if (response) {
            check(response.value().metrics.planner_mode == "adaptive",
                  "request metrics expose adaptive planner mode");
            check(response.value().metrics.strategy_id == strategy.strategy_id,
                  "request metrics expose qualified strategy id");
            check(response.value().metrics.backend == "reference",
                  "qualified strategy controls actual backend");
        }
        const auto snapshot = service_result.value()->snapshot();
        check(snapshot.manifest_status == "loaded" && snapshot.manifest_id == manifest.manifest_id,
              "runtime snapshot exposes loaded manifest identity");
        check(snapshot.strategy_id == strategy.strategy_id && snapshot.planned_kv_page_tokens == 2,
              "runtime snapshot exposes qualified execution geometry");
    }
    std::filesystem::remove(path);
}

} // namespace

int main() {
    test_paged_kv_fork_and_cow();
    test_service_scheduler_prefix_and_streaming();
    test_benchmark_uses_serving_pipeline();
    test_cancellation_and_reclamation();
    test_shutdown_cancels_work();
    test_stream_delivery_isolated_and_bounded();
    test_repeated_service_lifecycle();
    test_atomic_cohort_preserves_exact_concurrency_region();
    test_corrupt_manifest_fallback_and_require();
    test_adaptive_manifest_is_consumed_by_service();
    if (failures != 0) {
        std::cerr << failures << " serving test(s) failed\n";
        return 1;
    }
    std::cout << "serving tests passed\n";
    return 0;
}

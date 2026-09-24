#include "air/format.hpp"
#include "air/model.hpp"
#include "air/manifest.hpp"
#include "air/version.hpp"
#include "../src/statistics.hpp"
#include "air/runtime.hpp"
#include "air/tokenizer.hpp"

#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
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

class BinaryWriter {
public:
    explicit BinaryWriter(const std::filesystem::path& path) : out_(path, std::ios::binary) {
        if (!out_) throw std::runtime_error("unable to create test GGUF");
    }

    void raw(std::string_view data) { out_.write(data.data(), static_cast<std::streamsize>(data.size())); }
    void u8(std::uint8_t value) { out_.put(static_cast<char>(value)); }
    void u32(std::uint32_t value) {
        for (unsigned i = 0; i < 4; ++i) u8(static_cast<std::uint8_t>((value >> (i * 8U)) & 0xffU));
    }
    void i32(std::int32_t value) { u32(std::bit_cast<std::uint32_t>(value)); }
    void u64(std::uint64_t value) {
        for (unsigned i = 0; i < 8; ++i) u8(static_cast<std::uint8_t>((value >> (i * 8U)) & 0xffU));
    }
    void f32(float value) { u32(std::bit_cast<std::uint32_t>(value)); }
    void string(std::string_view value) { u64(value.size()); raw(value); }
    void pad_to(std::uint64_t alignment) {
        const auto pos = static_cast<std::uint64_t>(out_.tellp());
        const auto remainder = pos % alignment;
        if (remainder == 0) return;
        for (std::uint64_t i = 0; i < alignment - remainder; ++i) u8(0);
    }

    void kv_string(std::string_view key, std::string_view value) {
        string(key); u32(8); string(value);
    }
    void kv_u32(std::string_view key, std::uint32_t value) {
        string(key); u32(4); u32(value);
    }
    void kv_f32(std::string_view key, float value) {
        string(key); u32(6); f32(value);
    }
    void kv_bool(std::string_view key, bool value) {
        string(key); u32(7); u8(value ? 1U : 0U);
    }
    void kv_strings(std::string_view key, const std::vector<std::string>& values) {
        string(key); u32(9); u32(8); u64(values.size());
        for (const auto& value : values) string(value);
    }
    void kv_i32s(std::string_view key, const std::vector<std::int32_t>& values) {
        string(key); u32(9); u32(5); u64(values.size());
        for (const auto value : values) i32(value);
    }

private:
    std::ofstream out_;
};

std::filesystem::path write_test_gguf(std::uint32_t version = 3,
                                      std::uint32_t tensor_type = 0,
                                      std::uint32_t bos_id = 0,
                                      std::uint32_t sliding_window = 0) {
    const auto path = std::filesystem::temp_directory_path() / "air-prompt2-test.gguf";
    BinaryWriter out(path);
    out.raw("GGUF");
    out.u32(version);
    out.u64(1);   // tensors
    out.u64(20U + (sliding_window == 0U ? 0U : 1U));  // metadata entries

    out.kv_string("general.architecture", "qwen2");
    out.kv_string("general.name", "AIR synthetic model");
    out.kv_u32("general.alignment", 32);
    out.kv_u32("qwen2.block_count", 1);
    out.kv_u32("qwen2.context_length", 128);
    out.kv_u32("qwen2.embedding_length", 4);
    out.kv_u32("qwen2.feed_forward_length", 8);
    out.kv_u32("qwen2.attention.head_count", 2);
    out.kv_u32("qwen2.attention.head_count_kv", 1);
    if (sliding_window != 0U) out.kv_u32("qwen2.attention.sliding_window", sliding_window);
    out.kv_u32("qwen2.rope.dimension_count", 2);
    out.kv_f32("qwen2.rope.freq_base", 1000000.0F);
    out.kv_f32("qwen2.attention.layer_norm_rms_epsilon", 0.000001F);
    out.kv_string("tokenizer.ggml.model", "gpt2");
    out.kv_string("tokenizer.ggml.pre", "gpt2");
    out.kv_strings("tokenizer.ggml.tokens", {"a", "b"});
    out.kv_strings("tokenizer.ggml.merges", {});
    out.kv_i32s("tokenizer.ggml.token_type", {1, 1});
    out.kv_u32("tokenizer.ggml.bos_token_id", bos_id);
    out.kv_u32("tokenizer.ggml.eos_token_id", 1);
    out.kv_bool("tokenizer.ggml.add_bos_token", false);

    out.string("token_embd.weight");
    out.u32(2);
    out.u64(4);
    out.u64(2);
    out.u32(tensor_type);
    out.u64(0); // relative tensor-data offset
    out.pad_to(32);
    for (std::uint32_t i = 0; i < 8; ++i) out.f32(static_cast<float>(i));
    return path;
}

air::ModelDefinition valid_model() {
    air::ModelFingerprint fingerprint{"test", "test_arch", "unit"};
    air::ModelConfig config;
    config.architecture = "test_arch";
    config.layer_count = 1;
    config.embedding_size = 8;
    config.attention_head_count = 2;
    config.kv_head_count = 1;
    config.context_length = 32;
    config.vocabulary_size = 4;

    air::TokenizerDefinition tokenizer;
    tokenizer.model = "test";
    tokenizer.vocabulary = {"a", "b", "c", "d"};

    air::TensorDescriptor tensor;
    tensor.name = "weight";
    tensor.type = air::DataType::f32;
    tensor.format_type = 0;
    tensor.shape.dimensions = {2, 2};
    tensor.byte_size = 16;
    tensor.byte_size_exact = true;
    return {std::move(fingerprint), std::move(config), std::move(tokenizer), {std::move(tensor)}};
}

std::shared_ptr<const air::TokenizerDefinition> test_bpe_definition() {
    auto definition = std::make_shared<air::TokenizerDefinition>();
    definition->model = "gpt2";
    definition->pre_tokenizer = "gpt2";
    definition->vocabulary = {
        "h", "e", "l", "o", "w", "r", "d", "Ġ",
        "hello", "Ġworld", "<|x|>", "<s>", "Ã", "©"
    };
    definition->merges = {
        "h e", "he l", "hel l", "hell o",
        "Ġ w", "Ġw o", "Ġwo r", "Ġwor l", "Ġworl d"
    };
    definition->token_types.assign(definition->vocabulary.size(), 1);
    definition->token_types[10] = 3;
    definition->token_types[11] = 3;
    definition->special_ids.bos = 11;
    return definition;
}

void test_tensor_shape() {
    air::TensorShape shape{{2, 3, 4}};
    check(shape.element_count() == 24, "tensor element count");

    air::TensorShape invalid{{2, 0, 4}};
    check(invalid.element_count() == 0, "zero dimension rejected");
}

void test_model_validation_and_lookup() {
    auto model = valid_model();
    check(model.validate().is_ok(), "valid model passes validation");
    check(model.find_tensor("weight") != nullptr, "existing tensor is found");
    check(model.find_tensor("missing") == nullptr, "missing tensor is absent");
}

void test_static_planner_boundary() {
    auto model = std::make_shared<air::ModelDefinition>(valid_model());
    air::Runtime runtime(model, std::make_unique<air::StaticPlanner>());
    const air::RequestProfile request{128, 32, 1};
    const air::RuntimeSnapshot snapshot{};
    const auto plan = runtime.plan(request, snapshot);
    check(plan.backend == air::BackendKind::reference, "static planner owns backend choice");
}

void test_gguf_load_and_mapping() {
    const auto path = write_test_gguf();
    air::GgufFormat gguf;
    check(gguf.can_open(path), "GGUF magic is recognized");
    auto result = gguf.load(path);
    check(result.is_ok(), "GGUF V3 model loads");
    if (result) {
        const auto& model = result.value();
        check(model.fingerprint().format == "gguf-v3", "GGUF version enters fingerprint");
        check(model.fingerprint().architecture == "qwen2", "architecture is extracted");
        check(model.fingerprint().model_id == "AIR synthetic model", "model name is extracted");
        check(model.config().layer_count == 1, "layer count is extracted");
        check(model.config().feed_forward_size == 8, "feed-forward size is extracted");
        check(model.config().vocabulary_size == 2, "vocabulary size comes from tokenizer");
        check(model.tokenizer().model == "gpt2", "tokenizer model is extracted");
        check(model.tensors().size() == 1, "tensor table is extracted");
        const auto* tensor = model.find_tensor("token_embd.weight");
        check(tensor != nullptr, "tensor lookup works after GGUF load");
        if (tensor) {
            check(tensor->type == air::DataType::f32, "GGML tensor type maps to AIR type");
            check(tensor->byte_size == 32 && tensor->byte_size_exact, "F32 tensor byte size is exact");
            auto bytes = model.tensor_bytes(*tensor);
            check(bytes.is_ok() && bytes.value().size() == 32, "mapped tensor bytes are accessible");
        }
        check(model.storage() && model.storage()->size_bytes() == std::filesystem::file_size(path),
              "model retains read-only mapped storage");
    }
    std::filesystem::remove(path);
}

void test_sliding_window_metadata_is_preserved() {
    const auto path = write_test_gguf(3, 0, 0, 64);
    air::GgufFormat gguf;
    auto result = gguf.load(path);
    check(result && result.value().config().attention_sliding_window == 64,
          "GGUF sliding-window metadata enters canonical model config");
    std::filesystem::remove(path);
}

void test_unsupported_gguf_version() {
    const auto path = write_test_gguf(2);
    air::GgufFormat gguf;
    const auto result = gguf.load(path);
    check(!result.is_ok(), "unsupported GGUF version is rejected");
    check(result.status().code() == air::ErrorCode::unsupported,
          "unsupported GGUF version reports unsupported");
    std::filesystem::remove(path);
}


void test_unknown_tensor_encoding_remains_inspectable() {
    const auto path = write_test_gguf(3, 99);
    air::GgufFormat gguf;
    auto result = gguf.load(path);
    check(result.is_ok(), "unknown GGML tensor encoding remains parseable");
    if (result) {
        const auto& tensor = result.value().tensors().front();
        check(tensor.type == air::DataType::unknown, "unknown encoding is not misrepresented as executable type");
        check(tensor.format_type == 99, "raw GGML encoding id is preserved");
        check(!tensor.byte_size_exact && tensor.byte_size == 32, "unknown encoding retains validated storage span");
    }
    std::filesystem::remove(path);
}

void test_invalid_special_token_id_is_rejected() {
    const auto path = write_test_gguf(3, 0, 99);
    air::GgufFormat gguf;
    auto result = gguf.load(path);
    check(!result.is_ok(), "out-of-range special token id is rejected");
    check(result.status().code() == air::ErrorCode::data_error,
          "invalid special token id reports corrupt data");
    std::filesystem::remove(path);
}

void test_unsupported_tokenizer_family_is_explicit() {
    auto definition = std::make_shared<air::TokenizerDefinition>();
    definition->model = "llama";
    definition->vocabulary = {"a"};
    auto result = air::create_tokenizer(definition);
    check(!result.is_ok() && result.status().code() == air::ErrorCode::unsupported,
          "unsupported tokenizer family is explicit rather than approximated");
}

void test_gpt2_bpe_tokenizer() {
    auto tokenizer_result = air::create_tokenizer(test_bpe_definition());
    check(tokenizer_result.is_ok(), "GPT-2 BPE tokenizer is constructed");
    if (!tokenizer_result) return;
    auto& tokenizer = tokenizer_result.value();

    auto tokens = tokenizer->encode("hello world");
    check(tokens.is_ok(), "GPT-2 BPE encodes basic text");
    if (tokens) {
        check(tokens.value() == std::vector<air::TokenId>({8, 9}), "BPE merge ranks produce expected tokens");
        auto decoded = tokenizer->decode(tokens.value());
        check(decoded.is_ok() && decoded.value() == "hello world", "GPT-2 byte codec round-trips text");
    }

    auto unicode = tokenizer->encode("é");
    check(unicode.is_ok() && unicode.value() == std::vector<air::TokenId>({12, 13}),
          "GPT-2 byte encoding handles multi-byte UTF-8");
    if (unicode) {
        auto decoded = tokenizer->decode(unicode.value());
        check(decoded.is_ok() && decoded.value() == "é", "multi-byte UTF-8 round-trips through byte codec");
    }

    auto special = tokenizer->encode("hello<|x|> world");
    check(special.is_ok() && special.value() == std::vector<air::TokenId>({8, 10, 9}),
          "special tokens are recognized without crossing BPE boundaries");
    if (special) {
        auto decoded = tokenizer->decode(special.value());
        check(decoded.is_ok() && decoded.value() == "hello<|x|> world", "special token rendering round-trips");
        auto hidden = tokenizer->decode(special.value(), false);
        check(hidden.is_ok() && hidden.value() == "hello world", "special tokens can be omitted during decode");
    }

    air::TokenizeOptions with_bos;
    with_bos.add_bos = true;
    auto bos = tokenizer->encode("hello", with_bos);
    check(bos.is_ok() && bos.value() == std::vector<air::TokenId>({11, 8}), "explicit BOS insertion uses metadata id");
}


void test_shared_percentile_semantics() {
    const std::vector<double> even{10.0, 20.0};
    check(air::detail::percentile_linear(even, 0.50) == 15.0,
          "shared percentile helper interpolates the median of an even sample");
    const std::vector<double> four{10.0, 20.0, 30.0, 40.0};
    check(std::abs(air::detail::percentile_linear(four, 0.95) - 38.5) < 1e-12,
          "shared percentile helper linearly interpolates fractional percentile ranks");
}

void test_manifest_roundtrip_and_planner() {
    auto model = std::make_shared<air::ModelDefinition>(valid_model());
    air::ExecutionManifest manifest;
    manifest.schema_version = air::execution_manifest_schema_version;
    manifest.air_version = air::version_string();
    manifest.model_digest = air::model_digest(*model);
    manifest.hardware_digest = air::hardware_digest();
    manifest.manifest_id = "manifest:test";
    air::QualifiedStrategy strategy;
    strategy.workload = air::WorkloadClass::small;
    strategy.strategy_id = "small-reference-p16-k16";
    strategy.plan.backend = air::BackendKind::reference;
    strategy.plan.strategy_id = strategy.strategy_id;
    strategy.plan.scheduling.prefill_quantum_tokens = 16;
    strategy.plan.kv.page_tokens = 16;
    strategy.plan.linear.prefill_block = air::QuantizedLinearExecutionKind::baseline;
    strategy.plan.linear.decode_block = air::QuantizedLinearExecutionKind::baseline;
    strategy.plan.linear.decode_output = air::QuantizedLinearExecutionKind::baseline;
    strategy.strict_qualified = true;
    strategy.region = air::WorkloadRegion{0, 256, 1, 1};
    strategy.samples = 3;
    strategy.p50_ttft_ms = 10.0;
    strategy.p50_total_ms = 20.0;
    strategy.mean_prefill_tokens_per_second = 100.0;
    strategy.mean_decode_tokens_per_second = 50.0;
    strategy.evidence_seed = 42;
    strategy.evidence_id = "evidence:test";
    manifest.strategies.push_back(strategy);

    const auto path = std::filesystem::temp_directory_path() / "air-manifest-test.json";
    check(air::save_manifest(manifest, path).is_ok(), "execution manifest saves atomically");
    auto loaded = air::load_manifest(path);
    check(loaded.is_ok(), "execution manifest loads");
    if (loaded) {
        auto valid = air::validate_manifest(loaded.value(), *model);
        check(valid.manifest.has_value() && valid.status == "loaded", "fresh manifest validates against model and hardware");
        if (!loaded.value().strategies.empty()) {
            const auto& evidence = loaded.value().strategies.front();
            check(evidence.strict_qualified && evidence.evidence_seed == 42 &&
                  evidence.evidence_id == "evidence:test" &&
                  evidence.region.max_prompt_tokens == 256 &&
                  evidence.plan.linear.prefill_block == air::QuantizedLinearExecutionKind::baseline &&
                  evidence.plan.linear.decode_block == air::QuantizedLinearExecutionKind::baseline &&
                  evidence.plan.linear.decode_output == air::QuantizedLinearExecutionKind::baseline,
                  "current manifest schema round-trips qualification and tactic evidence");
        }
        auto stale = loaded.value();
        stale.air_version = "0.0.0";
        auto rejected = air::validate_manifest(stale, *model);
        check(!rejected.manifest && rejected.status == "stale:air-version",
              "version mismatch invalidates persisted execution evidence");
        air::ExecutionPlan fallback;
        fallback.backend = air::BackendKind::reference;
        fallback.strategy_id = "static";
        fallback.scheduling.prefill_quantum_tokens = 32;
        fallback.kv.page_tokens = 32;
        air::Runtime runtime(model, std::make_unique<air::ManifestPlanner>(loaded.value(), fallback));
        auto plan = runtime.plan(air::RequestProfile{128, 16, 1}, air::RuntimeSnapshot{});
        check(plan.strategy_id == strategy.strategy_id && plan.scheduling.prefill_quantum_tokens == 16,
              "manifest planner selects workload-qualified strategy");
        auto fallback_plan = runtime.plan(air::RequestProfile{1024, 16, 1}, air::RuntimeSnapshot{});
        check(fallback_plan.strategy_id == "static", "manifest planner falls back when workload is unqualified");
    }
    std::filesystem::remove(path);
}

void test_quantized_linear_tactic_parsing() {
    auto baseline = air::quantized_linear_execution_kind_from_string("baseline");
    auto reuse4 = air::quantized_linear_execution_kind_from_string("reuse4");
    auto reuse8 = air::quantized_linear_execution_kind_from_string("batch-reuse8");
    auto hybrid = air::quantized_linear_execution_kind_from_string("q5q8-dp4a-hybrid");
    auto dense = air::quantized_linear_execution_kind_from_string("dense-f32-cublas");
    auto bad = air::quantized_linear_execution_kind_from_string("magic-fast");
    check(baseline && baseline.value() == air::QuantizedLinearExecutionKind::baseline,
          "linear tactic parser accepts baseline");
    check(reuse4 && reuse4.value() == air::QuantizedLinearExecutionKind::batch_reuse4,
          "linear tactic parser accepts reuse4 alias");
    check(reuse8 && reuse8.value() == air::QuantizedLinearExecutionKind::batch_reuse8,
          "linear tactic parser accepts canonical reuse8 name");
    check(!hybrid,
          "rejected q5q8 DP4A hybrid is not production parseable");
    check(dense && dense.value() == air::QuantizedLinearExecutionKind::dense_f32_cublas,
          "linear tactic parser accepts dense FP32 cuBLAS name");
    check(!bad, "linear tactic parser rejects unknown tactics");
}

void test_attention_tactic_parsing() {
    auto baseline = air::attention_execution_kind_from_string("baseline");
    auto online = air::attention_execution_kind_from_string("online-softmax");
    auto alias = air::attention_execution_kind_from_string("online");
    auto bad = air::attention_execution_kind_from_string("magic-attention");
    check(baseline && baseline.value() == air::AttentionExecutionKind::baseline,
          "attention tactic parser accepts baseline");
    check(online && online.value() == air::AttentionExecutionKind::online_softmax,
          "attention tactic parser accepts canonical online-softmax");
    check(alias && alias.value() == air::AttentionExecutionKind::online_softmax,
          "attention tactic parser accepts online alias");
    check(!bad, "attention tactic parser rejects unknown tactics");
}

void test_execution_plan_capability_validation() {
    air::BackendCapabilities reference;
    reference.backend = air::BackendKind::reference;
    reference.kv_storage = air::KvStorageKind::paged;
    reference.sequence_checkpointing = true;
    reference.exact_prefix_reuse = true;

    air::ExecutionPlan reference_plan;
    reference_plan.backend = air::BackendKind::reference;
    reference_plan.strategy_id = "reference-test";
    reference_plan.scheduling.prefill_quantum_tokens = 8;
    check(!air::validate_execution_plan(reference_plan, reference).is_ok(),
          "paged backend rejects plan without physical page geometry");
    reference_plan.kv.page_tokens = 16;
    check(air::validate_execution_plan(reference_plan, reference).is_ok(),
          "paged backend accepts explicit physical page geometry");

    air::BackendCapabilities cuda;
    cuda.backend = air::BackendKind::cuda;
    cuda.prefill_execution = air::PrefillExecutionKind::native_batch;
    cuda.kv_storage = air::KvStorageKind::paged;
    cuda.max_prefill_batch_width = 128;
    cuda.sequence_checkpointing = true;
    cuda.exact_prefix_reuse = true;
    air::ExecutionPlan cuda_plan;
    cuda_plan.backend = air::BackendKind::cuda;
    cuda_plan.strategy_id = "cuda-test";
    cuda_plan.scheduling.prefill_quantum_tokens = 32;
    check(!air::validate_execution_plan(cuda_plan, cuda).is_ok(),
          "paged CUDA backend rejects plan without physical page geometry");
    cuda_plan.kv.page_tokens = 16;
    check(air::validate_execution_plan(cuda_plan, cuda).is_ok(),
          "native-batch paged CUDA accepts truthful execution geometry");
    cuda_plan.linear.prefill_block = air::QuantizedLinearExecutionKind::batch_reuse4;
    check(!air::validate_execution_plan(cuda_plan, cuda).is_ok(),
          "CUDA plan rejects a quantized linear tactic not exposed by backend capabilities");
    cuda.prefill_block_quantized_linear.push_back(air::QuantizedLinearExecutionKind::batch_reuse4);
    check(air::validate_execution_plan(cuda_plan, cuda).is_ok(),
          "CUDA plan accepts an explicitly exposed quantized linear tactic");
    cuda_plan.linear.prefill_block = air::QuantizedLinearExecutionKind::q5q8_dp4a_hybrid;
    check(!air::validate_execution_plan(cuda_plan, cuda).is_ok(),
          "CUDA plan rejects q5q8 DP4A hybrid until the backend advertises it");
    cuda.prefill_block_quantized_linear.push_back(air::QuantizedLinearExecutionKind::q5q8_dp4a_hybrid);
    check(air::validate_execution_plan(cuda_plan, cuda).is_ok(),
          "CUDA plan accepts q5q8 DP4A hybrid only on an explicitly advertised block surface");
    cuda_plan.linear.decode_output = air::QuantizedLinearExecutionKind::dense_f32_cublas;
    check(!air::validate_execution_plan(cuda_plan, cuda).is_ok(),
          "CUDA plan rejects a block-only dense tactic for the terminal vocabulary projection");
    cuda_plan.linear.decode_output = air::QuantizedLinearExecutionKind::q5q8_dp4a_hybrid;
    check(!air::validate_execution_plan(cuda_plan, cuda).is_ok(),
          "CUDA plan rejects the block-only q5q8 DP4A hybrid for the terminal vocabulary projection");
    cuda_plan.linear.decode_output = air::QuantizedLinearExecutionKind::baseline;
    cuda_plan.attention.prefill = air::AttentionExecutionKind::online_softmax;
    check(!air::validate_execution_plan(cuda_plan, cuda).is_ok(),
          "CUDA plan rejects an attention tactic not exposed by backend capabilities");
    cuda.prefill_attention.push_back(air::AttentionExecutionKind::online_softmax);
    check(air::validate_execution_plan(cuda_plan, cuda).is_ok(),
          "CUDA plan accepts an explicitly exposed attention tactic");
}



void test_manifest_semantic_corruption_is_rejected() {
    const auto dir = std::filesystem::temp_directory_path();
    auto write_and_load = [&](const std::string& name, const std::string& json) {
        const auto path = dir / ("air-manifest-corrupt-" + name + ".json");
        { std::ofstream out(path, std::ios::trunc); out << json << '\n'; }
        auto loaded = air::load_manifest(path);
        std::filesystem::remove(path);
        return loaded;
    };

    const std::string prefix =
        "{\"schema_version\":10,\"air_version\":\"" + air::version_string() +
        "\",\"model_digest\":\"m\",\"hardware_digest\":\"h\",\"manifest_id\":\"destruction\",\"strategies\":[";
    const std::string strategy =
        "{\"workload\":\"medium\",\"strategy_id\":\"s1\",\"strict_qualified\":true,"
        "\"region_min_prompt_tokens\":262,\"region_max_prompt_tokens\":262,"
        "\"region_min_active_sequences\":1,\"region_max_active_sequences\":1,"
        "\"backend\":\"cuda\",\"prefill_quantum_tokens\":32,"
        "\"prefill_block_quantized_linear\":\"batch-reuse8\","
        "\"decode_block_quantized_linear\":\"batch-reuse8\","
        "\"decode_output_quantized_linear\":\"batch-reuse8\","
        "\"prefill_attention\":\"online-softmax\",\"decode_attention\":\"baseline\","
        "\"samples\":5,\"mean_prefill_tokens_per_second\":100.0,"
        "\"prefill_tokens_per_second_confidence_half_width\":1.0,"
        "\"p50_ttft_ms\":10.0,\"p50_total_ms\":20.0,"
        "\"preparation_measured\":true,\"eviction_measured\":true,"
        "\"evidence_id\":\"e1\"}";

    auto duplicate_id = write_and_load("duplicate-strategy", prefix + strategy + "," + strategy + "]}");
    check(!duplicate_id, "manifest loader rejects duplicate strategy IDs");

    std::string evidence2 = strategy;
    const std::string sid_from = "\"strategy_id\":\"s1\"";
    const std::string sid_to = "\"strategy_id\":\"s2\"";
    const auto sid = evidence2.find(sid_from);
    if (sid != std::string::npos) evidence2.replace(sid, sid_from.size(), sid_to);
    auto duplicate_evidence = write_and_load("duplicate-evidence", prefix + strategy + "," + evidence2 + "]}");
    check(!duplicate_evidence, "manifest loader rejects duplicate evidence IDs");

    std::string negative = strategy;
    const auto perf = negative.find("100.0");
    negative.replace(perf, 5, "-1.0");
    auto negative_perf = write_and_load("negative-performance", prefix + negative + "]}");
    check(!negative_perf, "manifest loader rejects negative performance statistics");

    std::string no_perf = strategy;
    const std::string perf_field = "\"mean_prefill_tokens_per_second\":100.0,";
    no_perf.erase(no_perf.find(perf_field), perf_field.size());
    auto missing_perf = write_and_load("missing-performance", prefix + no_perf + "]}");
    check(!missing_perf, "strict-qualified manifest strategy must carry applicable performance evidence");

    std::string dense_missing_prep = strategy;
    auto replace_once = [&](std::string& text, const std::string& from, const std::string& to) {
        const auto pos = text.find(from); if (pos != std::string::npos) text.replace(pos, from.size(), to);
    };
    replace_once(dense_missing_prep, "\"strategy_id\":\"s1\"", "\"strategy_id\":\"dense\"");
    replace_once(dense_missing_prep, "\"prefill_block_quantized_linear\":\"batch-reuse8\"", "\"prefill_block_quantized_linear\":\"dense-f32-cublas\"");
    replace_once(dense_missing_prep, "\"decode_block_quantized_linear\":\"batch-reuse8\"", "\"decode_block_quantized_linear\":\"dense-f32-cublas\"");
    replace_once(dense_missing_prep, "\"preparation_measured\":true", "\"prepared_artifact_bytes\":1431306240,\"preparation_measured\":false");
    auto missing_prep = write_and_load("missing-preparation", prefix + dense_missing_prep + "]}");
    check(!missing_prep, "optional prepared strategy must not load without measured preparation evidence");
}

void test_legacy_manifest_schemas_are_rejected() {
    auto model = std::make_shared<air::ModelDefinition>(valid_model());
    for (const std::uint32_t schema : {1U, 2U, 3U, 4U}) {
        const auto path = std::filesystem::temp_directory_path() /
            ("air-manifest-legacy-" + std::to_string(schema) + ".json");
        std::ofstream out(path, std::ios::trunc);
        out << "{\"schema_version\":" << schema << ","
            << "\"air_version\":\"" << air::version_string() << "\","
            << "\"model_digest\":\"" << air::model_digest(*model) << "\","
            << "\"hardware_digest\":\"" << air::hardware_digest() << "\","
            << "\"manifest_id\":\"legacy\",\"strategies\":[]}" << '\n';
        out.close();
        auto loaded = air::load_manifest(path);
        check(!loaded && loaded.status().code() == air::ErrorCode::unsupported,
              "pre-release manifest schemas are rejected explicitly");
        std::filesystem::remove(path);
    }

    air::ExecutionManifest old_write;
    old_write.schema_version = 2;
    const auto write_path = std::filesystem::temp_directory_path() / "air-manifest-old-write.json";
    check(!air::save_manifest(old_write, write_path).is_ok(),
          "AIR only writes the frozen public manifest schema");
    std::filesystem::remove(write_path);
}

} // namespace

int main() {
    test_tensor_shape();
    test_model_validation_and_lookup();
    test_static_planner_boundary();
    test_quantized_linear_tactic_parsing();
    test_attention_tactic_parsing();
    test_execution_plan_capability_validation();
    test_shared_percentile_semantics();
    test_manifest_roundtrip_and_planner();
    test_manifest_semantic_corruption_is_rejected();
    test_legacy_manifest_schemas_are_rejected();
    test_gguf_load_and_mapping();
    test_sliding_window_metadata_is_preserved();
    test_unsupported_gguf_version();
    test_unknown_tensor_encoding_remains_inspectable();
    test_invalid_special_token_id_is_rejected();
    test_unsupported_tokenizer_family_is_explicit();
    test_gpt2_bpe_tokenizer();

    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }

    std::cout << "all AIR core tests passed\n";
    return 0;
}

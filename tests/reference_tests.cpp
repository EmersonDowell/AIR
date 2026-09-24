#include "air/reference.hpp"
#include "air/storage.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

bool near(float left, float right, float tolerance = 1.0e-5F) {
    return std::fabs(left - right) <= tolerance;
}

class ModelBuilder {
public:
    void add_f32(std::string name, std::vector<std::uint64_t> shape, std::span<const float> values) {
        std::vector<std::byte> raw;
        raw.reserve(values.size() * 4U);
        for (const float value : values) {
            const auto bits = std::bit_cast<std::uint32_t>(value);
            for (unsigned shift = 0; shift < 32; shift += 8) {
                raw.push_back(static_cast<std::byte>((bits >> shift) & 0xffU));
            }
        }
        add_raw(std::move(name), air::DataType::f32, 0, std::move(shape), std::move(raw));
    }

    void add_raw(std::string name,
                 air::DataType type,
                 std::uint32_t format_type,
                 std::vector<std::uint64_t> shape,
                 std::vector<std::byte> raw) {
        air::TensorDescriptor descriptor;
        descriptor.name = std::move(name);
        descriptor.type = type;
        descriptor.format_type = format_type;
        descriptor.shape.dimensions = std::move(shape);
        descriptor.byte_offset = bytes_.size();
        descriptor.byte_size = raw.size();
        descriptor.byte_size_exact = true;
        tensors_.push_back(std::move(descriptor));
        bytes_.insert(bytes_.end(), raw.begin(), raw.end());
    }

    std::shared_ptr<const air::ModelDefinition> finish(air::ModelConfig config,
                                                       air::TokenizerDefinition tokenizer,
                                                       std::string file_name) {
        const auto path = std::filesystem::temp_directory_path() / std::move(file_name);
        {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(bytes_.data()),
                      static_cast<std::streamsize>(bytes_.size()));
        }
        auto storage = air::ModelStorage::map_read_only(path);
        if (!storage) throw std::runtime_error(storage.status().message());
        auto model = std::make_shared<air::ModelDefinition>(
            air::ModelFingerprint{"test", config.architecture, "reference-fixture"},
            std::move(config), std::move(tokenizer), std::move(tensors_), std::move(storage).value());
        std::filesystem::remove(path);
        return model;
    }

private:
    std::vector<std::byte> bytes_;
    std::vector<air::TensorDescriptor> tensors_;
};

std::vector<float> zeros(std::size_t count) { return std::vector<float>(count, 0.0F); }
std::vector<float> ones(std::size_t count) { return std::vector<float>(count, 1.0F); }

std::shared_ptr<const air::ModelDefinition> make_tiny_qwen2() {
    constexpr std::uint32_t embedding = 4;
    constexpr std::uint32_t vocab = 4;
    constexpr std::uint32_t ffn = 6;

    air::ModelConfig config;
    config.architecture = "qwen2";
    config.layer_count = 1;
    config.embedding_size = embedding;
    config.feed_forward_size = ffn;
    config.attention_head_count = 2;
    config.kv_head_count = 1;
    config.rope_dimension_count = 2;
    config.context_length = 8;
    config.vocabulary_size = vocab;
    config.rope_frequency_base = 10000.0;
    config.rms_norm_epsilon = 1.0e-5;

    air::TokenizerDefinition tokenizer;
    tokenizer.model = "gpt2";
    tokenizer.pre_tokenizer = "gpt2";
    tokenizer.vocabulary = {"a", "b", "c", "d"};
    tokenizer.token_types = {1, 1, 1, 1};
    tokenizer.special_ids.eos = 3;

    ModelBuilder builder;
    const std::vector<float> identity = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    builder.add_f32("token_embd.weight", {embedding, vocab}, identity);
    builder.add_f32("output_norm.weight", {embedding}, ones(embedding));
    builder.add_f32("blk.0.attn_norm.weight", {embedding}, ones(embedding));
    builder.add_f32("blk.0.attn_q.weight", {embedding, embedding}, zeros(embedding * embedding));
    builder.add_f32("blk.0.attn_k.weight", {embedding, 2}, zeros(embedding * 2));
    builder.add_f32("blk.0.attn_v.weight", {embedding, 2}, zeros(embedding * 2));
    builder.add_f32("blk.0.attn_output.weight", {embedding, embedding}, zeros(embedding * embedding));
    builder.add_f32("blk.0.ffn_norm.weight", {embedding}, ones(embedding));
    builder.add_f32("blk.0.ffn_gate.weight", {embedding, ffn}, zeros(embedding * ffn));
    builder.add_f32("blk.0.ffn_up.weight", {embedding, ffn}, zeros(embedding * ffn));
    builder.add_f32("blk.0.ffn_down.weight", {ffn, embedding}, zeros(ffn * embedding));
    auto model = builder.finish(std::move(config), std::move(tokenizer), "air-reference-qwen2.bin");
    const auto validation = model->validate();
    if (!validation) throw std::runtime_error(validation.message());
    return model;
}

std::shared_ptr<const air::ModelDefinition> make_tensor_fixture() {
    air::ModelConfig config;
    config.architecture = "tensor-test";
    config.layer_count = 1;
    config.embedding_size = 2;
    config.attention_head_count = 1;
    config.kv_head_count = 1;
    config.context_length = 4;
    config.vocabulary_size = 2;

    air::TokenizerDefinition tokenizer;
    tokenizer.model = "test";
    tokenizer.vocabulary = {"a", "b"};

    ModelBuilder builder;
    const std::vector<float> matrix = {1, 2, 3, 4};
    builder.add_f32("matrix", {2, 2}, matrix);

    std::vector<std::byte> q4(18, std::byte{0});
    q4[0] = std::byte{0x00};
    q4[1] = std::byte{0x3c}; // fp16 1.0
    for (std::size_t i = 2; i < q4.size(); ++i) q4[i] = std::byte{0x98};
    builder.add_raw("q4", air::DataType::q4_0, 2, {32}, q4);

    std::vector<std::byte> q5(22, std::byte{0});
    q5[0] = std::byte{0x00};
    q5[1] = std::byte{0x3c}; // fp16 1.0
    for (std::size_t i = 2; i < 6; ++i) q5[i] = std::byte{0xff}; // fifth bit set for all 32 quants
    for (std::size_t i = 6; i < q5.size(); ++i) q5[i] = std::byte{0x10}; // q[0:16]=16, q[16:32]=17
    builder.add_raw("q5", air::DataType::q5_0, 6, {32}, q5);

    std::vector<std::byte> q8(34, std::byte{0});
    q8[0] = std::byte{0x00};
    q8[1] = std::byte{0x3c}; // fp16 1.0
    for (std::size_t i = 0; i < 32; ++i) q8[2 + i] = static_cast<std::byte>(static_cast<std::uint8_t>(i));
    builder.add_raw("q8", air::DataType::q8_0, 8, {32}, q8);

    std::vector<std::byte> q4k(144, std::byte{0});
    q4k[0] = std::byte{0x00};
    q4k[1] = std::byte{0x3c}; // block scale fp16 1.0
    for (std::size_t i = 4; i < 16; ++i) q4k[i] = std::byte{0x01};
    for (std::size_t i = 16; i < q4k.size(); ++i) q4k[i] = std::byte{0x21};
    builder.add_raw("q4k", air::DataType::q4_k, 12, {256}, q4k);

    std::vector<std::byte> q6k(210, std::byte{0});
    for (std::size_t i = 0; i < 128; ++i) q6k[i] = std::byte{0x11};
    for (std::size_t i = 128; i < 192; ++i) q6k[i] = std::byte{0xaa};
    for (std::size_t i = 192; i < 208; ++i) q6k[i] = std::byte{0x01};
    q6k[208] = std::byte{0x00};
    q6k[209] = std::byte{0x3c}; // block scale fp16 1.0
    builder.add_raw("q6k", air::DataType::q6_k, 14, {256}, q6k);

    return builder.finish(std::move(config), std::move(tokenizer), "air-reference-tensors.bin");
}

void test_tensor_reader() {
    auto model = make_tensor_fixture();
    air::ReferenceTensorReader reader(model);
    auto product = reader.matvec("matrix", std::vector<float>{1.0F, 1.0F});
    check(product && product.value().size() == 2 && near(product.value()[0], 3.0F) && near(product.value()[1], 7.0F),
          "reference matvec follows GGML row-major matrix convention");

    auto q4 = reader.vector("q4");
    check(q4 && q4.value().size() == 32, "Q4_0 block decodes");
    if (q4) {
        check(std::all_of(q4.value().begin(), q4.value().begin() + 16, [](float value) { return near(value, 0.0F); }),
              "Q4_0 low nibbles decode with -8 offset");
        check(std::all_of(q4.value().begin() + 16, q4.value().end(), [](float value) { return near(value, 1.0F); }),
              "Q4_0 high nibbles decode with -8 offset");
    }

    auto q5 = reader.vector("q5");
    check(q5 && q5.value().size() == 32, "Q5_0 block decodes");
    if (q5) {
        check(std::all_of(q5.value().begin(), q5.value().begin() + 16, [](float value) { return near(value, 0.0F); }),
              "Q5_0 first half reconstructs fifth bit and low nibbles");
        check(std::all_of(q5.value().begin() + 16, q5.value().end(), [](float value) { return near(value, 1.0F); }),
              "Q5_0 second half reconstructs fifth bit and high nibbles");
    }

    auto q8 = reader.vector("q8");
    check(q8 && q8.value().size() == 32 && near(q8.value()[0], 0.0F) && near(q8.value()[31], 31.0F),
          "Q8_0 block decodes signed bytes with fp16 scale");

    auto q4k = reader.vector("q4k");
    check(q4k && q4k.value().size() == 256, "Q4_K block decodes");
    if (q4k) {
        bool pattern_ok = true;
        for (std::size_t group = 0; group < 4; ++group) {
            for (std::size_t i = 0; i < 32; ++i) {
                pattern_ok = pattern_ok && near(q4k.value()[group * 64 + i], 1.0F);
                pattern_ok = pattern_ok && near(q4k.value()[group * 64 + 32 + i], 2.0F);
            }
        }
        check(pattern_ok, "Q4_K scale packing and low/high nibbles match GGML block geometry");
    }

    auto q6k = reader.vector("q6k");
    check(q6k && q6k.value().size() == 256 &&
              std::all_of(q6k.value().begin(), q6k.value().end(), [](float value) { return near(value, 1.0F); }),
          "Q6_K low/high bit planes and signed scales match GGML block geometry");
}


void test_randomized_quantized_blocks() {
    constexpr std::size_t cases = 32U;
    std::mt19937 rng(0xA17C0DEU);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<int> scale_dist(-8, 8);

    ModelBuilder builder;
    air::ModelConfig config;
    config.architecture = "tensor-random";
    config.layer_count = 1;
    config.embedding_size = 2;
    config.attention_head_count = 1;
    config.kv_head_count = 1;
    config.context_length = 4;
    config.vocabulary_size = 2;
    air::TokenizerDefinition tokenizer;
    tokenizer.model = "test";
    tokenizer.vocabulary = {"a", "b"};

    std::vector<std::byte> q4(cases * 18U);
    std::vector<float> q4_expected(cases * 32U);
    for (std::size_t block = 0; block < cases; ++block) {
        auto* raw = q4.data() + static_cast<std::ptrdiff_t>(block * 18U);
        raw[0] = std::byte{0x00}; raw[1] = std::byte{0x3c};
        for (std::size_t i = 0; i < 16U; ++i) {
            const auto packed = static_cast<std::uint8_t>(byte_dist(rng));
            raw[2 + static_cast<std::ptrdiff_t>(i)] = static_cast<std::byte>(packed);
            q4_expected[block * 32U + i] = static_cast<float>(static_cast<int>(packed & 0x0fU) - 8);
            q4_expected[block * 32U + 16U + i] = static_cast<float>(static_cast<int>(packed >> 4U) - 8);
        }
    }
    builder.add_raw("random_q4", air::DataType::q4_0, 2, {cases * 32U}, q4);

    std::vector<std::byte> q5(cases * 22U);
    std::vector<float> q5_expected(cases * 32U);
    for (std::size_t block = 0; block < cases; ++block) {
        auto* raw = q5.data() + static_cast<std::ptrdiff_t>(block * 22U);
        raw[0] = std::byte{0x00}; raw[1] = std::byte{0x3c};
        for (std::size_t i = 0; i < 4U; ++i) raw[2 + static_cast<std::ptrdiff_t>(i)] = static_cast<std::byte>(byte_dist(rng));
        for (std::size_t i = 0; i < 16U; ++i) raw[6 + static_cast<std::ptrdiff_t>(i)] = static_cast<std::byte>(byte_dist(rng));
        for (std::uint32_t i = 0; i < 32U; ++i) {
            const auto packed = std::to_integer<std::uint8_t>(raw[6 + static_cast<std::ptrdiff_t>(i % 16U)]);
            const auto low = i < 16U ? static_cast<std::uint8_t>(packed & 0x0fU) : static_cast<std::uint8_t>(packed >> 4U);
            const auto high_byte = std::to_integer<std::uint8_t>(raw[2 + static_cast<std::ptrdiff_t>(i / 8U)]);
            const auto high = static_cast<std::uint8_t>(((high_byte >> (i % 8U)) & 1U) << 4U);
            q5_expected[block * 32U + i] = static_cast<float>(static_cast<int>(low | high) - 16);
        }
    }
    builder.add_raw("random_q5", air::DataType::q5_0, 6, {cases * 32U}, q5);

    std::vector<std::byte> q8(cases * 34U);
    std::vector<float> q8_expected(cases * 32U);
    for (std::size_t block = 0; block < cases; ++block) {
        auto* raw = q8.data() + static_cast<std::ptrdiff_t>(block * 34U);
        raw[0] = std::byte{0x00}; raw[1] = std::byte{0x3c};
        for (std::size_t i = 0; i < 32U; ++i) {
            const auto value = static_cast<std::uint8_t>(byte_dist(rng));
            raw[2 + static_cast<std::ptrdiff_t>(i)] = static_cast<std::byte>(value);
            q8_expected[block * 32U + i] = static_cast<float>(static_cast<std::int8_t>(value));
        }
    }
    builder.add_raw("random_q8", air::DataType::q8_0, 8, {cases * 32U}, q8);

    std::vector<std::byte> q4k(cases * 144U);
    std::vector<float> q4k_expected(cases * 256U);
    const auto unpack_scale_min = [](std::uint32_t index, const std::byte* packed,
                                     std::uint8_t& scale, std::uint8_t& minimum) {
        const auto q = [packed](std::uint32_t i) { return static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(packed[i])); };
        if (index < 4U) {
            scale = static_cast<std::uint8_t>(q(index) & 63U);
            minimum = static_cast<std::uint8_t>(q(index + 4U) & 63U);
        } else {
            scale = static_cast<std::uint8_t>((q(index + 4U) & 0x0fU) | ((q(index - 4U) >> 6U) << 4U));
            minimum = static_cast<std::uint8_t>((q(index + 4U) >> 4U) | ((q(index) >> 6U) << 4U));
        }
    };
    for (std::size_t block = 0; block < cases; ++block) {
        auto* raw = q4k.data() + static_cast<std::ptrdiff_t>(block * 144U);
        raw[0] = std::byte{0x00}; raw[1] = std::byte{0x3c}; // d = 1
        raw[2] = std::byte{0x00}; raw[3] = std::byte{0x38}; // dmin = 0.5
        for (std::size_t i = 0; i < 12U; ++i) raw[4 + static_cast<std::ptrdiff_t>(i)] = static_cast<std::byte>(byte_dist(rng));
        for (std::size_t i = 0; i < 128U; ++i) raw[16 + static_cast<std::ptrdiff_t>(i)] = static_cast<std::byte>(byte_dist(rng));
        std::uint32_t scale_index = 0U;
        for (std::uint32_t group = 0; group < 4U; ++group) {
            std::uint8_t s0=0,m0=0,s1=0,m1=0;
            unpack_scale_min(scale_index, raw + 4, s0, m0);
            unpack_scale_min(scale_index + 1U, raw + 4, s1, m1);
            for (std::uint32_t i = 0; i < 32U; ++i) {
                const auto packed = std::to_integer<std::uint8_t>(raw[16 + static_cast<std::ptrdiff_t>(group * 32U + i)]);
                q4k_expected[block * 256U + group * 64U + i] = static_cast<float>(s0) * static_cast<float>(packed & 0x0fU) - 0.5F * static_cast<float>(m0);
                q4k_expected[block * 256U + group * 64U + 32U + i] = static_cast<float>(s1) * static_cast<float>(packed >> 4U) - 0.5F * static_cast<float>(m1);
            }
            scale_index += 2U;
        }
    }
    builder.add_raw("random_q4k", air::DataType::q4_k, 12, {cases * 256U}, q4k);

    std::vector<std::byte> q6k(cases * 210U);
    std::vector<float> q6k_expected(cases * 256U);
    for (std::size_t block = 0; block < cases; ++block) {
        auto* raw = q6k.data() + static_cast<std::ptrdiff_t>(block * 210U);
        for (std::size_t i = 0; i < 192U; ++i) raw[static_cast<std::ptrdiff_t>(i)] = static_cast<std::byte>(byte_dist(rng));
        for (std::size_t i = 0; i < 16U; ++i) raw[192 + static_cast<std::ptrdiff_t>(i)] = static_cast<std::byte>(static_cast<std::uint8_t>(static_cast<std::int8_t>(scale_dist(rng))));
        raw[208] = std::byte{0x00}; raw[209] = std::byte{0x38}; // d = 0.5
        for (std::uint32_t half = 0; half < 2U; ++half) {
            const auto* low = raw + half * 64U;
            const auto* high = raw + 128U + half * 32U;
            const auto* scales = raw + 192U + half * 8U;
            for (std::uint32_t i = 0; i < 32U; ++i) {
                const auto lo0 = std::to_integer<std::uint8_t>(low[i]);
                const auto lo1 = std::to_integer<std::uint8_t>(low[i + 32U]);
                const auto hi = std::to_integer<std::uint8_t>(high[i]);
                const std::uint32_t pair = i / 16U;
                const auto decode = [](std::uint8_t lo, std::uint8_t hi2) { return static_cast<std::int32_t>(lo | static_cast<std::uint8_t>(hi2 << 4U)) - 32; };
                const std::int32_t values[4] = {
                    decode(lo0 & 0x0fU, (hi >> 0U) & 3U), decode(lo1 & 0x0fU, (hi >> 2U) & 3U),
                    decode(lo0 >> 4U, (hi >> 4U) & 3U), decode(lo1 >> 4U, (hi >> 6U) & 3U)};
                const std::uint32_t offsets[4] = {0U,32U,64U,96U};
                const std::uint32_t scale_offsets[4] = {0U,2U,4U,6U};
                for (std::uint32_t lane = 0; lane < 4U; ++lane) {
                    const auto sc = static_cast<std::int8_t>(std::to_integer<std::uint8_t>(scales[pair + scale_offsets[lane]]));
                    const auto out_index = block * 256U + static_cast<std::size_t>(half) * 128U + offsets[lane] + i;
                    q6k_expected[out_index] = 0.5F * static_cast<float>(sc) * static_cast<float>(values[lane]);
                }
            }
        }
    }
    builder.add_raw("random_q6k", air::DataType::q6_k, 14, {cases * 256U}, q6k);

    auto model = builder.finish(std::move(config), std::move(tokenizer), "air-random-quant-blocks.bin");
    air::ReferenceTensorReader reader(model);
    const auto verify = [&](const char* name, const std::vector<float>& expected) {
        auto actual = reader.vector(name);
        check(actual && actual.value().size() == expected.size(), std::string(name) + " randomized decode shape");
        if (!actual) return;
        bool equal = true;
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (!near(actual.value()[i], expected[i], 1.0e-5F)) { equal = false; break; }
        }
        check(equal, std::string(name) + " randomized blocks match independent scalar oracle");
    };
    verify("random_q4", q4_expected);
    verify("random_q5", q5_expected);
    verify("random_q8", q8_expected);
    verify("random_q4k", q4k_expected);
    verify("random_q6k", q6k_expected);
}

void test_kv_transaction() {
    air::ReferenceKvCache cache(2, 1, 2, 4);
    const std::vector<float> key = {1.0F, 2.0F};
    const std::vector<float> value = {3.0F, 4.0F};
    check(cache.append_pending(0, key, value).is_ok(), "KV layer can stage a token");
    check(!cache.commit_token().is_ok(), "KV token cannot commit with an incomplete layer transaction");
    cache.rollback_pending();
    check(cache.size() == 0, "KV rollback preserves committed position");
    check(cache.append_pending(0, key, value).is_ok() && cache.append_pending(1, key, value).is_ok(),
          "all KV layers can stage the same token");
    check(cache.commit_token().is_ok() && cache.size() == 1, "complete KV token transaction commits atomically");
    auto cached = cache.key(1, 0, 0);
    check(cached && near(cached.value()[0], 1.0F) && near(cached.value()[1], 2.0F),
          "committed KV data is addressable by layer/token/head");
}

void test_qwen2_reference_executor() {
    auto model = make_tiny_qwen2();
    auto executor_result = air::ReferenceExecutor::create(model);
    check(executor_result.is_ok(), "supported Qwen2 model creates a reference executor");
    if (!executor_result) return;
    auto& executor = executor_result.value();

    air::ReferenceKvCache cache(1, 1, 2, 8);
    auto first = executor->step(2, cache);
    check(first && cache.size() == 1, "reference step commits one KV token");
    if (first) {
        const auto best = static_cast<air::TokenId>(std::distance(first.value().begin(),
            std::max_element(first.value().begin(), first.value().end())));
        check(best == 2, "identity fixture preserves the input token as greedy next-token winner");
    }

    air::ReferenceKvCache verified_cache(1, 1, 2, 8);
    air::ReferenceKvCache plain_cache(1, 1, 2, 8);
    air::VerificationTrace verified_trace;
    auto verified = executor->step_verified(2, verified_cache, verified_trace);
    auto plain = executor->step(2, plain_cache);
    check(verified && plain && verified.value() == plain.value(),
          "verified reference step preserves production logits exactly");
    check(verified_trace.snapshots().size() == 18U,
          "verified reference step emits every canonical Qwen2 stage once per layer");
    if (verified_trace.snapshots().size() == 18U) {
        check(verified_trace.snapshots().front().stage == air::VerificationStage::embedding &&
              verified_trace.snapshots().back().stage == air::VerificationStage::logits,
              "verification trace ordering begins at embedding and ends at logits");
    }

    auto second = executor->step(1, cache);
    check(second && cache.size() == 2, "incremental decode advances existing KV state");
    if (second) {
        const auto best = static_cast<air::TokenId>(std::distance(second.value().begin(),
            std::max_element(second.value().begin(), second.value().end())));
        check(best == 1, "incremental logits remain deterministic for fixture");
    }

    air::ReferenceKvCache full_cache(1, 1, 2, 1);
    auto full_first = executor->step(0, full_cache);
    auto overflow = executor->step(0, full_cache);
    check(full_first && !overflow && full_cache.size() == 1,
          "context overflow is rejected without advancing committed KV state");

    air::ReferenceKvCache prefill_cache(1, 1, 2, 8);
    const std::vector<air::TokenId> prompt = {0, 2};
    auto prefill = executor->prefill(prompt, prefill_cache);
    check(prefill && prefill_cache.size() == prompt.size(), "prefill consumes every prompt token through the same step path");

    air::GenerationConfig generation;
    generation.max_new_tokens = 3;
    generation.sampling.temperature = 0.0;
    auto generated = executor->generate(std::vector<air::TokenId>{2}, generation);
    check(generated && generated.value().tokens == std::vector<air::TokenId>({2, 2, 2}),
          "greedy generation is deterministic and uses executor logits");

    generation.max_new_tokens = 1;
    generation.sampling.temperature = 1.0;
    generation.sampling.top_p = 0.0;
    auto invalid_sampling = executor->generate(std::vector<air::TokenId>{2}, generation);
    check(!invalid_sampling && invalid_sampling.status().code() == air::ErrorCode::invalid_argument,
          "invalid probabilistic sampling configuration fails explicitly");
}

void test_output_bias_and_scaled_rope_contracts() {
    auto base = make_tiny_qwen2();

    ModelBuilder builder;
    air::ModelConfig config = base->config();
    air::TokenizerDefinition tokenizer = base->tokenizer();
    for (const auto& tensor : base->tensors()) {
        auto bytes = base->tensor_bytes(tensor);
        if (!bytes) throw std::runtime_error(bytes.status().message());
        builder.add_raw(tensor.name, tensor.type, tensor.format_type, tensor.shape.dimensions,
                        std::vector<std::byte>(bytes.value().begin(), bytes.value().end()));
    }
    const std::vector<float> output_bias = {0.0F, 0.0F, 0.0F, 10.0F};
    builder.add_f32("output.bias", {config.vocabulary_size}, output_bias);
    auto biased = builder.finish(config, tokenizer, "air-reference-qwen2-bias.bin");
    auto biased_executor = air::ReferenceExecutor::create(biased);
    check(biased_executor.is_ok(), "optional Qwen2 output bias is accepted");
    if (biased_executor) {
        air::ReferenceKvCache cache(1, 1, 2, 8);
        auto logits = biased_executor.value()->step(0, cache);
        check(logits && std::distance(logits.value().begin(),
              std::max_element(logits.value().begin(), logits.value().end())) == 3,
              "optional output bias is applied before token selection");
    }

    config = base->config();
    config.rope_scaling_type = "linear";
    auto scaled = std::make_shared<air::ModelDefinition>(
        air::ModelFingerprint{"test", "qwen2", "scaled-rope"}, std::move(config), base->tokenizer(),
        base->tensors(), base->storage());
    auto scaled_result = air::ReferenceExecutor::create(scaled);
    check(!scaled_result && scaled_result.status().code() == air::ErrorCode::unsupported,
          "unimplemented RoPE scaling is rejected instead of approximated");

    config = base->config();
    config.attention_sliding_window = 4;
    auto sliding = std::make_shared<air::ModelDefinition>(
        air::ModelFingerprint{"test", "qwen2", "sliding-window"}, std::move(config), base->tokenizer(),
        base->tensors(), base->storage());
    auto sliding_result = air::ReferenceExecutor::create(sliding);
    check(!sliding_result && sliding_result.status().code() == air::ErrorCode::unsupported,
          "Qwen2 sliding-window attention is rejected until its semantics are implemented");
}

void test_reference_rejects_wrong_architecture() {
    auto model = make_tiny_qwen2();
    air::ModelConfig config = model->config();
    config.architecture = "qwen3";
    air::TokenizerDefinition tokenizer = model->tokenizer();
    std::vector<air::TensorDescriptor> tensors = model->tensors();
    auto altered = std::make_shared<air::ModelDefinition>(
        air::ModelFingerprint{"test", "qwen3", "wrong-arch"}, std::move(config), std::move(tokenizer),
        std::move(tensors), model->storage());
    auto result = air::ReferenceExecutor::create(altered);
    check(!result && result.status().code() == air::ErrorCode::unsupported,
          "reference executor explicitly rejects unimplemented architectures");
}

} // namespace

int main() {
    test_tensor_reader();
    test_randomized_quantized_blocks();
    test_kv_transaction();
    test_qwen2_reference_executor();
    test_output_bias_and_scaled_rope_contracts();
    test_reference_rejects_wrong_architecture();

    if (failures != 0) {
        std::cerr << failures << " reference test(s) failed\n";
        return 1;
    }
    std::cout << "all AIR reference tests passed\n";
    return 0;
}

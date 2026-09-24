#include "air/format.hpp"

#include "air/storage.hpp"
#include "formats/gguf/gguf_reader.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace air {
namespace {

struct EncodingGeometry {
    DataType type{DataType::unknown};
    std::uint64_t block_elements{0};
    std::uint64_t block_bytes{0};
};

EncodingGeometry encoding_geometry(std::uint32_t ggml_type) {
    switch (ggml_type) {
    case 0: return {DataType::f32, 1, 4};
    case 1: return {DataType::f16, 1, 2};
    case 2: return {DataType::q4_0, 32, 18};
    case 3: return {DataType::q4_1, 32, 20};
    case 6: return {DataType::q5_0, 32, 22};
    case 7: return {DataType::q5_1, 32, 24};
    case 8: return {DataType::q8_0, 32, 34};
    case 9: return {DataType::q8_1, 32, 40};
    case 10: return {DataType::q2_k, 256, 84};
    case 11: return {DataType::q3_k, 256, 110};
    case 12: return {DataType::q4_k, 256, 144};
    case 13: return {DataType::q5_k, 256, 176};
    case 14: return {DataType::q6_k, 256, 210};
    case 15: return {DataType::q8_k, 256, 292};
    case 16: return {DataType::iq2_xxs, 256, 66};
    case 17: return {DataType::iq2_xs, 256, 74};
    case 18: return {DataType::iq3_xxs, 256, 98};
    case 19: return {DataType::iq1_s, 256, 50};
    case 20: return {DataType::iq4_nl, 32, 18};
    case 21: return {DataType::iq3_s, 256, 110};
    case 22: return {DataType::iq2_s, 256, 82};
    case 23: return {DataType::iq4_xs, 256, 136};
    case 24: return {DataType::i8, 1, 1};
    case 25: return {DataType::i16, 1, 2};
    case 26: return {DataType::i32, 1, 4};
    case 27: return {DataType::i64, 1, 8};
    case 28: return {DataType::f64, 1, 8};
    case 29: return {DataType::iq1_m, 256, 56};
    case 30: return {DataType::bf16, 1, 2};
    case 34: return {DataType::tq1_0, 256, 54};
    case 35: return {DataType::tq2_0, 256, 66};
    case 39: return {DataType::mxfp4, 0, 0};
    case 40: return {DataType::nvfp4, 0, 0};
    case 41: return {DataType::q1_0, 0, 0};
    case 42: return {DataType::q2_0, 0, 0};
    default: return {};
    }
}

Result<std::uint32_t> narrow_u32(std::uint64_t value, std::string_view key) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        return Status::data_error("GGUF metadata exceeds AIR integer range: " + std::string(key));
    }
    return static_cast<std::uint32_t>(value);
}

Result<std::optional<TokenId>> token_id(const gguf::Document& document,
                                        std::string_view key,
                                        std::size_t vocabulary_size) {
    const auto* raw = document.find(key);
    if (!raw) return std::optional<TokenId>{};
    const auto value = document.optional_u64(key);
    if (!value) {
        return Status::data_error("GGUF special-token id has wrong type: " + std::string(key));
    }
    if (*value >= vocabulary_size ||
        *value > static_cast<std::uint64_t>(std::numeric_limits<TokenId>::max())) {
        return Status::data_error("GGUF special-token id is outside vocabulary: " + std::string(key));
    }
    return std::optional<TokenId>{static_cast<TokenId>(*value)};
}

Result<ModelConfig> build_config(const gguf::Document& document,
                                 const std::string& architecture,
                                 std::size_t vocabulary_size) {
    ModelConfig config;
    config.architecture = architecture;

    const auto make_key = [&architecture](std::string_view suffix) {
        return architecture + "." + std::string(suffix);
    };

    auto layers = document.require_u64(make_key("block_count"));
    auto embedding = document.require_u64(make_key("embedding_length"));
    auto heads = document.require_u64(make_key("attention.head_count"));
    auto context = document.require_u64(make_key("context_length"));
    if (!layers) return layers.status();
    if (!embedding) return embedding.status();
    if (!heads) return heads.status();
    if (!context) return context.status();

    auto layer_count = narrow_u32(layers.value(), make_key("block_count"));
    auto embedding_size = narrow_u32(embedding.value(), make_key("embedding_length"));
    auto attention_heads = narrow_u32(heads.value(), make_key("attention.head_count"));
    if (!layer_count) return layer_count.status();
    if (!embedding_size) return embedding_size.status();
    if (!attention_heads) return attention_heads.status();

    config.layer_count = layer_count.value();
    config.embedding_size = embedding_size.value();
    config.attention_head_count = attention_heads.value();
    config.context_length = context.value();
    config.vocabulary_size = vocabulary_size;

    const auto kv_heads = document.optional_u64(make_key("attention.head_count_kv")).value_or(heads.value());
    auto kv_head_count = narrow_u32(kv_heads, make_key("attention.head_count_kv"));
    if (!kv_head_count) return kv_head_count.status();
    config.kv_head_count = kv_head_count.value();
    config.attention_sliding_window =
        document.optional_u64(make_key("attention.sliding_window")).value_or(0U);

    if (const auto ff = document.optional_u64(make_key("feed_forward_length"))) {
        auto value = narrow_u32(*ff, make_key("feed_forward_length"));
        if (!value) return value.status();
        config.feed_forward_size = value.value();
    }

    const auto default_rope = config.attention_head_count == 0
                                  ? 0U
                                  : config.embedding_size / config.attention_head_count;
    const auto rope = document.optional_u64(make_key("rope.dimension_count")).value_or(default_rope);
    auto rope_dimensions = narrow_u32(rope, make_key("rope.dimension_count"));
    if (!rope_dimensions) return rope_dimensions.status();
    config.rope_dimension_count = rope_dimensions.value();

    config.rope_frequency_base = document.optional_number(make_key("rope.freq_base")).value_or(0.0);
    config.rope_scaling_type = document.optional_string(make_key("rope.scaling.type")).value_or("none");
    config.rope_scaling_factor = document.optional_number(make_key("rope.scaling.factor")).value_or(1.0);
    config.rope_scale_linear = document.optional_number(make_key("rope.scale_linear")).value_or(1.0);
    config.rms_norm_epsilon =
        document.optional_number(make_key("attention.layer_norm_rms_epsilon")).value_or(0.0);

    if (const auto declared_vocab = document.optional_u64(make_key("vocab_size"));
        declared_vocab && *declared_vocab != vocabulary_size) {
        return Status::data_error("GGUF architecture vocab_size disagrees with tokenizer token count");
    }

    return config;
}

Result<TokenizerDefinition> build_tokenizer(const gguf::Document& document) {
    auto model = document.require_string("tokenizer.ggml.model");
    if (!model) return model.status();
    const auto* tokens = document.string_array("tokenizer.ggml.tokens");
    if (!tokens || tokens->empty()) {
        return Status::data_error("missing or empty tokenizer.ggml.tokens");
    }
    if (tokens->size() > static_cast<std::size_t>(std::numeric_limits<TokenId>::max())) {
        return Status::unsupported("tokenizer vocabulary exceeds AIR TokenId range");
    }

    TokenizerDefinition definition;
    definition.model = std::move(model).value();
    definition.pre_tokenizer = document.optional_string("tokenizer.ggml.pre").value_or("");
    definition.chat_template = document.optional_string("tokenizer.chat_template").value_or("");
    definition.vocabulary = *tokens;
    definition.add_bos = document.optional_bool("tokenizer.ggml.add_bos_token").value_or(false);
    definition.add_eos = document.optional_bool("tokenizer.ggml.add_eos_token").value_or(false);

    if (const auto* merges = document.string_array("tokenizer.ggml.merges")) {
        definition.merges = *merges;
    }
    if (const auto* scores = document.float_array("tokenizer.ggml.scores")) {
        definition.scores.reserve(scores->size());
        for (const double score : *scores) definition.scores.push_back(static_cast<float>(score));
    }
    if (const auto* types = document.signed_array("tokenizer.ggml.token_type")) {
        definition.token_types.reserve(types->size());
        for (const auto type : *types) {
            if (type < std::numeric_limits<std::int32_t>::min() ||
                type > std::numeric_limits<std::int32_t>::max()) {
                return Status::data_error("tokenizer token type exceeds int32 range");
            }
            definition.token_types.push_back(static_cast<std::int32_t>(type));
        }
    }

    const auto vocabulary_size = definition.vocabulary.size();
    auto bos = token_id(document, "tokenizer.ggml.bos_token_id", vocabulary_size);
    auto eos = token_id(document, "tokenizer.ggml.eos_token_id", vocabulary_size);
    auto unknown = token_id(document, "tokenizer.ggml.unknown_token_id", vocabulary_size);
    auto padding = token_id(document, "tokenizer.ggml.padding_token_id", vocabulary_size);
    auto eot = token_id(document, "tokenizer.ggml.eot_token_id", vocabulary_size);
    auto eom = token_id(document, "tokenizer.ggml.eom_token_id", vocabulary_size);
    if (!bos) return bos.status();
    if (!eos) return eos.status();
    if (!unknown) return unknown.status();
    if (!padding) return padding.status();
    if (!eot) return eot.status();
    if (!eom) return eom.status();
    definition.special_ids.bos = bos.value();
    definition.special_ids.eos = eos.value();
    definition.special_ids.unknown = unknown.value();
    definition.special_ids.padding = padding.value();
    definition.special_ids.eot = eot.value();
    definition.special_ids.eom = eom.value();
    return definition;
}

Result<std::vector<TensorDescriptor>> build_tensors(const gguf::Document& document) {
    std::vector<std::size_t> order(document.tensors.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&document](std::size_t a, std::size_t b) {
        return document.tensors[a].relative_offset < document.tensors[b].relative_offset;
    });

    std::vector<std::uint64_t> storage_spans(document.tensors.size(), 0);
    for (std::size_t sorted = 0; sorted < order.size(); ++sorted) {
        const auto index = order[sorted];
        const auto start = document.tensors[index].relative_offset;
        const auto end = sorted + 1 < order.size()
                             ? document.tensors[order[sorted + 1]].relative_offset
                             : document.file_size - document.data_offset;
        if (end < start) return Status::data_error("GGUF tensor offsets are not monotonic");
        storage_spans[index] = end - start;
    }

    std::vector<TensorDescriptor> tensors;
    tensors.reserve(document.tensors.size());
    for (std::size_t i = 0; i < document.tensors.size(); ++i) {
        const auto& source = document.tensors[i];
        TensorDescriptor tensor;
        tensor.name = source.name;
        tensor.format_type = source.type;
        tensor.shape.dimensions = source.dimensions;
        tensor.byte_offset = document.data_offset + source.relative_offset;

        const auto geometry = encoding_geometry(source.type);
        tensor.type = geometry.type;
        const auto elements = tensor.shape.element_count();
        if (geometry.block_elements != 0 && elements != 0 && elements % geometry.block_elements == 0) {
            const auto blocks = elements / geometry.block_elements;
            if (blocks > std::numeric_limits<std::uint64_t>::max() / geometry.block_bytes) {
                return Status::data_error("GGUF tensor byte size overflows: " + tensor.name);
            }
            tensor.byte_size = blocks * geometry.block_bytes;
            tensor.byte_size_exact = true;
            if (tensor.byte_size > storage_spans[i]) {
                return Status::data_error("GGUF tensor overlaps following tensor or file end: " + tensor.name);
            }
        } else {
            tensor.byte_size = storage_spans[i];
            tensor.byte_size_exact = false;
        }

        if (tensor.byte_size == 0) {
            return Status::data_error("GGUF tensor occupies zero bytes: " + tensor.name);
        }
        tensors.push_back(std::move(tensor));
    }
    return tensors;
}

} // namespace

bool GgufFormat::can_open(const std::filesystem::path& path) const {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return false;
    }

    std::array<char, 4> magic{};
    stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    return stream.gcount() == static_cast<std::streamsize>(magic.size()) &&
           magic[0] == 'G' && magic[1] == 'G' && magic[2] == 'U' && magic[3] == 'F';
}

Result<ModelDefinition> GgufFormat::load(const std::filesystem::path& path) const {
    auto document = gguf::read_document(path);
    if (!document) return document.status();

    auto architecture = document.value().require_string("general.architecture");
    if (!architecture) return architecture.status();
    auto tokenizer = build_tokenizer(document.value());
    if (!tokenizer) return tokenizer.status();
    auto config = build_config(document.value(), architecture.value(), tokenizer.value().vocabulary.size());
    if (!config) return config.status();
    auto tensors = build_tensors(document.value());
    if (!tensors) return tensors.status();
    auto storage = ModelStorage::map_read_only(path);
    if (!storage) return storage.status();

    ModelFingerprint fingerprint;
    fingerprint.format = "gguf-v" + std::to_string(document.value().version);
    fingerprint.architecture = architecture.value();
    fingerprint.model_id = document.value().optional_string("general.name").value_or(path.filename().string());

    ModelDefinition model(std::move(fingerprint),
                          std::move(config).value(),
                          std::move(tokenizer).value(),
                          std::move(tensors).value(),
                          std::move(storage).value());
    const auto validation = model.validate();
    if (!validation) return validation;
    return model;
}

} // namespace air

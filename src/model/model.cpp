#include "air/model.hpp"

#include <limits>
#include <unordered_set>

namespace air {

const char* to_string(DataType type) noexcept {
    switch (type) {
    case DataType::f32: return "f32";
    case DataType::f16: return "f16";
    case DataType::bf16: return "bf16";
    case DataType::q4_0: return "q4_0";
    case DataType::q4_1: return "q4_1";
    case DataType::q5_0: return "q5_0";
    case DataType::q5_1: return "q5_1";
    case DataType::q8_0: return "q8_0";
    case DataType::q8_1: return "q8_1";
    case DataType::q2_k: return "q2_K";
    case DataType::q3_k: return "q3_K";
    case DataType::q4_k: return "q4_K";
    case DataType::q5_k: return "q5_K";
    case DataType::q6_k: return "q6_K";
    case DataType::q8_k: return "q8_K";
    case DataType::iq2_xxs: return "iq2_xxs";
    case DataType::iq2_xs: return "iq2_xs";
    case DataType::iq3_xxs: return "iq3_xxs";
    case DataType::iq1_s: return "iq1_s";
    case DataType::iq4_nl: return "iq4_nl";
    case DataType::iq3_s: return "iq3_s";
    case DataType::iq2_s: return "iq2_s";
    case DataType::iq4_xs: return "iq4_xs";
    case DataType::i8: return "i8";
    case DataType::i16: return "i16";
    case DataType::i32: return "i32";
    case DataType::i64: return "i64";
    case DataType::f64: return "f64";
    case DataType::iq1_m: return "iq1_m";
    case DataType::tq1_0: return "tq1_0";
    case DataType::tq2_0: return "tq2_0";
    case DataType::mxfp4: return "mxfp4";
    case DataType::nvfp4: return "nvfp4";
    case DataType::q1_0: return "q1_0";
    case DataType::q2_0: return "q2_0";
    case DataType::unknown: break;
    }
    return "unknown";
}

std::uint64_t TensorShape::element_count() const noexcept {
    if (dimensions.empty()) {
        return 0;
    }

    std::uint64_t count = 1;
    for (const auto dimension : dimensions) {
        if (dimension == 0 || count > std::numeric_limits<std::uint64_t>::max() / dimension) {
            return 0;
        }
        count *= dimension;
    }
    return count;
}

ModelDefinition::ModelDefinition(ModelFingerprint fingerprint,
                                 ModelConfig config,
                                 TokenizerDefinition tokenizer,
                                 std::vector<TensorDescriptor> tensors,
                                 std::shared_ptr<const ModelStorage> storage)
    : fingerprint_(std::move(fingerprint)),
      config_(std::move(config)),
      tokenizer_(std::make_shared<const TokenizerDefinition>(std::move(tokenizer))),
      tensors_(std::move(tensors)),
      storage_(std::move(storage)) {
    tensor_index_.reserve(tensors_.size());
    for (std::size_t i = 0; i < tensors_.size(); ++i) {
        tensor_index_.emplace(tensors_[i].name, i);
    }
}

const TensorDescriptor* ModelDefinition::find_tensor(const std::string& name) const noexcept {
    const auto it = tensor_index_.find(name);
    if (it == tensor_index_.end()) {
        return nullptr;
    }
    return &tensors_[it->second];
}

Status ModelDefinition::validate() const {
    if (fingerprint_.format.empty()) {
        return Status::invalid_argument("model format is empty");
    }
    if (config_.architecture.empty()) {
        return Status::invalid_argument("model architecture is empty");
    }
    if (config_.layer_count == 0) {
        return Status::invalid_argument("layer count must be greater than zero");
    }
    if (config_.embedding_size == 0) {
        return Status::invalid_argument("embedding size must be greater than zero");
    }
    if (config_.attention_head_count == 0) {
        return Status::invalid_argument("attention head count must be greater than zero");
    }
    if (config_.kv_head_count == 0 || config_.kv_head_count > config_.attention_head_count) {
        return Status::invalid_argument("KV head count must be in [1, attention_head_count]");
    }
    if (config_.context_length == 0) {
        return Status::invalid_argument("context length must be greater than zero");
    }
    if (config_.vocabulary_size == 0) {
        return Status::invalid_argument("vocabulary size must be greater than zero");
    }
    if (tokenizer_->vocabulary.size() != config_.vocabulary_size) {
        return Status::invalid_argument("tokenizer vocabulary size does not match model vocabulary size");
    }
    if (!tokenizer_->scores.empty() && tokenizer_->scores.size() != tokenizer_->vocabulary.size()) {
        return Status::invalid_argument("tokenizer scores length does not match vocabulary");
    }
    if (!tokenizer_->token_types.empty() && tokenizer_->token_types.size() != tokenizer_->vocabulary.size()) {
        return Status::invalid_argument("tokenizer token-type length does not match vocabulary");
    }
    const auto validate_token_id = [this](std::optional<TokenId> id, const char* name) -> Status {
        if (!id) return Status::ok();
        if (*id < 0 || static_cast<std::size_t>(*id) >= tokenizer_->vocabulary.size()) {
            return Status::invalid_argument(std::string("tokenizer ") + name + " id is outside vocabulary");
        }
        return Status::ok();
    };
    for (const auto& [id, name] : std::initializer_list<std::pair<std::optional<TokenId>, const char*>>{
             {tokenizer_->special_ids.bos, "BOS"},
             {tokenizer_->special_ids.eos, "EOS"},
             {tokenizer_->special_ids.unknown, "unknown"},
             {tokenizer_->special_ids.padding, "padding"},
             {tokenizer_->special_ids.eot, "EOT"},
             {tokenizer_->special_ids.eom, "EOM"}}) {
        const auto status = validate_token_id(id, name);
        if (!status) return status;
    }
    if (tokenizer_->add_bos && !tokenizer_->special_ids.bos) {
        return Status::invalid_argument("tokenizer requests automatic BOS but has no BOS id");
    }
    if (tokenizer_->add_eos && !tokenizer_->special_ids.eos) {
        return Status::invalid_argument("tokenizer requests automatic EOS but has no EOS id");
    }

    std::unordered_set<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& tensor : tensors_) {
        if (tensor.name.empty()) {
            return Status::invalid_argument("tensor name is empty");
        }
        if (!names.emplace(tensor.name).second) {
            return Status::invalid_argument("duplicate tensor name: " + tensor.name);
        }
        if (tensor.shape.element_count() == 0) {
            return Status::invalid_argument("tensor has invalid shape: " + tensor.name);
        }
        if (tensor.byte_size == 0) {
            return Status::invalid_argument("tensor has zero storage size: " + tensor.name);
        }
        if (storage_ &&
            (tensor.byte_offset > storage_->size_bytes() ||
             tensor.byte_size > storage_->size_bytes() - tensor.byte_offset)) {
            return Status::data_error("tensor storage range is outside mapped model: " + tensor.name);
        }
    }

    return Status::ok();
}

Result<std::span<const std::byte>> ModelDefinition::tensor_bytes(
    const TensorDescriptor& tensor) const {
    if (!storage_) {
        return Status::invalid_state("model has no backing storage");
    }
    return storage_->view(tensor.byte_offset, tensor.byte_size);
}

} // namespace air

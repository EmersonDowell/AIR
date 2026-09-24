#pragma once

#include "air/status.hpp"
#include "air/storage.hpp"
#include "air/tensor.hpp"
#include "air/types.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace air {

struct ModelConfig {
    std::string architecture;
    std::uint32_t layer_count{0};
    std::uint32_t embedding_size{0};
    std::uint32_t feed_forward_size{0};
    std::uint32_t attention_head_count{0};
    std::uint32_t kv_head_count{0};
    std::uint64_t attention_sliding_window{0};
    std::uint32_t rope_dimension_count{0};
    std::uint64_t context_length{0};
    std::uint64_t vocabulary_size{0};
    double rope_frequency_base{0.0};
    std::string rope_scaling_type{"none"};
    double rope_scaling_factor{1.0};
    double rope_scale_linear{1.0};
    double rms_norm_epsilon{0.0};
};

struct TokenizerSpecialIds {
    std::optional<TokenId> bos;
    std::optional<TokenId> eos;
    std::optional<TokenId> unknown;
    std::optional<TokenId> padding;
    std::optional<TokenId> eot;
    std::optional<TokenId> eom;
};

struct TokenizerDefinition {
    std::string model;
    std::string pre_tokenizer;
    std::string chat_template;
    std::vector<std::string> vocabulary;
    std::vector<float> scores;
    std::vector<std::int32_t> token_types;
    std::vector<std::string> merges;
    TokenizerSpecialIds special_ids;
    bool add_bos{false};
    bool add_eos{false};
};

class ModelDefinition {
public:
    ModelDefinition() = default;
    ModelDefinition(ModelFingerprint fingerprint,
                    ModelConfig config,
                    TokenizerDefinition tokenizer,
                    std::vector<TensorDescriptor> tensors,
                    std::shared_ptr<const ModelStorage> storage = {});

    [[nodiscard]] const ModelFingerprint& fingerprint() const noexcept { return fingerprint_; }
    [[nodiscard]] const ModelConfig& config() const noexcept { return config_; }
    [[nodiscard]] const TokenizerDefinition& tokenizer() const noexcept { return *tokenizer_; }
    [[nodiscard]] const std::shared_ptr<const TokenizerDefinition>& tokenizer_handle() const noexcept { return tokenizer_; }
    [[nodiscard]] const std::vector<TensorDescriptor>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] const std::shared_ptr<const ModelStorage>& storage() const noexcept { return storage_; }
    [[nodiscard]] const TensorDescriptor* find_tensor(const std::string& name) const noexcept;
    [[nodiscard]] Status validate() const;
    [[nodiscard]] Result<std::span<const std::byte>> tensor_bytes(const TensorDescriptor& tensor) const;

private:
    ModelFingerprint fingerprint_;
    ModelConfig config_;
    std::shared_ptr<const TokenizerDefinition> tokenizer_{std::make_shared<TokenizerDefinition>()};
    std::vector<TensorDescriptor> tensors_;
    std::unordered_map<std::string, std::size_t> tensor_index_;
    std::shared_ptr<const ModelStorage> storage_;
};

} // namespace air

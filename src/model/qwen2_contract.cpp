#include "model/qwen2_contract.hpp"

#include <cmath>
#include <cstdint>
#include <span>

namespace air::detail {
namespace {

std::string block_name(std::uint32_t layer, const char* suffix) {
    return "blk." + std::to_string(layer) + "." + suffix;
}

Status require_shape(const ModelDefinition& model,
                     const std::string& name,
                     std::span<const std::uint64_t> dimensions,
                     bool optional = false) {
    const auto* tensor = model.find_tensor(name);
    if (!tensor) return optional ? Status::ok() : Status::data_error("missing required Qwen2 tensor: " + name);
    if (tensor->shape.dimensions.size() != dimensions.size()) {
        return Status::data_error("rank mismatch for tensor: " + name);
    }
    for (std::size_t i = 0; i < dimensions.size(); ++i) {
        if (tensor->shape.dimensions[i] != dimensions[i]) {
            return Status::data_error("shape mismatch for tensor: " + name);
        }
    }
    return Status::ok();
}

} // namespace

Status validate_qwen2_structure(const ModelDefinition& model) {
    const auto& config = model.config();
    if (config.architecture != "qwen2") return Status::unsupported("AIR Qwen2 execution supports only qwen2 architecture");
    if (config.layer_count == 0U || config.embedding_size == 0U || config.attention_head_count == 0U ||
        config.kv_head_count == 0U || config.context_length == 0U || config.vocabulary_size == 0U) {
        return Status::data_error("Qwen2 model geometry is incomplete");
    }
    if (config.embedding_size % config.attention_head_count != 0U) {
        return Status::data_error("Qwen2 embedding size is not divisible by attention head count");
    }
    if (config.attention_head_count % config.kv_head_count != 0U) {
        return Status::data_error("Qwen2 attention head count must be divisible by KV head count");
    }
    const std::uint64_t embedding = config.embedding_size;
    const std::uint64_t head_dimension = config.embedding_size / config.attention_head_count;
    const std::uint64_t q_width = config.attention_head_count * head_dimension;
    const std::uint64_t kv_width = config.kv_head_count * head_dimension;
    const std::uint64_t ffn = config.feed_forward_size;
    const std::uint64_t vocab = config.vocabulary_size;
    if (ffn == 0U) return Status::data_error("Qwen2 feed-forward size is missing");
    if (!(config.rms_norm_epsilon > 0.0)) return Status::data_error("Qwen2 RMS norm epsilon is missing or invalid");
    if (!(config.rope_frequency_base > 0.0)) return Status::data_error("Qwen2 RoPE base is missing or invalid");
    if (!config.rope_scaling_type.empty() && config.rope_scaling_type != "none") {
        return Status::unsupported("AIR Qwen2 execution does not implement RoPE scaling type: " + config.rope_scaling_type);
    }
    if (std::fabs(config.rope_scaling_factor - 1.0) > 1.0e-12 ||
        std::fabs(config.rope_scale_linear - 1.0) > 1.0e-12) {
        return Status::unsupported("AIR Qwen2 execution does not implement scaled RoPE");
    }
    if (config.attention_sliding_window != 0U) {
        return Status::unsupported("AIR Qwen2 execution does not implement sliding-window attention");
    }
    if (config.rope_dimension_count != head_dimension || config.rope_dimension_count % 2U != 0U) {
        return Status::unsupported("AIR Qwen2 execution requires full, even head-dimension RoPE");
    }

    const std::uint64_t embedding_shape[] = {embedding, vocab};
    auto status = require_shape(model, "token_embd.weight", embedding_shape);
    if (!status) return status;
    const std::uint64_t norm_shape[] = {embedding};
    status = require_shape(model, "output_norm.weight", norm_shape);
    if (!status) return status;
    const std::uint64_t output_shape[] = {embedding, vocab};
    if (model.find_tensor("output.weight")) {
        status = require_shape(model, "output.weight", output_shape);
        if (!status) return status;
    }
    const std::uint64_t output_bias_shape[] = {vocab};
    status = require_shape(model, "output.bias", output_bias_shape, true);
    if (!status) return status;

    for (std::uint32_t layer = 0; layer < config.layer_count; ++layer) {
        status = require_shape(model, block_name(layer, "attn_norm.weight"), norm_shape);
        if (!status) return status;
        const std::uint64_t q_shape[] = {embedding, q_width};
        status = require_shape(model, block_name(layer, "attn_q.weight"), q_shape);
        if (!status) return status;
        const std::uint64_t kv_shape[] = {embedding, kv_width};
        status = require_shape(model, block_name(layer, "attn_k.weight"), kv_shape);
        if (!status) return status;
        status = require_shape(model, block_name(layer, "attn_v.weight"), kv_shape);
        if (!status) return status;
        const std::uint64_t q_bias_shape[] = {q_width};
        status = require_shape(model, block_name(layer, "attn_q.bias"), q_bias_shape, true);
        if (!status) return status;
        const std::uint64_t kv_bias_shape[] = {kv_width};
        status = require_shape(model, block_name(layer, "attn_k.bias"), kv_bias_shape, true);
        if (!status) return status;
        status = require_shape(model, block_name(layer, "attn_v.bias"), kv_bias_shape, true);
        if (!status) return status;
        const std::uint64_t projection_shape[] = {q_width, embedding};
        status = require_shape(model, block_name(layer, "attn_output.weight"), projection_shape);
        if (!status) return status;
        status = require_shape(model, block_name(layer, "ffn_norm.weight"), norm_shape);
        if (!status) return status;
        const std::uint64_t ffn_in_shape[] = {embedding, ffn};
        status = require_shape(model, block_name(layer, "ffn_gate.weight"), ffn_in_shape);
        if (!status) return status;
        status = require_shape(model, block_name(layer, "ffn_up.weight"), ffn_in_shape);
        if (!status) return status;
        const std::uint64_t ffn_out_shape[] = {ffn, embedding};
        status = require_shape(model, block_name(layer, "ffn_down.weight"), ffn_out_shape);
        if (!status) return status;
    }
    return Status::ok();
}

std::vector<std::string> qwen2_execution_tensor_names(const ModelDefinition& model) {
    std::vector<std::string> names;
    names.reserve(static_cast<std::size_t>(model.config().layer_count) * 10U + 4U);
    names.emplace_back("token_embd.weight");
    names.emplace_back("output_norm.weight");
    if (model.find_tensor("output.weight")) names.emplace_back("output.weight");
    if (model.find_tensor("output.bias")) names.emplace_back("output.bias");
    for (std::uint32_t layer = 0; layer < model.config().layer_count; ++layer) {
        for (const char* suffix : {"attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                                  "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight",
                                  "ffn_up.weight", "ffn_down.weight"}) {
            names.push_back(block_name(layer, suffix));
        }
        for (const char* suffix : {"attn_q.bias", "attn_k.bias", "attn_v.bias"}) {
            const auto name = block_name(layer, suffix);
            if (model.find_tensor(name)) names.push_back(name);
        }
    }
    return names;
}

} // namespace air::detail

#include "air/reference.hpp"
#include "model/qwen2_contract.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>

namespace air {
namespace {

std::string block_name(std::uint32_t layer, const char* suffix) {
    return "blk." + std::to_string(layer) + "." + suffix;
}

std::vector<float> rms_norm(std::span<const float> input,
                            std::span<const float> weight,
                            double epsilon) {
    double square_sum = 0.0;
    for (const float value : input) square_sum += static_cast<double>(value) * value;
    const double mean = square_sum / static_cast<double>(input.size());
    const float scale = static_cast<float>(1.0 / std::sqrt(mean + epsilon));
    std::vector<float> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) output[i] = input[i] * scale * weight[i];
    return output;
}

void add_in_place(std::vector<float>& target, std::span<const float> value) {
    for (std::size_t i = 0; i < target.size(); ++i) target[i] += value[i];
}

float silu(float value) noexcept {
    return value / (1.0F + std::exp(-value));
}

Status apply_rope(std::vector<float>& q,
                  std::vector<float>& k,
                  std::uint32_t q_heads,
                  std::uint32_t kv_heads,
                  std::uint32_t head_dimension,
                  std::uint32_t rope_dimensions,
                  std::uint64_t position,
                  double rope_base) {
    if (rope_dimensions == 0U || rope_dimensions > head_dimension || rope_dimensions % 2U != 0U) {
        return Status::invalid_state("Qwen2 reference executor requires an even RoPE dimension within each head");
    }
    if (!(rope_base > 0.0)) return Status::invalid_state("Qwen2 reference executor requires a positive RoPE base");

    const std::uint32_t half = rope_dimensions / 2U;
    const auto rotate_heads = [&](std::vector<float>& values, std::uint32_t heads) {
        for (std::uint32_t head = 0; head < heads; ++head) {
            const std::size_t base = static_cast<std::size_t>(head) * head_dimension;
            for (std::uint32_t i = 0; i < half; ++i) {
                const double exponent = (2.0 * static_cast<double>(i)) /
                                        static_cast<double>(rope_dimensions);
                const double inverse_frequency = 1.0 / std::pow(rope_base, exponent);
                const double angle = static_cast<double>(position) * inverse_frequency;
                const float cosine = static_cast<float>(std::cos(angle));
                const float sine = static_cast<float>(std::sin(angle));
                const float first = values[base + i];
                const float second = values[base + i + half];
                values[base + i] = first * cosine - second * sine;
                values[base + i + half] = first * sine + second * cosine;
            }
        }
    };

    rotate_heads(q, q_heads);
    rotate_heads(k, kv_heads);
    return Status::ok();
}

Result<std::vector<float>> attention(std::span<const float> q,
                                     std::span<const float> current_k,
                                     std::span<const float> current_v,
                                     const ReferenceKvCache& cache,
                                     std::uint32_t layer,
                                     std::uint32_t q_heads,
                                     std::uint32_t kv_heads,
                                     std::uint32_t head_dimension) {
    const auto previous_tokens = cache.size();
    const auto total_tokens = previous_tokens + 1U;
    std::vector<float> output(static_cast<std::size_t>(q_heads) * head_dimension, 0.0F);
    std::vector<double> scores(static_cast<std::size_t>(total_tokens));
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dimension));

    for (std::uint32_t q_head = 0; q_head < q_heads; ++q_head) {
        const std::uint32_t kv_head = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(q_head) * kv_heads) / q_heads);
        const auto q_base = static_cast<std::size_t>(q_head) * head_dimension;
        const auto kv_base = static_cast<std::size_t>(kv_head) * head_dimension;

        double maximum = -std::numeric_limits<double>::infinity();
        for (std::uint64_t token = 0; token < total_tokens; ++token) {
            std::span<const float> key;
            if (token == previous_tokens) {
                key = current_k.subspan(kv_base, head_dimension);
            } else {
                auto cached = cache.key(layer, token, kv_head);
                if (!cached) return cached.status();
                key = cached.value();
            }
            double score = 0.0;
            for (std::uint32_t d = 0; d < head_dimension; ++d) {
                score += static_cast<double>(q[q_base + d]) * static_cast<double>(key[d]);
            }
            score *= scale;
            scores[static_cast<std::size_t>(token)] = score;
            maximum = std::max(maximum, score);
        }

        double denominator = 0.0;
        for (double& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        if (!(denominator > 0.0) || !std::isfinite(denominator)) {
            return Status::internal_error("attention softmax produced a non-finite denominator");
        }

        for (std::uint64_t token = 0; token < total_tokens; ++token) {
            std::span<const float> value;
            if (token == previous_tokens) {
                value = current_v.subspan(kv_base, head_dimension);
            } else {
                auto cached = cache.value(layer, token, kv_head);
                if (!cached) return cached.status();
                value = cached.value();
            }
            const float probability = static_cast<float>(scores[static_cast<std::size_t>(token)] / denominator);
            for (std::uint32_t d = 0; d < head_dimension; ++d) {
                output[q_base + d] += probability * value[d];
            }
        }
    }
    return output;
}

} // namespace

ReferenceKvCache::ReferenceKvCache(std::uint32_t layer_count,
                                   std::uint32_t kv_head_count,
                                   std::uint32_t head_dimension,
                                   std::uint64_t context_length,
                                   std::uint32_t page_tokens)
    : layer_count_(layer_count),
      kv_head_count_(kv_head_count),
      head_dimension_(head_dimension),
      context_length_(context_length),
      page_tokens_(page_tokens == 0U ? 32U : page_tokens),
      pending_keys_(layer_count),
      pending_values_(layer_count) {}

std::size_t ReferenceKvCache::stride() const noexcept {
    return static_cast<std::size_t>(kv_head_count_) * head_dimension_;
}

std::size_t ReferenceKvCache::page_layer_offset(std::uint32_t token_in_page,
                                                 std::uint32_t layer) const noexcept {
    return (static_cast<std::size_t>(token_in_page) * layer_count_ + layer) * stride();
}

std::shared_ptr<ReferenceKvCache::Page> ReferenceKvCache::new_page() const {
    auto page = std::make_shared<Page>();
    const auto elements = static_cast<std::size_t>(page_tokens_) * layer_count_ * stride();
    page->keys.resize(elements);
    page->values.resize(elements);
    return page;
}

std::size_t ReferenceKvCache::shared_page_count() const noexcept {
    std::size_t shared = 0;
    for (const auto& page : pages_) {
        if (page.use_count() > 1) ++shared;
    }
    return shared;
}

std::uint64_t ReferenceKvCache::resident_bytes() const noexcept {
    const auto elements_per_page = static_cast<std::uint64_t>(page_tokens_) * layer_count_ *
                                   kv_head_count_ * head_dimension_;
    return static_cast<std::uint64_t>(pages_.size()) * elements_per_page * sizeof(float) * 2U;
}

Result<std::span<const float>> ReferenceKvCache::key(std::uint32_t layer,
                                                     std::uint64_t token,
                                                     std::uint32_t kv_head) const {
    if (layer >= layer_count_ || kv_head >= kv_head_count_ || token >= token_count_) {
        return Status::invalid_argument("KV key lookup is outside committed cache");
    }
    const auto page_index = static_cast<std::size_t>(token / page_tokens_);
    const auto token_in_page = static_cast<std::uint32_t>(token % page_tokens_);
    const auto offset = page_layer_offset(token_in_page, layer) +
                        static_cast<std::size_t>(kv_head) * head_dimension_;
    return std::span<const float>(pages_[page_index]->keys.data() + offset, head_dimension_);
}

Result<std::span<const float>> ReferenceKvCache::value(std::uint32_t layer,
                                                       std::uint64_t token,
                                                       std::uint32_t kv_head) const {
    if (layer >= layer_count_ || kv_head >= kv_head_count_ || token >= token_count_) {
        return Status::invalid_argument("KV value lookup is outside committed cache");
    }
    const auto page_index = static_cast<std::size_t>(token / page_tokens_);
    const auto token_in_page = static_cast<std::uint32_t>(token % page_tokens_);
    const auto offset = page_layer_offset(token_in_page, layer) +
                        static_cast<std::size_t>(kv_head) * head_dimension_;
    return std::span<const float>(pages_[page_index]->values.data() + offset, head_dimension_);
}

Status ReferenceKvCache::append_pending(std::uint32_t layer,
                                        std::span<const float> key_values,
                                        std::span<const float> value_values) {
    if (layer >= layer_count_) return Status::invalid_argument("KV layer index is outside cache");
    if (token_count_ >= context_length_) return Status::invalid_state("KV cache context capacity exceeded");
    const auto expected = stride();
    if (key_values.size() != expected || value_values.size() != expected) {
        return Status::invalid_argument("KV append width does not match cache geometry");
    }
    if (!pending_keys_[layer].empty() || !pending_values_[layer].empty()) {
        return Status::invalid_state("KV layer already has pending state for current token");
    }
    pending_keys_[layer].assign(key_values.begin(), key_values.end());
    pending_values_[layer].assign(value_values.begin(), value_values.end());
    return Status::ok();
}

Status ReferenceKvCache::ensure_writable_tail() {
    if (pages_.empty() || pages_.back()->used_tokens == page_tokens_) {
        pages_.push_back(new_page());
        return Status::ok();
    }
    if (pages_.back().use_count() > 1) {
        pages_.back() = std::make_shared<Page>(*pages_.back());
    }
    return Status::ok();
}

Status ReferenceKvCache::commit_token() {
    if (token_count_ >= context_length_) return Status::invalid_state("KV cache context capacity exceeded");
    for (std::uint32_t layer = 0; layer < layer_count_; ++layer) {
        if (pending_keys_[layer].size() != stride() || pending_values_[layer].size() != stride()) {
            return Status::invalid_state("cannot commit incomplete KV token transaction");
        }
    }
    const auto writable = ensure_writable_tail();
    if (!writable) return writable;
    auto& page = *pages_.back();
    const auto token_in_page = page.used_tokens;
    for (std::uint32_t layer = 0; layer < layer_count_; ++layer) {
        const auto offset = page_layer_offset(token_in_page, layer);
        std::copy(pending_keys_[layer].begin(), pending_keys_[layer].end(), page.keys.begin() + static_cast<std::ptrdiff_t>(offset));
        std::copy(pending_values_[layer].begin(), pending_values_[layer].end(), page.values.begin() + static_cast<std::ptrdiff_t>(offset));
    }
    ++page.used_tokens;
    ++token_count_;
    rollback_pending();
    return Status::ok();
}

Result<ReferenceKvCache> ReferenceKvCache::fork(std::uint64_t committed_tokens) const {
    if (committed_tokens > token_count_) {
        return Status::invalid_argument("cannot fork KV cache beyond committed tokens");
    }
    ReferenceKvCache forked(layer_count_, kv_head_count_, head_dimension_, context_length_, page_tokens_);
    forked.token_count_ = committed_tokens;
    const auto full_pages = static_cast<std::size_t>(committed_tokens / page_tokens_);
    const auto tail_tokens = static_cast<std::uint32_t>(committed_tokens % page_tokens_);
    forked.pages_.insert(forked.pages_.end(), pages_.begin(), pages_.begin() + static_cast<std::ptrdiff_t>(full_pages));
    if (tail_tokens != 0U) {
        auto tail = std::make_shared<Page>(*pages_[full_pages]);
        tail->used_tokens = tail_tokens;
        forked.pages_.push_back(std::move(tail));
    }
    return forked;
}

void ReferenceKvCache::rollback_pending() noexcept {
    for (auto& values : pending_keys_) values.clear();
    for (auto& values : pending_values_) values.clear();
}

void ReferenceKvCache::reset() noexcept {
    pages_.clear();
    rollback_pending();
    token_count_ = 0;
}

Result<std::unique_ptr<ReferenceExecutor>> ReferenceExecutor::create(
    std::shared_ptr<const ModelDefinition> model) {
    if (!model) return Status::invalid_argument("reference executor requires a model");
    const auto canonical_validation = model->validate();
    if (!canonical_validation) return canonical_validation;
    auto executor = std::unique_ptr<ReferenceExecutor>(new ReferenceExecutor(std::move(model)));
    const auto validation = executor->validate_model();
    if (!validation) return validation;
    return executor;
}

Status ReferenceExecutor::validate_model() const {
    auto status = detail::validate_qwen2_structure(*model_);
    if (!status) return status;
    for (const auto& name : detail::qwen2_execution_tensor_names(*model_)) {
        const auto* tensor = model_->find_tensor(name);
        if (!tensor) return Status::internal_error("Qwen2 execution contract omitted required tensor: " + name);
        if (!ReferenceTensorReader::supports(tensor->type)) {
            return Status::unsupported("reference executor does not support " +
                                       std::string(to_string(tensor->type)) + " tensor: " + name);
        }
    }
    return Status::ok();
}

Result<std::vector<float>> ReferenceExecutor::add_optional_bias(
    std::vector<float> values, const std::string& tensor_name) const {
    if (!model_->find_tensor(tensor_name)) return values;
    auto bias = tensors_.vector(tensor_name);
    if (!bias) return bias.status();
    if (bias.value().size() != values.size()) return Status::data_error("bias width mismatch: " + tensor_name);
    add_in_place(values, bias.value());
    return values;
}

Result<std::vector<float>> ReferenceExecutor::step_impl(TokenId token, ReferenceKvCache& cache,
                                                        VerificationTrace* trace) const {
    const auto& config = model_->config();
    if (token < 0 || static_cast<std::uint64_t>(token) >= config.vocabulary_size) {
        return Status::invalid_argument("input token is outside model vocabulary");
    }
    if (cache.layer_count() != config.layer_count || cache.kv_head_count() != config.kv_head_count ||
        cache.head_dimension() != config.embedding_size / config.attention_head_count) {
        return Status::invalid_argument("KV cache geometry does not match model");
    }
    if (cache.size() >= config.context_length) return Status::invalid_state("model context length exceeded");

    const auto position = cache.size();
    const auto head_dimension = config.embedding_size / config.attention_head_count;
    auto hidden = tensors_.row("token_embd.weight", static_cast<std::uint64_t>(token));
    if (!hidden) return hidden.status();
    if (trace) trace->record(position, -1, VerificationStage::embedding, hidden.value());

    for (std::uint32_t layer = 0; layer < config.layer_count; ++layer) {
        auto attention_norm_weight = tensors_.vector(block_name(layer, "attn_norm.weight"));
        if (!attention_norm_weight) { cache.rollback_pending(); return attention_norm_weight.status(); }
        auto normalized = rms_norm(hidden.value(), attention_norm_weight.value(), config.rms_norm_epsilon);
        if (trace) trace->record(position, static_cast<std::int32_t>(layer), VerificationStage::attention_norm, normalized);

        auto q = tensors_.matvec(block_name(layer, "attn_q.weight"), normalized);
        auto k = tensors_.matvec(block_name(layer, "attn_k.weight"), normalized);
        auto v = tensors_.matvec(block_name(layer, "attn_v.weight"), normalized);
        if (!q) { cache.rollback_pending(); return q.status(); }
        if (!k) { cache.rollback_pending(); return k.status(); }
        if (!v) { cache.rollback_pending(); return v.status(); }
        q = add_optional_bias(std::move(q).value(), block_name(layer, "attn_q.bias"));
        k = add_optional_bias(std::move(k).value(), block_name(layer, "attn_k.bias"));
        v = add_optional_bias(std::move(v).value(), block_name(layer, "attn_v.bias"));
        if (!q) { cache.rollback_pending(); return q.status(); }
        if (!k) { cache.rollback_pending(); return k.status(); }
        if (!v) { cache.rollback_pending(); return v.status(); }
        if (trace) {
            const auto layer_index = static_cast<std::int32_t>(layer);
            trace->record(position, layer_index, VerificationStage::q_projection, q.value());
            trace->record(position, layer_index, VerificationStage::k_projection, k.value());
            trace->record(position, layer_index, VerificationStage::v_projection, v.value());
        }

        auto rope_status = apply_rope(q.value(), k.value(), config.attention_head_count,
                                      config.kv_head_count, head_dimension,
                                      config.rope_dimension_count, position,
                                      config.rope_frequency_base);
        if (!rope_status) { cache.rollback_pending(); return rope_status; }
        if (trace) {
            const auto layer_index = static_cast<std::int32_t>(layer);
            trace->record(position, layer_index, VerificationStage::q_rope, q.value());
            trace->record(position, layer_index, VerificationStage::k_rope, k.value());
        }

        auto attended = attention(q.value(), k.value(), v.value(), cache, layer,
                                  config.attention_head_count, config.kv_head_count,
                                  head_dimension);
        if (!attended) { cache.rollback_pending(); return attended.status(); }
        if (trace) trace->record(position, static_cast<std::int32_t>(layer), VerificationStage::attention, attended.value());
        auto projected = tensors_.matvec(block_name(layer, "attn_output.weight"), attended.value());
        if (!projected) { cache.rollback_pending(); return projected.status(); }
        if (trace) trace->record(position, static_cast<std::int32_t>(layer), VerificationStage::attention_projection, projected.value());
        add_in_place(hidden.value(), projected.value());
        if (trace) trace->record(position, static_cast<std::int32_t>(layer), VerificationStage::attention_residual, hidden.value());

        auto ffn_norm_weight = tensors_.vector(block_name(layer, "ffn_norm.weight"));
        if (!ffn_norm_weight) { cache.rollback_pending(); return ffn_norm_weight.status(); }
        normalized = rms_norm(hidden.value(), ffn_norm_weight.value(), config.rms_norm_epsilon);
        if (trace) trace->record(position, static_cast<std::int32_t>(layer), VerificationStage::ffn_norm, normalized);
        auto gate = tensors_.matvec(block_name(layer, "ffn_gate.weight"), normalized);
        auto up = tensors_.matvec(block_name(layer, "ffn_up.weight"), normalized);
        if (!gate) { cache.rollback_pending(); return gate.status(); }
        if (!up) { cache.rollback_pending(); return up.status(); }
        if (trace) {
            const auto layer_index = static_cast<std::int32_t>(layer);
            trace->record(position, layer_index, VerificationStage::ffn_gate, gate.value());
            trace->record(position, layer_index, VerificationStage::ffn_up, up.value());
        }
        for (std::size_t i = 0; i < gate.value().size(); ++i) gate.value()[i] = silu(gate.value()[i]) * up.value()[i];
        if (trace) trace->record(position, static_cast<std::int32_t>(layer), VerificationStage::ffn_activation, gate.value());
        auto down = tensors_.matvec(block_name(layer, "ffn_down.weight"), gate.value());
        if (!down) { cache.rollback_pending(); return down.status(); }
        if (trace) trace->record(position, static_cast<std::int32_t>(layer), VerificationStage::ffn_down, down.value());
        add_in_place(hidden.value(), down.value());
        if (trace) trace->record(position, static_cast<std::int32_t>(layer), VerificationStage::ffn_residual, hidden.value());

        auto append_status = cache.append_pending(layer, k.value(), v.value());
        if (!append_status) { cache.rollback_pending(); return append_status; }
    }

    auto final_norm_weight = tensors_.vector("output_norm.weight");
    if (!final_norm_weight) { cache.rollback_pending(); return final_norm_weight.status(); }
    auto final_hidden = rms_norm(hidden.value(), final_norm_weight.value(), config.rms_norm_epsilon);
    if (trace) trace->record(position, -1, VerificationStage::final_norm, final_hidden);
    const std::string output_tensor = model_->find_tensor("output.weight") ? "output.weight" : "token_embd.weight";
    auto logits = tensors_.matvec(output_tensor, final_hidden);
    if (!logits) { cache.rollback_pending(); return logits.status(); }
    logits = add_optional_bias(std::move(logits).value(), "output.bias");
    if (!logits) { cache.rollback_pending(); return logits.status(); }
    if (trace) trace->record(position, -1, VerificationStage::logits, logits.value());
    auto commit = cache.commit_token();
    if (!commit) { cache.rollback_pending(); return commit; }
    return logits;
}

Result<std::vector<float>> ReferenceExecutor::step(TokenId token, ReferenceKvCache& cache) const {
    return step_impl(token, cache, nullptr);
}

Result<std::vector<float>> ReferenceExecutor::step_verified(TokenId token, ReferenceKvCache& cache,
                                                            VerificationTrace& trace) const {
    return step_impl(token, cache, &trace);
}

Result<std::vector<float>> ReferenceExecutor::prefill(std::span<const TokenId> tokens,
                                                      ReferenceKvCache& cache) const {
    if (tokens.empty()) return Status::invalid_argument("prefill requires at least one token");
    std::vector<float> logits;
    for (const auto token : tokens) {
        auto step_result = step(token, cache);
        if (!step_result) return step_result.status();
        logits = std::move(step_result).value();
    }
    return logits;
}

Result<GenerationResult> ReferenceExecutor::generate(std::span<const TokenId> prompt,
                                                     const GenerationConfig& config) const {
    const auto& model_config = model_->config();
    const auto head_dimension = model_config.embedding_size / model_config.attention_head_count;
    ReferenceKvCache cache(model_config.layer_count, model_config.kv_head_count,
                           head_dimension, model_config.context_length);
    auto trace = run_autoregressive(
        prompt, config, model_->tokenizer().special_ids.eos,
        [this, &cache](std::span<const TokenId> tokens) { return prefill(tokens, cache); },
        [this, &cache](TokenId token) { return step(token, cache); });
    if (!trace) return trace.status();
    return std::move(trace).value().result;
}

} // namespace air

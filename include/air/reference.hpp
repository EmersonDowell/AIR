#pragma once

#include "air/generation.hpp"
#include "air/model.hpp"
#include "air/result.hpp"
#include "air/types.hpp"
#include "air/verification.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace air {

class ReferenceTensorReader final {
public:
    explicit ReferenceTensorReader(std::shared_ptr<const ModelDefinition> model)
        : model_(std::move(model)) {}

    [[nodiscard]] static bool supports(DataType type) noexcept;
    [[nodiscard]] Result<std::vector<float>> vector(const std::string& name) const;
    [[nodiscard]] Result<std::vector<float>> row(const std::string& name,
                                                 std::uint64_t row_index) const;
    [[nodiscard]] Result<std::vector<float>> matvec(const std::string& name,
                                                    std::span<const float> input) const;

private:
    [[nodiscard]] Result<std::vector<float>> decode_tensor(const TensorDescriptor& tensor) const;
    [[nodiscard]] Result<std::vector<float>> decode_range(const TensorDescriptor& tensor,
                                                          std::uint64_t element_offset,
                                                          std::uint64_t element_count) const;

    std::shared_ptr<const ModelDefinition> model_;
};

class ReferenceKvCache final {
public:
    ReferenceKvCache(std::uint32_t layer_count,
                     std::uint32_t kv_head_count,
                     std::uint32_t head_dimension,
                     std::uint64_t context_length,
                     std::uint32_t page_tokens = 32U);

    [[nodiscard]] std::uint64_t size() const noexcept { return token_count_; }
    [[nodiscard]] std::uint64_t capacity() const noexcept { return context_length_; }
    [[nodiscard]] std::uint32_t layer_count() const noexcept { return layer_count_; }
    [[nodiscard]] std::uint32_t kv_head_count() const noexcept { return kv_head_count_; }
    [[nodiscard]] std::uint32_t head_dimension() const noexcept { return head_dimension_; }
    [[nodiscard]] std::uint32_t page_tokens() const noexcept { return page_tokens_; }
    [[nodiscard]] std::size_t page_count() const noexcept { return pages_.size(); }
    [[nodiscard]] std::size_t shared_page_count() const noexcept;
    [[nodiscard]] std::uint64_t resident_bytes() const noexcept;

    [[nodiscard]] Result<std::span<const float>> key(std::uint32_t layer,
                                                     std::uint64_t token,
                                                     std::uint32_t kv_head) const;
    [[nodiscard]] Result<std::span<const float>> value(std::uint32_t layer,
                                                       std::uint64_t token,
                                                       std::uint32_t kv_head) const;

    [[nodiscard]] Status append_pending(std::uint32_t layer,
                                        std::span<const float> key,
                                        std::span<const float> value);
    [[nodiscard]] Status commit_token();
    [[nodiscard]] Result<ReferenceKvCache> fork(std::uint64_t committed_tokens) const;
    void rollback_pending() noexcept;
    void reset() noexcept;

private:
    struct Page {
        std::vector<float> keys;
        std::vector<float> values;
        std::uint32_t used_tokens{0};
    };

    [[nodiscard]] std::size_t stride() const noexcept;
    [[nodiscard]] std::size_t page_layer_offset(std::uint32_t token_in_page,
                                                std::uint32_t layer) const noexcept;
    [[nodiscard]] std::shared_ptr<Page> new_page() const;
    [[nodiscard]] Status ensure_writable_tail();

    std::uint32_t layer_count_{0};
    std::uint32_t kv_head_count_{0};
    std::uint32_t head_dimension_{0};
    std::uint64_t context_length_{0};
    std::uint32_t page_tokens_{32};
    std::uint64_t token_count_{0};
    std::vector<std::shared_ptr<Page>> pages_;
    std::vector<std::vector<float>> pending_keys_;
    std::vector<std::vector<float>> pending_values_;
};

class ReferenceExecutor final {
public:
    [[nodiscard]] static Result<std::unique_ptr<ReferenceExecutor>> create(
        std::shared_ptr<const ModelDefinition> model);

    [[nodiscard]] const ModelDefinition& model() const noexcept { return *model_; }
    [[nodiscard]] Result<std::vector<float>> step(TokenId token, ReferenceKvCache& cache) const;
    [[nodiscard]] Result<std::vector<float>> step_verified(TokenId token, ReferenceKvCache& cache,
                                                           VerificationTrace& trace) const;
    [[nodiscard]] Result<std::vector<float>> prefill(std::span<const TokenId> tokens,
                                                     ReferenceKvCache& cache) const;
    [[nodiscard]] Result<GenerationResult> generate(std::span<const TokenId> prompt,
                                                    const GenerationConfig& config) const;

private:
    explicit ReferenceExecutor(std::shared_ptr<const ModelDefinition> model)
        : model_(std::move(model)), tensors_(model_) {}

    [[nodiscard]] Status validate_model() const;
    [[nodiscard]] Result<std::vector<float>> step_impl(TokenId token, ReferenceKvCache& cache,
                                                        VerificationTrace* trace) const;
    [[nodiscard]] Result<std::vector<float>> add_optional_bias(
        std::vector<float> values, const std::string& tensor_name) const;

    std::shared_ptr<const ModelDefinition> model_;
    ReferenceTensorReader tensors_;
};

} // namespace air

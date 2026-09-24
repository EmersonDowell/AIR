#pragma once

#include "air/model.hpp"
#include "air/result.hpp"
#include "air/types.hpp"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace air {

struct TokenizeOptions {
    std::optional<bool> add_bos;
    std::optional<bool> add_eos;
    bool parse_special{true};
};

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    [[nodiscard]] virtual Result<std::vector<TokenId>> encode(
        std::string_view text, TokenizeOptions options = {}) const = 0;
    [[nodiscard]] virtual Result<std::string> decode(
        std::span<const TokenId> tokens, bool render_special = true) const = 0;
};

[[nodiscard]] Result<std::unique_ptr<Tokenizer>> create_tokenizer(
    std::shared_ptr<const TokenizerDefinition> definition);

} // namespace air

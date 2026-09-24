#include "air/tokenizer.hpp"

#include <pcre2.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace air {
namespace {

std::string utf8_from_codepoint(std::uint32_t cp) {
    std::string out;
    if (cp <= 0x7FU) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FFU) {
        out.push_back(static_cast<char>(0xC0U | (cp >> 6U)));
        out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    } else if (cp <= 0xFFFFU) {
        out.push_back(static_cast<char>(0xE0U | (cp >> 12U)));
        out.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    } else {
        out.push_back(static_cast<char>(0xF0U | (cp >> 18U)));
        out.push_back(static_cast<char>(0x80U | ((cp >> 12U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    }
    return out;
}

Result<std::vector<std::pair<std::uint32_t, std::string_view>>> utf8_codepoints(std::string_view text) {
    std::vector<std::pair<std::uint32_t, std::string_view>> out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        const auto lead = static_cast<std::uint8_t>(text[i]);
        std::size_t length = 0;
        std::uint32_t cp = 0;
        if (lead < 0x80U) {
            length = 1;
            cp = lead;
        } else if ((lead & 0xE0U) == 0xC0U) {
            length = 2;
            cp = lead & 0x1FU;
        } else if ((lead & 0xF0U) == 0xE0U) {
            length = 3;
            cp = lead & 0x0FU;
        } else if ((lead & 0xF8U) == 0xF0U) {
            length = 4;
            cp = lead & 0x07U;
        } else {
            return Status::invalid_argument("invalid UTF-8 leading byte at offset " + std::to_string(i));
        }
        if (i + length > text.size()) {
            return Status::invalid_argument("truncated UTF-8 sequence at offset " + std::to_string(i));
        }
        for (std::size_t j = 1; j < length; ++j) {
            const auto byte = static_cast<std::uint8_t>(text[i + j]);
            if ((byte & 0xC0U) != 0x80U) {
                return Status::invalid_argument("invalid UTF-8 continuation byte at offset " +
                                                std::to_string(i + j));
            }
            cp = (cp << 6U) | (byte & 0x3FU);
        }
        const bool overlong = (length == 2 && cp < 0x80U) || (length == 3 && cp < 0x800U) ||
                              (length == 4 && cp < 0x10000U);
        if (overlong || cp > 0x10FFFFU || (cp >= 0xD800U && cp <= 0xDFFFU)) {
            return Status::invalid_argument("invalid UTF-8 code point at offset " + std::to_string(i));
        }
        out.emplace_back(cp, text.substr(i, length));
        i += length;
    }
    return out;
}

struct ByteCodec {
    std::array<std::string, 256> encoded;
    std::unordered_map<std::uint32_t, std::uint8_t> decoded;

    ByteCodec() {
        std::vector<std::uint32_t> bytes;
        for (std::uint32_t b = 33; b <= 126; ++b) bytes.push_back(b);
        for (std::uint32_t b = 161; b <= 172; ++b) bytes.push_back(b);
        for (std::uint32_t b = 174; b <= 255; ++b) bytes.push_back(b);

        std::unordered_set<std::uint32_t> direct(bytes.begin(), bytes.end());
        std::uint32_t extra = 0;
        for (std::uint32_t b = 0; b <= 255; ++b) {
            const std::uint32_t cp = direct.contains(b) ? b : 256U + extra++;
            encoded[b] = utf8_from_codepoint(cp);
            decoded.emplace(cp, static_cast<std::uint8_t>(b));
        }
    }
};

const ByteCodec& byte_codec() {
    static const ByteCodec codec;
    return codec;
}

std::optional<std::string_view> pattern_for(std::string_view pre) {
    if (pre == "gpt2" || pre == "mpt" || pre == "olmo" || pre == "jais" || pre == "") {
        return R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";
    }
    if (pre == "qwen2" || pre == "stablelm2" || pre == "hunyuan" || pre == "solar-open") {
        return R"((?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    }
    if (pre == "qwen35" || pre == "qwen3.5") {
        return R"((?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    }
    if (pre == "llama-bpe" || pre == "llama3") {
        return R"((?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    }
    return std::nullopt;
}

class Regex final {
public:
    static Result<std::unique_ptr<Regex>> compile(std::string_view pattern) {
        int error_code = 0;
        PCRE2_SIZE error_offset = 0;
        pcre2_code* code = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.data()),
                                         pattern.size(),
                                         PCRE2_UTF | PCRE2_UCP,
                                         &error_code,
                                         &error_offset,
                                         nullptr);
        if (!code) {
            std::array<PCRE2_UCHAR, 256> message{};
            pcre2_get_error_message(error_code, message.data(), message.size());
            return Status::internal_error("failed to compile tokenizer regex at offset " +
                                          std::to_string(error_offset) + ": " +
                                          reinterpret_cast<const char*>(message.data()));
        }
        auto regex = std::unique_ptr<Regex>(new Regex(code));
        return regex;
    }

    ~Regex() { pcre2_code_free(code_); }

    Result<std::vector<std::string_view>> split(std::string_view text) const {
        std::vector<std::string_view> pieces;
        if (text.empty()) return pieces;

        pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(code_, nullptr);
        if (!match_data) return Status::internal_error("failed to allocate tokenizer regex match data");

        std::size_t cursor = 0;
        while (cursor < text.size()) {
            const int rc = pcre2_match(code_,
                                       reinterpret_cast<PCRE2_SPTR>(text.data()),
                                       text.size(),
                                       cursor,
                                       0,
                                       match_data,
                                       nullptr);
            if (rc == PCRE2_ERROR_NOMATCH) {
                pieces.push_back(text.substr(cursor));
                break;
            }
            if (rc < 0) {
                pcre2_match_data_free(match_data);
                return Status::invalid_argument("tokenizer regex rejected input near byte " +
                                                std::to_string(cursor));
            }
            PCRE2_SIZE* offsets = pcre2_get_ovector_pointer(match_data);
            const auto begin = static_cast<std::size_t>(offsets[0]);
            const auto end = static_cast<std::size_t>(offsets[1]);
            if (begin > cursor) pieces.push_back(text.substr(cursor, begin - cursor));
            if (end <= begin) {
                pcre2_match_data_free(match_data);
                return Status::internal_error("tokenizer regex produced an empty match");
            }
            pieces.push_back(text.substr(begin, end - begin));
            cursor = end;
        }

        pcre2_match_data_free(match_data);
        return pieces;
    }

private:
    explicit Regex(pcre2_code* code) : code_(code) {}
    pcre2_code* code_{nullptr};
};

std::string pair_key(std::string_view left, std::string_view right) {
    std::string key;
    key.reserve(left.size() + right.size() + 1);
    key.append(left);
    key.push_back('\x1f');
    key.append(right);
    return key;
}

class Gpt2BpeTokenizer final : public Tokenizer {
public:
    static Result<std::unique_ptr<Tokenizer>> create(std::shared_ptr<const TokenizerDefinition> definition) {
        if (!definition || definition->vocabulary.empty()) {
            return Status::data_error("cannot construct tokenizer with empty vocabulary");
        }
        const auto pattern = pattern_for(definition->pre_tokenizer);
        if (!pattern) {
            return Status::unsupported("unsupported GPT-2 BPE pre-tokenizer: " +
                                       definition->pre_tokenizer);
        }
        auto regex = Regex::compile(*pattern);
        if (!regex) return regex.status();

        auto tokenizer = std::unique_ptr<Gpt2BpeTokenizer>(
            new Gpt2BpeTokenizer(std::move(definition), std::move(regex).value()));
        const auto status = tokenizer->initialize();
        if (!status) return status;
        return std::unique_ptr<Tokenizer>(std::move(tokenizer));
    }

    Result<std::vector<TokenId>> encode(std::string_view text,
                                        TokenizeOptions options) const override {
        std::vector<TokenId> output;
        const bool add_bos = options.add_bos.value_or(definition_->add_bos);
        const bool add_eos = options.add_eos.value_or(definition_->add_eos);
        if (add_bos) {
            if (!definition_->special_ids.bos) return Status::invalid_state("tokenizer requests BOS but has no BOS id");
            output.push_back(*definition_->special_ids.bos);
        }

        if (options.parse_special && !special_by_first_.empty()) {
            std::size_t plain_start = 0;
            std::size_t cursor = 0;
            while (cursor < text.size()) {
                const auto match = special_at(text, cursor);
                if (!match) {
                    ++cursor;
                    continue;
                }
                if (cursor > plain_start) {
                    const auto status = encode_plain(text.substr(plain_start, cursor - plain_start), output);
                    if (!status) return status;
                }
                output.push_back(match->second);
                cursor += match->first.size();
                plain_start = cursor;
            }
            if (plain_start < text.size()) {
                const auto status = encode_plain(text.substr(plain_start), output);
                if (!status) return status;
            }
        } else {
            const auto status = encode_plain(text, output);
            if (!status) return status;
        }

        if (add_eos) {
            if (!definition_->special_ids.eos) return Status::invalid_state("tokenizer requests EOS but has no EOS id");
            output.push_back(*definition_->special_ids.eos);
        }
        return output;
    }

    Result<std::string> decode(std::span<const TokenId> tokens, bool render_special) const override {
        std::string output;
        for (const TokenId token : tokens) {
            if (token < 0 || static_cast<std::size_t>(token) >= definition_->vocabulary.size()) {
                return Status::invalid_argument("token id is outside tokenizer vocabulary: " + std::to_string(token));
            }
            if (!render_special && special_ids_.contains(token)) continue;
            const std::string& piece = definition_->vocabulary[static_cast<std::size_t>(token)];
            auto codepoints = utf8_codepoints(piece);
            if (!codepoints) return codepoints.status();
            for (const auto& [cp, bytes] : codepoints.value()) {
                const auto it = byte_codec().decoded.find(cp);
                if (it != byte_codec().decoded.end()) {
                    output.push_back(static_cast<char>(it->second));
                } else {
                    output.append(bytes);
                }
            }
        }
        return output;
    }

private:
    Gpt2BpeTokenizer(std::shared_ptr<const TokenizerDefinition> definition, std::unique_ptr<Regex> regex)
        : definition_(std::move(definition)), regex_(std::move(regex)) {}

    Status initialize() {
        token_to_id_.reserve(definition_->vocabulary.size());
        for (std::size_t i = 0; i < definition_->vocabulary.size(); ++i) {
            if (i > static_cast<std::size_t>(std::numeric_limits<TokenId>::max())) {
                return Status::unsupported("tokenizer vocabulary exceeds TokenId range");
            }
            const auto [_, inserted] = token_to_id_.emplace(definition_->vocabulary[i], static_cast<TokenId>(i));
            if (!inserted) return Status::data_error("duplicate tokenizer vocabulary piece");
        }

        merge_rank_.reserve(definition_->merges.size());
        for (std::size_t rank = 0; rank < definition_->merges.size(); ++rank) {
            const auto& merge = definition_->merges[rank];
            const auto separator = merge.find(' ');
            if (separator == std::string::npos || separator == 0 || separator + 1 >= merge.size()) {
                return Status::data_error("invalid tokenizer merge at rank " + std::to_string(rank));
            }
            merge_rank_.emplace(pair_key(std::string_view(merge).substr(0, separator),
                                         std::string_view(merge).substr(separator + 1)),
                                rank);
        }

        auto mark_special = [this](std::optional<TokenId> id) {
            if (id) special_ids_.insert(*id);
        };
        mark_special(definition_->special_ids.bos);
        mark_special(definition_->special_ids.eos);
        mark_special(definition_->special_ids.unknown);
        mark_special(definition_->special_ids.padding);
        mark_special(definition_->special_ids.eot);
        mark_special(definition_->special_ids.eom);

        if (!definition_->token_types.empty()) {
            for (std::size_t i = 0; i < definition_->token_types.size(); ++i) {
                const auto type = definition_->token_types[i];
                if (type == 3 || type == 4) special_ids_.insert(static_cast<TokenId>(i));
            }
        }

        for (const TokenId id : special_ids_) {
            if (id < 0 || static_cast<std::size_t>(id) >= definition_->vocabulary.size()) continue;
            const auto& text = definition_->vocabulary[static_cast<std::size_t>(id)];
            if (text.empty()) continue;
            special_by_first_[text.front()].emplace_back(text, id);
        }
        for (auto& [_, candidates] : special_by_first_) {
            std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                return a.first.size() > b.first.size();
            });
        }
        return Status::ok();
    }

    std::optional<std::pair<std::string_view, TokenId>> special_at(std::string_view text,
                                                                    std::size_t offset) const {
        const auto it = special_by_first_.find(text[offset]);
        if (it == special_by_first_.end()) return std::nullopt;
        for (const auto& [candidate, id] : it->second) {
            if (candidate.size() <= text.size() - offset &&
                text.compare(offset, candidate.size(), candidate) == 0) {
                return std::pair<std::string_view, TokenId>{candidate, id};
            }
        }
        return std::nullopt;
    }

    Status encode_plain(std::string_view text, std::vector<TokenId>& output) const {
        if (text.empty()) return Status::ok();
        auto pieces = regex_->split(text);
        if (!pieces) return pieces.status();
        for (const auto piece : pieces.value()) {
            std::string byte_encoded;
            for (const char value : piece) {
                const auto byte = static_cast<unsigned char>(value);
                byte_encoded += byte_codec().encoded[byte];
            }
            auto symbols_result = utf8_codepoints(byte_encoded);
            if (!symbols_result) return symbols_result.status();

            std::vector<std::string> symbols;
            symbols.reserve(symbols_result.value().size());
            for (const auto& [_, bytes] : symbols_result.value()) symbols.emplace_back(bytes);

            while (symbols.size() > 1) {
                std::optional<std::pair<std::string, std::string>> best_pair;
                std::size_t best_rank = std::numeric_limits<std::size_t>::max();
                for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
                    const auto rank_it = merge_rank_.find(pair_key(symbols[i], symbols[i + 1]));
                    if (rank_it != merge_rank_.end() && rank_it->second < best_rank) {
                        best_rank = rank_it->second;
                        best_pair = std::pair<std::string, std::string>{symbols[i], symbols[i + 1]};
                    }
                }
                if (!best_pair) break;

                std::vector<std::string> merged;
                merged.reserve(symbols.size());
                for (std::size_t i = 0; i < symbols.size();) {
                    if (i + 1 < symbols.size() && symbols[i] == best_pair->first &&
                        symbols[i + 1] == best_pair->second) {
                        merged.push_back(symbols[i] + symbols[i + 1]);
                        i += 2;
                    } else {
                        merged.push_back(std::move(symbols[i]));
                        ++i;
                    }
                }
                symbols = std::move(merged);
            }

            for (const auto& symbol : symbols) {
                const auto token = token_to_id_.find(symbol);
                if (token == token_to_id_.end()) {
                    return Status::data_error("BPE produced a piece absent from vocabulary");
                }
                output.push_back(token->second);
            }
        }
        return Status::ok();
    }

    std::shared_ptr<const TokenizerDefinition> definition_;
    std::unique_ptr<Regex> regex_;
    std::unordered_map<std::string, TokenId> token_to_id_;
    std::unordered_map<std::string, std::size_t> merge_rank_;
    std::unordered_set<TokenId> special_ids_;
    std::unordered_map<char, std::vector<std::pair<std::string, TokenId>>> special_by_first_;
};

} // namespace

Result<std::unique_ptr<Tokenizer>> create_tokenizer(std::shared_ptr<const TokenizerDefinition> definition) {
    if (!definition) return Status::invalid_argument("tokenizer definition is null");
    if (definition->model == "gpt2") {
        return Gpt2BpeTokenizer::create(std::move(definition));
    }
    return Status::unsupported("AIR tokenizer execution supports GGUF GPT-2 BPE models; found: " +
                               definition->model);
}

} // namespace air

#pragma once

#include "air/result.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace air::gguf {

enum class ValueType : std::uint32_t {
    u8 = 0,
    i8 = 1,
    u16 = 2,
    i16 = 3,
    u32 = 4,
    i32 = 5,
    f32 = 6,
    boolean = 7,
    string = 8,
    array = 9,
    u64 = 10,
    i64 = 11,
    f64 = 12,
};

using UnsignedArray = std::vector<std::uint64_t>;
using SignedArray = std::vector<std::int64_t>;
using FloatArray = std::vector<double>;
using BoolArray = std::vector<std::uint8_t>;
using StringArray = std::vector<std::string>;

using Scalar = std::variant<std::uint64_t, std::int64_t, double, bool, std::string>;
using Array = std::variant<UnsignedArray, SignedArray, FloatArray, BoolArray, StringArray>;

struct Value {
    ValueType type{ValueType::u8};
    bool is_array{false};
    Scalar scalar{std::uint64_t{0}};
    Array array{UnsignedArray{}};
};

struct TensorInfo {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    std::uint32_t type{0};
    std::uint64_t relative_offset{0};
};

struct Document {
    std::uint32_t version{0};
    std::uint32_t alignment{32};
    std::uint64_t data_offset{0};
    std::uint64_t file_size{0};
    std::unordered_map<std::string, Value> metadata;
    std::vector<TensorInfo> tensors;

    [[nodiscard]] const Value* find(std::string_view key) const noexcept;
    [[nodiscard]] Result<std::string> require_string(std::string_view key) const;
    [[nodiscard]] Result<std::uint64_t> require_u64(std::string_view key) const;
    [[nodiscard]] std::optional<std::string> optional_string(std::string_view key) const;
    [[nodiscard]] std::optional<std::uint64_t> optional_u64(std::string_view key) const;
    [[nodiscard]] std::optional<double> optional_number(std::string_view key) const;
    [[nodiscard]] std::optional<bool> optional_bool(std::string_view key) const;
    [[nodiscard]] const StringArray* string_array(std::string_view key) const noexcept;
    [[nodiscard]] const SignedArray* signed_array(std::string_view key) const noexcept;
    [[nodiscard]] const FloatArray* float_array(std::string_view key) const noexcept;
};

[[nodiscard]] Result<Document> read_document(const std::filesystem::path& path);

} // namespace air::gguf

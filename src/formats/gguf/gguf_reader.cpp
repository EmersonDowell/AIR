#include "formats/gguf/gguf_reader.hpp"

#include <algorithm>
#include <bit>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <unordered_set>

namespace air::gguf {
namespace {

constexpr std::uint64_t kMaxStringBytes = 256ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kMaxArrayElements = 16ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kMaxMetadataEntries = 1ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kMaxTensors = 4ULL * 1024ULL * 1024ULL;
constexpr std::uint32_t kMaxDimensions = 8;

class Reader {
public:
    explicit Reader(const std::filesystem::path& path)
        : stream_(path, std::ios::binary | std::ios::ate) {
        if (stream_) {
            const auto end = stream_.tellg();
            if (end >= 0) {
                size_ = static_cast<std::uint64_t>(end);
                stream_.seekg(0, std::ios::beg);
            }
        }
    }

    [[nodiscard]] bool good() const noexcept { return stream_.is_open() && size_ != 0; }
    [[nodiscard]] std::uint64_t size() const noexcept { return size_; }
    [[nodiscard]] std::uint64_t position() const noexcept { return position_; }

    Result<std::vector<std::byte>> bytes(std::uint64_t count) {
        if (count > remaining()) {
            return Status::data_error("GGUF ends unexpectedly at byte " + std::to_string(position_));
        }
        if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            return Status::unsupported("GGUF field exceeds addressable memory");
        }
        std::vector<std::byte> out(static_cast<std::size_t>(count));
        if (count != 0) {
            stream_.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count));
            if (!stream_) {
                return Status::io_error("failed while reading GGUF at byte " + std::to_string(position_));
            }
        }
        position_ += count;
        return out;
    }

    Result<std::uint8_t> u8() {
        auto data = bytes(1);
        if (!data) return data.status();
        return static_cast<std::uint8_t>(data.value()[0]);
    }

    Result<std::uint16_t> u16() {
        auto data = bytes(2);
        if (!data) return data.status();
        const auto& b = data.value();
        const auto value = static_cast<std::uint32_t>(static_cast<std::uint8_t>(b[0])) |
                           (static_cast<std::uint32_t>(static_cast<std::uint8_t>(b[1])) << 8U);
        return static_cast<std::uint16_t>(value);
    }

    Result<std::uint32_t> u32() {
        auto data = bytes(4);
        if (!data) return data.status();
        const auto& b = data.value();
        std::uint32_t value = 0;
        for (std::uint32_t i = 0; i < 4; ++i) {
            value |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(b[i])) << (8U * i);
        }
        return value;
    }

    Result<std::uint64_t> u64() {
        auto data = bytes(8);
        if (!data) return data.status();
        const auto& b = data.value();
        std::uint64_t value = 0;
        for (std::uint32_t i = 0; i < 8; ++i) {
            value |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(b[i])) << (8U * i);
        }
        return value;
    }

    Result<std::int8_t> i8() {
        auto value = u8();
        if (!value) return value.status();
        return std::bit_cast<std::int8_t>(value.value());
    }

    Result<std::int16_t> i16() {
        auto value = u16();
        if (!value) return value.status();
        return std::bit_cast<std::int16_t>(value.value());
    }

    Result<std::int32_t> i32() {
        auto value = u32();
        if (!value) return value.status();
        return std::bit_cast<std::int32_t>(value.value());
    }

    Result<std::int64_t> i64() {
        auto value = u64();
        if (!value) return value.status();
        return std::bit_cast<std::int64_t>(value.value());
    }

    Result<float> f32() {
        auto value = u32();
        if (!value) return value.status();
        return std::bit_cast<float>(value.value());
    }

    Result<double> f64() {
        auto value = u64();
        if (!value) return value.status();
        return std::bit_cast<double>(value.value());
    }

    Result<std::string> string() {
        auto length = u64();
        if (!length) return length.status();
        if (length.value() > kMaxStringBytes) {
            return Status::data_error("GGUF string exceeds AIR safety limit");
        }
        auto data = bytes(length.value());
        if (!data) return data.status();
        const auto& bytes_value = data.value();
        return std::string(reinterpret_cast<const char*>(bytes_value.data()), bytes_value.size());
    }

private:
    [[nodiscard]] std::uint64_t remaining() const noexcept {
        return position_ <= size_ ? size_ - position_ : 0;
    }

    std::ifstream stream_;
    std::uint64_t size_{0};
    std::uint64_t position_{0};
};

Result<Scalar> read_scalar(Reader& reader, ValueType type) {
    switch (type) {
    case ValueType::u8: {
        auto value = reader.u8(); if (!value) return value.status(); return Scalar{std::uint64_t{value.value()}};
    }
    case ValueType::i8: {
        auto value = reader.i8(); if (!value) return value.status(); return Scalar{std::int64_t{value.value()}};
    }
    case ValueType::u16: {
        auto value = reader.u16(); if (!value) return value.status(); return Scalar{std::uint64_t{value.value()}};
    }
    case ValueType::i16: {
        auto value = reader.i16(); if (!value) return value.status(); return Scalar{std::int64_t{value.value()}};
    }
    case ValueType::u32: {
        auto value = reader.u32(); if (!value) return value.status(); return Scalar{std::uint64_t{value.value()}};
    }
    case ValueType::i32: {
        auto value = reader.i32(); if (!value) return value.status(); return Scalar{std::int64_t{value.value()}};
    }
    case ValueType::f32: {
        auto value = reader.f32(); if (!value) return value.status(); return Scalar{double{value.value()}};
    }
    case ValueType::boolean: {
        auto value = reader.u8(); if (!value) return value.status(); return Scalar{value.value() != 0};
    }
    case ValueType::string: {
        auto value = reader.string(); if (!value) return value.status(); return Scalar{std::move(value).value()};
    }
    case ValueType::u64: {
        auto value = reader.u64(); if (!value) return value.status(); return Scalar{value.value()};
    }
    case ValueType::i64: {
        auto value = reader.i64(); if (!value) return value.status(); return Scalar{value.value()};
    }
    case ValueType::f64: {
        auto value = reader.f64(); if (!value) return value.status(); return Scalar{value.value()};
    }
    case ValueType::array:
        break;
    }
    return Status::data_error("invalid nested GGUF array value type");
}

Result<Array> read_array(Reader& reader, ValueType element_type, std::uint64_t count) {
    if (count > kMaxArrayElements) {
        return Status::data_error("GGUF array exceeds AIR safety limit");
    }
    if (element_type == ValueType::array) {
        return Status::data_error("nested GGUF arrays are invalid");
    }

    if (element_type == ValueType::string) {
        StringArray out;
        out.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t i = 0; i < count; ++i) {
            auto value = reader.string();
            if (!value) return value.status();
            out.push_back(std::move(value).value());
        }
        return Array{std::move(out)};
    }

    if (element_type == ValueType::f32 || element_type == ValueType::f64) {
        FloatArray out;
        out.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t i = 0; i < count; ++i) {
            auto value = read_scalar(reader, element_type);
            if (!value) return value.status();
            out.push_back(std::get<double>(value.value()));
        }
        return Array{std::move(out)};
    }

    if (element_type == ValueType::boolean) {
        BoolArray out;
        out.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t i = 0; i < count; ++i) {
            auto value = reader.u8();
            if (!value) return value.status();
            out.push_back(value.value() != 0 ? 1U : 0U);
        }
        return Array{std::move(out)};
    }

    const bool is_signed = element_type == ValueType::i8 || element_type == ValueType::i16 ||
                           element_type == ValueType::i32 || element_type == ValueType::i64;
    if (is_signed) {
        SignedArray out;
        out.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t i = 0; i < count; ++i) {
            auto value = read_scalar(reader, element_type);
            if (!value) return value.status();
            out.push_back(std::get<std::int64_t>(value.value()));
        }
        return Array{std::move(out)};
    }

    UnsignedArray out;
    out.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        auto value = read_scalar(reader, element_type);
        if (!value) return value.status();
        out.push_back(std::get<std::uint64_t>(value.value()));
    }
    return Array{std::move(out)};
}

Result<ValueType> read_value_type(Reader& reader) {
    auto raw = reader.u32();
    if (!raw) return raw.status();
    if (raw.value() > static_cast<std::uint32_t>(ValueType::f64)) {
        return Status::data_error("unknown GGUF metadata value type: " + std::to_string(raw.value()));
    }
    return static_cast<ValueType>(raw.value());
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    const auto remainder = value % alignment;
    return remainder == 0 ? value : value + (alignment - remainder);
}

} // namespace

const Value* Document::find(std::string_view key) const noexcept {
    const auto it = metadata.find(std::string(key));
    return it == metadata.end() ? nullptr : &it->second;
}

Result<std::string> Document::require_string(std::string_view key) const {
    const auto* value = find(key);
    if (!value) return Status::data_error("missing GGUF metadata: " + std::string(key));
    if (value->is_array || !std::holds_alternative<std::string>(value->scalar)) {
        return Status::data_error("GGUF metadata has wrong type: " + std::string(key));
    }
    return std::get<std::string>(value->scalar);
}

Result<std::uint64_t> Document::require_u64(std::string_view key) const {
    const auto value = optional_u64(key);
    if (!value) return Status::data_error("missing or non-integer GGUF metadata: " + std::string(key));
    return *value;
}

std::optional<std::string> Document::optional_string(std::string_view key) const {
    const auto* value = find(key);
    if (!value || value->is_array || !std::holds_alternative<std::string>(value->scalar)) return std::nullopt;
    return std::get<std::string>(value->scalar);
}

std::optional<std::uint64_t> Document::optional_u64(std::string_view key) const {
    const auto* value = find(key);
    if (!value || value->is_array) return std::nullopt;
    if (const auto* unsigned_value = std::get_if<std::uint64_t>(&value->scalar)) return *unsigned_value;
    if (const auto* signed_value = std::get_if<std::int64_t>(&value->scalar); signed_value && *signed_value >= 0) {
        return static_cast<std::uint64_t>(*signed_value);
    }
    return std::nullopt;
}

std::optional<double> Document::optional_number(std::string_view key) const {
    const auto* value = find(key);
    if (!value || value->is_array) return std::nullopt;
    if (const auto* float_value = std::get_if<double>(&value->scalar)) return *float_value;
    if (const auto* unsigned_value = std::get_if<std::uint64_t>(&value->scalar)) return static_cast<double>(*unsigned_value);
    if (const auto* signed_value = std::get_if<std::int64_t>(&value->scalar)) return static_cast<double>(*signed_value);
    return std::nullopt;
}

std::optional<bool> Document::optional_bool(std::string_view key) const {
    const auto* value = find(key);
    if (!value || value->is_array) return std::nullopt;
    if (const auto* boolean_value = std::get_if<bool>(&value->scalar)) return *boolean_value;
    return std::nullopt;
}

const StringArray* Document::string_array(std::string_view key) const noexcept {
    const auto* value = find(key);
    if (!value || !value->is_array) return nullptr;
    return std::get_if<StringArray>(&value->array);
}

const SignedArray* Document::signed_array(std::string_view key) const noexcept {
    const auto* value = find(key);
    if (!value || !value->is_array) return nullptr;
    return std::get_if<SignedArray>(&value->array);
}

const FloatArray* Document::float_array(std::string_view key) const noexcept {
    const auto* value = find(key);
    if (!value || !value->is_array) return nullptr;
    return std::get_if<FloatArray>(&value->array);
}

Result<Document> read_document(const std::filesystem::path& path) {
    Reader reader(path);
    if (!reader.good()) {
        return Status::io_error("unable to open or read GGUF file: " + path.string());
    }

    auto magic = reader.bytes(4);
    if (!magic) return magic.status();
    const auto& m = magic.value();
    if (m.size() != 4 || static_cast<char>(m[0]) != 'G' || static_cast<char>(m[1]) != 'G' ||
        static_cast<char>(m[2]) != 'U' || static_cast<char>(m[3]) != 'F') {
        return Status::invalid_argument("file is not GGUF: " + path.string());
    }

    auto version = reader.u32();
    auto tensor_count = reader.u64();
    auto metadata_count = reader.u64();
    if (!version) return version.status();
    if (!tensor_count) return tensor_count.status();
    if (!metadata_count) return metadata_count.status();

    if (version.value() != 3) {
        return Status::unsupported("AIR currently supports GGUF V3; file uses V" +
                                   std::to_string(version.value()));
    }
    if (tensor_count.value() > kMaxTensors || metadata_count.value() > kMaxMetadataEntries) {
        return Status::data_error("GGUF header counts exceed AIR safety limits");
    }

    Document document;
    document.version = version.value();
    document.file_size = reader.size();
    document.metadata.reserve(static_cast<std::size_t>(metadata_count.value()));

    for (std::uint64_t i = 0; i < metadata_count.value(); ++i) {
        auto key = reader.string();
        if (!key) return key.status();
        if (key.value().empty()) return Status::data_error("GGUF metadata key is empty");
        if (document.metadata.contains(key.value())) {
            return Status::data_error("duplicate GGUF metadata key: " + key.value());
        }

        auto type = read_value_type(reader);
        if (!type) return type.status();

        Value value;
        if (type.value() == ValueType::array) {
            auto element_type = read_value_type(reader);
            auto count = reader.u64();
            if (!element_type) return element_type.status();
            if (!count) return count.status();
            auto array = read_array(reader, element_type.value(), count.value());
            if (!array) return array.status();
            value.type = element_type.value();
            value.is_array = true;
            value.array = std::move(array).value();
        } else {
            auto scalar = read_scalar(reader, type.value());
            if (!scalar) return scalar.status();
            value.type = type.value();
            value.scalar = std::move(scalar).value();
        }

        document.metadata.emplace(std::move(key).value(), std::move(value));
    }

    if (const auto alignment = document.optional_u64("general.alignment")) {
        if (*alignment == 0 || (*alignment & (*alignment - 1)) != 0 || *alignment > 4096) {
            return Status::data_error("general.alignment must be a power of two in [1, 4096]");
        }
        document.alignment = static_cast<std::uint32_t>(*alignment);
    }

    if (const auto split_count = document.optional_u64("split.count"); split_count && *split_count > 1) {
        return Status::unsupported("multi-file GGUF model sets are deferred until model execution requires them");
    }

    document.tensors.reserve(static_cast<std::size_t>(tensor_count.value()));
    std::unordered_set<std::string> tensor_names;
    tensor_names.reserve(static_cast<std::size_t>(tensor_count.value()));

    for (std::uint64_t i = 0; i < tensor_count.value(); ++i) {
        TensorInfo tensor;
        auto name = reader.string();
        auto dimensions = reader.u32();
        if (!name) return name.status();
        if (!dimensions) return dimensions.status();
        if (name.value().empty()) return Status::data_error("GGUF tensor name is empty");
        if (!tensor_names.emplace(name.value()).second) {
            return Status::data_error("duplicate GGUF tensor name: " + name.value());
        }
        if (dimensions.value() == 0 || dimensions.value() > kMaxDimensions) {
            return Status::data_error("GGUF tensor has invalid dimension count: " + name.value());
        }

        tensor.name = std::move(name).value();
        tensor.dimensions.reserve(dimensions.value());
        for (std::uint32_t d = 0; d < dimensions.value(); ++d) {
            auto dimension = reader.u64();
            if (!dimension) return dimension.status();
            if (dimension.value() == 0) return Status::data_error("GGUF tensor has zero dimension: " + tensor.name);
            tensor.dimensions.push_back(dimension.value());
        }

        auto type = reader.u32();
        auto offset = reader.u64();
        if (!type) return type.status();
        if (!offset) return offset.status();
        tensor.type = type.value();
        tensor.relative_offset = offset.value();
        document.tensors.push_back(std::move(tensor));
    }

    document.data_offset = align_up(reader.position(), document.alignment);
    if (document.data_offset > document.file_size) {
        return Status::data_error("GGUF tensor-data offset exceeds file size");
    }

    const auto data_bytes = document.file_size - document.data_offset;
    for (const auto& tensor : document.tensors) {
        if (tensor.relative_offset > data_bytes) {
            return Status::data_error("GGUF tensor offset exceeds tensor-data blob: " + tensor.name);
        }
    }

    return document;
}

} // namespace air::gguf

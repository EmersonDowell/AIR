#include "air/reference.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

namespace air {
namespace {

float f16_to_f32(std::uint16_t value) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
    const std::uint32_t exponent = (value >> 10U) & 0x1fU;
    const std::uint32_t mantissa = value & 0x03ffU;

    std::uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            std::uint32_t m = mantissa;
            std::uint32_t e = 127U - 15U + 1U;
            while ((m & 0x0400U) == 0U) {
                m <<= 1U;
                --e;
            }
            m &= 0x03ffU;
            bits = sign | (e << 23U) | (m << 13U);
        }
    } else if (exponent == 0x1fU) {
        bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + (127U - 15U)) << 23U) | (mantissa << 13U);
    }
    return std::bit_cast<float>(bits);
}

std::uint16_t read_u16(const std::byte* data) noexcept {
    return static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(data[0])) |
           static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(data[1]) << 8U);
}

float read_f32(const std::byte* data) noexcept {
    std::uint32_t bits = static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[0])) |
                         (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[1])) << 8U) |
                         (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[2])) << 16U) |
                         (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[3])) << 24U);
    return std::bit_cast<float>(bits);
}

Result<std::vector<float>> decode_all(std::span<const std::byte> bytes,
                                      DataType type,
                                      std::uint64_t elements) {
    if (elements > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return Status::unsupported("tensor is too large for host address space");
    }
    std::vector<float> output(static_cast<std::size_t>(elements));

    switch (type) {
    case DataType::f32: {
        const auto expected = elements * 4U;
        if (bytes.size() != expected) return Status::data_error("F32 tensor storage size mismatch");
        for (std::uint64_t i = 0; i < elements; ++i) {
            output[static_cast<std::size_t>(i)] = read_f32(bytes.data() + i * 4U);
        }
        return output;
    }
    case DataType::f16: {
        const auto expected = elements * 2U;
        if (bytes.size() != expected) return Status::data_error("F16 tensor storage size mismatch");
        for (std::uint64_t i = 0; i < elements; ++i) {
            output[static_cast<std::size_t>(i)] = f16_to_f32(read_u16(bytes.data() + i * 2U));
        }
        return output;
    }
    case DataType::bf16: {
        const auto expected = elements * 2U;
        if (bytes.size() != expected) return Status::data_error("BF16 tensor storage size mismatch");
        for (std::uint64_t i = 0; i < elements; ++i) {
            const auto upper = static_cast<std::uint32_t>(read_u16(bytes.data() + i * 2U)) << 16U;
            output[static_cast<std::size_t>(i)] = std::bit_cast<float>(upper);
        }
        return output;
    }
    case DataType::q4_0: {
        if (elements % 32U != 0U || bytes.size() != (elements / 32U) * 18U) {
            return Status::data_error("Q4_0 tensor storage geometry mismatch");
        }
        for (std::uint64_t block = 0; block < elements / 32U; ++block) {
            const std::byte* source = bytes.data() + block * 18U;
            const float scale = f16_to_f32(read_u16(source));
            source += 2;
            const auto base = block * 32U;
            for (std::uint64_t i = 0; i < 16U; ++i) {
                const auto packed = std::to_integer<std::uint8_t>(source[i]);
                const auto low = static_cast<int>(packed & 0x0fU) - 8;
                const auto high = static_cast<int>(packed >> 4U) - 8;
                output[static_cast<std::size_t>(base + i)] = scale * static_cast<float>(low);
                output[static_cast<std::size_t>(base + i + 16U)] = scale * static_cast<float>(high);
            }
        }
        return output;
    }
    case DataType::q5_0: {
        if (elements % 32U != 0U || bytes.size() != (elements / 32U) * 22U) {
            return Status::data_error("Q5_0 tensor storage geometry mismatch");
        }
        for (std::uint64_t block = 0; block < elements / 32U; ++block) {
            const std::byte* source = bytes.data() + block * 22U;
            const float scale = f16_to_f32(read_u16(source));
            const std::byte* high_bits = source + 2U;
            const std::byte* quants = source + 6U;
            const auto base = block * 32U;
            for (std::uint32_t i = 0U; i < 16U; ++i) {
                const auto packed = std::to_integer<std::uint8_t>(quants[i]);
                const auto high0 = static_cast<std::uint8_t>(
                    ((std::to_integer<std::uint8_t>(high_bits[i / 8U]) >> (i % 8U)) & 0x01U) << 4U);
                const auto second = i + 16U;
                const auto high1 = static_cast<std::uint8_t>(
                    ((std::to_integer<std::uint8_t>(high_bits[second / 8U]) >> (second % 8U)) & 0x01U) << 4U);
                const auto low0 = static_cast<std::uint8_t>(packed & 0x0fU);
                const auto low1 = static_cast<std::uint8_t>(packed >> 4U);
                const auto q0 = static_cast<int>(low0 | high0) - 16;
                const auto q1 = static_cast<int>(low1 | high1) - 16;
                output[static_cast<std::size_t>(base + i)] = scale * static_cast<float>(q0);
                output[static_cast<std::size_t>(base + second)] = scale * static_cast<float>(q1);
            }
        }
        return output;
    }
    case DataType::q8_0: {
        if (elements % 32U != 0U || bytes.size() != (elements / 32U) * 34U) {
            return Status::data_error("Q8_0 tensor storage geometry mismatch");
        }
        for (std::uint64_t block = 0; block < elements / 32U; ++block) {
            const std::byte* source = bytes.data() + block * 34U;
            const float scale = f16_to_f32(read_u16(source));
            source += 2;
            const auto base = block * 32U;
            for (std::uint64_t i = 0; i < 32U; ++i) {
                const auto quantized = static_cast<std::int8_t>(std::to_integer<std::uint8_t>(source[i]));
                output[static_cast<std::size_t>(base + i)] = scale * static_cast<float>(quantized);
            }
        }
        return output;
    }
    case DataType::q4_k: {
        constexpr std::uint64_t block_elements = 256U;
        constexpr std::uint64_t block_bytes = 144U;
        if (elements % block_elements != 0U || bytes.size() != (elements / block_elements) * block_bytes) {
            return Status::data_error("Q4_K tensor storage geometry mismatch");
        }
        const auto scale_min = [](std::uint32_t index, const std::byte* packed,
                                  std::uint8_t& scale, std::uint8_t& minimum) {
            const auto q = [packed](std::uint32_t i) -> std::uint32_t {
                return static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(packed[i]));
            };
            if (index < 4U) {
                scale = q(index) & 63U;
                minimum = q(index + 4U) & 63U;
            } else {
                scale = static_cast<std::uint8_t>((q(index + 4U) & 0x0fU) | ((q(index - 4U) >> 6U) << 4U));
                minimum = static_cast<std::uint8_t>((q(index + 4U) >> 4U) | ((q(index) >> 6U) << 4U));
            }
        };
        for (std::uint64_t block = 0; block < elements / block_elements; ++block) {
            const std::byte* source = bytes.data() + block * block_bytes;
            const float block_scale = f16_to_f32(read_u16(source));
            const float block_minimum = f16_to_f32(read_u16(source + 2U));
            const std::byte* scales = source + 4U;
            const std::byte* quants = source + 16U;
            const auto base = block * block_elements;
            std::uint32_t scale_index = 0U;
            for (std::uint32_t group = 0U; group < 4U; ++group) {
                std::uint8_t scale_low = 0U;
                std::uint8_t minimum_low = 0U;
                std::uint8_t scale_high = 0U;
                std::uint8_t minimum_high = 0U;
                scale_min(scale_index, scales, scale_low, minimum_low);
                scale_min(scale_index + 1U, scales, scale_high, minimum_high);
                const float d0 = block_scale * static_cast<float>(scale_low);
                const float m0 = block_minimum * static_cast<float>(minimum_low);
                const float d1 = block_scale * static_cast<float>(scale_high);
                const float m1 = block_minimum * static_cast<float>(minimum_high);
                for (std::uint32_t i = 0U; i < 32U; ++i) {
                    const auto packed = std::to_integer<std::uint8_t>(quants[group * 32U + i]);
                    output[static_cast<std::size_t>(base + group * 64U + i)] =
                        d0 * static_cast<float>(packed & 0x0fU) - m0;
                    output[static_cast<std::size_t>(base + group * 64U + 32U + i)] =
                        d1 * static_cast<float>(packed >> 4U) - m1;
                }
                scale_index += 2U;
            }
        }
        return output;
    }
    case DataType::q6_k: {
        constexpr std::uint64_t block_elements = 256U;
        constexpr std::uint64_t block_bytes = 210U;
        if (elements % block_elements != 0U || bytes.size() != (elements / block_elements) * block_bytes) {
            return Status::data_error("Q6_K tensor storage geometry mismatch");
        }
        for (std::uint64_t block = 0; block < elements / block_elements; ++block) {
            const std::byte* source = bytes.data() + block * block_bytes;
            const std::byte* ql = source;
            const std::byte* qh = source + 128U;
            const std::byte* scales = source + 192U;
            const float block_scale = f16_to_f32(read_u16(source + 208U));
            const auto base = block * block_elements;
            for (std::uint32_t half = 0U; half < 2U; ++half) {
                const std::byte* low = ql + half * 64U;
                const std::byte* high = qh + half * 32U;
                const std::byte* scale = scales + half * 8U;
                for (std::uint32_t i = 0U; i < 32U; ++i) {
                    const auto lo0 = std::to_integer<std::uint8_t>(low[i]);
                    const auto lo1 = std::to_integer<std::uint8_t>(low[i + 32U]);
                    const auto hi = std::to_integer<std::uint8_t>(high[i]);
                    const std::uint32_t scale_pair = i / 16U;
                    const auto decode = [](std::uint8_t low_bits, std::uint8_t high_bits) {
                        return static_cast<std::int32_t>(low_bits | static_cast<std::uint8_t>(high_bits << 4U)) - 32;
                    };
                    const std::int32_t q0 = decode(lo0 & 0x0fU, (hi >> 0U) & 0x03U);
                    const std::int32_t q1 = decode(lo1 & 0x0fU, (hi >> 2U) & 0x03U);
                    const std::int32_t q2 = decode(lo0 >> 4U, (hi >> 4U) & 0x03U);
                    const std::int32_t q3 = decode(lo1 >> 4U, (hi >> 6U) & 0x03U);
                    const auto signed_scale = [scale](std::uint32_t index) {
                        return static_cast<std::int8_t>(std::to_integer<std::uint8_t>(scale[index]));
                    };
                    const auto half_base = base + static_cast<std::uint64_t>(half) * 128U;
                    output[static_cast<std::size_t>(half_base + i)] =
                        block_scale * static_cast<float>(signed_scale(scale_pair + 0U)) * static_cast<float>(q0);
                    output[static_cast<std::size_t>(half_base + 32U + i)] =
                        block_scale * static_cast<float>(signed_scale(scale_pair + 2U)) * static_cast<float>(q1);
                    output[static_cast<std::size_t>(half_base + 64U + i)] =
                        block_scale * static_cast<float>(signed_scale(scale_pair + 4U)) * static_cast<float>(q2);
                    output[static_cast<std::size_t>(half_base + 96U + i)] =
                        block_scale * static_cast<float>(signed_scale(scale_pair + 6U)) * static_cast<float>(q3);
                }
            }
        }
        return output;
    }
    default:
        return Status::unsupported("reference decoder does not support tensor type " +
                                   std::string(to_string(type)));
    }
}

} // namespace

bool ReferenceTensorReader::supports(DataType type) noexcept {
    return type == DataType::f32 || type == DataType::f16 || type == DataType::bf16 ||
           type == DataType::q4_0 || type == DataType::q5_0 || type == DataType::q8_0 ||
           type == DataType::q4_k || type == DataType::q6_k;
}

Result<std::vector<float>> ReferenceTensorReader::decode_tensor(const TensorDescriptor& tensor) const {
    if (!supports(tensor.type)) {
        return Status::unsupported("reference execution cannot decode tensor " + tensor.name +
                                   " of type " + to_string(tensor.type));
    }
    if (!tensor.byte_size_exact) {
        return Status::unsupported("reference execution requires exact tensor geometry: " + tensor.name);
    }
    auto bytes = model_->tensor_bytes(tensor);
    if (!bytes) return bytes.status();
    return decode_all(bytes.value(), tensor.type, tensor.shape.element_count());
}

Result<std::vector<float>> ReferenceTensorReader::decode_range(const TensorDescriptor& tensor,
                                                               std::uint64_t element_offset,
                                                               std::uint64_t element_count) const {
    const auto total = tensor.shape.element_count();
    if (element_offset > total || element_count > total - element_offset) {
        return Status::invalid_argument("tensor element range is outside tensor: " + tensor.name);
    }
    if (!supports(tensor.type)) {
        return Status::unsupported("reference execution cannot decode tensor " + tensor.name +
                                   " of type " + to_string(tensor.type));
    }
    if (!tensor.byte_size_exact) {
        return Status::unsupported("reference execution requires exact tensor geometry: " + tensor.name);
    }

    std::uint64_t byte_offset = 0;
    std::uint64_t byte_count = 0;
    switch (tensor.type) {
    case DataType::f32:
        byte_offset = element_offset * 4U;
        byte_count = element_count * 4U;
        break;
    case DataType::f16:
    case DataType::bf16:
        byte_offset = element_offset * 2U;
        byte_count = element_count * 2U;
        break;
    case DataType::q4_0:
        if (element_offset % 32U != 0U || element_count % 32U != 0U) {
            return Status::unsupported("Q4_0 reference range must align to 32-element blocks: " + tensor.name);
        }
        byte_offset = (element_offset / 32U) * 18U;
        byte_count = (element_count / 32U) * 18U;
        break;
    case DataType::q5_0:
        if (element_offset % 32U != 0U || element_count % 32U != 0U) {
            return Status::unsupported("Q5_0 reference range must align to 32-element blocks: " + tensor.name);
        }
        byte_offset = (element_offset / 32U) * 22U;
        byte_count = (element_count / 32U) * 22U;
        break;
    case DataType::q8_0:
        if (element_offset % 32U != 0U || element_count % 32U != 0U) {
            return Status::unsupported("Q8_0 reference range must align to 32-element blocks: " + tensor.name);
        }
        byte_offset = (element_offset / 32U) * 34U;
        byte_count = (element_count / 32U) * 34U;
        break;
    case DataType::q4_k:
        if (element_offset % 256U != 0U || element_count % 256U != 0U) {
            return Status::unsupported("Q4_K reference range must align to 256-element blocks: " + tensor.name);
        }
        byte_offset = (element_offset / 256U) * 144U;
        byte_count = (element_count / 256U) * 144U;
        break;
    case DataType::q6_k:
        if (element_offset % 256U != 0U || element_count % 256U != 0U) {
            return Status::unsupported("Q6_K reference range must align to 256-element blocks: " + tensor.name);
        }
        byte_offset = (element_offset / 256U) * 210U;
        byte_count = (element_count / 256U) * 210U;
        break;
    default:
        return Status::unsupported("reference decoder does not support tensor type " +
                                   std::string(to_string(tensor.type)));
    }

    auto bytes = model_->tensor_bytes(tensor);
    if (!bytes) return bytes.status();
    if (byte_offset > bytes.value().size() || byte_count > bytes.value().size() - byte_offset) {
        return Status::data_error("decoded tensor range exceeds storage: " + tensor.name);
    }
    return decode_all(bytes.value().subspan(static_cast<std::size_t>(byte_offset),
                                            static_cast<std::size_t>(byte_count)),
                      tensor.type, element_count);
}

Result<std::vector<float>> ReferenceTensorReader::vector(const std::string& name) const {
    const auto* tensor = model_->find_tensor(name);
    if (!tensor) return Status::data_error("missing tensor: " + name);
    if (tensor->shape.dimensions.size() != 1U) {
        return Status::data_error("expected rank-1 tensor: " + name);
    }
    return decode_tensor(*tensor);
}

Result<std::vector<float>> ReferenceTensorReader::row(const std::string& name,
                                                      std::uint64_t row_index) const {
    const auto* tensor = model_->find_tensor(name);
    if (!tensor) return Status::data_error("missing tensor: " + name);
    if (tensor->shape.dimensions.size() != 2U) {
        return Status::data_error("expected rank-2 tensor: " + name);
    }
    const auto columns = tensor->shape.dimensions[0];
    const auto rows = tensor->shape.dimensions[1];
    if (row_index >= rows) return Status::invalid_argument("tensor row is outside matrix: " + name);
    return decode_range(*tensor, row_index * columns, columns);
}

Result<std::vector<float>> ReferenceTensorReader::matvec(const std::string& name,
                                                         std::span<const float> input) const {
    const auto* tensor = model_->find_tensor(name);
    if (!tensor) return Status::data_error("missing tensor: " + name);
    if (tensor->shape.dimensions.size() != 2U) {
        return Status::data_error("expected rank-2 matrix tensor: " + name);
    }
    const auto columns = tensor->shape.dimensions[0];
    const auto rows = tensor->shape.dimensions[1];
    if (columns != input.size()) {
        return Status::data_error("matrix input width mismatch for tensor: " + name);
    }

    std::vector<float> output(static_cast<std::size_t>(rows), 0.0F);
    for (std::uint64_t row_index = 0; row_index < rows; ++row_index) {
        auto decoded_row = decode_range(*tensor, row_index * columns, columns);
        if (!decoded_row) return decoded_row.status();
        double sum = 0.0;
        for (std::uint64_t column = 0; column < columns; ++column) {
            sum += static_cast<double>(decoded_row.value()[static_cast<std::size_t>(column)]) *
                   static_cast<double>(input[static_cast<std::size_t>(column)]);
        }
        output[static_cast<std::size_t>(row_index)] = static_cast<float>(sum);
    }
    return output;
}

} // namespace air

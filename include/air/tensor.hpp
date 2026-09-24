#pragma once

#include "air/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace air {

struct TensorShape {
    std::vector<std::uint64_t> dimensions;

    [[nodiscard]] std::uint64_t element_count() const noexcept;
};

struct TensorDescriptor {
    std::string name;
    DataType type{DataType::unknown};
    std::uint32_t format_type{0};
    TensorShape shape;
    std::uint64_t byte_offset{0};
    std::uint64_t byte_size{0};
    bool byte_size_exact{false};
};

} // namespace air

#pragma once

#include <cstdint>
#include <string>

namespace air {

enum class DeviceKind {
    cpu = 0,
    cuda,
};

struct DeviceInfo {
    DeviceKind kind{DeviceKind::cpu};
    int ordinal{0};
    std::string name;
    int compute_major{0};
    int compute_minor{0};
    std::uint64_t total_memory_bytes{0};
    std::uint64_t free_memory_bytes{0};
};

} // namespace air

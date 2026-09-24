#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace air {

inline constexpr std::uint32_t hardware_topology_schema_version = 1U;

enum class HardwareNodeKind {
    cpu = 0,
    host_memory,
    accelerator,
    storage,
    remote_accelerator,
};

enum class HardwareLinkKind {
    memory_access = 0,
    host_device,
    peer_device,
    storage_host,
    storage_device,
    remote,
};

[[nodiscard]] const char* to_string(HardwareNodeKind kind) noexcept;
[[nodiscard]] const char* to_string(HardwareLinkKind kind) noexcept;

struct HardwareNode {
    std::string id;
    HardwareNodeKind kind{HardwareNodeKind::cpu};
    std::string name;
    std::string backend;
    std::string architecture;
    std::int32_t ordinal{-1};
    std::int32_t numa_node{-1};
    std::uint64_t total_bytes{0};
    std::uint64_t available_bytes{0};
    std::vector<std::string> capabilities;
};

struct HardwareLink {
    std::string source_id;
    std::string target_id;
    HardwareLinkKind kind{HardwareLinkKind::memory_access};
    bool measured{false};
    double bandwidth_bytes_per_second{0.0};
    double latency_microseconds{0.0};
};

struct HardwareTopology {
    std::uint32_t schema_version{hardware_topology_schema_version};
    std::string fingerprint;
    std::vector<HardwareNode> nodes;
    std::vector<HardwareLink> links;
};

struct HardwareTopologyValidation {
    bool valid{false};
    std::string message;
};

[[nodiscard]] const HardwareNode* find_hardware_node(
    const HardwareTopology& topology, std::string_view id) noexcept;

// Structural validation only. This function does not select placement,
// scheduling, execution tactics, or infer missing performance.
[[nodiscard]] HardwareTopologyValidation validate_hardware_topology(
    const HardwareTopology& topology);

} // namespace air

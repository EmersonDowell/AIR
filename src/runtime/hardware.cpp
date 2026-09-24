#include "air/hardware.hpp"

#include <cmath>
#include <unordered_set>

namespace air {

const char* to_string(HardwareNodeKind kind) noexcept {
    switch (kind) {
    case HardwareNodeKind::cpu: return "cpu";
    case HardwareNodeKind::host_memory: return "host-memory";
    case HardwareNodeKind::accelerator: return "accelerator";
    case HardwareNodeKind::storage: return "storage";
    case HardwareNodeKind::remote_accelerator: return "remote-accelerator";
    }
    return "unknown";
}

const char* to_string(HardwareLinkKind kind) noexcept {
    switch (kind) {
    case HardwareLinkKind::memory_access: return "memory-access";
    case HardwareLinkKind::host_device: return "host-device";
    case HardwareLinkKind::peer_device: return "peer-device";
    case HardwareLinkKind::storage_host: return "storage-host";
    case HardwareLinkKind::storage_device: return "storage-device";
    case HardwareLinkKind::remote: return "remote";
    }
    return "unknown";
}

const HardwareNode* find_hardware_node(
    const HardwareTopology& topology, std::string_view id) noexcept {
    for (const auto& node : topology.nodes) {
        if (node.id == id) return &node;
    }
    return nullptr;
}

HardwareTopologyValidation validate_hardware_topology(const HardwareTopology& topology) {
    if (topology.schema_version != hardware_topology_schema_version) {
        return {false, "unsupported hardware topology schema version"};
    }
    if (topology.nodes.empty()) {
        return {false, "hardware topology has no nodes"};
    }

    std::unordered_set<std::string> ids;
    ids.reserve(topology.nodes.size());
    for (const auto& node : topology.nodes) {
        if (node.id.empty()) return {false, "hardware topology node has empty id"};
        if (!ids.insert(node.id).second) {
            return {false, "hardware topology contains duplicate node id: " + node.id};
        }
        if (node.total_bytes != 0U && node.available_bytes > node.total_bytes) {
            return {false, "hardware topology node available bytes exceed total bytes: " + node.id};
        }
    }

    for (const auto& link : topology.links) {
        if (link.source_id.empty() || link.target_id.empty()) {
            return {false, "hardware topology link has empty endpoint"};
        }
        if (link.source_id == link.target_id) {
            return {false, "hardware topology link endpoints must differ"};
        }
        if (!ids.contains(link.source_id) || !ids.contains(link.target_id)) {
            return {false, "hardware topology link references an unknown node"};
        }
        if (!std::isfinite(link.bandwidth_bytes_per_second) ||
            !std::isfinite(link.latency_microseconds) ||
            link.bandwidth_bytes_per_second < 0.0 ||
            link.latency_microseconds < 0.0) {
            return {false, "hardware topology link contains invalid measurement"};
        }
        if (link.measured && link.bandwidth_bytes_per_second <= 0.0) {
            return {false, "measured hardware topology link requires positive bandwidth"};
        }
    }
    return {true, "ok"};
}

} // namespace air

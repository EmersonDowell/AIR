#include "air/hardware.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

air::HardwareTopology laptop_fixture() {
    air::HardwareTopology topology;
    topology.fingerprint = "synthetic:laptop";
    topology.nodes = {
        {"cpu0", air::HardwareNodeKind::cpu, "synthetic cpu", "cpu", "x86_64",
         0, 0, 0, 0, {"avx2"}},
        {"ram0", air::HardwareNodeKind::host_memory, "host memory", "host", "dram",
         0, 0, 36ULL * 1024ULL * 1024ULL * 1024ULL,
         30ULL * 1024ULL * 1024ULL * 1024ULL, {"pageable", "pinned-capable"}},
        {"gpu0", air::HardwareNodeKind::accelerator, "synthetic gpu", "cuda", "sm86",
         0, -1, 16ULL * 1024ULL * 1024ULL * 1024ULL,
         12ULL * 1024ULL * 1024ULL * 1024ULL, {"graphs", "async-copy"}},
        {"nvme0", air::HardwareNodeKind::storage, "synthetic nvme", "filesystem", "nvme",
         0, -1, 1024ULL * 1024ULL * 1024ULL * 1024ULL,
         512ULL * 1024ULL * 1024ULL * 1024ULL, {"mmap"}},
    };
    topology.links = {
        {"cpu0", "ram0", air::HardwareLinkKind::memory_access, true, 40.0e9, 0.08},
        {"ram0", "gpu0", air::HardwareLinkKind::host_device, true, 12.0e9, 8.0},
        {"nvme0", "ram0", air::HardwareLinkKind::storage_host, true, 5.0e9, 80.0},
    };
    return topology;
}

} // namespace

int main() {
    auto valid = laptop_fixture();
    auto result = air::validate_hardware_topology(valid);
    require(result.valid, "valid synthetic laptop topology rejected");
    require(air::find_hardware_node(valid, "gpu0") != nullptr, "node lookup failed");
    require(air::find_hardware_node(valid, "missing") == nullptr,
            "missing node lookup should be null");

    auto duplicate = laptop_fixture();
    duplicate.nodes.push_back(duplicate.nodes.front());
    require(!air::validate_hardware_topology(duplicate).valid, "duplicate node id accepted");

    auto capacity = laptop_fixture();
    capacity.nodes[1].available_bytes = capacity.nodes[1].total_bytes + 1U;
    require(!air::validate_hardware_topology(capacity).valid,
            "impossible available capacity accepted");

    auto dangling = laptop_fixture();
    dangling.links.push_back(
        {"gpu0", "gpu1", air::HardwareLinkKind::peer_device, true, 1.0e9, 5.0});
    require(!air::validate_hardware_topology(dangling).valid, "dangling link accepted");

    auto invalid_measurement = laptop_fixture();
    invalid_measurement.links[0].bandwidth_bytes_per_second =
        std::numeric_limits<double>::quiet_NaN();
    require(!air::validate_hardware_topology(invalid_measurement).valid,
            "NaN measurement accepted");

    auto unmeasured = laptop_fixture();
    unmeasured.links.push_back(
        {"cpu0", "gpu0", air::HardwareLinkKind::remote, false, 0.0, 0.0});
    require(air::validate_hardware_topology(unmeasured).valid,
            "unmeasured capability edge rejected");

    auto multigpu = laptop_fixture();
    multigpu.nodes.push_back(
        {"gpu1", air::HardwareNodeKind::accelerator, "second gpu", "cuda", "sm90",
         1, -1, 80ULL * 1024ULL * 1024ULL * 1024ULL,
         70ULL * 1024ULL * 1024ULL * 1024ULL, {"graphs", "peer-access"}});
    multigpu.links.push_back(
        {"gpu0", "gpu1", air::HardwareLinkKind::peer_device, true, 50.0e9, 2.0});
    require(air::validate_hardware_topology(multigpu).valid,
            "valid multi-gpu graph rejected");

    std::cout << "hardware topology tests PASS\n";
    return 0;
}

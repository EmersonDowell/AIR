#include "air/cuda.hpp"

#include <iostream>

int main() {
#if AIR_HAS_CUDA
    if (!air::cuda_compiled()) {
        std::cerr << "AIR_HAS_CUDA is true but cuda_compiled() is false\n";
        return 1;
    }
    auto devices = air::cuda_devices();
    if (!devices) {
        // A CUDA build may run on a machine without an NVIDIA device/driver. The build contract is still testable.
        std::cout << "CUDA backend compiled; runtime device query unavailable: " << devices.status().message() << '\n';
        return 0;
    }
    std::cout << "CUDA backend compiled; devices=" << devices.value().size() << '\n';
#else
    if (air::cuda_compiled()) {
        std::cerr << "CPU-only build incorrectly reports compiled CUDA support\n";
        return 1;
    }
    auto devices = air::cuda_devices();
    if (devices || devices.status().code() != air::ErrorCode::unsupported) {
        std::cerr << "CPU-only CUDA device query must fail explicitly as unsupported\n";
        return 1;
    }
    std::cout << "CUDA-disabled contract behaves explicitly\n";
#endif
    return 0;
}

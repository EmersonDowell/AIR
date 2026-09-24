#include "air/cuda.hpp"
#include "model/qwen2_contract.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#if AIR_HAS_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace air {
namespace {

class ScopedProfileRange final {
public:
    explicit ScopedProfileRange(const char* name) noexcept {
#if AIR_HAS_NVTX
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }
    ~ScopedProfileRange() {
#if AIR_HAS_NVTX
        nvtxRangePop();
#endif
    }
    ScopedProfileRange(const ScopedProfileRange&) = delete;
    ScopedProfileRange& operator=(const ScopedProfileRange&) = delete;
};

Status cuda_status(cudaError_t code, const char* operation) {
    if (code == cudaSuccess) return Status::ok();
    return Status::internal_error(std::string(operation) + ": " + cudaGetErrorString(code));
}

Status cublas_status(cublasStatus_t code, const char* operation) {
    if (code == CUBLAS_STATUS_SUCCESS) return Status::ok();
    return Status::internal_error(std::string(operation) + " failed with cuBLAS status " + std::to_string(static_cast<int>(code)));
}

Status select_device(int ordinal) {
    int count = 0;
    auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) return cuda_status(status, "cudaGetDeviceCount");
    if (ordinal < 0 || ordinal >= count) return Status::invalid_argument("CUDA device ordinal is outside available devices");
    return cuda_status(cudaSetDevice(ordinal), "cudaSetDevice");
}

Status activate_device(int ordinal) {
    return cuda_status(cudaSetDevice(ordinal), "cudaSetDevice");
}

std::uint64_t quant_block_elements(DataType type) noexcept {
    switch (type) {
    case DataType::q4_0:
    case DataType::q5_0:
    case DataType::q8_0:
        return 32U;
    case DataType::q4_k:
    case DataType::q6_k:
        return 256U;
    default:
        return 1U;
    }
}

bool supports_cuda_tensor(DataType type) noexcept {
    return type == DataType::f32 || type == DataType::f16 || type == DataType::bf16 ||
           type == DataType::q4_0 || type == DataType::q5_0 || type == DataType::q8_0 ||
           type == DataType::q4_k || type == DataType::q6_k;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) noexcept {
    return (value + alignment - 1U) & ~(alignment - 1U);
}

class DeviceAllocation final {
public:
    DeviceAllocation() = default;
    ~DeviceAllocation() { reset(); }
    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;
    DeviceAllocation(DeviceAllocation&& other) noexcept { *this = std::move(other); }
    DeviceAllocation& operator=(DeviceAllocation&& other) noexcept {
        if (this != &other) {
            reset();
            pointer_ = std::exchange(other.pointer_, nullptr);
            bytes_ = std::exchange(other.bytes_, 0U);
            device_ordinal_ = std::exchange(other.device_ordinal_, -1);
        }
        return *this;
    }

    Status allocate(std::uint64_t bytes) {
        reset();
        if (bytes == 0U) return Status::ok();
        if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            return Status::unsupported("CUDA allocation exceeds host size_t range");
        }
        void* pointer = nullptr;
        auto status = cudaMalloc(&pointer, static_cast<std::size_t>(bytes));
        if (status != cudaSuccess) return cuda_status(status, "cudaMalloc");
        pointer_ = pointer;
        bytes_ = bytes;
        cudaGetDevice(&device_ordinal_);
        return Status::ok();
    }

    void reset() noexcept {
        if (pointer_) {
            int original = 0;
            const bool have_original = cudaGetDevice(&original) == cudaSuccess;
            if (device_ordinal_ >= 0 && (!have_original || original != device_ordinal_)) cudaSetDevice(device_ordinal_);
            cudaFree(pointer_);
            if (have_original && original != device_ordinal_) cudaSetDevice(original);
        }
        pointer_ = nullptr;
        bytes_ = 0U;
        device_ordinal_ = -1;
    }

    [[nodiscard]] void* get() noexcept { return pointer_; }
    [[nodiscard]] std::uint64_t size_bytes() const noexcept { return bytes_; }

private:
    void* pointer_{nullptr};
    std::uint64_t bytes_{0};
    int device_ordinal_{-1};
};

enum class CudaTensorKernel : std::uint8_t {
    f32 = 0,
    f16,
    bf16,
    q4_0,
    q5_0,
    q8_0,
    q4_k,
    q6_k,
};

Result<CudaTensorKernel> cuda_tensor_kernel(DataType type) {
    switch (type) {
    case DataType::f32: return CudaTensorKernel::f32;
    case DataType::f16: return CudaTensorKernel::f16;
    case DataType::bf16: return CudaTensorKernel::bf16;
    case DataType::q4_0: return CudaTensorKernel::q4_0;
    case DataType::q5_0: return CudaTensorKernel::q5_0;
    case DataType::q8_0: return CudaTensorKernel::q8_0;
    case DataType::q4_k: return CudaTensorKernel::q4_k;
    case DataType::q6_k: return CudaTensorKernel::q6_k;
    default: return Status::unsupported("CUDA tensor has no prepared kernel path");
    }
}

struct ResidentTensor {
    const TensorDescriptor* descriptor{nullptr};
    const std::uint8_t* data{nullptr};
    CudaTensorKernel kernel{CudaTensorKernel::f32};
};

struct PackedDp4aTensor {
    const std::int8_t* codes{nullptr};
    const float* scales{nullptr};
    std::uint64_t input_width{0};
    std::uint64_t output_width{0};
};

constexpr std::uint32_t kMaxNativePrefillBatch = 128U;
constexpr std::uint32_t kMaxNativeDecodeBatch = 8U;
constexpr std::uint32_t kArgmaxBlocks = 256U;

struct Workspace {
    DeviceAllocation arena;
    DeviceAllocation token_arena;
    TokenId* token_ids{nullptr};
    float* hidden{nullptr};
    float* normalized{nullptr};
    float* q{nullptr};
    float* k{nullptr};
    float* v{nullptr};
    float* attended{nullptr};
    float* projection{nullptr};
    float* gate{nullptr};
    float* up{nullptr};
    float* logits{nullptr};
    float* scores{nullptr};
    float* argmax_values{nullptr};
    TokenId* argmax_indices{nullptr};
    TokenId* selected_token{nullptr};
    TokenId* nonfinite_flag{nullptr};
    std::uint64_t bytes{0};
};

__device__ float decode_f16_bits(std::uint16_t value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
    const std::uint32_t exponent = (value >> 10U) & 0x1fU;
    const std::uint32_t mantissa = value & 0x03ffU;
    std::uint32_t bits = 0U;
    if (exponent == 0U) {
        if (mantissa == 0U) {
            bits = sign;
        } else {
            std::uint32_t m = mantissa;
            std::uint32_t e = 127U - 15U + 1U;
            while ((m & 0x0400U) == 0U) { m <<= 1U; --e; }
            m &= 0x03ffU;
            bits = sign | (e << 23U) | (m << 13U);
        }
    } else if (exponent == 0x1fU) {
        bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + (127U - 15U)) << 23U) | (mantissa << 13U);
    }
    return __uint_as_float(bits);
}

__device__ std::uint16_t load_u16(const std::uint8_t* data) {
    return static_cast<std::uint16_t>(data[0]) | (static_cast<std::uint16_t>(data[1]) << 8U);
}

__device__ __forceinline__ float decode_element(const std::uint8_t* data, int type, std::uint64_t index) {
    switch (type) {
    case static_cast<int>(DataType::f32):
        return reinterpret_cast<const float*>(data)[index];
    case static_cast<int>(DataType::f16):
        return __half2float(reinterpret_cast<const __half*>(data)[index]);
    case static_cast<int>(DataType::bf16):
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(data)[index]);
    case static_cast<int>(DataType::q4_0): {
        const std::uint64_t block = index / 32U;
        const std::uint32_t local = static_cast<std::uint32_t>(index % 32U);
        const auto* source = data + block * 18U;
        const float scale = decode_f16_bits(load_u16(source));
        const std::uint8_t packed = source[2U + (local % 16U)];
        const int q = local < 16U ? static_cast<int>(packed & 0x0fU) - 8 : static_cast<int>(packed >> 4U) - 8;
        return scale * static_cast<float>(q);
    }
    case static_cast<int>(DataType::q5_0): {
        const std::uint64_t block = index / 32U;
        const std::uint32_t local = static_cast<std::uint32_t>(index % 32U);
        const auto* source = data + block * 22U;
        const float scale = decode_f16_bits(load_u16(source));
        const std::uint8_t high = static_cast<std::uint8_t>(
            ((source[2U + local / 8U] >> (local % 8U)) & 0x01U) << 4U);
        const std::uint8_t packed = source[6U + (local % 16U)];
        const std::uint8_t low = local < 16U ? static_cast<std::uint8_t>(packed & 0x0fU)
                                             : static_cast<std::uint8_t>(packed >> 4U);
        const int q = static_cast<int>(low | high) - 16;
        return scale * static_cast<float>(q);
    }
    case static_cast<int>(DataType::q8_0): {
        const std::uint64_t block = index / 32U;
        const std::uint32_t local = static_cast<std::uint32_t>(index % 32U);
        const auto* source = data + block * 34U;
        const float scale = decode_f16_bits(load_u16(source));
        const int raw = static_cast<int>(source[2U + local]);
        const int quantized = raw < 128 ? raw : raw - 256;
        return scale * static_cast<float>(quantized);
    }
    case static_cast<int>(DataType::q4_k): {
        const std::uint64_t block = index / 256U;
        const std::uint32_t local = static_cast<std::uint32_t>(index % 256U);
        const auto* source = data + block * 144U;
        const float block_scale = decode_f16_bits(load_u16(source));
        const float block_minimum = decode_f16_bits(load_u16(source + 2U));
        const auto* scales = source + 4U;
        const auto* quants = source + 16U;
        const std::uint32_t group = local / 64U;
        const std::uint32_t lane = local % 64U;
        const std::uint32_t scale_index = group * 2U + (lane >= 32U ? 1U : 0U);
        std::uint8_t scale = 0U;
        std::uint8_t minimum = 0U;
        if (scale_index < 4U) {
            scale = scales[scale_index] & 63U;
            minimum = scales[scale_index + 4U] & 63U;
        } else {
            scale = static_cast<std::uint8_t>((scales[scale_index + 4U] & 0x0fU) |
                                              ((scales[scale_index - 4U] >> 6U) << 4U));
            minimum = static_cast<std::uint8_t>((scales[scale_index + 4U] >> 4U) |
                                                ((scales[scale_index] >> 6U) << 4U));
        }
        const std::uint8_t packed = quants[group * 32U + (lane % 32U)];
        const std::uint8_t q = lane < 32U ? static_cast<std::uint8_t>(packed & 0x0fU)
                                          : static_cast<std::uint8_t>(packed >> 4U);
        return block_scale * static_cast<float>(scale) * static_cast<float>(q) -
               block_minimum * static_cast<float>(minimum);
    }
    case static_cast<int>(DataType::q6_k): {
        const std::uint64_t block = index / 256U;
        const std::uint32_t local = static_cast<std::uint32_t>(index % 256U);
        const auto* source = data + block * 210U;
        const auto* ql = source;
        const auto* qh = source + 128U;
        const auto* scales = source + 192U;
        const float block_scale = decode_f16_bits(load_u16(source + 208U));
        const std::uint32_t half = local / 128U;
        const std::uint32_t position = local % 128U;
        const auto* low = ql + half * 64U;
        const auto* high = qh + half * 32U;
        const auto* scale = scales + half * 8U;
        std::uint32_t i = position % 32U;
        const std::uint32_t section = position / 32U;
        const std::uint8_t hi = high[i];
        std::uint8_t low_bits = 0U;
        std::uint8_t high_bits = 0U;
        std::uint32_t scale_index = 0U;
        if (section == 0U) {
            low_bits = low[i] & 0x0fU; high_bits = hi & 0x03U; scale_index = i / 16U;
        } else if (section == 1U) {
            low_bits = low[i + 32U] & 0x0fU; high_bits = (hi >> 2U) & 0x03U; scale_index = i / 16U + 2U;
        } else if (section == 2U) {
            low_bits = low[i] >> 4U; high_bits = (hi >> 4U) & 0x03U; scale_index = i / 16U + 4U;
        } else {
            low_bits = low[i + 32U] >> 4U; high_bits = (hi >> 6U) & 0x03U; scale_index = i / 16U + 6U;
        }
        const int q = static_cast<int>(low_bits | static_cast<std::uint8_t>(high_bits << 4U)) - 32;
        const int raw_scale = static_cast<int>(scale[scale_index]);
        const int signed_scale = raw_scale < 128 ? raw_scale : raw_scale - 256;
        return block_scale * static_cast<float>(signed_scale) * static_cast<float>(q);
    }
    default:
        return 0.0F;
    }
}

__global__ void load_row_kernel(const std::uint8_t* tensor, int type, std::uint64_t row,
                                std::uint64_t width, float* output) {
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < width; i += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        output[i] = decode_element(tensor, type, row * width + i);
    }
}


__global__ void load_rows_kernel(const std::uint8_t* tensor, int type, const TokenId* tokens,
                                 std::uint32_t batch, std::uint64_t width, float* output) {
    const std::uint64_t total = static_cast<std::uint64_t>(batch) * width;
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        const std::uint32_t row = static_cast<std::uint32_t>(i / width);
        const std::uint64_t col = i % width;
        const auto token = tokens[row];
        output[i] = decode_element(tensor, type, static_cast<std::uint64_t>(token) * width + col);
    }
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}


__global__ void dequantize_to_f32_kernel(const std::uint8_t* matrix, int type,
                                         std::uint64_t count, float* output) {
    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        output[index] = decode_element(matrix, type, index);
    }
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
    }
    return value;
}

__device__ __forceinline__ int warp_reduce_int_sum(int value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

template <DataType Type>
__global__ void pack_signed32_weights_kernel(const std::uint8_t* source,
                                             std::uint64_t block_count,
                                             std::int8_t* codes,
                                             float* scales) {
    for (std::uint64_t block = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         block < block_count;
         block += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        if constexpr (Type == DataType::q5_0) {
            const auto* raw = source + block * 22U;
            scales[block] = decode_f16_bits(load_u16(raw));
            const auto* high = raw + 2U;
            const auto* quants = raw + 6U;
            auto* out = codes + block * 32U;
            for (std::uint32_t i = 0U; i < 16U; ++i) {
                const std::uint8_t packed = quants[i];
                const std::uint8_t hi0 = static_cast<std::uint8_t>(((high[i / 8U] >> (i % 8U)) & 1U) << 4U);
                const std::uint32_t j = i + 16U;
                const std::uint8_t hi1 = static_cast<std::uint8_t>(((high[j / 8U] >> (j % 8U)) & 1U) << 4U);
                out[i] = static_cast<std::int8_t>(static_cast<int>((packed & 0x0fU) | hi0) - 16);
                out[j] = static_cast<std::int8_t>(static_cast<int>((packed >> 4U) | hi1) - 16);
            }
        } else if constexpr (Type == DataType::q8_0) {
            const auto* raw = source + block * 34U;
            scales[block] = decode_f16_bits(load_u16(raw));
            auto* out = codes + block * 32U;
            for (std::uint32_t i = 0U; i < 32U; ++i) out[i] = static_cast<std::int8_t>(raw[2U + i]);
        }
    }
}

__global__ void quantize_q8_activation_kernel(const float* input,
                                              std::uint64_t input_width,
                                              std::uint32_t batch,
                                              std::int8_t* codes,
                                              float* scales) {
    const unsigned int lane = threadIdx.x & 31U;
    const std::uint64_t blocks_per_item = input_width / 32U;
    const std::uint64_t logical = static_cast<std::uint64_t>(blockIdx.x);
    const std::uint32_t item = static_cast<std::uint32_t>(logical / blocks_per_item);
    const std::uint64_t block = logical % blocks_per_item;
    if (item >= batch) return;
    const std::uint64_t index = static_cast<std::uint64_t>(item) * input_width + block * 32U + lane;
    const float value = input[index];
    float max_abs = warp_reduce_max(fabsf(value));
    max_abs = __shfl_sync(0xffffffffU, max_abs, 0);
    const float scale = max_abs == 0.0F ? 0.0F : max_abs / 127.0F;
    int q = scale == 0.0F ? 0 : __float2int_rn(value / scale);
    q = max(-127, min(127, q));
    codes[index] = static_cast<std::int8_t>(q);
    if (lane == 0U) scales[logical] = scale;
}

template <unsigned int ItemsPerWarp>
__global__ void q5q8_dp4a_hybrid_matmul_kernel(const std::int8_t* weight_codes,
                                             const float* weight_scales,
                                             std::uint64_t input_width,
                                             std::uint64_t output_width,
                                             const std::int8_t* activation_codes,
                                             const float* activation_scales,
                                             float* output,
                                             std::uint32_t batch) {
    static_assert(ItemsPerWarp == 8U);
    constexpr unsigned int warps_per_block = 4U;
    const unsigned int lane = threadIdx.x & 31U;
    const unsigned int warp = threadIdx.x / 32U;
    if (warp >= warps_per_block) return;
    const std::uint64_t row = static_cast<std::uint64_t>(blockIdx.x) * warps_per_block + warp;
    if (row >= output_width) return;
    const std::uint32_t first_item = static_cast<std::uint32_t>(blockIdx.y) * ItemsPerWarp;
    if (first_item >= batch) return;
    const std::uint64_t blocks_per_row = input_width / 32U;
    float sums[ItemsPerWarp] = {};
    for (std::uint64_t block = lane; block < blocks_per_row; block += 32U) {
        const auto* w = weight_codes + row * input_width + block * 32U;
        const float ws = weight_scales[row * blocks_per_row + block];
        const auto* wi = reinterpret_cast<const int*>(w);
#pragma unroll
        for (unsigned int item_offset = 0U; item_offset < ItemsPerWarp; ++item_offset) {
            const std::uint32_t item = first_item + item_offset;
            if (item >= batch) continue;
            const auto* a = activation_codes + static_cast<std::uint64_t>(item) * input_width + block * 32U;
            const auto* ai = reinterpret_cast<const int*>(a);
            int idot = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) idot = __dp4a(wi[k], ai[k], idot);
            const float as = activation_scales[static_cast<std::uint64_t>(item) * blocks_per_row + block];
            sums[item_offset] += static_cast<float>(idot) * ws * as;
        }
    }
#pragma unroll
    for (unsigned int item_offset = 0U; item_offset < ItemsPerWarp; ++item_offset) {
        sums[item_offset] = warp_reduce_sum(sums[item_offset]);
    }
    if (lane == 0U) {
#pragma unroll
        for (unsigned int item_offset = 0U; item_offset < ItemsPerWarp; ++item_offset) {
            const std::uint32_t item = first_item + item_offset;
            if (item < batch) output[static_cast<std::uint64_t>(item) * output_width + row] = sums[item_offset];
        }
    }
}

template <DataType Type>
__global__ void specialized_matmul_kernel(const std::uint8_t* matrix,
                                          std::uint64_t input_width,
                                          std::uint64_t output_width,
                                          const float* input,
                                          float* output,
                                          std::uint32_t batch) {
    constexpr unsigned int warp_size = 32U;
    constexpr unsigned int warps_per_block = 4U;
    const unsigned int lane = threadIdx.x & (warp_size - 1U);
    const unsigned int warp = threadIdx.x / warp_size;
    if (warp >= warps_per_block) return;
    const std::uint64_t row = static_cast<std::uint64_t>(blockIdx.x) * warps_per_block + warp;
    const std::uint32_t item = blockIdx.y;
    if (row >= output_width || item >= batch) return;

    const std::uint64_t base = row * input_width;
    const float* item_input = input + static_cast<std::uint64_t>(item) * input_width;
    float sum = 0.0F;
    for (std::uint64_t col = lane; col < input_width; col += warp_size) {
        sum += decode_element(matrix, static_cast<int>(Type), base + col) * item_input[col];
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0U) {
        output[static_cast<std::uint64_t>(item) * output_width + row] = sum;
    }
}

template <DataType Type, unsigned int ItemsPerWarp>
__global__ void specialized_matmul_batch_reuse_kernel(const std::uint8_t* matrix,
                                                       std::uint64_t input_width,
                                                       std::uint64_t output_width,
                                                       const float* input,
                                                       float* output,
                                                       std::uint32_t batch) {
    static_assert(ItemsPerWarp == 4U || ItemsPerWarp == 8U);
    constexpr unsigned int warp_size = 32U;
    constexpr unsigned int warps_per_block = 4U;
    const unsigned int lane = threadIdx.x & (warp_size - 1U);
    const unsigned int warp = threadIdx.x / warp_size;
    if (warp >= warps_per_block) return;

    const std::uint64_t row = static_cast<std::uint64_t>(blockIdx.x) * warps_per_block + warp;
    if (row >= output_width) return;

    const std::uint32_t first_item = static_cast<std::uint32_t>(blockIdx.y) * ItemsPerWarp;
    if (first_item >= batch) return;

    float sums[ItemsPerWarp] = {};
    const std::uint64_t base = row * input_width;

    // The baseline native-prefill kernel decodes the same quantized weight once
    // per prompt item. This tactic keeps the canonical GGUF bytes resident and
    // reuses each decoded weight across a small prompt-item tile. Per-item
    // accumulation and warp-reduction order are unchanged.
    for (std::uint64_t col = lane; col < input_width; col += warp_size) {
        const float weight = decode_element(matrix, static_cast<int>(Type), base + col);
#pragma unroll
        for (unsigned int item_offset = 0U; item_offset < ItemsPerWarp; ++item_offset) {
            const std::uint32_t item = first_item + item_offset;
            if (item < batch) {
                sums[item_offset] +=
                    weight * input[static_cast<std::uint64_t>(item) * input_width + col];
            }
        }
    }

#pragma unroll
    for (unsigned int item_offset = 0U; item_offset < ItemsPerWarp; ++item_offset) {
        sums[item_offset] = warp_reduce_sum(sums[item_offset]);
    }

    if (lane == 0U) {
#pragma unroll
        for (unsigned int item_offset = 0U; item_offset < ItemsPerWarp; ++item_offset) {
            const std::uint32_t item = first_item + item_offset;
            if (item < batch) {
                output[static_cast<std::uint64_t>(item) * output_width + row] =
                    sums[item_offset];
            }
        }
    }
}

template <DataType Type>
__global__ void specialized_matvec_kernel(const std::uint8_t* matrix,
                                          std::uint64_t input_width,
                                          std::uint64_t output_width,
                                          const float* input,
                                          float* output) {
    constexpr unsigned int warp_size = 32U;
    constexpr unsigned int warps_per_block = 4U;
    const unsigned int lane = threadIdx.x & (warp_size - 1U);
    const unsigned int warp = threadIdx.x / warp_size;
    if (warp >= warps_per_block) return;
    const std::uint64_t row = static_cast<std::uint64_t>(blockIdx.x) * warps_per_block + warp;
    if (row >= output_width) return;

    const std::uint64_t base = row * input_width;
    float sum = 0.0F;
    for (std::uint64_t col = lane; col < input_width; col += warp_size) {
        sum += decode_element(matrix, static_cast<int>(Type), base + col) * input[col];
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0U) output[row] = sum;
}

__device__ __forceinline__ bool better_argmax(float lhs_value, TokenId lhs_index,
                                              float rhs_value, TokenId rhs_index) {
    return lhs_value > rhs_value || (lhs_value == rhs_value && lhs_index < rhs_index);
}

__global__ void argmax_stage1_kernel(const float* logits, std::uint64_t count,
                                     float* values, TokenId* indices, TokenId* nonfinite_flag) {
    float best_value = -CUDART_INF_F;
    TokenId best_index = 0;
    bool have_value = false;
    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        const float value = logits[index];
        if (!isfinite(value)) {
            atomicExch(reinterpret_cast<int*>(nonfinite_flag), 1);
            continue;
        }
        const auto token = static_cast<TokenId>(index);
        if (!have_value || better_argmax(value, token, best_value, best_index)) {
            best_value = value;
            best_index = token;
            have_value = true;
        }
    }

    constexpr unsigned int warp_size = 32U;
    __shared__ float warp_values[32];
    __shared__ TokenId warp_indices[32];
    const unsigned int lane = threadIdx.x & (warp_size - 1U);
    const unsigned int warp = threadIdx.x / warp_size;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_value = __shfl_down_sync(0xffffffffU, best_value, offset);
        const TokenId other_index = __shfl_down_sync(0xffffffffU, best_index, offset);
        if (better_argmax(other_value, other_index, best_value, best_index)) {
            best_value = other_value;
            best_index = other_index;
        }
    }
    if (lane == 0U) {
        warp_values[warp] = best_value;
        warp_indices[warp] = best_index;
    }
    __syncthreads();

    if (warp == 0U) {
        const unsigned int warp_count = (blockDim.x + warp_size - 1U) / warp_size;
        best_value = lane < warp_count ? warp_values[lane] : -CUDART_INF_F;
        best_index = lane < warp_count ? warp_indices[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_value = __shfl_down_sync(0xffffffffU, best_value, offset);
            const TokenId other_index = __shfl_down_sync(0xffffffffU, best_index, offset);
            if (better_argmax(other_value, other_index, best_value, best_index)) {
                best_value = other_value;
                best_index = other_index;
            }
        }
        if (lane == 0U) {
            values[blockIdx.x] = best_value;
            indices[blockIdx.x] = best_index;
        }
    }
}

__global__ void argmax_stage2_kernel(const float* values, const TokenId* indices,
                                     std::uint32_t count, TokenId* selected_token) {
    float best_value = -CUDART_INF_F;
    TokenId best_index = 0;
    bool have_value = false;
    for (std::uint32_t index = threadIdx.x; index < count; index += blockDim.x) {
        const float value = values[index];
        const TokenId token = indices[index];
        if (!have_value || better_argmax(value, token, best_value, best_index)) {
            best_value = value;
            best_index = token;
            have_value = true;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_value = __shfl_down_sync(0xffffffffU, best_value, offset);
        const TokenId other_index = __shfl_down_sync(0xffffffffU, best_index, offset);
        if (better_argmax(other_value, other_index, best_value, best_index)) {
            best_value = other_value;
            best_index = other_index;
        }
    }
    constexpr unsigned int warp_size = 32U;
    __shared__ float warp_values[32];
    __shared__ TokenId warp_indices[32];
    const unsigned int lane = threadIdx.x & (warp_size - 1U);
    const unsigned int warp = threadIdx.x / warp_size;
    if (lane == 0U) {
        warp_values[warp] = best_value;
        warp_indices[warp] = best_index;
    }
    __syncthreads();
    if (warp == 0U) {
        const unsigned int warp_count = (blockDim.x + warp_size - 1U) / warp_size;
        best_value = lane < warp_count ? warp_values[lane] : -CUDART_INF_F;
        best_index = lane < warp_count ? warp_indices[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_value = __shfl_down_sync(0xffffffffU, best_value, offset);
            const TokenId other_index = __shfl_down_sync(0xffffffffU, best_index, offset);
            if (better_argmax(other_value, other_index, best_value, best_index)) {
                best_value = other_value;
                best_index = other_index;
            }
        }
        if (lane == 0U) selected_token[0] = best_index;
    }
}

__global__ void target_logprobs_kernel(const float* logits, std::uint64_t count,
                                       const TokenId* target_tokens,
                                       std::uint32_t target_count,
                                       float* output,
                                       TokenId* nonfinite_flag) {
    constexpr unsigned int warp_size = 32U;
    __shared__ float warp_values[32];
    __shared__ float block_max;
    __shared__ float block_log_z;

    float local_max = -CUDART_INF_F;
    for (std::uint64_t index = threadIdx.x; index < count; index += blockDim.x) {
        const float value = logits[index];
        if (!isfinite(value)) {
            atomicExch(reinterpret_cast<int*>(nonfinite_flag), 1);
            continue;
        }
        local_max = fmaxf(local_max, value);
    }
    float reduced_max = warp_reduce_max(local_max);
    const unsigned int lane = threadIdx.x & (warp_size - 1U);
    const unsigned int warp = threadIdx.x / warp_size;
    if (lane == 0U) warp_values[warp] = reduced_max;
    __syncthreads();

    if (warp == 0U) {
        const unsigned int warp_count = (blockDim.x + warp_size - 1U) / warp_size;
        reduced_max = lane < warp_count ? warp_values[lane] : -CUDART_INF_F;
        reduced_max = warp_reduce_max(reduced_max);
        if (lane == 0U) block_max = reduced_max;
    }
    __syncthreads();

    float local_sum = 0.0F;
    for (std::uint64_t index = threadIdx.x; index < count; index += blockDim.x) {
        const float value = logits[index];
        if (isfinite(value)) local_sum += expf(value - block_max);
    }
    float reduced_sum = warp_reduce_sum(local_sum);
    if (lane == 0U) warp_values[warp] = reduced_sum;
    __syncthreads();

    if (warp == 0U) {
        const unsigned int warp_count = (blockDim.x + warp_size - 1U) / warp_size;
        reduced_sum = lane < warp_count ? warp_values[lane] : 0.0F;
        reduced_sum = warp_reduce_sum(reduced_sum);
        if (lane == 0U) {
            if (!isfinite(reduced_sum) || !(reduced_sum > 0.0F)) {
                atomicExch(reinterpret_cast<int*>(nonfinite_flag), 1);
                block_log_z = CUDART_NAN_F;
            } else {
                block_log_z = block_max + logf(reduced_sum);
            }
        }
    }
    __syncthreads();

    for (std::uint32_t i = threadIdx.x; i < target_count; i += blockDim.x) {
        const TokenId token = target_tokens[i];
        const float value = logits[static_cast<std::uint64_t>(token)];
        if (!isfinite(value) || !isfinite(block_log_z)) {
            atomicExch(reinterpret_cast<int*>(nonfinite_flag), 1);
            output[i] = CUDART_NAN_F;
        } else {
            output[i] = value - block_log_z;
        }
    }
}

__global__ void rms_norm_kernel(const float* input, const std::uint8_t* weight, int weight_type,
                                std::uint64_t width, float epsilon, float* output) {
    float sum = 0.0F;
    for (std::uint64_t i = threadIdx.x; i < width; i += blockDim.x) sum += input[i] * input[i];
    extern __shared__ float scratch[];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0U) scratch[0] = rsqrtf(scratch[0] / static_cast<float>(width) + epsilon);
    __syncthreads();
    const float scale = scratch[0];
    for (std::uint64_t i = threadIdx.x; i < width; i += blockDim.x) {
        output[i] = input[i] * scale * decode_element(weight, weight_type, i);
    }
}


__global__ void rms_norm_batch_kernel(const float* input, const std::uint8_t* weight, int weight_type,
                                      std::uint64_t width, float epsilon, float* output,
                                      std::uint32_t batch) {
    const std::uint32_t item = blockIdx.x;
    if (item >= batch) return;
    const float* item_input = input + static_cast<std::uint64_t>(item) * width;
    float* item_output = output + static_cast<std::uint64_t>(item) * width;
    float sum = 0.0F;
    for (std::uint64_t i = threadIdx.x; i < width; i += blockDim.x) sum += item_input[i] * item_input[i];
    extern __shared__ float scratch[];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0U) scratch[0] = rsqrtf(scratch[0] / static_cast<float>(width) + epsilon);
    __syncthreads();
    const float scale = scratch[0];
    for (std::uint64_t i = threadIdx.x; i < width; i += blockDim.x) {
        item_output[i] = item_input[i] * scale * decode_element(weight, weight_type, i);
    }
}

__global__ void add_tensor_batch_kernel(float* target, const std::uint8_t* value, int value_type,
                                        std::uint64_t width, std::uint32_t batch) {
    const std::uint64_t total = static_cast<std::uint64_t>(batch) * width;
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        target[i] += decode_element(value, value_type, i % width);
    }
}

__global__ void add_tensor_kernel(float* target, const std::uint8_t* value, int value_type, std::uint64_t width) {
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < width; i += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        target[i] += decode_element(value, value_type, i);
    }
}

__global__ void add_vector_kernel(float* target, const float* value, std::uint64_t width) {
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < width; i += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) target[i] += value[i];
}

__global__ void silu_mul_kernel(float* gate, const float* up, std::uint64_t width) {
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < width; i += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        const float x = gate[i];
        gate[i] = (x / (1.0F + expf(-x))) * up[i];
    }
}

__global__ void rope_kernel(float* values, std::uint32_t heads, std::uint32_t head_dimension,
                            std::uint32_t rope_dimensions, std::uint64_t position, float rope_base) {
    const std::uint32_t half = rope_dimensions / 2U;
    const std::uint64_t pairs = static_cast<std::uint64_t>(heads) * half;
    for (std::uint64_t pair = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < pairs; pair += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        const std::uint32_t head = static_cast<std::uint32_t>(pair / half);
        const std::uint32_t i = static_cast<std::uint32_t>(pair % half);
        const float exponent = (2.0F * static_cast<float>(i)) / static_cast<float>(rope_dimensions);
        const float angle = static_cast<float>(position) / powf(rope_base, exponent);
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        const std::uint64_t base = static_cast<std::uint64_t>(head) * head_dimension;
        const float first = values[base + i];
        const float second = values[base + i + half];
        values[base + i] = first * cosine - second * sine;
        values[base + i + half] = first * sine + second * cosine;
    }
}


__global__ void rope_batch_kernel(float* values, std::uint32_t batch, std::uint32_t heads,
                                  std::uint32_t head_dimension, std::uint32_t rope_dimensions,
                                  std::uint64_t position_start, float rope_base) {
    const std::uint32_t half = rope_dimensions / 2U;
    const std::uint64_t pairs_per_item = static_cast<std::uint64_t>(heads) * half;
    const std::uint64_t total_pairs = static_cast<std::uint64_t>(batch) * pairs_per_item;
    for (std::uint64_t pair = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < total_pairs; pair += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        const std::uint32_t item = static_cast<std::uint32_t>(pair / pairs_per_item);
        const std::uint64_t local_pair = pair % pairs_per_item;
        const std::uint32_t head = static_cast<std::uint32_t>(local_pair / half);
        const std::uint32_t i = static_cast<std::uint32_t>(local_pair % half);
        const float exponent = (2.0F * static_cast<float>(i)) / static_cast<float>(rope_dimensions);
        const float angle = static_cast<float>(position_start + item) / powf(rope_base, exponent);
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        const std::uint64_t base = static_cast<std::uint64_t>(item) * heads * head_dimension +
                                   static_cast<std::uint64_t>(head) * head_dimension;
        const float first = values[base + i];
        const float second = values[base + i + half];
        values[base + i] = first * cosine - second * sine;
        values[base + i + half] = first * sine + second * cosine;
    }
}


__device__ const float* paged_kv_ptr(float* const* pages, std::uint64_t token,
                                    std::uint32_t page_tokens, std::uint32_t layer,
                                    std::uint64_t kv_width, std::uint64_t kv_base) {
    const std::uint64_t page = token / page_tokens;
    const std::uint64_t within = token % page_tokens;
    return pages[page] + (static_cast<std::uint64_t>(layer) * page_tokens + within) * kv_width + kv_base;
}

__global__ void attention_scores_paged_kernel(const float* q, const float* current_k,
                                              float* const* key_pages, std::uint32_t page_tokens,
                                              std::uint64_t previous_tokens, std::uint32_t layer,
                                              std::uint32_t q_heads, std::uint32_t kv_heads,
                                              std::uint32_t head_dimension, float* scores) {
    const std::uint64_t total_tokens = previous_tokens + 1U;
    const std::uint64_t item = blockIdx.x;
    const std::uint32_t q_head = static_cast<std::uint32_t>(item / total_tokens);
    const std::uint64_t token = item % total_tokens;
    if (q_head >= q_heads) return;
    const std::uint32_t kv_head = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(q_head) * kv_heads) / q_heads);
    const std::uint64_t q_base = static_cast<std::uint64_t>(q_head) * head_dimension;
    const std::uint64_t kv_base = static_cast<std::uint64_t>(kv_head) * head_dimension;
    const std::uint64_t kv_width = static_cast<std::uint64_t>(kv_heads) * head_dimension;
    const float* key = token == previous_tokens
        ? current_k + kv_base
        : paged_kv_ptr(key_pages, token, page_tokens, layer, kv_width, kv_base);
    float sum = 0.0F;
    for (std::uint32_t d = threadIdx.x; d < head_dimension; d += blockDim.x) sum += q[q_base + d] * key[d];
    extern __shared__ float scratch[];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0U) scores[static_cast<std::uint64_t>(q_head) * total_tokens + token] =
        scratch[0] * rsqrtf(static_cast<float>(head_dimension));
}

__global__ void attention_value_paged_kernel(const float* scores, const float* current_v,
                                             float* const* value_pages, std::uint32_t page_tokens,
                                             std::uint64_t previous_tokens, std::uint32_t layer,
                                             std::uint32_t q_heads, std::uint32_t kv_heads,
                                             std::uint32_t head_dimension, float* output) {
    const std::uint32_t q_head = blockIdx.x;
    if (q_head >= q_heads) return;
    const std::uint64_t total_tokens = previous_tokens + 1U;
    const std::uint32_t kv_head = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(q_head) * kv_heads) / q_heads);
    const std::uint64_t kv_width = static_cast<std::uint64_t>(kv_heads) * head_dimension;
    const std::uint64_t kv_base = static_cast<std::uint64_t>(kv_head) * head_dimension;
    const float* head_scores = scores + static_cast<std::uint64_t>(q_head) * total_tokens;
    extern __shared__ float scratch[];

    float local_max = -CUDART_INF_F;
    for (std::uint64_t token = threadIdx.x; token < total_tokens; token += blockDim.x) {
        local_max = fmaxf(local_max, head_scores[token]);
    }
    scratch[threadIdx.x] = local_max;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        __syncthreads();
    }
    const float maximum = scratch[0];

    float local_sum = 0.0F;
    for (std::uint64_t token = threadIdx.x; token < total_tokens; token += blockDim.x) {
        local_sum += expf(head_scores[token] - maximum);
    }
    scratch[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float denominator = scratch[0];

    for (std::uint32_t d = threadIdx.x; d < head_dimension; d += blockDim.x) {
        float sum = 0.0F;
        for (std::uint64_t token = 0; token < total_tokens; ++token) {
            const float probability = expf(head_scores[token] - maximum) / denominator;
            const float* value = token == previous_tokens
                ? current_v + kv_base
                : paged_kv_ptr(value_pages, token, page_tokens, layer, kv_width, kv_base);
            sum += probability * value[d];
        }
        output[static_cast<std::uint64_t>(q_head) * head_dimension + d] = sum;
    }
}

__device__ const float* batch_kv_ptr(float* const* pages, const float* current,
                                    std::uint64_t token, std::uint64_t previous_tokens,
                                    std::uint32_t page_tokens, std::uint32_t layer,
                                    std::uint64_t kv_width, std::uint64_t kv_base) {
    if (token < previous_tokens) {
        return paged_kv_ptr(pages, token, page_tokens, layer, kv_width, kv_base);
    }
    const std::uint64_t current_index = token - previous_tokens;
    return current + current_index * kv_width + kv_base;
}

// Native multi-token causal prefill. Each block owns one (query token, query
// head) pair. It computes softmax statistics without materializing a
// batch*context score tensor, then accumulates the attended value in-place.
__global__ void attention_batch_paged_kernel(const float* q, const float* current_k,
                                             const float* current_v, float* const* key_pages,
                                             float* const* value_pages, std::uint32_t page_tokens,
                                             std::uint64_t previous_tokens, std::uint32_t batch,
                                             std::uint32_t layer, std::uint32_t q_heads,
                                             std::uint32_t kv_heads, std::uint32_t head_dimension,
                                             float* output) {
    const std::uint64_t block = blockIdx.x;
    const std::uint32_t item = static_cast<std::uint32_t>(block / q_heads);
    const std::uint32_t q_head = static_cast<std::uint32_t>(block % q_heads);
    if (item >= batch) return;
    const std::uint32_t kv_head = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(q_head) * kv_heads) / q_heads);
    const std::uint64_t q_width = static_cast<std::uint64_t>(q_heads) * head_dimension;
    const std::uint64_t kv_width = static_cast<std::uint64_t>(kv_heads) * head_dimension;
    const std::uint64_t q_base = static_cast<std::uint64_t>(item) * q_width +
                                 static_cast<std::uint64_t>(q_head) * head_dimension;
    const std::uint64_t kv_base = static_cast<std::uint64_t>(kv_head) * head_dimension;
    const std::uint64_t total_tokens = previous_tokens + static_cast<std::uint64_t>(item) + 1U;
    extern __shared__ float scratch[];

    float local_max = -CUDART_INF_F;
    for (std::uint64_t token = threadIdx.x; token < total_tokens; token += blockDim.x) {
        const float* key = batch_kv_ptr(key_pages, current_k, token, previous_tokens, page_tokens, layer, kv_width, kv_base);
        float dot = 0.0F;
        for (std::uint32_t d = 0; d < head_dimension; ++d) dot += q[q_base + d] * key[d];
        local_max = fmaxf(local_max, dot * rsqrtf(static_cast<float>(head_dimension)));
    }
    scratch[threadIdx.x] = local_max;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        __syncthreads();
    }
    const float maximum = scratch[0];

    float local_sum = 0.0F;
    for (std::uint64_t token = threadIdx.x; token < total_tokens; token += blockDim.x) {
        const float* key = batch_kv_ptr(key_pages, current_k, token, previous_tokens, page_tokens, layer, kv_width, kv_base);
        float dot = 0.0F;
        for (std::uint32_t d = 0; d < head_dimension; ++d) dot += q[q_base + d] * key[d];
        local_sum += expf(dot * rsqrtf(static_cast<float>(head_dimension)) - maximum);
    }
    scratch[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float denominator = scratch[0];

    float* item_output = output + static_cast<std::uint64_t>(item) * q_width +
                         static_cast<std::uint64_t>(q_head) * head_dimension;
    for (std::uint32_t d = threadIdx.x; d < head_dimension; d += blockDim.x) item_output[d] = 0.0F;
    __syncthreads();

    // The dot product is reduced cooperatively once per key token, then all
    // threads update their disjoint output dimensions with the shared weight.
    for (std::uint64_t token = 0; token < total_tokens; ++token) {
        const float* key = batch_kv_ptr(key_pages, current_k, token, previous_tokens, page_tokens, layer, kv_width, kv_base);
        float partial = 0.0F;
        for (std::uint32_t d = threadIdx.x; d < head_dimension; d += blockDim.x) {
            partial += q[q_base + d] * key[d];
        }
        scratch[threadIdx.x] = partial;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
            if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            __syncthreads();
        }
        const float probability = expf(scratch[0] * rsqrtf(static_cast<float>(head_dimension)) - maximum) / denominator;
        const float* value = batch_kv_ptr(value_pages, current_v, token, previous_tokens, page_tokens, layer, kv_width, kv_base);
        for (std::uint32_t d = threadIdx.x; d < head_dimension; d += blockDim.x) {
            item_output[d] += probability * value[d];
        }
        __syncthreads();
    }
}

// One-pass causal paged prefill attention using online softmax. Each block
// owns one (query token, query head). QK is computed exactly once per key token.
// The output numerator is rescaled only when the running maximum changes.
__global__ void attention_batch_paged_online_kernel(const float* q, const float* current_k,
                                                    const float* current_v, float* const* key_pages,
                                                    float* const* value_pages, std::uint32_t page_tokens,
                                                    std::uint64_t previous_tokens, std::uint32_t batch,
                                                    std::uint32_t layer, std::uint32_t q_heads,
                                                    std::uint32_t kv_heads, std::uint32_t head_dimension,
                                                    float* output) {
    const std::uint64_t block = blockIdx.x;
    const std::uint32_t item = static_cast<std::uint32_t>(block / q_heads);
    const std::uint32_t q_head = static_cast<std::uint32_t>(block % q_heads);
    if (item >= batch) return;
    const std::uint32_t kv_head = static_cast<std::uint32_t>((static_cast<std::uint64_t>(q_head) * kv_heads) / q_heads);
    const std::uint64_t q_width = static_cast<std::uint64_t>(q_heads) * head_dimension;
    const std::uint64_t kv_width = static_cast<std::uint64_t>(kv_heads) * head_dimension;
    const std::uint64_t q_base = static_cast<std::uint64_t>(item) * q_width + static_cast<std::uint64_t>(q_head) * head_dimension;
    const std::uint64_t kv_base = static_cast<std::uint64_t>(kv_head) * head_dimension;
    const std::uint64_t total_tokens = previous_tokens + static_cast<std::uint64_t>(item) + 1U;
    float* item_output = output + static_cast<std::uint64_t>(item) * q_width + static_cast<std::uint64_t>(q_head) * head_dimension;
    extern __shared__ float scratch[];
    __shared__ float state[3]; // alpha, beta, running denominator

    for (std::uint32_t d = threadIdx.x; d < head_dimension; d += blockDim.x) item_output[d] = 0.0F;
    if (threadIdx.x == 0U) { state[0] = 0.0F; state[1] = 0.0F; state[2] = 0.0F; }
    __syncthreads();

    float running_max = -CUDART_INF_F;
    float running_sum = 0.0F;
    const float scale = rsqrtf(static_cast<float>(head_dimension));
    for (std::uint64_t token = 0; token < total_tokens; ++token) {
        const float* key = batch_kv_ptr(key_pages, current_k, token, previous_tokens, page_tokens, layer, kv_width, kv_base);
        float partial = 0.0F;
        for (std::uint32_t d = threadIdx.x; d < head_dimension; d += blockDim.x) partial += q[q_base + d] * key[d];
        scratch[threadIdx.x] = partial;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
            if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0U) {
            const float score = scratch[0] * scale;
            const float next_max = fmaxf(running_max, score);
            const float alpha = running_max == -CUDART_INF_F ? 0.0F : expf(running_max - next_max);
            const float beta = expf(score - next_max);
            running_sum = running_sum * alpha + beta;
            running_max = next_max;
            state[0] = alpha;
            state[1] = beta;
            state[2] = running_sum;
        }
        __syncthreads();
        const float* value = batch_kv_ptr(value_pages, current_v, token, previous_tokens, page_tokens, layer, kv_width, kv_base);
        for (std::uint32_t d = threadIdx.x; d < head_dimension; d += blockDim.x) {
            item_output[d] = item_output[d] * state[0] + value[d] * state[1];
        }
        __syncthreads();
    }
    const float denominator = state[2];
    for (std::uint32_t d = threadIdx.x; d < head_dimension; d += blockDim.x) item_output[d] /= denominator;
}

__global__ void append_kv_batch_kernel(const float* current_k, const float* current_v,
                                       float* const* key_pages, float* const* value_pages,
                                       std::uint32_t page_tokens, std::uint64_t previous_tokens,
                                       std::uint32_t batch, std::uint32_t layer,
                                       std::uint64_t kv_width) {
    const std::uint64_t total = static_cast<std::uint64_t>(batch) * kv_width;
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        const std::uint32_t item = static_cast<std::uint32_t>(i / kv_width);
        const std::uint64_t element = i % kv_width;
        const std::uint64_t token = previous_tokens + item;
        const std::uint64_t page = token / page_tokens;
        const std::uint64_t within = token % page_tokens;
        const std::uint64_t offset = (static_cast<std::uint64_t>(layer) * page_tokens + within) * kv_width + element;
        key_pages[page][offset] = current_k[static_cast<std::uint64_t>(item) * kv_width + element];
        value_pages[page][offset] = current_v[static_cast<std::uint64_t>(item) * kv_width + element];
    }
}

unsigned int blocks_for(std::uint64_t elements) {
    constexpr std::uint64_t threads = 256U;
    return static_cast<unsigned int>(std::min<std::uint64_t>(65535U, (elements + threads - 1U) / threads));
}

Status launch_status(const char* operation) {
    return cuda_status(cudaGetLastError(), operation);
}

} // namespace

class CudaKvPagePool final {
public:
    struct PageView {
        float* keys{nullptr};
        float* values{nullptr};
    };

    CudaKvPagePool(int device_ordinal, std::uint32_t layer_count,
                   std::uint64_t kv_width, std::uint32_t page_tokens)
        : device_ordinal_(device_ordinal), layer_count_(layer_count),
          kv_width_(kv_width), page_tokens_(page_tokens) {
        one_side_bytes_ = static_cast<std::uint64_t>(layer_count_) * page_tokens_ *
                          kv_width_ * sizeof(float);
        page_bytes_ = one_side_bytes_ * 2U;
    }

    [[nodiscard]] std::uint32_t page_tokens() const noexcept { return page_tokens_; }
    [[nodiscard]] std::uint64_t page_bytes() const noexcept { return page_bytes_; }

    [[nodiscard]] Result<std::size_t> acquire() {
        std::lock_guard lock(mutex_);
        if (!free_.empty()) {
            const auto id = free_.back();
            free_.pop_back();
            pages_[id]->references = 1U;
            return id;
        }
        auto page = std::make_unique<Page>();
        auto status = page->storage.allocate(page_bytes_);
        if (!status) return status;
        auto* base = static_cast<std::uint8_t*>(page->storage.get());
        page->keys = reinterpret_cast<float*>(base);
        page->values = reinterpret_cast<float*>(base + one_side_bytes_);
        page->references = 1U;
        pages_.push_back(std::move(page));
        return pages_.size() - 1U;
    }

    [[nodiscard]] Status retain(std::size_t id) {
        std::lock_guard lock(mutex_);
        if (id >= pages_.size() || pages_[id]->references == 0U) {
            return Status::invalid_state("CUDA KV page retain references an unowned page");
        }
        if (pages_[id]->references == std::numeric_limits<std::uint32_t>::max()) {
            return Status::invalid_state("CUDA KV page reference count overflow");
        }
        ++pages_[id]->references;
        return Status::ok();
    }

    void release(std::size_t id) noexcept {
        std::lock_guard lock(mutex_);
        if (id >= pages_.size() || pages_[id]->references == 0U) return;
        --pages_[id]->references;
        if (pages_[id]->references == 0U) free_.push_back(id);
    }

    [[nodiscard]] std::uint32_t references(std::size_t id) const noexcept {
        std::lock_guard lock(mutex_);
        return id < pages_.size() ? pages_[id]->references : 0U;
    }

    [[nodiscard]] PageView view(std::size_t id) const noexcept {
        std::lock_guard lock(mutex_);
        if (id >= pages_.size()) return {};
        return PageView{pages_[id]->keys, pages_[id]->values};
    }

    [[nodiscard]] Result<std::size_t> clone_page(std::size_t source, cudaStream_t stream) {
        const auto source_view = view(source);
        if (!source_view.keys || !source_view.values) {
            return Status::invalid_state("CUDA KV copy-on-write source page is invalid");
        }
        auto acquired = acquire();
        if (!acquired) return acquired.status();
        const auto target = acquired.value();
        const auto target_view = view(target);
        auto status = cuda_status(cudaMemcpyAsync(target_view.keys, source_view.keys, one_side_bytes_,
                                                  cudaMemcpyDeviceToDevice, stream),
                                  "cudaMemcpyAsync(KV COW keys)");
        if (!status) { release(target); return status; }
        status = cuda_status(cudaMemcpyAsync(target_view.values, source_view.values, one_side_bytes_,
                                             cudaMemcpyDeviceToDevice, stream),
                             "cudaMemcpyAsync(KV COW values)");
        if (!status) { release(target); return status; }
        return target;
    }

    [[nodiscard]] std::uint64_t allocated_bytes() const noexcept {
        std::lock_guard lock(mutex_);
        return static_cast<std::uint64_t>(pages_.size()) * page_bytes_;
    }
    [[nodiscard]] std::uint64_t free_bytes() const noexcept {
        std::lock_guard lock(mutex_);
        return static_cast<std::uint64_t>(free_.size()) * page_bytes_;
    }

private:
    struct Page {
        DeviceAllocation storage;
        float* keys{nullptr};
        float* values{nullptr};
        std::uint32_t references{0};
    };

    int device_ordinal_{0};
    std::uint32_t layer_count_{0};
    std::uint64_t kv_width_{0};
    std::uint32_t page_tokens_{0};
    std::uint64_t one_side_bytes_{0};
    std::uint64_t page_bytes_{0};
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<Page>> pages_;
    std::vector<std::size_t> free_;
};

struct CudaKvCache::Impl {
    int device_ordinal{0};
    std::uint32_t layer_count{0};
    std::uint32_t kv_head_count{0};
    std::uint32_t head_dimension{0};
    std::uint64_t context_length{0};
    std::uint64_t token_count{0};
    std::uint32_t page_tokens{0};
    std::shared_ptr<CudaKvPagePool> pool;
    std::vector<std::size_t> pages;
    DeviceAllocation page_table;
    float** key_pages{nullptr};
    float** value_pages{nullptr};
    std::uint64_t max_pages{0};
    std::vector<std::uint8_t> pending;
    std::uint32_t staged_tokens{0};
    cudaStream_t stream{nullptr};

    ~Impl() { release_all(); }

    [[nodiscard]] std::uint64_t kv_width() const noexcept {
        return static_cast<std::uint64_t>(kv_head_count) * head_dimension;
    }
    [[nodiscard]] std::uint64_t bytes_per_token() const noexcept {
        return static_cast<std::uint64_t>(layer_count) * kv_width() * 2U * sizeof(float);
    }
    [[nodiscard]] std::uint64_t committed_bytes() const noexcept {
        return token_count * bytes_per_token();
    }
    [[nodiscard]] std::uint64_t resident_bytes() const noexcept {
        return static_cast<std::uint64_t>(pages.size()) * (pool ? pool->page_bytes() : 0U) +
               page_table.size_bytes();
    }
    [[nodiscard]] std::uint64_t committed_page_count() const noexcept {
        if (token_count == 0U || page_tokens == 0U) return 0U;
        return (token_count + page_tokens - 1U) / page_tokens;
    }

    Status set_page_pointer(std::size_t slot, std::size_t page_id) {
        if (!pool || slot >= max_pages) return Status::invalid_state("CUDA KV page table slot is invalid");
        const auto view = pool->view(page_id);
        if (!view.keys || !view.values) return Status::invalid_state("CUDA KV page view is invalid");
        auto* key = view.keys;
        auto* value = view.values;
        auto status = cuda_status(cudaMemcpyAsync(key_pages + slot, &key, sizeof(float*),
                                                  cudaMemcpyHostToDevice, stream),
                                  "cudaMemcpyAsync(KV key page table)");
        if (!status) return status;
        return cuda_status(cudaMemcpyAsync(value_pages + slot, &value, sizeof(float*),
                                           cudaMemcpyHostToDevice, stream),
                           "cudaMemcpyAsync(KV value page table)");
    }

    Status attach_existing(std::size_t page_id) {
        auto status = pool->retain(page_id);
        if (!status) return status;
        pages.push_back(page_id);
        status = set_page_pointer(pages.size() - 1U, page_id);
        if (!status) {
            pages.pop_back();
            pool->release(page_id);
        }
        return status;
    }

    Status acquire_page() {
        auto acquired = pool->acquire();
        if (!acquired) return acquired.status();
        pages.push_back(acquired.value());
        auto status = set_page_pointer(pages.size() - 1U, acquired.value());
        if (!status) {
            pages.pop_back();
            pool->release(acquired.value());
        }
        return status;
    }

    void trim_to_committed() noexcept {
        const auto keep = committed_page_count();
        while (pages.size() > keep) {
            pool->release(pages.back());
            pages.pop_back();
        }
    }

    void release_all() noexcept {
        if (pool) {
            for (const auto page : pages) pool->release(page);
        }
        pages.clear();
        token_count = 0U;
        staged_tokens = 0U;
        std::fill(pending.begin(), pending.end(), 0U);
    }

    Status begin_transaction(std::uint32_t tokens) {
        if (tokens == 0U) return Status::invalid_argument("CUDA KV transaction requires tokens");
        if (staged_tokens != 0U) return Status::invalid_state("CUDA KV transaction already active");
        if (token_count + tokens > context_length) return Status::invalid_state("CUDA KV context capacity exceeded");

        // A shared partial tail must become private before any append mutates it.
        if (token_count != 0U && token_count % page_tokens != 0U && !pages.empty() &&
            pool->references(pages.back()) > 1U) {
            const auto old = pages.back();
            auto cloned = pool->clone_page(old, stream);
            if (!cloned) return cloned.status();
            pages.back() = cloned.value();
            auto status = set_page_pointer(pages.size() - 1U, cloned.value());
            if (!status) {
                pool->release(cloned.value());
                pages.back() = old;
                return status;
            }
            pool->release(old);
        }

        const auto required = (token_count + tokens + page_tokens - 1U) / page_tokens;
        while (pages.size() < required) {
            const auto status = acquire_page();
            if (!status) { trim_to_committed(); return status; }
        }
        staged_tokens = tokens;
        std::fill(pending.begin(), pending.end(), 0U);
        return Status::ok();
    }

    void rollback() noexcept {
        staged_tokens = 0U;
        std::fill(pending.begin(), pending.end(), 0U);
        trim_to_committed();
    }
};

struct CudaExecutor::Impl {
    std::shared_ptr<const ModelDefinition> model;
    int device_ordinal{0};
    cudaStream_t stream{nullptr};
    cublasHandle_t cublas{nullptr};
    DeviceAllocation model_arena;
    std::unordered_map<std::string, ResidentTensor> tensors;
    Workspace workspace;
    // Static geometry is initialized before the executor is published. Hot-path
    // counters are atomic because runtime telemetry may sample them while the
    // scheduler thread is launching work.
    CudaExecutionStats stats;
    std::atomic<std::uint64_t> host_to_device_bytes{0};
    std::atomic<std::uint64_t> device_to_host_bytes{0};
    std::atomic<std::uint64_t> f32_matvec_calls{0};
    std::atomic<std::uint64_t> specialized_matvec_calls{0};
    std::atomic<std::uint64_t> f32_matmul_calls{0};
    std::atomic<std::uint64_t> specialized_matmul_calls{0};
    std::atomic<std::uint64_t> batch_reuse4_matmul_calls{0};
    std::atomic<std::uint64_t> batch_reuse8_matmul_calls{0};
    std::atomic<std::uint64_t> q5q8_dp4a_hybrid_matmul_calls{0};
    std::atomic<std::uint64_t> q5q8_dp4a_hybrid_fallback_calls{0};
    std::atomic<std::uint64_t> dense_f32_cublas_matmul_calls{0};
    std::atomic<std::uint64_t> dense_f32_prepared_bytes{0};
    DeviceAllocation dense_f32_arena;
    std::unordered_map<std::string, const float*> dense_f32_tensors;
    std::mutex dense_f32_mutex;
    std::atomic<std::uint64_t> packed_dp4a_prepared_bytes{0};
    DeviceAllocation packed_dp4a_arena;
    DeviceAllocation packed_dp4a_activation_arena;
    std::unordered_map<std::string, PackedDp4aTensor> packed_dp4a_tensors;
    std::mutex packed_dp4a_mutex;
    std::uint64_t packed_dp4a_activation_capacity_bytes{0};
    std::int8_t* packed_dp4a_activation_codes{nullptr};
    float* packed_dp4a_activation_scales{nullptr};
    std::atomic<std::uint64_t> full_logit_readbacks{0};
    std::atomic<std::uint64_t> greedy_token_readbacks{0};
    std::atomic<std::uint64_t> target_logprob_readbacks{0};
    std::atomic<std::uint64_t> outputless_prefill_chunks{0};
    std::atomic<std::uint64_t> prefill_batch_calls{0};
    std::atomic<std::uint64_t> prefill_batched_sequences{0};
    std::atomic<std::uint64_t> prefill_batched_tokens{0};
    std::atomic<std::uint64_t> decode_batch_calls{0};
    std::atomic<std::uint64_t> decode_batched_sequences{0};
    mutable std::mutex pools_mutex;
    std::unordered_map<std::uint32_t, std::shared_ptr<CudaKvPagePool>> kv_pools;

    ~Impl() {
        int original = 0;
        const bool have_original = cudaGetDevice(&original) == cudaSuccess;
        if (device_ordinal >= 0 && (!have_original || original != device_ordinal)) cudaSetDevice(device_ordinal);
        if (cublas) cublasDestroy(cublas);
        if (stream) cudaStreamDestroy(stream);
        if (have_original && original != device_ordinal) cudaSetDevice(original);
    }

    [[nodiscard]] const ResidentTensor* tensor(const std::string& name) const noexcept {
        const auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : &it->second;
    }

    [[nodiscard]] static bool is_transformer_block_linear_name(std::string_view name) noexcept {
        if (!name.starts_with("blk.")) return false;
        return name.ends_with(".attn_q.weight") ||
               name.ends_with(".attn_k.weight") ||
               name.ends_with(".attn_v.weight") ||
               name.ends_with(".attn_output.weight") ||
               name.ends_with(".ffn_gate.weight") ||
               name.ends_with(".ffn_up.weight") ||
               name.ends_with(".ffn_down.weight");
    }

    [[nodiscard]] std::uint64_t dense_f32_required_bytes() const noexcept {
        std::uint64_t total_bytes = 0U;
        for (const auto& [name, resident] : tensors) {
            const auto& dims = resident.descriptor->shape.dimensions;
            if (dims.size() != 2U || !is_transformer_block_linear_name(name)) continue;
            if (dims[0] != 0U && dims[1] > std::numeric_limits<std::uint64_t>::max() / dims[0]) {
                return std::numeric_limits<std::uint64_t>::max();
            }
            const std::uint64_t elements = dims[0] * dims[1];
            if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
                return std::numeric_limits<std::uint64_t>::max();
            }
            total_bytes = align_up(total_bytes, 256U);
            const std::uint64_t bytes = elements * sizeof(float);
            if (bytes > std::numeric_limits<std::uint64_t>::max() - total_bytes) {
                return std::numeric_limits<std::uint64_t>::max();
            }
            total_bytes += bytes;
        }
        return total_bytes;
    }

    [[nodiscard]] Status prepare_dense_f32_tensors() {
        std::lock_guard lock(dense_f32_mutex);
        if (dense_f32_arena.size_bytes() != 0U) return Status::ok();

        struct DensePlacement {
            const ResidentTensor* tensor{nullptr};
            std::uint64_t offset{0};
            std::uint64_t elements{0};
        };
        std::vector<DensePlacement> placements;
        std::uint64_t total_bytes = 0U;
        for (const auto& [name, resident] : tensors) {
            const auto& dims = resident.descriptor->shape.dimensions;
            if (dims.size() != 2U || !is_transformer_block_linear_name(name)) continue;
            // Prepared block-linear state follows the explicit operation scope.
            // Embeddings, vocabulary projection, and unrelated rank-2 tensors are
            // not materialized merely because they share a matrix shape.
            if (dims[0] != 0U && dims[1] > std::numeric_limits<std::uint64_t>::max() / dims[0]) {
                return Status::unsupported("dense FP32 prepared tensor element count overflow");
            }
            const std::uint64_t elements = dims[0] * dims[1];
            if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
                return Status::unsupported("dense FP32 prepared tensor byte count overflow");
            }
            total_bytes = align_up(total_bytes, 256U);
            const std::uint64_t bytes = elements * sizeof(float);
            if (bytes > std::numeric_limits<std::uint64_t>::max() - total_bytes) {
                return Status::unsupported("dense FP32 prepared tensor arena overflow");
            }
            placements.push_back(DensePlacement{&resident, total_bytes, elements});
            total_bytes += bytes;
        }

        DeviceAllocation candidate;
        auto status = candidate.allocate(total_bytes);
        if (!status) return status;
        auto* base = static_cast<std::uint8_t*>(candidate.get());
        constexpr unsigned int threads = 256U;
        for (const auto& placement : placements) {
            auto* destination = reinterpret_cast<float*>(base + placement.offset);
            dequantize_to_f32_kernel<<<blocks_for(placement.elements), threads, 0, stream>>>(
                placement.tensor->data,
                static_cast<int>(placement.tensor->descriptor->type),
                placement.elements,
                destination);
            status = launch_status("dequantize_to_f32_kernel");
            if (!status) return status;
        }
        status = cuda_status(cudaStreamSynchronize(stream), "cudaStreamSynchronize(prepare dense-f32-cublas)");
        if (!status) return status;

        dense_f32_tensors.clear();
        for (const auto& placement : placements) {
            dense_f32_tensors.emplace(
                placement.tensor->descriptor->name,
                reinterpret_cast<const float*>(
                    static_cast<std::uint8_t*>(candidate.get()) + placement.offset));
        }
        dense_f32_arena = std::move(candidate);
        dense_f32_prepared_bytes.store(total_bytes, std::memory_order_relaxed);
        return Status::ok();
    }

    [[nodiscard]] Result<const float*> dense_f32_tensor(const ResidentTensor& tensor) {
        auto status = prepare_dense_f32_tensors();
        if (!status) return status;
        const auto found = dense_f32_tensors.find(tensor.descriptor->name);
        if (found == dense_f32_tensors.end()) {
            return Status::unsupported("dense FP32 cuBLAS tactic has no prepared matrix: " + tensor.descriptor->name);
        }
        return found->second;
    }

    [[nodiscard]] static bool packed_dp4a_eligible(const ResidentTensor& resident) noexcept {
        const auto& dims = resident.descriptor->shape.dimensions;
        return dims.size() == 2U &&
               is_transformer_block_linear_name(resident.descriptor->name) &&
               (resident.descriptor->type == DataType::q5_0 || resident.descriptor->type == DataType::q8_0);
    }

    [[nodiscard]] std::uint64_t packed_dp4a_required_bytes() const noexcept {
        std::uint64_t total = 0U;
        std::uint64_t max_input_width = 0U;
        for (const auto& [_, resident] : tensors) {
            if (!packed_dp4a_eligible(resident)) continue;
            const auto& dims = resident.descriptor->shape.dimensions;
            if (dims[0] != 0U && dims[1] > std::numeric_limits<std::uint64_t>::max() / dims[0]) {
                return std::numeric_limits<std::uint64_t>::max();
            }
            const std::uint64_t elements = dims[0] * dims[1];
            const std::uint64_t blocks = elements / 32U;
            total = align_up(total, 256U);
            if (elements > std::numeric_limits<std::uint64_t>::max() - total) return std::numeric_limits<std::uint64_t>::max();
            total += elements;
            total = align_up(total, 256U);
            if (blocks > (std::numeric_limits<std::uint64_t>::max() - total) / sizeof(float)) return std::numeric_limits<std::uint64_t>::max();
            total += blocks * sizeof(float);
            max_input_width = std::max(max_input_width, dims[0]);
        }
        if (max_input_width == 0U) return total;
        if (max_input_width > std::numeric_limits<std::uint64_t>::max() / kMaxNativePrefillBatch) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        const std::uint64_t activation_elements = max_input_width * kMaxNativePrefillBatch;
        const std::uint64_t activation_blocks = activation_elements / 32U;
        std::uint64_t workspace = align_up(activation_elements, 256U);
        if (activation_blocks > (std::numeric_limits<std::uint64_t>::max() - workspace) / sizeof(float)) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        workspace += activation_blocks * sizeof(float);
        if (workspace > std::numeric_limits<std::uint64_t>::max() - total) return std::numeric_limits<std::uint64_t>::max();
        return total + workspace;
    }

    [[nodiscard]] Status prepare_packed_dp4a_tensors() {
        std::lock_guard lock(packed_dp4a_mutex);
        if (packed_dp4a_prepared_bytes.load(std::memory_order_relaxed) != 0U) return Status::ok();
        struct Placement {
            const ResidentTensor* tensor{nullptr};
            std::uint64_t codes_offset{0};
            std::uint64_t scales_offset{0};
            std::uint64_t elements{0};
            std::uint64_t blocks{0};
        };
        std::vector<Placement> placements;
        std::uint64_t arena_bytes = 0U;
        std::uint64_t max_input_width = 0U;
        for (const auto& [_, resident] : tensors) {
            if (!packed_dp4a_eligible(resident)) continue;
            const auto& dims = resident.descriptor->shape.dimensions;
            if (dims[0] % 32U != 0U) return Status::unsupported("q5q8-dp4a-hybrid requires K aligned to 32");
            if (dims[0] != 0U && dims[1] > std::numeric_limits<std::uint64_t>::max() / dims[0]) {
                return Status::unsupported("q5q8-dp4a-hybrid prepared tensor element count overflow");
            }
            const std::uint64_t elements = dims[0] * dims[1];
            const std::uint64_t blocks = elements / 32U;
            arena_bytes = align_up(arena_bytes, 256U);
            const std::uint64_t codes_offset = arena_bytes;
            if (elements > std::numeric_limits<std::uint64_t>::max() - arena_bytes) {
                return Status::unsupported("q5q8-dp4a-hybrid code arena overflow");
            }
            arena_bytes += elements;
            arena_bytes = align_up(arena_bytes, 256U);
            const std::uint64_t scales_offset = arena_bytes;
            if (blocks > (std::numeric_limits<std::uint64_t>::max() - arena_bytes) / sizeof(float)) {
                return Status::unsupported("q5q8-dp4a-hybrid scale arena overflow");
            }
            arena_bytes += blocks * sizeof(float);
            placements.push_back(Placement{&resident, codes_offset, scales_offset, elements, blocks});
            max_input_width = std::max(max_input_width, dims[0]);
        }
        DeviceAllocation candidate;
        auto status = candidate.allocate(arena_bytes);
        if (!status) return status;
        auto* base = static_cast<std::uint8_t*>(candidate.get());
        constexpr unsigned int threads = 128U;
        for (const auto& placement : placements) {
            auto* codes = reinterpret_cast<std::int8_t*>(base + placement.codes_offset);
            auto* scales = reinterpret_cast<float*>(base + placement.scales_offset);
            const auto grid64 = std::min<std::uint64_t>((placement.blocks + threads - 1U) / threads, 65535U);
            const auto grid = static_cast<unsigned int>(std::max<std::uint64_t>(grid64, 1U));
            if (placement.tensor->descriptor->type == DataType::q5_0) {
                pack_signed32_weights_kernel<DataType::q5_0><<<grid, threads, 0, stream>>>(
                    placement.tensor->data, placement.blocks, codes, scales);
            } else {
                pack_signed32_weights_kernel<DataType::q8_0><<<grid, threads, 0, stream>>>(
                    placement.tensor->data, placement.blocks, codes, scales);
            }
            status = launch_status("pack_signed32_weights_kernel");
            if (!status) return status;
        }
        DeviceAllocation activation;
        std::uint64_t activation_bytes = 0U;
        std::uint64_t codes_bytes = 0U;
        std::uint64_t activation_blocks = 0U;
        if (max_input_width != 0U) {
            if (max_input_width > std::numeric_limits<std::uint64_t>::max() / kMaxNativePrefillBatch) {
                return Status::unsupported("q5q8-dp4a-hybrid activation workspace overflow");
            }
            codes_bytes = max_input_width * kMaxNativePrefillBatch;
            activation_blocks = codes_bytes / 32U;
            activation_bytes = align_up(codes_bytes, 256U);
            if (activation_blocks > (std::numeric_limits<std::uint64_t>::max() - activation_bytes) / sizeof(float)) {
                return Status::unsupported("q5q8-dp4a-hybrid activation scale workspace overflow");
            }
            activation_bytes += activation_blocks * sizeof(float);
            status = activation.allocate(activation_bytes);
            if (!status) return status;
        }
        status = cuda_status(cudaStreamSynchronize(stream), "cudaStreamSynchronize(prepare q5q8-dp4a-hybrid)");
        if (!status) return status;
        packed_dp4a_tensors.clear();
        for (const auto& placement : placements) {
            const auto& dims = placement.tensor->descriptor->shape.dimensions;
            packed_dp4a_tensors.emplace(placement.tensor->descriptor->name, PackedDp4aTensor{
                reinterpret_cast<const std::int8_t*>(base + placement.codes_offset),
                reinterpret_cast<const float*>(base + placement.scales_offset), dims[0], dims[1]});
        }
        packed_dp4a_arena = std::move(candidate);
        packed_dp4a_activation_arena = std::move(activation);
        if (activation_bytes != 0U) {
            auto* activation_base = static_cast<std::uint8_t*>(packed_dp4a_activation_arena.get());
            packed_dp4a_activation_codes = reinterpret_cast<std::int8_t*>(activation_base);
            const auto scales_offset = align_up(codes_bytes, 256U);
            packed_dp4a_activation_scales = reinterpret_cast<float*>(activation_base + scales_offset);
            packed_dp4a_activation_capacity_bytes = activation_bytes;
        }
        packed_dp4a_prepared_bytes.store(arena_bytes + activation_bytes, std::memory_order_relaxed);
        return Status::ok();
    }

    [[nodiscard]] Result<PackedDp4aTensor> packed_dp4a_tensor(const ResidentTensor& tensor) {
        auto status = prepare_packed_dp4a_tensors();
        if (!status) return status;
        const auto found = packed_dp4a_tensors.find(tensor.descriptor->name);
        if (found == packed_dp4a_tensors.end()) {
            return Status::unsupported("q5q8-dp4a-hybrid has no prepared matrix: " + tensor.descriptor->name);
        }
        return found->second;
    }

    [[nodiscard]] std::shared_ptr<CudaKvPagePool> pool_for(std::uint32_t page_tokens) {
        std::lock_guard lock(pools_mutex);
        const auto found = kv_pools.find(page_tokens);
        if (found != kv_pools.end()) return found->second;
        const auto& config = model->config();
        const auto head_dimension = config.embedding_size / config.attention_head_count;
        const auto kv_width = static_cast<std::uint64_t>(config.kv_head_count) * head_dimension;
        auto pool = std::make_shared<CudaKvPagePool>(device_ordinal, config.layer_count, kv_width, page_tokens);
        kv_pools.emplace(page_tokens, pool);
        return pool;
    }

    [[nodiscard]] std::pair<std::uint64_t, std::uint64_t> pool_bytes() const noexcept {
        std::lock_guard lock(pools_mutex);
        std::uint64_t allocated = 0U;
        std::uint64_t free = 0U;
        for (const auto& [_, pool] : kv_pools) {
            allocated += pool->allocated_bytes();
            free += pool->free_bytes();
        }
        return {allocated, free};
    }

    Status initialize() {
        auto status = select_device(device_ordinal);
        if (!status) return status;
        status = cuda_status(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        if (!status) return status;
        status = cublas_status(cublasCreate(&cublas), "cublasCreate");
        if (!status) return status;
        status = cublas_status(cublasSetStream(cublas, stream), "cublasSetStream");
        if (!status) return status;

        status = detail::validate_qwen2_structure(*model);
        if (!status) return status;
        const auto names = detail::qwen2_execution_tensor_names(*model);
        std::uint64_t total_bytes = 0U;
        struct Placement { const TensorDescriptor* tensor; std::uint64_t offset; CudaTensorKernel kernel; };
        std::vector<Placement> placements;
        placements.reserve(names.size());
        for (const auto& name : names) {
            const auto* descriptor = model->find_tensor(name);
            if (!descriptor) return Status::internal_error("Qwen2 execution contract omitted required tensor: " + name);
            if (!descriptor->byte_size_exact || !supports_cuda_tensor(descriptor->type)) {
                return Status::unsupported("CUDA executor does not support " + std::string(to_string(descriptor->type)) +
                                           " tensor: " + name);
            }
            const auto block_elements = quant_block_elements(descriptor->type);
            if (descriptor->shape.dimensions.size() == 2U &&
                descriptor->shape.dimensions[0] % block_elements != 0U) {
                return Status::data_error("CUDA quantized matrix row is not block-aligned: " + name);
            }
            total_bytes = align_up(total_bytes, 256U);
            if (descriptor->byte_size > std::numeric_limits<std::uint64_t>::max() - total_bytes) {
                return Status::unsupported("CUDA model residency size overflow");
            }
            auto kernel = cuda_tensor_kernel(descriptor->type);
            if (!kernel) return kernel.status();
            placements.push_back({descriptor, total_bytes, kernel.value()});
            total_bytes += descriptor->byte_size;
        }
        status = model_arena.allocate(total_bytes);
        if (!status) return status;
        auto* base = static_cast<std::uint8_t*>(model_arena.get());
        for (const auto& placement : placements) {
            auto bytes = model->tensor_bytes(*placement.tensor);
            if (!bytes) return bytes.status();
            status = cuda_status(cudaMemcpy(base + placement.offset, bytes.value().data(), bytes.value().size(),
                                            cudaMemcpyHostToDevice), "cudaMemcpy(model tensor)");
            if (!status) return status;
            tensors.emplace(placement.tensor->name, ResidentTensor{placement.tensor, base + placement.offset, placement.kernel});
            host_to_device_bytes.fetch_add(placement.tensor->byte_size, std::memory_order_relaxed);
        }
        stats.resident_model_bytes = total_bytes;

        const auto& config = model->config();
        const std::uint64_t embedding = config.embedding_size;
        const std::uint64_t kv_width = static_cast<std::uint64_t>(config.kv_head_count) *
                                       (config.embedding_size / config.attention_head_count);
        const std::uint64_t ffn = config.feed_forward_size;
        const std::uint64_t vocab = config.vocabulary_size;
        const std::uint64_t scores = static_cast<std::uint64_t>(config.attention_head_count) * config.context_length;
        const std::uint64_t batch = kMaxNativePrefillBatch;
        const std::uint64_t batch_floats = batch * (embedding * 5U + kv_width * 2U + ffn * 2U);
        const std::uint64_t float_count = batch_floats +
            static_cast<std::uint64_t>(kMaxNativeDecodeBatch) * vocab + scores + kArgmaxBlocks;
        if (float_count > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
            return Status::unsupported("CUDA workspace size overflow");
        }
        status = workspace.arena.allocate(float_count * sizeof(float));
        if (!status) return status;
        status = workspace.token_arena.allocate((batch + kArgmaxBlocks + 2U) * sizeof(TokenId));
        if (!status) return status;
        workspace.token_ids = static_cast<TokenId*>(workspace.token_arena.get());
        workspace.argmax_indices = workspace.token_ids + batch;
        workspace.selected_token = workspace.argmax_indices + kArgmaxBlocks;
        workspace.nonfinite_flag = workspace.selected_token + 1U;
        auto* cursor = static_cast<float*>(workspace.arena.get());
        const auto take = [&cursor](std::uint64_t count) {
            float* result = cursor;
            cursor += count;
            return result;
        };
        workspace.hidden = take(batch * embedding);
        workspace.normalized = take(batch * embedding);
        workspace.q = take(batch * embedding);
        workspace.k = take(batch * kv_width);
        workspace.v = take(batch * kv_width);
        workspace.attended = take(batch * embedding);
        workspace.projection = take(batch * embedding);
        workspace.gate = take(batch * ffn);
        workspace.up = take(batch * ffn);
        workspace.logits = take(static_cast<std::uint64_t>(kMaxNativeDecodeBatch) * vocab);
        workspace.scores = take(scores);
        workspace.argmax_values = take(kArgmaxBlocks);
        workspace.bytes = workspace.arena.size_bytes() + workspace.token_arena.size_bytes();
        stats.workspace_bytes = workspace.bytes;
        stats.kv_logical_context_bytes = static_cast<std::uint64_t>(config.layer_count) *
            config.context_length * kv_width * 2U * sizeof(float);
        return Status::ok();
    }

    Status load_row(const ResidentTensor& tensor, std::uint64_t row, std::uint64_t width, float* output) {
        ScopedProfileRange range("air.cuda.embedding_row");
        constexpr unsigned int threads = 256U;
        load_row_kernel<<<blocks_for(width), threads, 0, stream>>>(tensor.data, static_cast<int>(tensor.descriptor->type), row, width, output);
        return launch_status("load_row_kernel");
    }

    Status load_rows(const ResidentTensor& tensor, std::span<const TokenId> tokens,
                     std::uint64_t width, float* output) {
        ScopedProfileRange range("air.cuda.embedding_rows");
        if (tokens.empty() || tokens.size() > kMaxNativePrefillBatch) {
            return Status::invalid_argument("CUDA native prefill batch width is unsupported");
        }
        auto status = cuda_status(cudaMemcpyAsync(workspace.token_ids, tokens.data(),
                                                  tokens.size_bytes(), cudaMemcpyHostToDevice, stream),
                                  "cudaMemcpyAsync(prefill tokens)");
        if (!status) return status;
        host_to_device_bytes.fetch_add(tokens.size_bytes(), std::memory_order_relaxed);
        const auto total = static_cast<std::uint64_t>(tokens.size()) * width;
        constexpr unsigned int threads = 256U;
        load_rows_kernel<<<blocks_for(total), threads, 0, stream>>>(
            tensor.data, static_cast<int>(tensor.descriptor->type), workspace.token_ids,
            static_cast<std::uint32_t>(tokens.size()), width, output);
        return launch_status("load_rows_kernel");
    }

    Status matvec(const ResidentTensor& matrix, const float* input, float* output) {
        ScopedProfileRange range("air.cuda.matvec");
        const auto& dims = matrix.descriptor->shape.dimensions;
        if (dims.size() != 2U) return Status::invalid_state("CUDA matvec requires rank-2 tensor: " + matrix.descriptor->name);
        const std::uint64_t input_width = dims[0];
        const std::uint64_t output_width = dims[1];
        if (matrix.kernel == CudaTensorKernel::f32) {
            if (input_width > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
                output_width > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
                return Status::unsupported("cuBLAS matvec dimensions exceed int range");
            }
            const float alpha = 1.0F;
            const float beta = 0.0F;
            const auto status = cublas_status(cublasSgemv(cublas, CUBLAS_OP_T,
                                                          static_cast<int>(input_width), static_cast<int>(output_width),
                                                          &alpha, reinterpret_cast<const float*>(matrix.data),
                                                          static_cast<int>(input_width), input, 1, &beta, output, 1),
                                              "cublasSgemv");
            if (status) f32_matvec_calls.fetch_add(1U, std::memory_order_relaxed);
            return status;
        }
        constexpr unsigned int threads = 128U;
        constexpr std::uint64_t rows_per_block = 4U;
        const auto grid64 = (output_width + rows_per_block - 1U) / rows_per_block;
        if (grid64 > static_cast<std::uint64_t>(std::numeric_limits<unsigned int>::max())) {
            return Status::unsupported("CUDA matvec output width exceeds grid range");
        }
        const auto grid = static_cast<unsigned int>(grid64);
        switch (matrix.kernel) {
        case CudaTensorKernel::f16:
            specialized_matvec_kernel<DataType::f16><<<grid, threads, 0, stream>>>(matrix.data, input_width, output_width, input, output);
            break;
        case CudaTensorKernel::bf16:
            specialized_matvec_kernel<DataType::bf16><<<grid, threads, 0, stream>>>(matrix.data, input_width, output_width, input, output);
            break;
        case CudaTensorKernel::q4_0:
            specialized_matvec_kernel<DataType::q4_0><<<grid, threads, 0, stream>>>(matrix.data, input_width, output_width, input, output);
            break;
        case CudaTensorKernel::q5_0:
            specialized_matvec_kernel<DataType::q5_0><<<grid, threads, 0, stream>>>(matrix.data, input_width, output_width, input, output);
            break;
        case CudaTensorKernel::q8_0:
            specialized_matvec_kernel<DataType::q8_0><<<grid, threads, 0, stream>>>(matrix.data, input_width, output_width, input, output);
            break;
        case CudaTensorKernel::q4_k:
            specialized_matvec_kernel<DataType::q4_k><<<grid, threads, 0, stream>>>(matrix.data, input_width, output_width, input, output);
            break;
        case CudaTensorKernel::q6_k:
            specialized_matvec_kernel<DataType::q6_k><<<grid, threads, 0, stream>>>(matrix.data, input_width, output_width, input, output);
            break;
        case CudaTensorKernel::f32:
            return Status::internal_error("prepared F32 tensor reached specialized matvec dispatch");
        }
        const auto status = launch_status("specialized_matvec_kernel");
        if (status) specialized_matvec_calls.fetch_add(1U, std::memory_order_relaxed);
        return status;
    }

    Status matmul(const ResidentTensor& matrix, const float* input, float* output,
                  std::uint32_t batch, QuantizedLinearExecutionKind linear) {
        ScopedProfileRange range("air.cuda.matmul");
        if (batch == 0U || batch > kMaxNativePrefillBatch) {
            return Status::invalid_argument("CUDA matmul batch width is unsupported");
        }
        const auto& dims = matrix.descriptor->shape.dimensions;
        if (dims.size() != 2U) return Status::invalid_state("CUDA matmul requires rank-2 tensor: " + matrix.descriptor->name);
        const std::uint64_t input_width = dims[0];
        const std::uint64_t output_width = dims[1];

        // Operation scope is explicit in ExecutionPlan. matmul executes the
        // tactic it is given and never performs tensor-name-based fallback.
        // Unsupported operation/tactic combinations are rejected by plan
        // capability validation before execution.
        if (linear == QuantizedLinearExecutionKind::dense_f32_cublas) {
            if (input_width > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
                output_width > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
                batch > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
                return Status::unsupported("dense FP32 cuBLAS dimensions exceed int range");
            }
            auto prepared = dense_f32_tensor(matrix);
            if (!prepared) return prepared.status();
            const float alpha = 1.0F;
            const float beta = 0.0F;
            const auto status = cublas_status(
                cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             static_cast<int>(output_width), static_cast<int>(batch),
                             static_cast<int>(input_width), &alpha,
                             prepared.value(), CUDA_R_32F, static_cast<int>(input_width),
                             input, CUDA_R_32F, static_cast<int>(input_width),
                             &beta, output, CUDA_R_32F, static_cast<int>(output_width),
                             CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT),
                "cublasGemmEx(dense-f32-cublas-pedantic)");
            if (status) dense_f32_cublas_matmul_calls.fetch_add(1U, std::memory_order_relaxed);
            return status;
        }
        if (linear == QuantizedLinearExecutionKind::q5q8_dp4a_hybrid) {
            if (matrix.descriptor->type == DataType::q5_0 || matrix.descriptor->type == DataType::q8_0) {
                ScopedProfileRange packed_range("air.cuda.q5q8_dp4a_hybrid");
                if (input_width % 32U != 0U) return Status::unsupported("q5q8-dp4a-hybrid requires K aligned to 32");
                auto prepared = packed_dp4a_tensor(matrix);
                if (!prepared) return prepared.status();
                const std::uint64_t activation_blocks = static_cast<std::uint64_t>(batch) * (input_width / 32U);
                if (activation_blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned int>::max())) {
                    return Status::unsupported("q5q8-dp4a-hybrid activation grid exceeds CUDA range");
                }
                quantize_q8_activation_kernel<<<static_cast<unsigned int>(activation_blocks), 32U, 0, stream>>>(
                    input, input_width, batch, packed_dp4a_activation_codes,
                    packed_dp4a_activation_scales);
                auto status = launch_status("quantize_q8_activation_kernel");
                if (!status) return status;
                constexpr unsigned int threads = 128U;
                constexpr std::uint64_t rows_per_block = 4U;
                const auto grid_x64 = (output_width + rows_per_block - 1U) / rows_per_block;
                if (grid_x64 > static_cast<std::uint64_t>(std::numeric_limits<unsigned int>::max())) {
                    return Status::unsupported("q5q8-dp4a-hybrid output grid exceeds CUDA range");
                }
                const dim3 grid(static_cast<unsigned int>(grid_x64), (batch + 7U) / 8U, 1U);
                q5q8_dp4a_hybrid_matmul_kernel<8U><<<grid, threads, 0, stream>>>(
                    prepared.value().codes, prepared.value().scales, input_width, output_width,
                    packed_dp4a_activation_codes, packed_dp4a_activation_scales, output, batch);
                status = launch_status("q5q8_dp4a_hybrid_matmul_kernel");
                if (status) q5q8_dp4a_hybrid_matmul_calls.fetch_add(1U, std::memory_order_relaxed);
                return status;
            }
            // Sprint-3 candidate is intentionally bounded: Q5_0/Q8_0 use packed
            // integer dots, while formats with affine/subgroup reconstruction stay
            // on the qualified reuse8 kernel until a separate CUDA implementation
            // proves their exact scale/minimum epilogue.
            q5q8_dp4a_hybrid_fallback_calls.fetch_add(1U, std::memory_order_relaxed);
            linear = QuantizedLinearExecutionKind::batch_reuse8;
        }
        if (matrix.kernel == CudaTensorKernel::f32) {
            if (input_width > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
                output_width > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
                batch > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
                return Status::unsupported("cuBLAS matmul dimensions exceed int range");
            }
            const float alpha = 1.0F;
            const float beta = 0.0F;
            const auto status = cublas_status(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                                          static_cast<int>(output_width), static_cast<int>(batch),
                                                          static_cast<int>(input_width), &alpha,
                                                          reinterpret_cast<const float*>(matrix.data), static_cast<int>(input_width),
                                                          input, static_cast<int>(input_width), &beta,
                                                          output, static_cast<int>(output_width)),
                                              "cublasSgemm");
            if (status) f32_matmul_calls.fetch_add(1U, std::memory_order_relaxed);
            return status;
        }
        constexpr unsigned int threads = 128U;
        constexpr std::uint64_t rows_per_block = 4U;
        const auto grid_x = (output_width + rows_per_block - 1U) / rows_per_block;
        if (grid_x > static_cast<std::uint64_t>(std::numeric_limits<unsigned int>::max())) {
            return Status::unsupported("CUDA matmul output width exceeds grid range");
        }

        const auto launch_baseline = [&](auto type_tag) {
            constexpr DataType Type = decltype(type_tag)::value;
            const dim3 grid(static_cast<unsigned int>(grid_x), batch, 1U);
            specialized_matmul_kernel<Type><<<grid, threads, 0, stream>>>(
                matrix.data, input_width, output_width, input, output, batch);
        };
        const auto launch_reuse4 = [&](auto type_tag) {
            constexpr DataType Type = decltype(type_tag)::value;
            const dim3 grid(static_cast<unsigned int>(grid_x), (batch + 3U) / 4U, 1U);
            specialized_matmul_batch_reuse_kernel<Type, 4U><<<grid, threads, 0, stream>>>(
                matrix.data, input_width, output_width, input, output, batch);
        };
        const auto launch_reuse8 = [&](auto type_tag) {
            constexpr DataType Type = decltype(type_tag)::value;
            const dim3 grid(static_cast<unsigned int>(grid_x), (batch + 7U) / 8U, 1U);
            specialized_matmul_batch_reuse_kernel<Type, 8U><<<grid, threads, 0, stream>>>(
                matrix.data, input_width, output_width, input, output, batch);
        };
        const auto dispatch = [&](auto type_tag) {
            switch (linear) {
            case QuantizedLinearExecutionKind::baseline:
                launch_baseline(type_tag);
                break;
            case QuantizedLinearExecutionKind::batch_reuse4:
                launch_reuse4(type_tag);
                break;
            case QuantizedLinearExecutionKind::batch_reuse8:
                launch_reuse8(type_tag);
                break;
            case QuantizedLinearExecutionKind::q5q8_dp4a_hybrid:
                break; // handled or reduced to reuse8 before specialized dispatch
            case QuantizedLinearExecutionKind::dense_f32_cublas:
                break; // handled before specialized dispatch
            }
        };

        switch (matrix.kernel) {
        case CudaTensorKernel::f16: dispatch(std::integral_constant<DataType, DataType::f16>{}); break;
        case CudaTensorKernel::bf16: dispatch(std::integral_constant<DataType, DataType::bf16>{}); break;
        case CudaTensorKernel::q4_0: dispatch(std::integral_constant<DataType, DataType::q4_0>{}); break;
        case CudaTensorKernel::q5_0: dispatch(std::integral_constant<DataType, DataType::q5_0>{}); break;
        case CudaTensorKernel::q8_0: dispatch(std::integral_constant<DataType, DataType::q8_0>{}); break;
        case CudaTensorKernel::q4_k: dispatch(std::integral_constant<DataType, DataType::q4_k>{}); break;
        case CudaTensorKernel::q6_k: dispatch(std::integral_constant<DataType, DataType::q6_k>{}); break;
        case CudaTensorKernel::f32:
            return Status::internal_error("prepared F32 tensor reached specialized matmul dispatch");
        }

        const auto status = launch_status("specialized_matmul_kernel");
        if (status) {
            specialized_matmul_calls.fetch_add(1U, std::memory_order_relaxed);
            if (linear == QuantizedLinearExecutionKind::batch_reuse4) {
                batch_reuse4_matmul_calls.fetch_add(1U, std::memory_order_relaxed);
            } else if (linear == QuantizedLinearExecutionKind::batch_reuse8) {
                batch_reuse8_matmul_calls.fetch_add(1U, std::memory_order_relaxed);
            }
        }
        return status;
    }

    Status linear_single(const ResidentTensor& matrix, const float* input, float* output,
                         QuantizedLinearExecutionKind linear) {
        // Preserve the historical scalar baseline exactly. Non-baseline
        // operation-scoped tactics use the already-qualified matmul machinery at
        // batch width one, so ExecutionPlan labels are executable semantics rather
        // than telemetry-only metadata.
        if (linear == QuantizedLinearExecutionKind::baseline) {
            return matvec(matrix, input, output);
        }
        return matmul(matrix, input, output, 1U, linear);
    }

    Status rms_norm(const float* input, const ResidentTensor& weight, std::uint64_t width, float epsilon, float* output) {
        ScopedProfileRange range("air.cuda.rms_norm");
        constexpr unsigned int threads = 256U;
        rms_norm_kernel<<<1U, threads, threads * sizeof(float), stream>>>(
            input, weight.data, static_cast<int>(weight.descriptor->type), width, epsilon, output);
        return launch_status("rms_norm_kernel");
    }

    Status rms_norm_batch(const float* input, const ResidentTensor& weight, std::uint64_t width,
                          float epsilon, float* output, std::uint32_t batch) {
        ScopedProfileRange range("air.cuda.rms_norm_batch");
        constexpr unsigned int threads = 256U;
        rms_norm_batch_kernel<<<batch, threads, threads * sizeof(float), stream>>>(
            input, weight.data, static_cast<int>(weight.descriptor->type), width, epsilon, output, batch);
        return launch_status("rms_norm_batch_kernel");
    }

    Status add_optional_tensor(float* target, const std::string& name, std::uint64_t width) {
        const auto* bias = tensor(name);
        if (!bias) return Status::ok();
        constexpr unsigned int threads = 256U;
        add_tensor_kernel<<<blocks_for(width), threads, 0, stream>>>(
            target, bias->data, static_cast<int>(bias->descriptor->type), width);
        return launch_status("add_tensor_kernel");
    }

    Status add_optional_tensor_batch(float* target, const std::string& name,
                                     std::uint64_t width, std::uint32_t batch) {
        const auto* bias = tensor(name);
        if (!bias) return Status::ok();
        constexpr unsigned int threads = 256U;
        const auto total = static_cast<std::uint64_t>(batch) * width;
        add_tensor_batch_kernel<<<blocks_for(total), threads, 0, stream>>>(
            target, bias->data, static_cast<int>(bias->descriptor->type), width, batch);
        return launch_status("add_tensor_batch_kernel");
    }

    Status add_vector(float* target, const float* value, std::uint64_t width) {
        ScopedProfileRange range("air.cuda.residual_add");
        constexpr unsigned int threads = 256U;
        add_vector_kernel<<<blocks_for(width), threads, 0, stream>>>(target, value, width);
        return launch_status("add_vector_kernel");
    }

    Status add_vector_batch(float* target, const float* value, std::uint64_t width, std::uint32_t batch) {
        ScopedProfileRange range("air.cuda.residual_add_batch");
        constexpr unsigned int threads = 256U;
        const auto total = static_cast<std::uint64_t>(batch) * width;
        add_vector_kernel<<<blocks_for(total), threads, 0, stream>>>(target, value, total);
        return launch_status("add_vector_kernel(batch)");
    }

    Status apply_rope(float* values, std::uint32_t heads, std::uint32_t head_dimension,
                      std::uint32_t rope_dimensions, std::uint64_t position, float rope_base) {
        ScopedProfileRange range("air.cuda.rope");
        const std::uint64_t pairs = static_cast<std::uint64_t>(heads) * (rope_dimensions / 2U);
        constexpr unsigned int threads = 256U;
        rope_kernel<<<blocks_for(pairs), threads, 0, stream>>>(values, heads, head_dimension, rope_dimensions, position, rope_base);
        return launch_status("rope_kernel");
    }

    Status apply_rope_batch(float* values, std::uint32_t batch, std::uint32_t heads,
                            std::uint32_t head_dimension, std::uint32_t rope_dimensions,
                            std::uint64_t position_start, float rope_base) {
        ScopedProfileRange range("air.cuda.rope_batch");
        const std::uint64_t pairs = static_cast<std::uint64_t>(batch) * heads * (rope_dimensions / 2U);
        constexpr unsigned int threads = 256U;
        rope_batch_kernel<<<blocks_for(pairs), threads, 0, stream>>>(
            values, batch, heads, head_dimension, rope_dimensions, position_start, rope_base);
        return launch_status("rope_batch_kernel");
    }

    Status attention_from(CudaKvCache::Impl& cache, std::uint32_t layer,
                          std::uint32_t q_heads, std::uint32_t kv_heads,
                          std::uint32_t head_dimension, const float* q, const float* k,
                          const float* v, float* attended) {
        ScopedProfileRange range("air.cuda.attention_decode");
        const std::uint64_t total_tokens = cache.token_count + 1U;
        const std::uint64_t score_blocks = static_cast<std::uint64_t>(q_heads) * total_tokens;
        if (score_blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned int>::max())) {
            return Status::unsupported("attention score grid exceeds CUDA grid range");
        }
        constexpr unsigned int threads = 256U;
        attention_scores_paged_kernel<<<static_cast<unsigned int>(score_blocks), threads, threads * sizeof(float), stream>>>(
            q, k, cache.key_pages, cache.page_tokens, cache.token_count,
            layer, q_heads, kv_heads, head_dimension, workspace.scores);
        auto status = launch_status("attention_scores_paged_kernel");
        if (!status) return status;
        attention_value_paged_kernel<<<q_heads, threads, threads * sizeof(float), stream>>>(
            workspace.scores, v, cache.value_pages, cache.page_tokens, cache.token_count,
            layer, q_heads, kv_heads, head_dimension, attended);
        return launch_status("attention_value_paged_kernel");
    }

    Status attention(CudaKvCache::Impl& cache, std::uint32_t layer,
                     std::uint32_t q_heads, std::uint32_t kv_heads, std::uint32_t head_dimension) {
        return attention_from(cache, layer, q_heads, kv_heads, head_dimension,
                              workspace.q, workspace.k, workspace.v, workspace.attended);
    }

    Status attention_batch_from(CudaKvCache::Impl& cache, std::uint32_t batch,
                                std::uint32_t layer, std::uint32_t q_heads,
                                std::uint32_t kv_heads, std::uint32_t head_dimension,
                                AttentionExecutionKind attention,
                                const float* current_q, const float* current_k,
                                const float* current_v, float* attended) {
        ScopedProfileRange range("air.cuda.attention_prefill");
        const std::uint64_t blocks = static_cast<std::uint64_t>(batch) * q_heads;
        if (blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned int>::max())) {
            return Status::unsupported("batched attention grid exceeds CUDA grid range");
        }
        constexpr unsigned int threads = 128U;
        if (attention == AttentionExecutionKind::online_softmax) {
            attention_batch_paged_online_kernel<<<static_cast<unsigned int>(blocks), threads,
                threads * sizeof(float), stream>>>(
                current_q, current_k, current_v, cache.key_pages, cache.value_pages,
                cache.page_tokens, cache.token_count, batch, layer, q_heads, kv_heads,
                head_dimension, attended);
            return launch_status("attention_batch_paged_online_kernel");
        }
        attention_batch_paged_kernel<<<static_cast<unsigned int>(blocks), threads,
            threads * sizeof(float), stream>>>(
            current_q, current_k, current_v, cache.key_pages, cache.value_pages,
            cache.page_tokens, cache.token_count, batch, layer, q_heads, kv_heads,
            head_dimension, attended);
        return launch_status("attention_batch_paged_kernel");
    }

    Status attention_batch(CudaKvCache::Impl& cache, std::uint32_t batch, std::uint32_t layer,
                           std::uint32_t q_heads, std::uint32_t kv_heads,
                           std::uint32_t head_dimension, AttentionExecutionKind attention) {
        return attention_batch_from(cache, batch, layer, q_heads, kv_heads, head_dimension,
                                    attention, workspace.q, workspace.k, workspace.v,
                                    workspace.attended);
    }

    Status append_kv_from(CudaKvCache::Impl& cache, std::uint32_t layer, std::uint32_t batch,
                          const float* current_k, const float* current_v) {
        ScopedProfileRange range("air.cuda.kv_append");
        if (layer >= cache.layer_count) return Status::invalid_argument("CUDA KV layer is outside cache");
        if (cache.pending[layer] != 0U) return Status::invalid_state("CUDA KV layer already staged for current transaction");
        if (cache.staged_tokens != batch || batch == 0U) return Status::invalid_state("CUDA KV staged batch width mismatch");
        constexpr unsigned int threads = 256U;
        const auto total = static_cast<std::uint64_t>(batch) * cache.kv_width();
        append_kv_batch_kernel<<<blocks_for(total), threads, 0, stream>>>(
            current_k, current_v, cache.key_pages, cache.value_pages, cache.page_tokens,
            cache.token_count, batch, layer, cache.kv_width());
        const auto status = launch_status("append_kv_batch_kernel");
        if (!status) return status;
        cache.pending[layer] = 1U;
        return Status::ok();
    }

    Status append_kv(CudaKvCache::Impl& cache, std::uint32_t layer, std::uint32_t batch) {
        return append_kv_from(cache, layer, batch, workspace.k, workspace.v);
    }

    Status validate_kv_commit(const CudaKvCache::Impl& cache) const {
        if (cache.staged_tokens == 0U) return Status::invalid_state("cannot commit empty CUDA KV transaction");
        if (!std::all_of(cache.pending.begin(), cache.pending.end(), [](std::uint8_t value) { return value != 0U; })) {
            return Status::invalid_state("cannot commit incomplete CUDA KV transaction");
        }
        return Status::ok();
    }

    Status commit_kv(CudaKvCache::Impl& cache) {
        auto status = validate_kv_commit(cache);
        if (!status) return status;
        cache.token_count += cache.staged_tokens;
        cache.staged_tokens = 0U;
        std::fill(cache.pending.begin(), cache.pending.end(), 0U);
        return Status::ok();
    }

    Status record_trace(VerificationTrace* trace, std::uint64_t position, std::int32_t layer,
                        VerificationStage stage, const float* device_values, std::uint64_t count) {
        if (!trace) return Status::ok();
        if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            return Status::unsupported("verification snapshot exceeds host address range");
        }
        std::vector<float> host(static_cast<std::size_t>(count));
        auto status = cuda_status(cudaMemcpyAsync(host.data(), device_values, host.size() * sizeof(float),
                                                  cudaMemcpyDeviceToHost, stream),
                                  "cudaMemcpyAsync(verification snapshot)");
        if (!status) return status;
        status = cuda_status(cudaStreamSynchronize(stream), "cudaStreamSynchronize(verification snapshot)");
        if (!status) return status;
        trace->record(position, layer, stage, host);
        return Status::ok();
    }

    Result<std::vector<float>> read_logits_host() {
        ScopedProfileRange range("air.cuda.logit_readback");
        const auto vocab = model->config().vocabulary_size;
        std::vector<float> logits(static_cast<std::size_t>(vocab));
        auto status = cuda_status(cudaMemcpyAsync(logits.data(), workspace.logits,
                                                  logits.size() * sizeof(float),
                                                  cudaMemcpyDeviceToHost, stream),
                                  "cudaMemcpyAsync(logits)");
        if (!status) return status;
        status = cuda_status(cudaStreamSynchronize(stream), "cudaStreamSynchronize(logits)");
        if (!status) return status;
        const auto bytes = static_cast<std::uint64_t>(logits.size()) * sizeof(float);
        device_to_host_bytes.fetch_add(bytes, std::memory_order_relaxed);
        full_logit_readbacks.fetch_add(1U, std::memory_order_relaxed);
        if (!std::all_of(logits.begin(), logits.end(), [](float value) { return std::isfinite(value); })) {
            return Status::internal_error("CUDA executor produced non-finite logits");
        }
        return logits;
    }

    Result<TokenId> select_greedy_device_from(const float* logits) {
        ScopedProfileRange range("air.cuda.greedy_select");
        const auto vocab = model->config().vocabulary_size;
        if (vocab == 0U || vocab > static_cast<std::uint64_t>(std::numeric_limits<TokenId>::max())) {
            return Status::unsupported("CUDA greedy selection vocabulary exceeds token-id range");
        }
        auto status = cuda_status(cudaMemsetAsync(workspace.nonfinite_flag, 0, sizeof(TokenId), stream),
                                  "cudaMemsetAsync(argmax nonfinite)");
        if (!status) return status;
        constexpr unsigned int threads = 256U;
        const auto requested_blocks = blocks_for(vocab);
        const auto blocks = std::max(1U, std::min(kArgmaxBlocks, requested_blocks));
        argmax_stage1_kernel<<<blocks, threads, 0, stream>>>(
            logits, vocab, workspace.argmax_values, workspace.argmax_indices,
            workspace.nonfinite_flag);
        status = launch_status("argmax_stage1_kernel");
        if (!status) return status;
        argmax_stage2_kernel<<<1U, threads, 0, stream>>>(
            workspace.argmax_values, workspace.argmax_indices, blocks, workspace.selected_token);
        status = launch_status("argmax_stage2_kernel");
        if (!status) return status;

        TokenId result[2]{-1, 0};
        status = cuda_status(cudaMemcpyAsync(result, workspace.selected_token, sizeof(result),
                                             cudaMemcpyDeviceToHost, stream),
                             "cudaMemcpyAsync(argmax result)");
        if (!status) return status;
        status = cuda_status(cudaStreamSynchronize(stream), "cudaStreamSynchronize(argmax)");
        if (!status) return status;
        device_to_host_bytes.fetch_add(sizeof(result), std::memory_order_relaxed);
        greedy_token_readbacks.fetch_add(1U, std::memory_order_relaxed);
        if (result[1] != 0) return Status::internal_error("CUDA executor produced non-finite logits");
        if (result[0] < 0 || static_cast<std::uint64_t>(result[0]) >= vocab) {
            return Status::internal_error("CUDA greedy selection produced an invalid token id");
        }
        return result[0];
    }

    Result<TokenId> select_greedy_device() {
        return select_greedy_device_from(workspace.logits);
    }

    Result<std::vector<float>> read_target_logprobs_device(
        std::span<const TokenId> target_tokens) {
        ScopedProfileRange range("air.cuda.target_logprobs");
        const auto vocab = model->config().vocabulary_size;
        if (target_tokens.empty()) {
            return Status::invalid_argument("CUDA target logprob request requires at least one token");
        }
        if (target_tokens.size() > kMaxNativePrefillBatch ||
            target_tokens.size() > kArgmaxBlocks) {
            return Status::unsupported("CUDA target logprob request exceeds workspace capacity");
        }
        for (const auto token : target_tokens) {
            if (token < 0 || static_cast<std::uint64_t>(token) >= vocab) {
                return Status::invalid_argument("CUDA target logprob token is outside vocabulary");
            }
        }
        const auto token_bytes = static_cast<std::uint64_t>(target_tokens.size()) * sizeof(TokenId);
        auto status = cuda_status(cudaMemcpyAsync(workspace.token_ids, target_tokens.data(),
                                                  static_cast<std::size_t>(token_bytes),
                                                  cudaMemcpyHostToDevice, stream),
                                  "cudaMemcpyAsync(target logprob tokens)");
        if (!status) return status;
        host_to_device_bytes.fetch_add(token_bytes, std::memory_order_relaxed);
        status = cuda_status(cudaMemsetAsync(workspace.nonfinite_flag, 0, sizeof(TokenId), stream),
                             "cudaMemsetAsync(target logprob nonfinite)");
        if (!status) return status;
        constexpr unsigned int threads = 256U;
        target_logprobs_kernel<<<1U, threads, 0, stream>>>(
            workspace.logits, vocab, workspace.token_ids,
            static_cast<std::uint32_t>(target_tokens.size()),
            workspace.argmax_values, workspace.nonfinite_flag);
        status = launch_status("target_logprobs_kernel");
        if (!status) return status;

        std::vector<float> scores(target_tokens.size());
        TokenId nonfinite = 0;
        const auto score_bytes = static_cast<std::uint64_t>(scores.size()) * sizeof(float);
        status = cuda_status(cudaMemcpyAsync(scores.data(), workspace.argmax_values,
                                             static_cast<std::size_t>(score_bytes),
                                             cudaMemcpyDeviceToHost, stream),
                             "cudaMemcpyAsync(target logprobs)");
        if (!status) return status;
        status = cuda_status(cudaMemcpyAsync(&nonfinite, workspace.nonfinite_flag, sizeof(nonfinite),
                                             cudaMemcpyDeviceToHost, stream),
                             "cudaMemcpyAsync(target logprob nonfinite)");
        if (!status) return status;
        status = cuda_status(cudaStreamSynchronize(stream), "cudaStreamSynchronize(target logprobs)");
        if (!status) return status;
        device_to_host_bytes.fetch_add(score_bytes + sizeof(nonfinite), std::memory_order_relaxed);
        target_logprob_readbacks.fetch_add(1U, std::memory_order_relaxed);
        if (nonfinite != 0 ||
            !std::all_of(scores.begin(), scores.end(), [](float value) { return std::isfinite(value); })) {
            return Status::internal_error("CUDA target logprob reduction produced non-finite evidence");
        }
        return scores;
    }

    Status synchronize_outputless_prefill() {
        const auto status = cuda_status(cudaStreamSynchronize(stream),
                                        "cudaStreamSynchronize(outputless prefill)");
        if (status) outputless_prefill_chunks.fetch_add(1U, std::memory_order_relaxed);
        return status;
    }

};

bool cuda_compiled() noexcept { return true; }

Result<std::vector<DeviceInfo>> cuda_devices() {
    int count = 0;
    auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice) return std::vector<DeviceInfo>{};
    if (status != cudaSuccess) return cuda_status(status, "cudaGetDeviceCount");
    std::vector<DeviceInfo> devices;
    devices.reserve(static_cast<std::size_t>(count));
    int original = 0;
    cudaGetDevice(&original);
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        cudaDeviceProp properties{};
        status = cudaGetDeviceProperties(&properties, ordinal);
        if (status != cudaSuccess) return cuda_status(status, "cudaGetDeviceProperties");
        status = cudaSetDevice(ordinal);
        if (status != cudaSuccess) return cuda_status(status, "cudaSetDevice");
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        status = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (status != cudaSuccess) return cuda_status(status, "cudaMemGetInfo");
        devices.push_back(DeviceInfo{DeviceKind::cuda, ordinal, properties.name, properties.major, properties.minor,
                                     static_cast<std::uint64_t>(total_bytes), static_cast<std::uint64_t>(free_bytes)});
    }
    cudaSetDevice(original);
    return devices;
}

CudaKvCache::CudaKvCache(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CudaKvCache::~CudaKvCache() = default;
CudaKvCache::CudaKvCache(CudaKvCache&&) noexcept = default;
CudaKvCache& CudaKvCache::operator=(CudaKvCache&&) noexcept = default;
std::uint64_t CudaKvCache::size() const noexcept { return impl_ ? impl_->token_count : 0U; }
std::uint64_t CudaKvCache::capacity() const noexcept { return impl_ ? impl_->context_length : 0U; }
std::uint32_t CudaKvCache::page_tokens() const noexcept { return impl_ ? impl_->page_tokens : 0U; }
std::uint64_t CudaKvCache::committed_bytes() const noexcept { return impl_ ? impl_->committed_bytes() : 0U; }
std::uint64_t CudaKvCache::resident_bytes() const noexcept { return impl_ ? impl_->resident_bytes() : 0U; }

Result<std::unique_ptr<CudaKvCache>> CudaKvCache::fork(std::uint64_t prefix_tokens) const {
    if (!impl_) return Status::invalid_state("CUDA KV cache is not initialized");
    if (prefix_tokens > impl_->token_count) return Status::invalid_argument("CUDA KV fork prefix exceeds committed tokens");
    auto status = activate_device(impl_->device_ordinal);
    if (!status) return status;
    auto clone = std::make_unique<CudaKvCache::Impl>();
    clone->device_ordinal = impl_->device_ordinal;
    clone->layer_count = impl_->layer_count;
    clone->kv_head_count = impl_->kv_head_count;
    clone->head_dimension = impl_->head_dimension;
    clone->context_length = impl_->context_length;
    clone->token_count = prefix_tokens;
    clone->page_tokens = impl_->page_tokens;
    clone->pool = impl_->pool;
    clone->stream = impl_->stream;
    clone->max_pages = impl_->max_pages;
    clone->pending.assign(impl_->layer_count, 0U);
    const auto table_bytes = clone->max_pages * 2U * sizeof(float*);
    status = clone->page_table.allocate(table_bytes);
    if (!status) return status;
    clone->key_pages = static_cast<float**>(clone->page_table.get());
    clone->value_pages = clone->key_pages + clone->max_pages;
    const auto required_pages = prefix_tokens == 0U ? 0U :
        (prefix_tokens + clone->page_tokens - 1U) / clone->page_tokens;
    for (std::uint64_t i = 0; i < required_pages; ++i) {
        status = clone->attach_existing(impl_->pages[static_cast<std::size_t>(i)]);
        if (!status) return status;
    }
    return std::unique_ptr<CudaKvCache>(new CudaKvCache(std::move(clone)));
}

void CudaKvCache::reset() noexcept {
    if (!impl_) return;
    impl_->release_all();
}

CudaExecutor::CudaExecutor(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CudaExecutor::~CudaExecutor() = default;
CudaExecutor::CudaExecutor(CudaExecutor&&) noexcept = default;
CudaExecutor& CudaExecutor::operator=(CudaExecutor&&) noexcept = default;

Result<std::unique_ptr<CudaExecutor>> CudaExecutor::create(std::shared_ptr<const ModelDefinition> model,
                                                           int device_ordinal) {
    if (!model) return Status::invalid_argument("CUDA executor requires a model");
    auto status = model->validate();
    if (!status) return status;
    auto impl = std::make_unique<Impl>();
    impl->model = std::move(model);
    impl->device_ordinal = device_ordinal;
    status = impl->initialize();
    if (!status) return status;
    return std::unique_ptr<CudaExecutor>(new CudaExecutor(std::move(impl)));
}

const ModelDefinition& CudaExecutor::model() const noexcept { return *impl_->model; }
int CudaExecutor::device_ordinal() const noexcept { return impl_->device_ordinal; }
CudaExecutionStats CudaExecutor::stats() const noexcept {
    auto result = impl_->stats;
    result.host_to_device_bytes = impl_->host_to_device_bytes.load(std::memory_order_relaxed);
    result.device_to_host_bytes = impl_->device_to_host_bytes.load(std::memory_order_relaxed);
    result.f32_matvec_calls = impl_->f32_matvec_calls.load(std::memory_order_relaxed);
    result.specialized_matvec_calls = impl_->specialized_matvec_calls.load(std::memory_order_relaxed);
    result.f32_matmul_calls = impl_->f32_matmul_calls.load(std::memory_order_relaxed);
    result.specialized_matmul_calls = impl_->specialized_matmul_calls.load(std::memory_order_relaxed);
    result.batch_reuse4_matmul_calls = impl_->batch_reuse4_matmul_calls.load(std::memory_order_relaxed);
    result.batch_reuse8_matmul_calls = impl_->batch_reuse8_matmul_calls.load(std::memory_order_relaxed);
    result.q5q8_dp4a_hybrid_matmul_calls = impl_->q5q8_dp4a_hybrid_matmul_calls.load(std::memory_order_relaxed);
    result.q5q8_dp4a_hybrid_fallback_calls = impl_->q5q8_dp4a_hybrid_fallback_calls.load(std::memory_order_relaxed);
    result.dense_f32_cublas_matmul_calls = impl_->dense_f32_cublas_matmul_calls.load(std::memory_order_relaxed);
    result.q5q8_dp4a_hybrid_prepared_bytes = impl_->packed_dp4a_prepared_bytes.load(std::memory_order_relaxed);
    result.prepared_linear_bytes = impl_->dense_f32_prepared_bytes.load(std::memory_order_relaxed) +
                                   result.q5q8_dp4a_hybrid_prepared_bytes;
    result.resident_model_bytes += result.prepared_linear_bytes;
    result.full_logit_readbacks = impl_->full_logit_readbacks.load(std::memory_order_relaxed);
    result.greedy_token_readbacks = impl_->greedy_token_readbacks.load(std::memory_order_relaxed);
    result.target_logprob_readbacks = impl_->target_logprob_readbacks.load(std::memory_order_relaxed);
    result.outputless_prefill_chunks = impl_->outputless_prefill_chunks.load(std::memory_order_relaxed);
    result.prefill_batch_calls = impl_->prefill_batch_calls.load(std::memory_order_relaxed);
    result.prefill_batched_sequences = impl_->prefill_batched_sequences.load(std::memory_order_relaxed);
    result.prefill_batched_tokens = impl_->prefill_batched_tokens.load(std::memory_order_relaxed);
    result.decode_batch_calls = impl_->decode_batch_calls.load(std::memory_order_relaxed);
    result.decode_batched_sequences = impl_->decode_batched_sequences.load(std::memory_order_relaxed);
    const auto [allocated, free] = impl_->pool_bytes();
    result.kv_pool_allocated_bytes = allocated;
    result.kv_pool_free_bytes = free;
    std::lock_guard lock(impl_->pools_mutex);
    if (impl_->kv_pools.size() == 1U) result.kv_page_bytes = impl_->kv_pools.begin()->second->page_bytes();
    return result;
}

Status CudaExecutor::prepare_linear_tactic(QuantizedLinearExecutionKind kind) {
    auto status = activate_device(impl_->device_ordinal);
    if (!status) return status;
    if (kind == QuantizedLinearExecutionKind::dense_f32_cublas) {
        return impl_->prepare_dense_f32_tensors();
    }
    if (kind == QuantizedLinearExecutionKind::q5q8_dp4a_hybrid) {
        return impl_->prepare_packed_dp4a_tensors();
    }
    return Status::ok();
}

std::uint64_t CudaExecutor::estimate_linear_tactic_preparation_bytes(
    QuantizedLinearExecutionKind kind) const noexcept {
    if (kind == QuantizedLinearExecutionKind::dense_f32_cublas) {
        if (impl_->dense_f32_prepared_bytes.load(std::memory_order_relaxed) != 0U) return 0U;
        return impl_->dense_f32_required_bytes();
    }
    if (kind == QuantizedLinearExecutionKind::q5q8_dp4a_hybrid) {
        if (impl_->packed_dp4a_prepared_bytes.load(std::memory_order_relaxed) != 0U) return 0U;
        return impl_->packed_dp4a_required_bytes();
    }
    return 0U;
}

Status CudaExecutor::trim_linear_tactics(
    std::span<const QuantizedLinearExecutionKind> required) {
    bool keep_dense = false;
    bool keep_dp4a = false;
    for (const auto tactic : required) {
        keep_dense = keep_dense || tactic == QuantizedLinearExecutionKind::dense_f32_cublas;
        keep_dp4a = keep_dp4a || tactic == QuantizedLinearExecutionKind::q5q8_dp4a_hybrid;
    }
    auto status = activate_device(impl_->device_ordinal);
    if (!status) return status;
    if (!keep_dense && impl_->dense_f32_prepared_bytes.load(std::memory_order_relaxed) != 0U) {
        std::lock_guard lock(impl_->dense_f32_mutex);
        status = cuda_status(cudaStreamSynchronize(impl_->stream),
                             "cudaStreamSynchronize(trim dense-f32-cublas)");
        if (!status) return status;
        impl_->dense_f32_tensors.clear();
        impl_->dense_f32_arena.reset();
        impl_->dense_f32_prepared_bytes.store(0U, std::memory_order_relaxed);
    }
    if (!keep_dp4a && impl_->packed_dp4a_prepared_bytes.load(std::memory_order_relaxed) != 0U) {
        std::lock_guard lock(impl_->packed_dp4a_mutex);
        status = cuda_status(cudaStreamSynchronize(impl_->stream),
                             "cudaStreamSynchronize(trim q5q8-dp4a-hybrid)");
        if (!status) return status;
        impl_->packed_dp4a_tensors.clear();
        impl_->packed_dp4a_arena.reset();
        impl_->packed_dp4a_activation_arena.reset();
        impl_->packed_dp4a_activation_capacity_bytes = 0U;
        impl_->packed_dp4a_activation_codes = nullptr;
        impl_->packed_dp4a_activation_scales = nullptr;
        impl_->packed_dp4a_prepared_bytes.store(0U, std::memory_order_relaxed);
    }
    return Status::ok();
}

Result<std::unique_ptr<CudaKvCache>> CudaExecutor::create_kv_cache(std::uint32_t page_tokens) const {
    auto status = activate_device(impl_->device_ordinal);
    if (!status) return status;
    if (page_tokens == 0U || page_tokens > 1024U) {
        return Status::invalid_argument("CUDA KV page tokens must be in [1,1024]");
    }
    const auto& config = impl_->model->config();
    auto cache = std::make_unique<CudaKvCache::Impl>();
    cache->device_ordinal = impl_->device_ordinal;
    cache->layer_count = config.layer_count;
    cache->kv_head_count = config.kv_head_count;
    cache->head_dimension = config.embedding_size / config.attention_head_count;
    cache->context_length = config.context_length;
    cache->page_tokens = page_tokens;
    cache->pool = impl_->pool_for(page_tokens);
    cache->stream = impl_->stream;
    cache->max_pages = (config.context_length + page_tokens - 1U) / page_tokens;
    cache->pending.assign(config.layer_count, 0U);
    if (cache->max_pages > std::numeric_limits<std::uint64_t>::max() / (2U * sizeof(float*))) {
        return Status::unsupported("CUDA KV page table size overflow");
    }
    status = cache->page_table.allocate(cache->max_pages * 2U * sizeof(float*));
    if (!status) return status;
    cache->key_pages = static_cast<float**>(cache->page_table.get());
    cache->value_pages = cache->key_pages + cache->max_pages;
    return std::unique_ptr<CudaKvCache>(new CudaKvCache(std::move(cache)));
}

Result<std::vector<float>> CudaExecutor::step_impl(
    TokenId token, CudaKvCache& cache, FinalOutput output, TokenId* greedy_token,
    std::span<const TokenId> target_tokens, std::vector<float>* target_logprobs,
    QuantizedLinearExecutionKind block_linear, QuantizedLinearExecutionKind output_linear,
    AttentionExecutionKind attention, VerificationTrace* trace) {
    if (block_linear != QuantizedLinearExecutionKind::baseline &&
        block_linear != QuantizedLinearExecutionKind::batch_reuse8 &&
        block_linear != QuantizedLinearExecutionKind::dense_f32_cublas) {
        return Status::unsupported("CUDA decode block-linear tactic is unsupported");
    }
    if (output_linear != QuantizedLinearExecutionKind::baseline &&
        output_linear != QuantizedLinearExecutionKind::batch_reuse8) {
        return Status::unsupported("CUDA decode output-projection tactic is unsupported");
    }
    if (attention != AttentionExecutionKind::baseline) {
        return Status::unsupported("CUDA decode attention tactic is not implemented");
    }
    ScopedProfileRange step_range("air.decode.step");
    auto status = activate_device(impl_->device_ordinal);
    if (!status) return status;
    if (!cache.impl_) return Status::invalid_argument("CUDA KV cache is not initialized");
    auto& kv = *cache.impl_;
    const auto& config = impl_->model->config();
    const std::uint32_t head_dimension = config.embedding_size / config.attention_head_count;
    if (kv.device_ordinal != impl_->device_ordinal || kv.layer_count != config.layer_count ||
        kv.kv_head_count != config.kv_head_count || kv.head_dimension != head_dimension ||
        kv.context_length != config.context_length) {
        return Status::invalid_argument("CUDA KV cache does not belong to this executor geometry/device");
    }
    if (token < 0 || static_cast<std::uint64_t>(token) >= config.vocabulary_size) {
        return Status::invalid_argument("input token is outside model vocabulary");
    }
    if (kv.token_count >= config.context_length) return Status::invalid_state("model context length exceeded");
    kv.rollback();
    status = kv.begin_transaction(1U);
    if (!status) return status;

    const auto fail = [&kv](Status failure) -> Result<std::vector<float>> { kv.rollback(); return failure; };
    const auto* embedding = impl_->tensor("token_embd.weight");
    if (!embedding) return fail(Status::internal_error("resident token embedding is missing"));
    status = impl_->load_row(*embedding, static_cast<std::uint64_t>(token), config.embedding_size, impl_->workspace.hidden);
    if (!status) return fail(status);

    const auto block_name = [](std::uint32_t layer, const char* suffix) {
        return "blk." + std::to_string(layer) + "." + suffix;
    };
    const std::uint64_t embedding_width = config.embedding_size;
    const std::uint64_t kv_width = static_cast<std::uint64_t>(config.kv_head_count) * head_dimension;
    const auto position = kv.token_count;
    status = impl_->record_trace(trace, position, -1, VerificationStage::embedding,
                                 impl_->workspace.hidden, embedding_width);
    if (!status) return fail(status);

    for (std::uint32_t layer = 0; layer < config.layer_count; ++layer) {
        ScopedProfileRange layer_range("air.decode.layer");
        const auto* attn_norm = impl_->tensor(block_name(layer, "attn_norm.weight"));
        if (!attn_norm) return fail(Status::internal_error("resident attention norm is missing"));
        status = impl_->rms_norm(impl_->workspace.hidden, *attn_norm, embedding_width,
                                 static_cast<float>(config.rms_norm_epsilon), impl_->workspace.normalized);
        if (!status) return fail(status);
        status = impl_->record_trace(trace, position, static_cast<std::int32_t>(layer),
                                     VerificationStage::attention_norm, impl_->workspace.normalized,
                                     embedding_width);
        if (!status) return fail(status);

        const auto* q_weight = impl_->tensor(block_name(layer, "attn_q.weight"));
        const auto* k_weight = impl_->tensor(block_name(layer, "attn_k.weight"));
        const auto* v_weight = impl_->tensor(block_name(layer, "attn_v.weight"));
        if (!q_weight || !k_weight || !v_weight) return fail(Status::internal_error("resident QKV weight is missing"));
        status = impl_->linear_single(*q_weight, impl_->workspace.normalized, impl_->workspace.q, block_linear); if (!status) return fail(status);
        status = impl_->linear_single(*k_weight, impl_->workspace.normalized, impl_->workspace.k, block_linear); if (!status) return fail(status);
        status = impl_->linear_single(*v_weight, impl_->workspace.normalized, impl_->workspace.v, block_linear); if (!status) return fail(status);
        status = impl_->add_optional_tensor(impl_->workspace.q, block_name(layer, "attn_q.bias"), embedding_width); if (!status) return fail(status);
        status = impl_->add_optional_tensor(impl_->workspace.k, block_name(layer, "attn_k.bias"), kv_width); if (!status) return fail(status);
        status = impl_->add_optional_tensor(impl_->workspace.v, block_name(layer, "attn_v.bias"), kv_width); if (!status) return fail(status);
        status = impl_->record_trace(trace, position, static_cast<std::int32_t>(layer), VerificationStage::q_projection,
                                     impl_->workspace.q, embedding_width); if (!status) return fail(status);
        status = impl_->record_trace(trace, position, static_cast<std::int32_t>(layer), VerificationStage::k_projection,
                                     impl_->workspace.k, kv_width); if (!status) return fail(status);
        status = impl_->record_trace(trace, position, static_cast<std::int32_t>(layer), VerificationStage::v_projection,
                                     impl_->workspace.v, kv_width); if (!status) return fail(status);
        status = impl_->apply_rope(impl_->workspace.q, config.attention_head_count, head_dimension,
                                   config.rope_dimension_count, position, static_cast<float>(config.rope_frequency_base));
        if (!status) return fail(status);
        status = impl_->apply_rope(impl_->workspace.k, config.kv_head_count, head_dimension,
                                   config.rope_dimension_count, position, static_cast<float>(config.rope_frequency_base));
        if (!status) return fail(status);
        status = impl_->record_trace(trace, position, static_cast<std::int32_t>(layer), VerificationStage::q_rope,
                                     impl_->workspace.q, embedding_width); if (!status) return fail(status);
        status = impl_->record_trace(trace, position, static_cast<std::int32_t>(layer), VerificationStage::k_rope,
                                     impl_->workspace.k, kv_width); if (!status) return fail(status);
        status = impl_->attention(kv, layer, config.attention_head_count, config.kv_head_count, head_dimension);
        if (!status) return fail(status);
        status = impl_->record_trace(trace, position, static_cast<std::int32_t>(layer), VerificationStage::attention,
                                     impl_->workspace.attended, embedding_width); if (!status) return fail(status);

        const auto* attn_output = impl_->tensor(block_name(layer, "attn_output.weight"));
        if (!attn_output) return fail(Status::internal_error("resident attention output weight is missing"));
        status = impl_->linear_single(*attn_output, impl_->workspace.attended, impl_->workspace.projection, block_linear); if (!status) return fail(status);
        status = impl_->record_trace(trace, position, static_cast<std::int32_t>(layer), VerificationStage::attention_projection,
                                     impl_->workspace.projection, embedding_width); if (!status) return fail(status);
        status = impl_->add_vector(impl_->workspace.hidden, impl_->workspace.projection, embedding_width); if (!status) return fail(status);
        status = impl_->record_trace(trace, position, static_cast<std::int32_t>(layer), VerificationStage::attention_residual,
                                     impl_->workspace.hidden, embedding_width); if (!status) return fail(status);

        const auto* ffn_norm = impl_->tensor(block_name(layer, "ffn_norm.weight"));
        const auto* gate_weight = impl_->tensor(block_name(layer, "ffn_gate.weight"));
        const auto* up_weight = impl_->tensor(block_name(layer, "ffn_up.weight"));
        const auto* down_weight = impl_->tensor(block_name(layer, "ffn_down.weight"));
        if (!ffn_norm || !gate_weight || !up_weight || !down_weight) return fail(Status::internal_error("resident FFN tensor is missing"));
        status = impl_->rms_norm(impl_->workspace.hidden, *ffn_norm, embedding_width,
                                 static_cast<float>(config.rms_norm_epsilon), impl_->workspace.normalized); if (!status) return fail(status);
        status = impl_->record_trace(trace, position, static_cast<std::int32_t>(layer), VerificationStage::ffn_norm,
                                     impl_->workspace.normalized, embedding_width); if (!status) return fail(status);
        status = impl_->linear_single(*gate_weight, impl_->workspace.normalized, impl_->workspace.gate, block_linear); if (!status) return fail(status);
        status = impl_->linear_single(*up_weight, impl_->workspace.normalized, impl_->workspace.up, block_linear); if (!status) return fail(status);
        status = impl_->record_trace(trace, position, static_cast<std::int32_t>(layer), VerificationStage::ffn_gate,
                                     impl_->workspace.gate, config.feed_forward_size); if (!status) return fail(status);
        status = impl_->record_trace(trace, position, static_cast<std::int32_t>(layer), VerificationStage::ffn_up,
                                     impl_->workspace.up, config.feed_forward_size); if (!status) return fail(status);
        constexpr unsigned int threads = 256U;
        silu_mul_kernel<<<blocks_for(config.feed_forward_size), threads, 0, impl_->stream>>>(
            impl_->workspace.gate, impl_->workspace.up, config.feed_forward_size);
        status = launch_status("silu_mul_kernel"); if (!status) return fail(status);
        status = impl_->record_trace(trace, position, static_cast<std::int32_t>(layer), VerificationStage::ffn_activation,
                                     impl_->workspace.gate, config.feed_forward_size); if (!status) return fail(status);
        status = impl_->linear_single(*down_weight, impl_->workspace.gate, impl_->workspace.projection, block_linear); if (!status) return fail(status);
        status = impl_->record_trace(trace, position, static_cast<std::int32_t>(layer), VerificationStage::ffn_down,
                                     impl_->workspace.projection, embedding_width); if (!status) return fail(status);
        status = impl_->add_vector(impl_->workspace.hidden, impl_->workspace.projection, embedding_width); if (!status) return fail(status);
        status = impl_->record_trace(trace, position, static_cast<std::int32_t>(layer), VerificationStage::ffn_residual,
                                     impl_->workspace.hidden, embedding_width); if (!status) return fail(status);
        status = impl_->append_kv(kv, layer, 1U); if (!status) return fail(status);
    }

    const auto* output_norm = impl_->tensor("output_norm.weight");
    if (!output_norm) return fail(Status::internal_error("resident output norm is missing"));
    status = impl_->rms_norm(impl_->workspace.hidden, *output_norm, embedding_width,
                             static_cast<float>(config.rms_norm_epsilon), impl_->workspace.normalized);
    if (!status) return fail(status);
    status = impl_->record_trace(trace, position, -1, VerificationStage::final_norm,
                                 impl_->workspace.normalized, embedding_width);
    if (!status) return fail(status);
    const auto* output_weight = impl_->tensor(impl_->model->find_tensor("output.weight") ? "output.weight" : "token_embd.weight");
    if (!output_weight) return fail(Status::internal_error("resident output weight is missing"));
    {
        ScopedProfileRange output_range("air.decode.output_projection");
        status = impl_->linear_single(*output_weight, impl_->workspace.normalized, impl_->workspace.logits, output_linear);
    }
    if (!status) return fail(status);
    status = impl_->add_optional_tensor(impl_->workspace.logits, "output.bias", config.vocabulary_size); if (!status) return fail(status);
    status = impl_->record_trace(trace, position, -1, VerificationStage::logits,
                                 impl_->workspace.logits, config.vocabulary_size);
    if (!status) return fail(status);

    std::vector<float> logits;
    if (output == FinalOutput::logits) {
        auto copied = impl_->read_logits_host();
        if (!copied) return fail(copied.status());
        logits = std::move(copied).value();
    } else if (output == FinalOutput::greedy) {
        if (!greedy_token) return fail(Status::invalid_argument("CUDA greedy step requires an output token"));
        auto selected = impl_->select_greedy_device();
        if (!selected) return fail(selected.status());
        *greedy_token = selected.value();
    } else if (output == FinalOutput::target_logprobs) {
        if (!target_logprobs) return fail(Status::invalid_argument("CUDA target-logprob step requires an output vector"));
        auto scores = impl_->read_target_logprobs_device(target_tokens);
        if (!scores) return fail(scores.status());
        *target_logprobs = std::move(scores).value();
    } else {
        return fail(Status::invalid_argument("CUDA decode does not support discard output mode"));
    }
    status = impl_->commit_kv(kv);
    if (!status) return fail(status);
    return logits;
}

Result<std::vector<float>> CudaExecutor::step(
    TokenId token, CudaKvCache& cache, QuantizedLinearExecutionKind linear, AttentionExecutionKind attention) {
    return step_impl(token, cache, FinalOutput::logits, nullptr, {}, nullptr, linear,
                     QuantizedLinearExecutionKind::baseline, attention, nullptr);
}

Result<std::vector<float>> CudaExecutor::step(
    TokenId token, CudaKvCache& cache, QuantizedLinearExecutionKind block_linear,
    QuantizedLinearExecutionKind output_linear, AttentionExecutionKind attention) {
    return step_impl(token, cache, FinalOutput::logits, nullptr, {}, nullptr, block_linear,
                     output_linear, attention, nullptr);
}

Result<std::vector<float>> CudaExecutor::step_verified(
    TokenId token, CudaKvCache& cache, VerificationTrace& trace,
    QuantizedLinearExecutionKind block_linear, QuantizedLinearExecutionKind output_linear,
    AttentionExecutionKind attention) {
    return step_impl(token, cache, FinalOutput::logits, nullptr, {}, nullptr, block_linear,
                     output_linear, attention, &trace);
}

Result<TokenId> CudaExecutor::step_greedy(
    TokenId token, CudaKvCache& cache, QuantizedLinearExecutionKind linear, AttentionExecutionKind attention) {
    TokenId selected = -1;
    auto result = step_impl(token, cache, FinalOutput::greedy, &selected, {}, nullptr, linear,
                            QuantizedLinearExecutionKind::baseline, attention, nullptr);
    if (!result) return result.status();
    return selected;
}

Result<TokenId> CudaExecutor::step_greedy(
    TokenId token, CudaKvCache& cache, QuantizedLinearExecutionKind block_linear,
    QuantizedLinearExecutionKind output_linear, AttentionExecutionKind attention) {
    TokenId selected = -1;
    auto result = step_impl(token, cache, FinalOutput::greedy, &selected, {}, nullptr, block_linear,
                            output_linear, attention, nullptr);
    if (!result) return result.status();
    return selected;
}

Result<std::vector<float>> CudaExecutor::step_target_logprobs(
    TokenId token, std::span<const TokenId> target_tokens, CudaKvCache& cache,
    QuantizedLinearExecutionKind block_linear, QuantizedLinearExecutionKind output_linear,
    AttentionExecutionKind attention) {
    std::vector<float> scores;
    auto result = step_impl(token, cache, FinalOutput::target_logprobs, nullptr, target_tokens, &scores,
                            block_linear, output_linear, attention, nullptr);
    if (!result) return result.status();
    return scores;
}

Result<std::vector<TokenId>> CudaExecutor::step_greedy_batch(
    std::span<const TokenId> tokens, std::span<CudaKvCache*> caches,
    QuantizedLinearExecutionKind block_linear, QuantizedLinearExecutionKind output_linear,
    AttentionExecutionKind attention) {
    ScopedProfileRange batch_range("air.decode.batch");
    if (tokens.empty() || tokens.size() != caches.size()) {
        return Status::invalid_argument("CUDA decode batch requires matching non-empty token/cache spans");
    }
    if (tokens.size() > kMaxNativeDecodeBatch) {
        return Status::unsupported("CUDA decode batch exceeds native width");
    }
    if (block_linear != QuantizedLinearExecutionKind::baseline &&
        block_linear != QuantizedLinearExecutionKind::batch_reuse8 &&
        block_linear != QuantizedLinearExecutionKind::dense_f32_cublas) {
        return Status::unsupported("CUDA native decode block linear tactic is unsupported");
    }
    if (output_linear != QuantizedLinearExecutionKind::baseline &&
        output_linear != QuantizedLinearExecutionKind::batch_reuse8) {
        return Status::unsupported("CUDA native decode output-projection tactic is unsupported");
    }
    if (attention != AttentionExecutionKind::baseline) {
        return Status::unsupported("CUDA native decode attention tactic is not implemented");
    }
    if (tokens.size() == 1U) {
        auto token = step_greedy(tokens.front(), *caches.front(), block_linear, output_linear, attention);
        if (!token) return token.status();
        return std::vector<TokenId>{token.value()};
    }

    auto status = activate_device(impl_->device_ordinal);
    if (!status) return status;
    const auto& config = impl_->model->config();
    const std::uint32_t count = static_cast<std::uint32_t>(tokens.size());
    const std::uint32_t head_dimension = config.embedding_size / config.attention_head_count;
    const std::uint64_t embedding_width = config.embedding_size;
    const std::uint64_t kv_width = static_cast<std::uint64_t>(config.kv_head_count) * head_dimension;

    std::vector<CudaKvCache::Impl*> kvs;
    kvs.reserve(caches.size());
    for (std::size_t index = 0; index < caches.size(); ++index) {
        auto* cache = caches[index];
        if (!cache || !cache->impl_) return Status::invalid_argument("CUDA decode batch contains an uninitialized cache");
        auto& kv = *cache->impl_;
        if (kv.device_ordinal != impl_->device_ordinal || kv.layer_count != config.layer_count ||
            kv.kv_head_count != config.kv_head_count || kv.head_dimension != head_dimension ||
            kv.context_length != config.context_length) {
            return Status::invalid_argument("CUDA decode batch cache does not match executor geometry/device");
        }
        if (tokens[index] < 0 || static_cast<std::uint64_t>(tokens[index]) >= config.vocabulary_size) {
            return Status::invalid_argument("CUDA decode batch token is outside model vocabulary");
        }
        if (kv.token_count >= config.context_length) {
            return Status::invalid_state("CUDA decode batch sequence exceeded model context length");
        }
        kv.rollback();
        kvs.push_back(&kv);
    }

    const auto rollback_all = [&]() noexcept {
        for (auto* kv : kvs) kv->rollback();
    };
    const auto fail = [&](Status failure) -> Result<std::vector<TokenId>> {
        rollback_all();
        return failure;
    };
    for (auto* kv : kvs) {
        status = kv->begin_transaction(1U);
        if (!status) return fail(status);
    }

    const auto* embedding = impl_->tensor("token_embd.weight");
    if (!embedding) return fail(Status::internal_error("resident token embedding is missing"));
    status = impl_->load_rows(*embedding, tokens, embedding_width, impl_->workspace.hidden);
    if (!status) return fail(status);

    const auto block_name = [](std::uint32_t layer, const char* suffix) {
        return "blk." + std::to_string(layer) + "." + suffix;
    };

    for (std::uint32_t layer = 0; layer < config.layer_count; ++layer) {
        ScopedProfileRange layer_range("air.decode.batch.layer");
        const auto* attn_norm = impl_->tensor(block_name(layer, "attn_norm.weight"));
        const auto* q_weight = impl_->tensor(block_name(layer, "attn_q.weight"));
        const auto* k_weight = impl_->tensor(block_name(layer, "attn_k.weight"));
        const auto* v_weight = impl_->tensor(block_name(layer, "attn_v.weight"));
        const auto* attn_output = impl_->tensor(block_name(layer, "attn_output.weight"));
        const auto* ffn_norm = impl_->tensor(block_name(layer, "ffn_norm.weight"));
        const auto* gate_weight = impl_->tensor(block_name(layer, "ffn_gate.weight"));
        const auto* up_weight = impl_->tensor(block_name(layer, "ffn_up.weight"));
        const auto* down_weight = impl_->tensor(block_name(layer, "ffn_down.weight"));
        if (!attn_norm || !q_weight || !k_weight || !v_weight || !attn_output ||
            !ffn_norm || !gate_weight || !up_weight || !down_weight) {
            return fail(Status::internal_error("resident Qwen2 layer tensor is missing"));
        }

        status = impl_->rms_norm_batch(impl_->workspace.hidden, *attn_norm, embedding_width,
                                       static_cast<float>(config.rms_norm_epsilon),
                                       impl_->workspace.normalized, count); if (!status) return fail(status);
        status = impl_->matmul(*q_weight, impl_->workspace.normalized, impl_->workspace.q, count, block_linear); if (!status) return fail(status);
        status = impl_->matmul(*k_weight, impl_->workspace.normalized, impl_->workspace.k, count, block_linear); if (!status) return fail(status);
        status = impl_->matmul(*v_weight, impl_->workspace.normalized, impl_->workspace.v, count, block_linear); if (!status) return fail(status);
        status = impl_->add_optional_tensor_batch(impl_->workspace.q, block_name(layer, "attn_q.bias"), embedding_width, count); if (!status) return fail(status);
        status = impl_->add_optional_tensor_batch(impl_->workspace.k, block_name(layer, "attn_k.bias"), kv_width, count); if (!status) return fail(status);
        status = impl_->add_optional_tensor_batch(impl_->workspace.v, block_name(layer, "attn_v.bias"), kv_width, count); if (!status) return fail(status);

        for (std::uint32_t item = 0; item < count; ++item) {
            const auto position = kvs[item]->token_count;
            float* q = impl_->workspace.q + static_cast<std::uint64_t>(item) * embedding_width;
            float* k = impl_->workspace.k + static_cast<std::uint64_t>(item) * kv_width;
            float* v = impl_->workspace.v + static_cast<std::uint64_t>(item) * kv_width;
            float* attended = impl_->workspace.attended + static_cast<std::uint64_t>(item) * embedding_width;
            status = impl_->apply_rope(q, config.attention_head_count, head_dimension,
                                       config.rope_dimension_count, position,
                                       static_cast<float>(config.rope_frequency_base)); if (!status) return fail(status);
            status = impl_->apply_rope(k, config.kv_head_count, head_dimension,
                                       config.rope_dimension_count, position,
                                       static_cast<float>(config.rope_frequency_base)); if (!status) return fail(status);
            status = impl_->attention_from(*kvs[item], layer, config.attention_head_count,
                                           config.kv_head_count, head_dimension, q, k, v, attended);
            if (!status) return fail(status);
        }

        status = impl_->matmul(*attn_output, impl_->workspace.attended, impl_->workspace.projection, count, block_linear); if (!status) return fail(status);
        status = impl_->add_vector_batch(impl_->workspace.hidden, impl_->workspace.projection, embedding_width, count); if (!status) return fail(status);
        status = impl_->rms_norm_batch(impl_->workspace.hidden, *ffn_norm, embedding_width,
                                       static_cast<float>(config.rms_norm_epsilon),
                                       impl_->workspace.normalized, count); if (!status) return fail(status);
        status = impl_->matmul(*gate_weight, impl_->workspace.normalized, impl_->workspace.gate, count, block_linear); if (!status) return fail(status);
        status = impl_->matmul(*up_weight, impl_->workspace.normalized, impl_->workspace.up, count, block_linear); if (!status) return fail(status);
        constexpr unsigned int threads = 256U;
        const auto ffn_total = static_cast<std::uint64_t>(count) * config.feed_forward_size;
        silu_mul_kernel<<<blocks_for(ffn_total), threads, 0, impl_->stream>>>(
            impl_->workspace.gate, impl_->workspace.up, ffn_total);
        status = launch_status("silu_mul_kernel(decode batch)"); if (!status) return fail(status);
        status = impl_->matmul(*down_weight, impl_->workspace.gate, impl_->workspace.projection, count, block_linear); if (!status) return fail(status);
        status = impl_->add_vector_batch(impl_->workspace.hidden, impl_->workspace.projection, embedding_width, count); if (!status) return fail(status);

        for (std::uint32_t item = 0; item < count; ++item) {
            const float* k = impl_->workspace.k + static_cast<std::uint64_t>(item) * kv_width;
            const float* v = impl_->workspace.v + static_cast<std::uint64_t>(item) * kv_width;
            status = impl_->append_kv_from(*kvs[item], layer, 1U, k, v);
            if (!status) return fail(status);
        }
    }

    const auto* output_norm = impl_->tensor("output_norm.weight");
    if (!output_norm) return fail(Status::internal_error("resident output norm is missing"));
    status = impl_->rms_norm_batch(impl_->workspace.hidden, *output_norm, embedding_width,
                                   static_cast<float>(config.rms_norm_epsilon),
                                   impl_->workspace.normalized, count); if (!status) return fail(status);
    const auto* output_weight = impl_->tensor(impl_->model->find_tensor("output.weight") ? "output.weight" : "token_embd.weight");
    if (!output_weight) return fail(Status::internal_error("resident output weight is missing"));
    {
        ScopedProfileRange output_range("air.decode.batch.output_projection");
        status = impl_->matmul(*output_weight, impl_->workspace.normalized, impl_->workspace.logits, count, output_linear);
    }
    if (!status) return fail(status);
    status = impl_->add_optional_tensor_batch(impl_->workspace.logits, "output.bias", config.vocabulary_size, count); if (!status) return fail(status);

    std::vector<TokenId> selected;
    selected.reserve(count);
    for (std::uint32_t item = 0; item < count; ++item) {
        const float* logits = impl_->workspace.logits + static_cast<std::uint64_t>(item) * config.vocabulary_size;
        auto token = impl_->select_greedy_device_from(logits);
        if (!token) return fail(token.status());
        selected.push_back(token.value());
    }
    // Validate every per-sequence transaction before mutating any token_count so
    // one malformed batch member cannot create a partially committed batch.
    for (auto* kv : kvs) {
        status = impl_->validate_kv_commit(*kv);
        if (!status) return fail(status);
    }
    for (auto* kv : kvs) {
        status = impl_->commit_kv(*kv);
        if (!status) return fail(status);
    }
    impl_->decode_batch_calls.fetch_add(1U, std::memory_order_relaxed);
    impl_->decode_batched_sequences.fetch_add(count, std::memory_order_relaxed);
    return selected;
}

Result<std::vector<float>> CudaExecutor::prefill_impl(
    std::span<const TokenId> tokens, CudaKvCache& cache, FinalOutput output,
    TokenId* greedy_token, std::span<const TokenId> target_tokens,
    std::vector<float>* target_logprobs, QuantizedLinearExecutionKind linear,
    AttentionExecutionKind attention) {
    ScopedProfileRange prefill_range("air.prefill");
    if (tokens.empty()) return Status::invalid_argument("prefill requires at least one token");
    if (!cache.impl_) return Status::invalid_argument("CUDA KV cache is not initialized");
    auto status = activate_device(impl_->device_ordinal);
    if (!status) return status;
    auto& kv = *cache.impl_;
    const auto& config = impl_->model->config();
    const std::uint32_t head_dimension = config.embedding_size / config.attention_head_count;
    if (kv.device_ordinal != impl_->device_ordinal || kv.layer_count != config.layer_count ||
        kv.kv_head_count != config.kv_head_count || kv.head_dimension != head_dimension ||
        kv.context_length != config.context_length) {
        return Status::invalid_argument("CUDA KV cache does not belong to this executor geometry/device");
    }
    for (const auto token : tokens) {
        if (token < 0 || static_cast<std::uint64_t>(token) >= config.vocabulary_size) {
            return Status::invalid_argument("prefill token is outside model vocabulary");
        }
    }

    std::vector<float> final_logits;
    std::size_t offset = 0U;
    while (offset < tokens.size()) {
        ScopedProfileRange chunk_range("air.prefill.chunk");
        const auto count = static_cast<std::uint32_t>(std::min<std::size_t>(
            kMaxNativePrefillBatch, tokens.size() - offset));
        const auto chunk = tokens.subspan(offset, count);
        kv.rollback();
        status = kv.begin_transaction(count);
        if (!status) return status;
        const auto fail = [&kv](Status failure) -> Result<std::vector<float>> { kv.rollback(); return failure; };

        const auto* embedding = impl_->tensor("token_embd.weight");
        if (!embedding) return fail(Status::internal_error("resident token embedding is missing"));
        status = impl_->load_rows(*embedding, chunk, config.embedding_size, impl_->workspace.hidden);
        if (!status) return fail(status);

        const auto block_name = [](std::uint32_t layer, const char* suffix) {
            return "blk." + std::to_string(layer) + "." + suffix;
        };
        const std::uint64_t embedding_width = config.embedding_size;
        const std::uint64_t kv_width = static_cast<std::uint64_t>(config.kv_head_count) * head_dimension;
        const auto position_start = kv.token_count;

        for (std::uint32_t layer = 0; layer < config.layer_count; ++layer) {
            ScopedProfileRange layer_range("air.prefill.layer");
            const auto* attn_norm = impl_->tensor(block_name(layer, "attn_norm.weight"));
            const auto* q_weight = impl_->tensor(block_name(layer, "attn_q.weight"));
            const auto* k_weight = impl_->tensor(block_name(layer, "attn_k.weight"));
            const auto* v_weight = impl_->tensor(block_name(layer, "attn_v.weight"));
            const auto* attn_output = impl_->tensor(block_name(layer, "attn_output.weight"));
            const auto* ffn_norm = impl_->tensor(block_name(layer, "ffn_norm.weight"));
            const auto* gate_weight = impl_->tensor(block_name(layer, "ffn_gate.weight"));
            const auto* up_weight = impl_->tensor(block_name(layer, "ffn_up.weight"));
            const auto* down_weight = impl_->tensor(block_name(layer, "ffn_down.weight"));
            if (!attn_norm || !q_weight || !k_weight || !v_weight || !attn_output ||
                !ffn_norm || !gate_weight || !up_weight || !down_weight) {
                return fail(Status::internal_error("resident Qwen2 layer tensor is missing"));
            }

            status = impl_->rms_norm_batch(impl_->workspace.hidden, *attn_norm, embedding_width,
                                           static_cast<float>(config.rms_norm_epsilon),
                                           impl_->workspace.normalized, count); if (!status) return fail(status);
            status = impl_->matmul(*q_weight, impl_->workspace.normalized, impl_->workspace.q, count, linear); if (!status) return fail(status);
            status = impl_->matmul(*k_weight, impl_->workspace.normalized, impl_->workspace.k, count, linear); if (!status) return fail(status);
            status = impl_->matmul(*v_weight, impl_->workspace.normalized, impl_->workspace.v, count, linear); if (!status) return fail(status);
            status = impl_->add_optional_tensor_batch(impl_->workspace.q, block_name(layer, "attn_q.bias"), embedding_width, count); if (!status) return fail(status);
            status = impl_->add_optional_tensor_batch(impl_->workspace.k, block_name(layer, "attn_k.bias"), kv_width, count); if (!status) return fail(status);
            status = impl_->add_optional_tensor_batch(impl_->workspace.v, block_name(layer, "attn_v.bias"), kv_width, count); if (!status) return fail(status);
            status = impl_->apply_rope_batch(impl_->workspace.q, count, config.attention_head_count, head_dimension,
                                             config.rope_dimension_count, position_start,
                                             static_cast<float>(config.rope_frequency_base)); if (!status) return fail(status);
            status = impl_->apply_rope_batch(impl_->workspace.k, count, config.kv_head_count, head_dimension,
                                             config.rope_dimension_count, position_start,
                                             static_cast<float>(config.rope_frequency_base)); if (!status) return fail(status);
            status = impl_->attention_batch(kv, count, layer, config.attention_head_count,
                                            config.kv_head_count, head_dimension, attention); if (!status) return fail(status);
            status = impl_->matmul(*attn_output, impl_->workspace.attended, impl_->workspace.projection, count, linear); if (!status) return fail(status);
            status = impl_->add_vector_batch(impl_->workspace.hidden, impl_->workspace.projection, embedding_width, count); if (!status) return fail(status);
            status = impl_->rms_norm_batch(impl_->workspace.hidden, *ffn_norm, embedding_width,
                                           static_cast<float>(config.rms_norm_epsilon),
                                           impl_->workspace.normalized, count); if (!status) return fail(status);
            status = impl_->matmul(*gate_weight, impl_->workspace.normalized, impl_->workspace.gate, count, linear); if (!status) return fail(status);
            status = impl_->matmul(*up_weight, impl_->workspace.normalized, impl_->workspace.up, count, linear); if (!status) return fail(status);
            constexpr unsigned int threads = 256U;
            const auto ffn_total = static_cast<std::uint64_t>(count) * config.feed_forward_size;
            silu_mul_kernel<<<blocks_for(ffn_total), threads, 0, impl_->stream>>>(
                impl_->workspace.gate, impl_->workspace.up, ffn_total);
            status = launch_status("silu_mul_kernel(batch)"); if (!status) return fail(status);
            status = impl_->matmul(*down_weight, impl_->workspace.gate, impl_->workspace.projection, count, linear); if (!status) return fail(status);
            status = impl_->add_vector_batch(impl_->workspace.hidden, impl_->workspace.projection, embedding_width, count); if (!status) return fail(status);
            status = impl_->append_kv(kv, layer, count); if (!status) return fail(status);
        }

        const bool final_chunk = offset + count == tokens.size();
        if (final_chunk && output != FinalOutput::discard) {
            const auto* output_norm = impl_->tensor("output_norm.weight");
            if (!output_norm) return fail(Status::internal_error("resident output norm is missing"));
            const float* final_hidden = impl_->workspace.hidden +
                static_cast<std::uint64_t>(count - 1U) * embedding_width;
            status = impl_->rms_norm(final_hidden, *output_norm, embedding_width,
                                     static_cast<float>(config.rms_norm_epsilon), impl_->workspace.normalized);
            if (!status) return fail(status);
            const auto* output_weight = impl_->tensor(impl_->model->find_tensor("output.weight") ? "output.weight" : "token_embd.weight");
            if (!output_weight) return fail(Status::internal_error("resident output weight is missing"));
            {
                ScopedProfileRange output_range("air.prefill.output_projection");
                status = impl_->matvec(*output_weight, impl_->workspace.normalized, impl_->workspace.logits);
            }
            if (!status) return fail(status);
            status = impl_->add_optional_tensor(impl_->workspace.logits, "output.bias", config.vocabulary_size); if (!status) return fail(status);

            if (output == FinalOutput::logits) {
                auto copied = impl_->read_logits_host();
                if (!copied) return fail(copied.status());
                final_logits = std::move(copied).value();
            } else if (output == FinalOutput::greedy) {
                if (!greedy_token) return fail(Status::invalid_argument("CUDA greedy prefill requires an output token"));
                auto selected = impl_->select_greedy_device();
                if (!selected) return fail(selected.status());
                *greedy_token = selected.value();
            } else if (output == FinalOutput::target_logprobs) {
                if (!target_logprobs) return fail(Status::invalid_argument("CUDA target-logprob prefill requires an output vector"));
                auto scores = impl_->read_target_logprobs_device(target_tokens);
                if (!scores) return fail(scores.status());
                *target_logprobs = std::move(scores).value();
            } else {
                return fail(Status::invalid_argument("CUDA prefill final output mode is invalid"));
            }
        } else {
            status = impl_->synchronize_outputless_prefill();
            if (!status) return fail(status);
        }
        status = impl_->commit_kv(kv);
        if (!status) return fail(status);
        offset += count;
    }
    return final_logits;
}

Result<std::vector<float>> CudaExecutor::prefill(
    std::span<const TokenId> tokens, CudaKvCache& cache,
    QuantizedLinearExecutionKind linear, AttentionExecutionKind attention) {
    return prefill_impl(tokens, cache, FinalOutput::logits, nullptr, {}, nullptr, linear, attention);
}

Status CudaExecutor::prefill_discard(
    std::span<const TokenId> tokens, CudaKvCache& cache,
    QuantizedLinearExecutionKind linear, AttentionExecutionKind attention) {
    auto result = prefill_impl(tokens, cache, FinalOutput::discard, nullptr, {}, nullptr, linear, attention);
    return result ? Status::ok() : result.status();
}

Result<TokenId> CudaExecutor::prefill_greedy(
    std::span<const TokenId> tokens, CudaKvCache& cache,
    QuantizedLinearExecutionKind linear, AttentionExecutionKind attention) {
    TokenId selected = -1;
    auto result = prefill_impl(tokens, cache, FinalOutput::greedy, &selected, {}, nullptr, linear, attention);
    if (!result) return result.status();
    return selected;
}

Result<std::vector<float>> CudaExecutor::prefill_target_logprobs(
    std::span<const TokenId> tokens, std::span<const TokenId> target_tokens,
    CudaKvCache& cache, QuantizedLinearExecutionKind linear, AttentionExecutionKind attention) {
    std::vector<float> scores;
    auto result = prefill_impl(tokens, cache, FinalOutput::target_logprobs, nullptr, target_tokens, &scores,
                               linear, attention);
    if (!result) return result.status();
    return scores;
}


Result<CudaPrefillBatchExecution> CudaExecutor::prefill_batch(
    std::span<const CudaPrefillBatchItem> items,
    QuantizedLinearExecutionKind linear,
    AttentionExecutionKind attention) {
    ScopedProfileRange batch_range("air.prefill.multi_sequence");
    if (items.size() < 2U) {
        return Status::invalid_argument("CUDA multi-sequence prefill requires at least two sequences");
    }
    if (items.size() > kMaxNativePrefillBatch) {
        return Status::unsupported("CUDA multi-sequence prefill sequence width is unsupported");
    }

    auto status = activate_device(impl_->device_ordinal);
    if (!status) return status;
    const auto& config = impl_->model->config();
    const std::uint32_t head_dimension =
        config.embedding_size / config.attention_head_count;
    const std::uint64_t embedding_width = config.embedding_size;
    const std::uint64_t kv_width =
        static_cast<std::uint64_t>(config.kv_head_count) * head_dimension;

    std::vector<std::size_t> offsets(items.size(), 0U);
    CudaPrefillBatchExecution result;
    result.items.resize(items.size());

    for (const auto& item : items) {
        if (!item.cache || !item.cache->impl_) {
            return Status::invalid_argument("CUDA prefill batch contains an uninitialized KV cache");
        }
        if (item.tokens.empty()) {
            return Status::invalid_argument("CUDA prefill batch contains an empty token slice");
        }
        auto& kv = *item.cache->impl_;
        if (kv.device_ordinal != impl_->device_ordinal ||
            kv.layer_count != config.layer_count ||
            kv.kv_head_count != config.kv_head_count ||
            kv.head_dimension != head_dimension ||
            kv.context_length != config.context_length) {
            return Status::invalid_argument(
                "CUDA prefill batch contains a KV cache with incompatible geometry/device");
        }
        if (kv.token_count + item.tokens.size() > config.context_length) {
            return Status::invalid_state("CUDA prefill batch exceeds model context length");
        }
        for (const auto token : item.tokens) {
            if (token < 0 || static_cast<std::uint64_t>(token) >= config.vocabulary_size) {
                return Status::invalid_argument(
                    "CUDA prefill batch contains a token outside model vocabulary");
            }
        }
    }

    const auto block_name = [](std::uint32_t layer, const char* suffix) {
        return "blk." + std::to_string(layer) + "." + suffix;
    };

    struct RoundPart {
        std::size_t item_index{0};
        std::uint32_t count{0};
        std::uint32_t flat_offset{0};
        std::uint64_t position_start{0};
    };

    while (true) {
        std::vector<std::size_t> active_indices;
        active_indices.reserve(items.size());
        for (std::size_t i = 0; i < items.size(); ++i) {
            if (offsets[i] < items[i].tokens.size()) active_indices.push_back(i);
        }
        if (active_indices.empty()) break;

        const auto fair_cap = std::max<std::uint32_t>(
            1U, kMaxNativePrefillBatch /
                    static_cast<std::uint32_t>(active_indices.size()));

        std::vector<RoundPart> parts;
        std::vector<TokenId> flattened;
        parts.reserve(active_indices.size());
        flattened.reserve(kMaxNativePrefillBatch);

        std::uint32_t flat_offset = 0U;
        for (const auto index : active_indices) {
            const auto remaining = items[index].tokens.size() - offsets[index];
            const auto count = static_cast<std::uint32_t>(
                std::min<std::size_t>(remaining, fair_cap));
            if (count == 0U) continue;
            auto& kv = *items[index].cache->impl_;
            parts.push_back(RoundPart{index, count, flat_offset, kv.token_count});
            const auto begin = items[index].tokens.begin() +
                static_cast<std::ptrdiff_t>(offsets[index]);
            flattened.insert(flattened.end(), begin,
                             begin + static_cast<std::ptrdiff_t>(count));
            flat_offset += count;
        }

        if (parts.size() < 2U && items.size() >= 2U && result.physical_batches == 0U) {
            return Status::unsupported(
                "CUDA multi-sequence prefill could not form a physical multi-sequence round");
        }
        if (flattened.empty() || flattened.size() > kMaxNativePrefillBatch) {
            return Status::internal_error("CUDA prefill batch round packing is invalid");
        }

        std::size_t begun = 0U;
        for (; begun < parts.size(); ++begun) {
            auto& kv = *items[parts[begun].item_index].cache->impl_;
            kv.rollback();
            status = kv.begin_transaction(parts[begun].count);
            if (!status) break;
        }
        if (!status) {
            for (std::size_t i = 0; i < begun; ++i) {
                items[parts[i].item_index].cache->impl_->rollback();
            }
            return status;
        }
        const auto fail_round = [&](Status failure)
            -> Result<CudaPrefillBatchExecution> {
            for (const auto& part : parts) {
                items[part.item_index].cache->impl_->rollback();
            }
            return failure;
        };

        const auto* embedding = impl_->tensor("token_embd.weight");
        if (!embedding) {
            return fail_round(
                Status::internal_error("resident token embedding is missing"));
        }
        status = impl_->load_rows(*embedding, flattened, embedding_width,
                                  impl_->workspace.hidden);
        if (!status) return fail_round(status);

        const auto total = static_cast<std::uint32_t>(flattened.size());

        for (std::uint32_t layer = 0; layer < config.layer_count; ++layer) {
            ScopedProfileRange layer_range("air.prefill.multi_sequence.layer");
            const auto* attn_norm =
                impl_->tensor(block_name(layer, "attn_norm.weight"));
            const auto* q_weight =
                impl_->tensor(block_name(layer, "attn_q.weight"));
            const auto* k_weight =
                impl_->tensor(block_name(layer, "attn_k.weight"));
            const auto* v_weight =
                impl_->tensor(block_name(layer, "attn_v.weight"));
            const auto* attn_output =
                impl_->tensor(block_name(layer, "attn_output.weight"));
            const auto* ffn_norm =
                impl_->tensor(block_name(layer, "ffn_norm.weight"));
            const auto* gate_weight =
                impl_->tensor(block_name(layer, "ffn_gate.weight"));
            const auto* up_weight =
                impl_->tensor(block_name(layer, "ffn_up.weight"));
            const auto* down_weight =
                impl_->tensor(block_name(layer, "ffn_down.weight"));
            if (!attn_norm || !q_weight || !k_weight || !v_weight ||
                !attn_output || !ffn_norm || !gate_weight || !up_weight ||
                !down_weight) {
                return fail_round(Status::internal_error(
                    "resident Qwen2 layer tensor is missing"));
            }

            status = impl_->rms_norm_batch(
                impl_->workspace.hidden, *attn_norm, embedding_width,
                static_cast<float>(config.rms_norm_epsilon),
                impl_->workspace.normalized, total);
            if (!status) return fail_round(status);

            status = impl_->matmul(
                *q_weight, impl_->workspace.normalized, impl_->workspace.q,
                total, linear);
            if (!status) return fail_round(status);
            status = impl_->matmul(
                *k_weight, impl_->workspace.normalized, impl_->workspace.k,
                total, linear);
            if (!status) return fail_round(status);
            status = impl_->matmul(
                *v_weight, impl_->workspace.normalized, impl_->workspace.v,
                total, linear);
            if (!status) return fail_round(status);

            status = impl_->add_optional_tensor_batch(
                impl_->workspace.q, block_name(layer, "attn_q.bias"),
                embedding_width, total);
            if (!status) return fail_round(status);
            status = impl_->add_optional_tensor_batch(
                impl_->workspace.k, block_name(layer, "attn_k.bias"),
                kv_width, total);
            if (!status) return fail_round(status);
            status = impl_->add_optional_tensor_batch(
                impl_->workspace.v, block_name(layer, "attn_v.bias"),
                kv_width, total);
            if (!status) return fail_round(status);

            for (const auto& part : parts) {
                const auto embed_offset =
                    static_cast<std::uint64_t>(part.flat_offset) *
                    embedding_width;
                const auto kv_offset =
                    static_cast<std::uint64_t>(part.flat_offset) * kv_width;
                status = impl_->apply_rope_batch(
                    impl_->workspace.q + embed_offset, part.count,
                    config.attention_head_count, head_dimension,
                    config.rope_dimension_count, part.position_start,
                    static_cast<float>(config.rope_frequency_base));
                if (!status) return fail_round(status);
                status = impl_->apply_rope_batch(
                    impl_->workspace.k + kv_offset, part.count,
                    config.kv_head_count, head_dimension,
                    config.rope_dimension_count, part.position_start,
                    static_cast<float>(config.rope_frequency_base));
                if (!status) return fail_round(status);
            }

            for (const auto& part : parts) {
                auto& kv = *items[part.item_index].cache->impl_;
                const auto embed_offset =
                    static_cast<std::uint64_t>(part.flat_offset) *
                    embedding_width;
                const auto kv_offset =
                    static_cast<std::uint64_t>(part.flat_offset) * kv_width;
                status = impl_->attention_batch_from(
                    kv, part.count, layer, config.attention_head_count,
                    config.kv_head_count, head_dimension, attention,
                    impl_->workspace.q + embed_offset,
                    impl_->workspace.k + kv_offset,
                    impl_->workspace.v + kv_offset,
                    impl_->workspace.attended + embed_offset);
                if (!status) return fail_round(status);
            }

            status = impl_->matmul(
                *attn_output, impl_->workspace.attended,
                impl_->workspace.projection, total, linear);
            if (!status) return fail_round(status);
            status = impl_->add_vector_batch(
                impl_->workspace.hidden, impl_->workspace.projection,
                embedding_width, total);
            if (!status) return fail_round(status);

            status = impl_->rms_norm_batch(
                impl_->workspace.hidden, *ffn_norm, embedding_width,
                static_cast<float>(config.rms_norm_epsilon),
                impl_->workspace.normalized, total);
            if (!status) return fail_round(status);
            status = impl_->matmul(
                *gate_weight, impl_->workspace.normalized, impl_->workspace.gate,
                total, linear);
            if (!status) return fail_round(status);
            status = impl_->matmul(
                *up_weight, impl_->workspace.normalized, impl_->workspace.up,
                total, linear);
            if (!status) return fail_round(status);

            constexpr unsigned int threads = 256U;
            const auto ffn_total =
                static_cast<std::uint64_t>(total) * config.feed_forward_size;
            silu_mul_kernel<<<blocks_for(ffn_total), threads, 0, impl_->stream>>>(
                impl_->workspace.gate, impl_->workspace.up, ffn_total);
            status = launch_status("silu_mul_kernel(multi-sequence prefill)");
            if (!status) return fail_round(status);

            status = impl_->matmul(
                *down_weight, impl_->workspace.gate,
                impl_->workspace.projection, total, linear);
            if (!status) return fail_round(status);
            status = impl_->add_vector_batch(
                impl_->workspace.hidden, impl_->workspace.projection,
                embedding_width, total);
            if (!status) return fail_round(status);

            for (const auto& part : parts) {
                auto& kv = *items[part.item_index].cache->impl_;
                const auto kv_offset =
                    static_cast<std::uint64_t>(part.flat_offset) * kv_width;
                status = impl_->append_kv_from(
                    kv, layer, part.count,
                    impl_->workspace.k + kv_offset,
                    impl_->workspace.v + kv_offset);
                if (!status) return fail_round(status);
            }
        }

        bool synchronized = false;
        for (const auto& part : parts) {
            const auto index = part.item_index;
            const bool completes =
                offsets[index] + part.count == items[index].tokens.size();
            if (!completes ||
                items[index].output == CudaPrefillBatchOutput::discard) {
                continue;
            }

            const auto* output_norm = impl_->tensor("output_norm.weight");
            if (!output_norm) {
                return fail_round(
                    Status::internal_error("resident output norm is missing"));
            }
            const auto* output_weight = impl_->tensor(
                impl_->model->find_tensor("output.weight")
                    ? "output.weight"
                    : "token_embd.weight");
            if (!output_weight) {
                return fail_round(
                    Status::internal_error("resident output weight is missing"));
            }

            const auto final_flat =
                static_cast<std::uint64_t>(part.flat_offset + part.count - 1U);
            const float* final_hidden =
                impl_->workspace.hidden + final_flat * embedding_width;
            status = impl_->rms_norm(
                final_hidden, *output_norm, embedding_width,
                static_cast<float>(config.rms_norm_epsilon),
                impl_->workspace.normalized);
            if (!status) return fail_round(status);
            status = impl_->matvec(
                *output_weight, impl_->workspace.normalized,
                impl_->workspace.logits);
            if (!status) return fail_round(status);
            status = impl_->add_optional_tensor(
                impl_->workspace.logits, "output.bias",
                config.vocabulary_size);
            if (!status) return fail_round(status);

            if (items[index].output == CudaPrefillBatchOutput::greedy) {
                auto selected = impl_->select_greedy_device();
                if (!selected) return fail_round(selected.status());
                result.items[index].greedy_token = selected.value();
                synchronized = true;
            } else {
                auto copied = impl_->read_logits_host();
                if (!copied) return fail_round(copied.status());
                result.items[index].logits = std::move(copied).value();
                synchronized = true;
            }
        }

        if (!synchronized) {
            status = impl_->synchronize_outputless_prefill();
            if (!status) return fail_round(status);
        }

        for (const auto& part : parts) {
            status = impl_->validate_kv_commit(
                *items[part.item_index].cache->impl_);
            if (!status) return fail_round(status);
        }
        for (const auto& part : parts) {
            status = impl_->commit_kv(*items[part.item_index].cache->impl_);
            if (!status) return fail_round(status);
        }

        for (const auto& part : parts) {
            offsets[part.item_index] += part.count;
        }

        ++result.physical_batches;
        result.physical_sequence_participations += parts.size();
        result.physical_tokens += flattened.size();
        result.max_sequences = std::max<std::uint32_t>(
            result.max_sequences, static_cast<std::uint32_t>(parts.size()));
        impl_->prefill_batch_calls.fetch_add(1U, std::memory_order_relaxed);
        impl_->prefill_batched_sequences.fetch_add(
            parts.size(), std::memory_order_relaxed);
        impl_->prefill_batched_tokens.fetch_add(
            flattened.size(), std::memory_order_relaxed);
    }

    return result;
}

Result<GenerationResult> CudaExecutor::generate(std::span<const TokenId> prompt, const GenerationConfig& config) {
    if (prompt.empty()) return Status::invalid_argument("generation requires a non-empty prompt");
    auto cache = create_kv_cache(16U);
    if (!cache) return cache.status();

    if (config.sampling.temperature <= 0.0) {
        GenerationResult result;
        result.tokens.reserve(config.max_new_tokens);
        if (config.max_new_tokens == 0U) return result;
        auto selected = prefill_greedy(prompt, *cache.value());
        if (!selected) return selected.status();
        for (std::uint32_t index = 0; index < config.max_new_tokens; ++index) {
            const TokenId token = selected.value();
            result.tokens.push_back(token);
            if (config.stop_on_eos && impl_->model->tokenizer().special_ids.eos &&
                token == *impl_->model->tokenizer().special_ids.eos) {
                result.hit_eos = true;
                break;
            }
            if (index + 1U == config.max_new_tokens) break;
            selected = step_greedy(token, *cache.value());
            if (!selected) return selected.status();
        }
        return result;
    }

    auto trace = run_autoregressive(
        prompt, config, impl_->model->tokenizer().special_ids.eos,
        [this, &cache](std::span<const TokenId> tokens) { return prefill(tokens, *cache.value()); },
        [this, &cache](TokenId token) { return step(token, *cache.value()); });
    if (!trace) return trace.status();
    return std::move(trace).value().result;
}

} // namespace air

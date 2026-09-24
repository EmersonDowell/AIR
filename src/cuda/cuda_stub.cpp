#include "air/cuda.hpp"

namespace air {

bool cuda_compiled() noexcept { return false; }

Result<std::vector<DeviceInfo>> cuda_devices() {
    return Status::unsupported("AIR was built without CUDA support");
}

struct CudaKvCache::Impl {};
CudaKvCache::CudaKvCache(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CudaKvCache::~CudaKvCache() = default;
CudaKvCache::CudaKvCache(CudaKvCache&&) noexcept = default;
CudaKvCache& CudaKvCache::operator=(CudaKvCache&&) noexcept = default;
std::uint64_t CudaKvCache::size() const noexcept { return 0U; }
std::uint64_t CudaKvCache::capacity() const noexcept { return 0U; }
std::uint32_t CudaKvCache::page_tokens() const noexcept { return 0U; }
std::uint64_t CudaKvCache::committed_bytes() const noexcept { return 0U; }
std::uint64_t CudaKvCache::resident_bytes() const noexcept { return 0U; }
Result<std::unique_ptr<CudaKvCache>> CudaKvCache::fork(std::uint64_t) const {
    return Status::unsupported("AIR was built without CUDA support");
}
void CudaKvCache::reset() noexcept {}

struct CudaExecutor::Impl {};
CudaExecutor::CudaExecutor(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CudaExecutor::~CudaExecutor() = default;
CudaExecutor::CudaExecutor(CudaExecutor&&) noexcept = default;
CudaExecutor& CudaExecutor::operator=(CudaExecutor&&) noexcept = default;
Result<std::unique_ptr<CudaExecutor>> CudaExecutor::create(std::shared_ptr<const ModelDefinition>, int) {
    return Status::unsupported("AIR was built without CUDA support");
}
const ModelDefinition& CudaExecutor::model() const noexcept { std::terminate(); }
int CudaExecutor::device_ordinal() const noexcept { return -1; }
CudaExecutionStats CudaExecutor::stats() const noexcept { return {}; }
Status CudaExecutor::prepare_linear_tactic(QuantizedLinearExecutionKind) {
    return Status::unsupported("AIR was built without CUDA support");
}
std::uint64_t CudaExecutor::estimate_linear_tactic_preparation_bytes(QuantizedLinearExecutionKind) const noexcept {
    return 0U;
}
Status CudaExecutor::trim_linear_tactics(std::span<const QuantizedLinearExecutionKind>) {
    return Status::ok();
}
Result<std::unique_ptr<CudaKvCache>> CudaExecutor::create_kv_cache(std::uint32_t) const {
    return Status::unsupported("AIR was built without CUDA support");
}
Result<std::vector<float>> CudaExecutor::step(TokenId, CudaKvCache&, QuantizedLinearExecutionKind, AttentionExecutionKind) {
    return Status::unsupported("AIR was built without CUDA support");
}
Result<std::vector<float>> CudaExecutor::step(TokenId, CudaKvCache&, QuantizedLinearExecutionKind, QuantizedLinearExecutionKind, AttentionExecutionKind) {
    return Status::unsupported("AIR was built without CUDA support");
}
Result<std::vector<float>> CudaExecutor::step_verified(TokenId, CudaKvCache&, VerificationTrace&, QuantizedLinearExecutionKind, QuantizedLinearExecutionKind, AttentionExecutionKind) {
    return Status::unsupported("AIR was built without CUDA support");
}
Result<TokenId> CudaExecutor::step_greedy(TokenId, CudaKvCache&, QuantizedLinearExecutionKind, AttentionExecutionKind) {
    return Status::unsupported("AIR was built without CUDA support");
}
Result<TokenId> CudaExecutor::step_greedy(TokenId, CudaKvCache&, QuantizedLinearExecutionKind, QuantizedLinearExecutionKind, AttentionExecutionKind) {
    return Status::unsupported("AIR was built without CUDA support");
}
Result<std::vector<TokenId>> CudaExecutor::step_greedy_batch(std::span<const TokenId>, std::span<CudaKvCache*>, QuantizedLinearExecutionKind, QuantizedLinearExecutionKind, AttentionExecutionKind) {
    return Status::unsupported("AIR was built without CUDA support");
}
Result<std::vector<float>> CudaExecutor::prefill(std::span<const TokenId>, CudaKvCache&, QuantizedLinearExecutionKind, AttentionExecutionKind) {
    return Status::unsupported("AIR was built without CUDA support");
}
Status CudaExecutor::prefill_discard(std::span<const TokenId>, CudaKvCache&, QuantizedLinearExecutionKind, AttentionExecutionKind) {
    return Status::unsupported("AIR was built without CUDA support");
}
Result<TokenId> CudaExecutor::prefill_greedy(std::span<const TokenId>, CudaKvCache&, QuantizedLinearExecutionKind, AttentionExecutionKind) {
    return Status::unsupported("AIR was built without CUDA support");
}
Result<CudaPrefillBatchExecution> CudaExecutor::prefill_batch(
    std::span<const CudaPrefillBatchItem>,
    QuantizedLinearExecutionKind, AttentionExecutionKind) {
    return Status::unsupported("AIR was built without CUDA support");
}
Result<GenerationResult> CudaExecutor::generate(std::span<const TokenId>, const GenerationConfig&) {
    return Status::unsupported("AIR was built without CUDA support");
}

} // namespace air

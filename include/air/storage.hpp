#pragma once

#include "air/result.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

namespace air {

class ModelStorage final {
public:
    ~ModelStorage();

    ModelStorage(const ModelStorage&) = delete;
    ModelStorage& operator=(const ModelStorage&) = delete;
    ModelStorage(ModelStorage&&) = delete;
    ModelStorage& operator=(ModelStorage&&) = delete;

    [[nodiscard]] static Result<std::shared_ptr<const ModelStorage>> map_read_only(
        const std::filesystem::path& path);

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
    [[nodiscard]] std::uint64_t size_bytes() const noexcept { return size_bytes_; }
    [[nodiscard]] Result<std::span<const std::byte>> view(std::uint64_t offset,
                                                         std::uint64_t size) const;

private:
    ModelStorage(std::filesystem::path path, int fd, const std::byte* data, std::uint64_t size_bytes)
        : path_(std::move(path)), fd_(fd), data_(data), size_bytes_(size_bytes) {}

    std::filesystem::path path_;
    int fd_{-1};
    const std::byte* data_{nullptr};
    std::uint64_t size_bytes_{0};
};

} // namespace air

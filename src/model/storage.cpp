#include "air/storage.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace air {

ModelStorage::~ModelStorage() {
    if (data_ != nullptr && size_bytes_ != 0) {
        ::munmap(const_cast<std::byte*>(data_), static_cast<std::size_t>(size_bytes_));
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

Result<std::shared_ptr<const ModelStorage>> ModelStorage::map_read_only(
    const std::filesystem::path& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return Status::io_error("failed to open model file: " + path.string() + ": " +
                                std::strerror(errno));
    }

    struct stat info {};
    if (::fstat(fd, &info) != 0) {
        const std::string message = std::strerror(errno);
        ::close(fd);
        return Status::io_error("failed to stat model file: " + path.string() + ": " + message);
    }
    if (info.st_size <= 0) {
        ::close(fd);
        return Status::data_error("model file is empty: " + path.string());
    }

    const auto size = static_cast<std::uint64_t>(info.st_size);
    if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        ::close(fd);
        return Status::unsupported("model file exceeds addressable process size");
    }

    void* mapping = ::mmap(nullptr, static_cast<std::size_t>(size), PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        const std::string message = std::strerror(errno);
        ::close(fd);
        return Status::io_error("failed to memory-map model file: " + path.string() + ": " + message);
    }

    auto storage = std::shared_ptr<const ModelStorage>(new ModelStorage(
        path, fd, static_cast<const std::byte*>(mapping), size));
    return storage;
}

Result<std::span<const std::byte>> ModelStorage::view(std::uint64_t offset,
                                                      std::uint64_t size) const {
    if (offset > size_bytes_ || size > size_bytes_ - offset) {
        return Status::invalid_argument("requested model-storage view is out of bounds");
    }
    return std::span<const std::byte>(data_ + static_cast<std::size_t>(offset),
                                      static_cast<std::size_t>(size));
}

} // namespace air

#pragma once

#include <string>
#include <utility>

namespace air {

enum class ErrorCode {
    ok = 0,
    invalid_argument,
    invalid_state,
    io_error,
    cancelled,
    data_error,
    unsupported,
    internal_error,
};

class Status {
public:
    Status() = default;

    static Status ok() { return {}; }
    static Status invalid_argument(std::string message) {
        return {ErrorCode::invalid_argument, std::move(message)};
    }
    static Status invalid_state(std::string message) {
        return {ErrorCode::invalid_state, std::move(message)};
    }
    static Status io_error(std::string message) {
        return {ErrorCode::io_error, std::move(message)};
    }
    static Status cancelled(std::string message) {
        return {ErrorCode::cancelled, std::move(message)};
    }
    static Status data_error(std::string message) {
        return {ErrorCode::data_error, std::move(message)};
    }
    static Status unsupported(std::string message) {
        return {ErrorCode::unsupported, std::move(message)};
    }
    static Status internal_error(std::string message) {
        return {ErrorCode::internal_error, std::move(message)};
    }

    [[nodiscard]] bool is_ok() const noexcept { return code_ == ErrorCode::ok; }
    [[nodiscard]] explicit operator bool() const noexcept { return is_ok(); }
    [[nodiscard]] ErrorCode code() const noexcept { return code_; }
    [[nodiscard]] const std::string& message() const noexcept { return message_; }

private:
    Status(ErrorCode code, std::string message)
        : code_(code), message_(std::move(message)) {}

    ErrorCode code_{ErrorCode::ok};
    std::string message_;
};

} // namespace air

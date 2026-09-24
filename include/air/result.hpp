#pragma once

#include "air/status.hpp"

#include <optional>
#include <stdexcept>
#include <utility>

namespace air {

template <typename T>
class Result {
public:
    Result(T value) : value_(std::move(value)), status_(Status::ok()) {}
    Result(Status status) : status_(std::move(status)) {
        if (status_.is_ok()) {
            throw std::invalid_argument("Result error constructor requires a non-ok status");
        }
    }

    [[nodiscard]] bool is_ok() const noexcept { return value_.has_value(); }
    [[nodiscard]] explicit operator bool() const noexcept { return is_ok(); }
    [[nodiscard]] const Status& status() const noexcept { return status_; }

    [[nodiscard]] const T& value() const & {
        if (!value_) {
            throw std::logic_error("attempted to read value from failed Result");
        }
        return *value_;
    }

    [[nodiscard]] T& value() & {
        if (!value_) {
            throw std::logic_error("attempted to read value from failed Result");
        }
        return *value_;
    }

    [[nodiscard]] T&& value() && {
        if (!value_) {
            throw std::logic_error("attempted to read value from failed Result");
        }
        return std::move(*value_);
    }

private:
    std::optional<T> value_;
    Status status_;
};

} // namespace air

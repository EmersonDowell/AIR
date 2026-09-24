#pragma once

#include "air/model.hpp"
#include "air/result.hpp"

#include <filesystem>
#include <string_view>

namespace air {

class GgufFormat final {
public:
    [[nodiscard]] std::string_view name() const noexcept { return "gguf"; }
    [[nodiscard]] bool can_open(const std::filesystem::path& path) const;
    [[nodiscard]] Result<ModelDefinition> load(const std::filesystem::path& path) const;
};

} // namespace air

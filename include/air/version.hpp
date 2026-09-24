#pragma once

#include <string>

namespace air {

[[nodiscard]] inline std::string version_string() {
    return std::to_string(AIR_VERSION_MAJOR) + "." +
           std::to_string(AIR_VERSION_MINOR) + "." +
           std::to_string(AIR_VERSION_PATCH);
}

} // namespace air

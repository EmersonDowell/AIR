#pragma once

#include <string>
#include <string_view>

namespace air::detail {

[[nodiscard]] inline std::string json_escape(std::string_view value) {
    std::string out;
    out.reserve(value.size() + 8U);
    for (const char raw : value) {
        const auto ch = static_cast<unsigned char>(raw);
        switch (ch) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (ch < 0x20U) {
                constexpr char hex[] = "0123456789abcdef";
                out += "\\u00";
                out += hex[(ch >> 4U) & 0x0fU];
                out += hex[ch & 0x0fU];
            } else {
                out.push_back(raw);
            }
        }
    }
    return out;
}

} // namespace air::detail

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace air::detail {

template <class Range>
[[nodiscard]] double percentile_linear(const Range& values, double fraction) {
    if (values.empty()) return 0.0;
    std::vector<double> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end());
    const double position = std::clamp(fraction, 0.0, 1.0) *
                            static_cast<double>(sorted.size() - 1U);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    if (lower == upper) return sorted[lower];
    const double weight = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

} // namespace air::detail

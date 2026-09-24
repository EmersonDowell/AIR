#pragma once

#include <cstdint>
#include <string>

namespace air {

using TokenId = std::int32_t;
using SequenceId = std::uint64_t;
using RequestId = std::uint64_t;

enum class DataType {
    unknown = 0,
    f32,
    f16,
    bf16,
    q4_0,
    q4_1,
    q5_0,
    q5_1,
    q8_0,
    q8_1,
    q2_k,
    q3_k,
    q4_k,
    q5_k,
    q6_k,
    q8_k,
    iq2_xxs,
    iq2_xs,
    iq3_xxs,
    iq1_s,
    iq4_nl,
    iq3_s,
    iq2_s,
    iq4_xs,
    i8,
    i16,
    i32,
    i64,
    f64,
    iq1_m,
    tq1_0,
    tq2_0,
    mxfp4,
    nvfp4,
    q1_0,
    q2_0,
};

[[nodiscard]] const char* to_string(DataType type) noexcept;

struct ModelFingerprint {
    std::string format;
    std::string architecture;
    std::string model_id;
};

} // namespace air

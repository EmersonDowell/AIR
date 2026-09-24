#pragma once

#include "air/serving.hpp"

#include <boost/json/value.hpp>

#include <cstdint>
#include <string>
#include <string_view>

namespace air::server {

enum class ApiFlavor {
    native = 0,
    completions,
    chat_completions,
};

struct ParsedDecisionRequest {
    DecisionRequest decision;
};

struct ParsedRequest {
    InferenceRequest inference;
    ApiFlavor flavor{ApiFlavor::native};
    bool stream{false};
    std::string response_id;
    std::uint64_t created{0};
};

[[nodiscard]] Result<ParsedRequest> parse_generation_request(std::string_view target,
                                                             std::string_view body);
[[nodiscard]] Result<ParsedDecisionRequest> parse_decision_request(
    std::string_view body);
[[nodiscard]] std::string decision_json(const DecisionResponse& response);
[[nodiscard]] std::string completion_json(const ParsedRequest& request,
                                          const InferenceResponse& response);
[[nodiscard]] std::string stream_chunk_json(const ParsedRequest& request,
                                            std::string_view delta,
                                            TokenId token,
                                            bool first_chunk);
[[nodiscard]] std::string stream_finish_json(const ParsedRequest& request,
                                             const InferenceResponse& response);
[[nodiscard]] std::string error_json(std::string_view message,
                                     std::string_view type = "invalid_request_error");
[[nodiscard]] std::string service_snapshot_json(const ServiceSnapshot& snapshot);
[[nodiscard]] std::string model_json(const ModelDefinition& model, std::string_view backend);
[[nodiscard]] std::string models_json(const ModelDefinition& model);
[[nodiscard]] std::string metrics_text(const ServiceSnapshot& snapshot);
[[nodiscard]] std::string events_json(const std::vector<RuntimeEvent>& events);

} // namespace air::server

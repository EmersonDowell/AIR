#pragma once

#include "air/execution.hpp"
#include "air/model.hpp"

#include <memory>
#include <string>
#include <vector>

namespace air {

struct PlanningInput {
    const ModelDefinition& model;
    const RequestProfile& request;
    const RuntimeSnapshot& runtime;
};

struct PlanningCandidateTrace {
    std::string strategy_id;
    std::string disposition;
    bool eligible{false};
    bool memory_feasible{false};
    bool prepared_state_hot{false};
    double estimated_transition_ms{0.0};
    double estimated_horizon_ms{0.0};
    double estimated_break_even_tokens{0.0};
    std::uint64_t prepared_artifact_bytes{0};
};

struct PlanningDecision {
    ExecutionPlan plan;
    std::string objective{"static"};
    std::string reason{"static"};
    std::uint32_t eligible_candidates{0};
    bool prepared_state_hot{false};
    double estimated_transition_ms{0.0};
    double estimated_break_even_tokens{0.0};
    std::vector<PlanningCandidateTrace> candidates;
};

class Planner {
public:
    virtual ~Planner() = default;
    [[nodiscard]] virtual PlanningDecision decide(const PlanningInput& input) const = 0;
};

class StaticPlanner final : public Planner {
public:
    explicit StaticPlanner(ExecutionPlan plan = {});
    [[nodiscard]] PlanningDecision decide(const PlanningInput& input) const override;
private:
    ExecutionPlan plan_;
};

class Runtime {
public:
    Runtime(std::shared_ptr<const ModelDefinition> model,
            std::unique_ptr<Planner> planner);

    [[nodiscard]] const ModelDefinition& model() const noexcept { return *model_; }
    [[nodiscard]] PlanningDecision decide(const RequestProfile& request,
                                          const RuntimeSnapshot& snapshot) const;
    [[nodiscard]] ExecutionPlan plan(const RequestProfile& request,
                                     const RuntimeSnapshot& snapshot) const;

private:
    std::shared_ptr<const ModelDefinition> model_;
    std::unique_ptr<Planner> planner_;
};

} // namespace air

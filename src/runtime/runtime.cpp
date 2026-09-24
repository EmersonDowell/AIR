#include "air/runtime.hpp"

#include <stdexcept>
#include <utility>

namespace air {

StaticPlanner::StaticPlanner(ExecutionPlan plan) : plan_(std::move(plan)) {}

PlanningDecision StaticPlanner::decide(const PlanningInput&) const {
    PlanningDecision out;
    out.plan = plan_;
    out.objective = "static";
    out.reason = "static-plan";
    out.eligible_candidates = 1U;
    return out;
}

Runtime::Runtime(std::shared_ptr<const ModelDefinition> model,
                 std::unique_ptr<Planner> planner)
    : model_(std::move(model)), planner_(std::move(planner)) {
    if (!model_) throw std::invalid_argument("Runtime requires a model");
    if (!planner_) throw std::invalid_argument("Runtime requires a planner");
}

PlanningDecision Runtime::decide(const RequestProfile& request,
                                 const RuntimeSnapshot& snapshot) const {
    return planner_->decide(PlanningInput{*model_, request, snapshot});
}

ExecutionPlan Runtime::plan(const RequestProfile& request,
                            const RuntimeSnapshot& snapshot) const {
    return decide(request, snapshot).plan;
}

} // namespace air

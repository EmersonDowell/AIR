#include "air/serving.hpp"
#include "air/version.hpp"

#include <boost/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace json = boost::json;

namespace {
struct Options {
    std::filesystem::path model;
    std::filesystem::path manifest;
    std::filesystem::path medium_prompt;
    std::filesystem::path small_prompt;
    std::filesystem::path output;
    std::string scenario{"eviction"};
    int device{0};
    std::uint64_t horizon{10000};
    bool qualification_allow_unknown_eviction{false};
    std::uint32_t cycles{20};
};

std::optional<std::string> read_text(const std::filesystem::path& path) {
    std::ifstream in(path); if (!in) return std::nullopt;
    std::ostringstream out; out << in.rdbuf(); return out.str();
}

void usage() {
    std::cout << "AIR Strategy Probe " << air::version_string() << "\n"
              << "Usage: air-strategy-probe -m MODEL --manifest FILE --medium-prompt-file FILE --small-prompt-file FILE [options]\n"
              << "  --scenario eviction|hot-dense|oscillation|hot-stress\n"
              << "  --cycles N (stress scenarios, default 20)\n"
              << "  --horizon-tokens N\n"
              << "  --qualification-allow-unknown-eviction\n"
              << "  --output FILE\n";
}

std::optional<Options> parse(int argc, char** argv) {
    Options o;
    for (int i=1;i<argc;++i) {
        std::string a=argv[i];
        auto next=[&]()->std::optional<std::string>{ if(i+1>=argc)return{}; return std::string(argv[++i]); };
        if(a=="-h"||a=="--help"){usage();std::exit(0);}
        if(a=="-m"||a=="--model"){auto v=next();if(!v)return{};o.model=*v;}
        else if(a=="--manifest"){auto v=next();if(!v)return{};o.manifest=*v;}
        else if(a=="--medium-prompt-file"){auto v=next();if(!v)return{};o.medium_prompt=*v;}
        else if(a=="--small-prompt-file"){auto v=next();if(!v)return{};o.small_prompt=*v;}
        else if(a=="--output"){auto v=next();if(!v)return{};o.output=*v;}
        else if(a=="--scenario"){auto v=next();if(!v)return{};o.scenario=*v;}
        else if(a=="--device"){auto v=next();if(!v)return{};try{o.device=std::stoi(*v);}catch(...){return{};}}
        else if(a=="--horizon-tokens"){auto v=next();if(!v)return{};try{o.horizon=std::stoull(*v);}catch(...){return{};}}
        else if(a=="--cycles"){auto v=next();if(!v)return{};try{o.cycles=static_cast<std::uint32_t>(std::stoul(*v));}catch(...){return{};} if(o.cycles==0)return{};}
        else if(a=="--qualification-allow-unknown-eviction") o.qualification_allow_unknown_eviction=true;
        else return {};
    }
    if(o.model.empty()||o.manifest.empty()||o.medium_prompt.empty()||o.small_prompt.empty()) return {};
    if(o.scenario!="eviction"&&o.scenario!="hot-dense"&&o.scenario!="oscillation"&&o.scenario!="hot-stress") return {};
    return o;
}

json::object metrics_json(const air::RequestMetrics& m) {
    json::object o;
    o["strategy_id"]=m.strategy_id;
    o["strategy_objective"]=m.strategy_objective;
    o["strategy_decision_reason"]=m.strategy_decision_reason;
    o["strategy_eligible_candidates"]=m.strategy_eligible_candidates;
    o["strategy_prepared_state_hot"]=m.strategy_prepared_state_hot;
    o["strategy_estimated_transition_ms"]=m.strategy_estimated_transition_ms;
    o["strategy_estimated_break_even_tokens"]=m.strategy_estimated_break_even_tokens;
    json::array candidates;
    for (const auto& c : m.strategy_candidates) {
        json::object x;
        x["strategy_id"]=c.strategy_id; x["disposition"]=c.disposition;
        x["eligible"]=c.eligible; x["memory_feasible"]=c.memory_feasible;
        x["prepared_state_hot"]=c.prepared_state_hot;
        x["estimated_transition_ms"]=c.estimated_transition_ms;
        x["estimated_horizon_ms"]=c.estimated_horizon_ms;
        x["estimated_break_even_tokens"]=c.estimated_break_even_tokens;
        x["prepared_artifact_bytes"]=c.prepared_artifact_bytes;
        candidates.push_back(std::move(x));
    }
    o["strategy_candidates"]=std::move(candidates);
    o["plan_preparation_bytes"]=m.plan_preparation_bytes;
    o["plan_preparation_ms"]=m.plan_preparation_ms;
    o["plan_eviction_ms"]=m.plan_eviction_ms;
    o["prompt_tokens"]=m.prompt_tokens;
    o["generated_tokens"]=m.generated_tokens;
    o["ttft_ms"]=m.ttft_ms;
    o["total_ms"]=m.total_ms;
    return o;
}

json::object snapshot_json(const air::ServiceSnapshot& s) {
    json::object o;
    o["planner_mode"]=s.planner_mode;
    o["manifest_status"]=s.manifest_status;
    o["manifest_id"]=s.manifest_id;
    o["strategy_id"]=s.strategy_id;
    o["strategy_objective"]=s.strategy_objective;
    o["strategy_decision_reason"]=s.strategy_decision_reason;
    o["strategy_eligible_candidates"]=s.strategy_eligible_candidates;
    o["strategy_prepared_state_hot"]=s.strategy_prepared_state_hot;
    o["strategy_estimated_transition_ms"]=s.strategy_estimated_transition_ms;
    o["strategy_estimated_break_even_tokens"]=s.strategy_estimated_break_even_tokens;
    json::array candidates;
    for (const auto& c : s.strategy_candidates) {
        json::object x; x["strategy_id"]=c.strategy_id; x["disposition"]=c.disposition;
        x["eligible"]=c.eligible; x["memory_feasible"]=c.memory_feasible;
        x["prepared_state_hot"]=c.prepared_state_hot; x["estimated_transition_ms"]=c.estimated_transition_ms;
        x["estimated_horizon_ms"]=c.estimated_horizon_ms; x["estimated_break_even_tokens"]=c.estimated_break_even_tokens;
        x["prepared_artifact_bytes"]=c.prepared_artifact_bytes; candidates.push_back(std::move(x));
    }
    o["strategy_candidates"]=std::move(candidates);
    o["current_prepared_artifact_bytes"]=s.current_prepared_artifact_bytes;
    o["current_device_bytes"]=s.current_device_bytes;
    o["peak_device_bytes"]=s.peak_device_bytes;
    o["current_kv_bytes"]=s.current_kv_bytes;
    o["peak_kv_bytes"]=s.peak_kv_bytes;
    o["completed_requests"]=s.completed_requests;
    o["failed_requests"]=s.failed_requests;
    o["native_decode_batches"]=s.native_decode_batches;
    o["native_decode_sequences"]=s.native_decode_sequences;
    o["planned_prefill_block_linear_tactic"]=s.planned_prefill_block_linear_tactic;
    o["planned_decode_block_linear_tactic"]=s.planned_decode_block_linear_tactic;
    o["planned_decode_output_linear_tactic"]=s.planned_decode_output_linear_tactic;
    return o;
}

std::uint64_t current_rss_kib() {
    std::ifstream in("/proc/self/status");
    std::string key;
    while (in >> key) {
        if (key == "VmRSS:") {
            std::uint64_t value = 0; std::string unit;
            in >> value >> unit;
            return value;
        }
        std::string rest; std::getline(in, rest);
    }
    return 0;
}
}

int main(int argc,char** argv){
    auto opts=parse(argc,argv); if(!opts){usage();return 2;}
    auto medium=read_text(opts->medium_prompt); auto small=read_text(opts->small_prompt);
    if(!medium||!small){std::cerr<<"unable to read prompt file\n";return 2;}
    air::ManifestConfig mc; mc.enabled=true; mc.require=true; mc.path=opts->manifest;
    mc.strategy.objective=air::StrategyObjective::maximum_throughput;
    mc.strategy.expected_horizon_tokens=opts->horizon;
    mc.strategy.allow_unmeasured_eviction_for_qualification=opts->qualification_allow_unknown_eviction;
    air::SchedulerConfig sc; sc.prefix_cache_entries=0; sc.max_active_requests=8; sc.token_budget_per_cycle=256; sc.prefill_quantum_tokens=32;
    auto service=air::InferenceService::create(opts->model,air::BackendPreference::automatic,opts->device,sc,{},mc,{});
    if(!service){std::cerr<<service.status().message()<<'\n';return 3;}
    auto request_for=[](std::string prompt){air::InferenceRequest r;r.prompt=std::move(prompt);r.generation.max_new_tokens=2;r.generation.sampling.temperature=0.0;return r;};
    json::object root;
    root["schema"]="air.strategy-probe.v2";
    root["air_version"]=air::version_string();
    root["scenario"]=opts->scenario;
    root["cycles"]=opts->cycles;
    root["qualification_allow_unknown_eviction"]=opts->qualification_allow_unknown_eviction;
    json::array steps;
    auto record = [&](std::string_view label, const std::string& prompt) -> bool {
        const auto before = service.value()->snapshot();
        auto response = service.value()->generate(request_for(prompt));
        if (!response) { std::cerr << response.status().message() << '\n'; return false; }
        const auto after = service.value()->snapshot();
        json::object step;
        step["label"] = label;
        step["metrics"] = metrics_json(response.value().metrics);
        step["before"] = snapshot_json(before);
        step["after"] = snapshot_json(after);
        step["rss_kib"] = current_rss_kib();
        steps.push_back(std::move(step));
        return true;
    };

    if (opts->scenario == "eviction" || opts->scenario == "hot-dense") {
        if (!record("first-medium", *medium)) return 4;
        const auto& second_prompt = opts->scenario == "hot-dense" ? *medium : *small;
        if (!record(opts->scenario == "hot-dense" ? "second-medium" : "second-small", second_prompt)) return 5;
        root["first"] = steps[0].as_object().at("metrics");
        root["second"] = steps[1].as_object().at("metrics");
    } else if (opts->scenario == "oscillation") {
        for (std::uint32_t cycle = 0; cycle < opts->cycles; ++cycle) {
            if (!record("medium", *medium)) return 4;
            if (!record("small", *small)) return 5;
        }
    } else {
        for (std::uint32_t cycle = 0; cycle < opts->cycles; ++cycle) {
            if (!record("medium", *medium)) return 4;
        }
    }
    root["steps"] = std::move(steps);
    root["snapshot"]=snapshot_json(service.value()->snapshot());
    root["rss_kib_final"] = current_rss_kib();
    const auto text=json::serialize(root)+"\n";
    if(opts->output.empty()) std::cout<<text;
    else {std::ofstream out(opts->output,std::ios::trunc); if(!out)return 6;out<<text;}
    return 0;
}

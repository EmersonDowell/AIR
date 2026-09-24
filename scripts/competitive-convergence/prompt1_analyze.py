#!/usr/bin/env python3
import argparse, json
from pathlib import Path

def load(path):
    try: return json.loads(path.read_text())
    except Exception: return None

def bench(path):
    j=load(path)
    if not j: return None
    s=j.get("summary",{})
    return {
        "prompt_tokens_total": s.get("prompt_tokens"),
        "generated_tokens_total": s.get("generated_tokens"),
        "wall_ms": s.get("wall_ms"),
        "aggregate_tps": s.get("aggregate_generated_tokens_per_second"),
        "prefill_tps": s.get("mean_prefill_tokens_per_second"),
        "decode_tps": s.get("mean_decode_tokens_per_second"),
        "ttft_p50_ms": s.get("p50_ttft_ms"),
        "total_p50_ms": s.get("p50_total_ms"),
        "native_decode_batches": s.get("native_decode_batches"),
        "native_decode_sequences": s.get("native_decode_sequences"),
        "strategy_id": j.get("strategy_id"),
        "planner_mode": j.get("planner_mode"),
        "planned_prefill_quantum_tokens": j.get("planned_prefill_quantum_tokens"),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args=ap.parse_args()
    out={"schema":"air.prompt1.preliminary.v1","benchmarks":{}}
    for p in sorted((args.root/"bench").glob("*.json")):
        row=bench(p)
        if row: out["benchmarks"][p.stem]=row
    for p in sorted((args.root/"multi").glob("*.json")):
        if p.name in ("health.json","runtime-final.json","events-final.json"):
            continue
        out["multi_" + p.stem.replace("-","_")] = load(p)
    topo=args.root/"hardware/hardware-topology.json"
    if topo.exists(): out["hardware_topology"]=load(topo)
    # Source-shape classifications only. Performance hypotheses remain unresolved
    # until raw profiler and machine evidence are reviewed.
    out["hypotheses_preliminary"]={
        "H1":"UNRESOLVED",
        "H2":"SUPPORTED_BY_SOURCE_SHAPE_PENDING_MEASUREMENT",
        "H3":"UNRESOLVED",
        "H4":"UNRESOLVED",
        "H5":"UNRESOLVED",
        "H6":"UNRESOLVED",
        "H7":"UNRESOLVED",
        "H8":"SUPPORTED_BY_CURRENT_SOURCE_LIMIT",
        "H9":"SUPPORTED_BY_CURRENT_SOURCE_LIMIT",
        "H10":"SUPPORTED_BY_CURRENT_SOURCE_SHAPE_PENDING_MEASUREMENT",
        "H11":"UNRESOLVED_PENDING_PROPERTY_AND_MACHINE_EVIDENCE",
    }
    (args.root/"PRELIMINARY.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")

if __name__=="__main__":
    main()

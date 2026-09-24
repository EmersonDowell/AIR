#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

def load(p: Path):
    try: return json.loads(p.read_text())
    except Exception: return None

def mean(v): return statistics.mean(v) if v else None

def analyze_manifest_integrity(root: Path) -> dict:
    d = root / "manifest-corruption"
    exits = {}
    for p in sorted(d.glob("*.exit")):
        try: exits[p.stem] = int(p.read_text().strip())
        except Exception: pass

    valid_control_rc = exits.pop("valid-control", None)
    expected_doc = load(d / "cases" / "EXPECTED.json") or {}
    expected_map = expected_doc.get("cases") if isinstance(expected_doc.get("cases"), dict) else {}
    expected = sorted(k for k, v in expected_map.items() if v == "reject")
    observed = sorted(exits)
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    corruptions_rejected = bool(expected) and not missing and all(exits.get(k, 0) != 0 for k in expected)
    valid_control_ok = valid_control_rc == 0
    return {
        "valid_control_rc": valid_control_rc,
        "valid_control_ok": valid_control_ok,
        "expected_corruptions": expected,
        "observed_corruptions": observed,
        "missing_corruptions": missing,
        "unexpected_corruption_exits": unexpected,
        "cases": exits,
        "all_corruptions_rejected": corruptions_rejected,
        "all_rejected": valid_control_ok and corruptions_rejected,
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("root"); args=ap.parse_args()
    r=Path(args.root)
    out={"schema":"air.prompt15.preliminary-summary.v2"}
    out["manifest_integrity"]=analyze_manifest_integrity(r)

    gaps=[]
    for p in sorted((r/"out-of-region").glob("*.json")):
        d=load(p)
        if not d: continue
        runs=d.get("runs",[]); row=runs[0] if runs else {}
        gaps.append({"file":p.name,"planner_mode":d.get("planner_mode"),
                     "strategy_id":row.get("strategy_id",d.get("strategy_id")),
                     "reason":row.get("strategy_decision_reason"),
                     "prompt_tokens":d.get("summary",{}).get("prompt_tokens"),
                     "concurrency":d.get("config",{}).get("concurrency")})
    out["evidence_region_destruction"]=gaps

    budgets=[]
    for p in sorted((r/"memory-budget").glob("*.json")):
        d=load(p)
        if not d: continue
        row=(d.get("runs") or [{}])[0]
        budgets.append({"file":p.name,"strategy_id":row.get("strategy_id"),
                        "prepared_bytes":d.get("summary",{}).get("prepared_artifact_bytes"),
                        "peak_device_bytes":d.get("summary",{}).get("peak_device_bytes"),
                        "candidates":row.get("strategy_candidates",[])})
    out["memory_budget_sweep"]=budgets

    conc=[]
    for p in sorted((r/"concurrency").glob("c*.json")):
        d=load(p)
        if not d: continue
        counts={}
        for run in d.get("runs",[]):
            sid=str(run.get("strategy_id"))
            counts[sid]=counts.get(sid,0)+1
        conc.append({"file":p.name,"concurrency":d.get("config",{}).get("concurrency"),
                     "strategy_id":d.get("strategy_id"),"request_strategy_counts":counts,
                     "planner_mode":d.get("planner_mode"),
                     "aggregate_tps":d.get("summary",{}).get("aggregate_generated_tokens_per_second"),
                     "native_decode_batches":d.get("summary",{}).get("native_decode_batches"),
                     "native_decode_sequences":d.get("summary",{}).get("native_decode_sequences")})
    out["concurrency"]=conc

    out["stop_rule"]={"status":"requires-final-analysis",
        "note":"Preliminary summary is non-promotional; final Prompt-15 classification requires causal archive analysis."}

    (r/"PROMPT15-SUMMARY.json").write_text(json.dumps(out,indent=2)+"\n")
    m=out["manifest_integrity"]
    lines=["# AIR Prompt 15 preliminary destruction summary","",
           f"Valid manifest control executes: **{m['valid_control_ok']}**",
           f"All expected corrupt manifests rejected: **{m['all_corruptions_rejected']}**",
           f"Manifest integrity preliminary gate: **{m['all_rejected']}**","",
           f"- out-of-region probes: {len(gaps)}",
           f"- memory-budget probes: {len(budgets)}",
           f"- concurrency reports: {len(conc)}","",
           "Final competitive/stop-rule classification remains deferred to archive analysis."]
    (r/"PROMPT15-SUMMARY.md").write_text("\n".join(lines)+"\n")
    return 0

if __name__=="__main__":
    raise SystemExit(main())

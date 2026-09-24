#!/usr/bin/env python3
import argparse, json, math, statistics
from pathlib import Path

def ci(values):
    n=len(values)
    if not values: return None
    mean=statistics.mean(values)
    if n<2: return {"n":n,"mean":mean,"ci95":[mean,mean]}
    # t critical for n=5 is 2.776; use conservative 2.776 for this small R2 set.
    half=2.776*statistics.stdev(values)/math.sqrt(n)
    return {"n":n,"mean":mean,"ci95":[mean-half,mean+half],
            "median":statistics.median(values)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root",type=Path)
    args=ap.parse_args()
    out={"schema":"air.prompt1.r2-summary.v1"}

    hp=args.root/"hardware/cuda-hardware.json"
    if hp.exists():
        try: out["cuda_hardware"]=json.loads(hp.read_text())
        except Exception as e: out["cuda_hardware_error"]=str(e)

    qp=args.root/"q/q-randomized.json"
    if qp.exists():
        q=json.loads(qp.read_text())
        grouped={}
        for target in (262,1042):
            grouped[str(target)]={}
            for quantum in (32,64,128):
                rows=[r for r in q["records"]
                      if r["target"]==target and r["q"]==quantum and r["rc"]==0]
                grouped[str(target)][str(quantum)]={
                    "prefill_tps":ci([r["prefill_tps"] for r in rows if r.get("prefill_tps")]),
                    "total_ms":ci([r["total_ms"] for r in rows if r.get("total_ms")]),
                    "ttft_ms":ci([r["ttft_ms"] for r in rows if r.get("ttft_ms")]),
                }
        out["q_randomized"]=grouped

    for name in ("ncu-attention-q32.csv","ncu-attention-q128.csv","ncu-linear-q32.csv"):
        p=args.root/"ncu"/name
        out.setdefault("ncu",{})[name]={"present":p.exists(),"bytes":p.stat().st_size if p.exists() else 0}

    (args.root/"R2-SUMMARY.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":
    main()

#!/usr/bin/env python3
import argparse, json, math, statistics
from pathlib import Path

T975 = {2:12.706,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365,9:2.306,10:2.262}
def ci(values):
    xs=[float(x) for x in values if math.isfinite(float(x))]
    n=len(xs); mean=statistics.fmean(xs) if xs else 0.0
    if n<2: return {"n":n,"mean":mean,"stdev":0.0,"low":mean,"high":mean}
    sd=statistics.stdev(xs); t=T975.get(n,1.96); half=t*sd/math.sqrt(n)
    return {"n":n,"mean":mean,"stdev":sd,"low":mean-half,"high":mean+half}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('bench'); ap.add_argument('--json',required=True); ap.add_argument('--markdown',required=True); a=ap.parse_args()
    rows=[]
    for path in sorted(Path(a.bench).glob('round-*-*.json')):
        parts=path.stem.split('-'); rnd=int(parts[1]); prompt=parts[2]; tactic=parts[3]
        d=json.loads(path.read_text()); s=d['summary']
        rows.append((rnd,prompt,tactic,float(s['mean_prefill_tokens_per_second']),float(s['p50_ttft_ms']),float(s['p50_total_ms']),float(s['aggregate_generated_tokens_per_second']),d.get('prefill_attention_tactic','')))
    out={"schema":"air.prompt11.attention-tactic-summary.v1","prompts":{}}
    md=['# AIR Prompt 11 Attention Tactic Summary','', 'Ratios above 1.0 favor online-softmax. Confidence intervals are paired two-sided 95% Student-t intervals.','']
    for prompt in sorted({r[1] for r in rows}):
        group=[r for r in rows if r[1]==prompt]; by={}
        for r in group: by.setdefault(r[2],[]).append(r)
        po={"tactics":{},"paired_vs_baseline":{}}
        md += [f'## {prompt}','', '| Tactic | Prefill tok/s mean | 95% CI | Paired speed ratio vs baseline | 95% CI |','|---|---:|---:|---:|---:|']
        for tactic, rs in sorted(by.items()):
            emitted={r[7] for r in rs}
            expected={'baseline':'baseline','online':'online-softmax'}.get(tactic,tactic)
            if emitted != {expected}: raise SystemExit(f'{prompt}/{tactic}: benchmark emitted {emitted}, expected {expected}')
            po['tactics'][tactic]={"prefill_tokens_per_second":ci([r[3] for r in rs]),"p50_ttft_ms":ci([r[4] for r in rs]),"p50_total_ms":ci([r[5] for r in rs]),"aggregate_output_tokens_per_second":ci([r[6] for r in rs])}
        base={r[0]:r for r in by.get('baseline',[])}
        for tactic, rs in sorted(by.items()):
            if tactic=='baseline': continue
            paired=[]
            for r in rs:
                b=base.get(r[0]);
                if b: paired.append((r,b))
            ratios=[r[3]/b[3] for r,b in paired]
            total=[b[5]/r[5] for r,b in paired]
            po['paired_vs_baseline'][tactic]={"prefill_speed_ratio":ci(ratios),"total_speed_ratio":ci(total)}
        for tactic in sorted(by):
            c=po['tactics'][tactic]['prefill_tokens_per_second']
            if tactic=='baseline': md.append(f"| {tactic} | {c['mean']:.3f} | [{c['low']:.3f}, {c['high']:.3f}] | baseline | baseline |")
            else:
                rc=po['paired_vs_baseline'][tactic]['prefill_speed_ratio']; md.append(f"| {tactic} | {c['mean']:.3f} | [{c['low']:.3f}, {c['high']:.3f}] | {rc['mean']:.4f} | [{rc['low']:.4f}, {rc['high']:.4f}] |")
        md.append(''); out['prompts'][prompt]=po
    Path(a.json).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); Path(a.markdown).write_text('\n'.join(md)+'\n')
if __name__=='__main__': main()

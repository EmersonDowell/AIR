#!/usr/bin/env python3
"""Summarize AIR Prompt 12 native multi-sequence decode evidence.

Consumes air.benchmark.v5 reports emitted by the production InferenceService.
Ratios above 1.0 favor the native-batch candidate.
"""
from __future__ import annotations
import argparse, json, math, statistics
from pathlib import Path

T975={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,18:2.101,19:2.093,20:2.086,21:2.080,22:2.074,23:2.069,24:2.064,25:2.060,26:2.056,27:2.052,28:2.048,29:2.045,30:2.042}
def ci95(xs):
    n=len(xs)
    if not n:return {"n":0,"mean":None,"low":None,"high":None,"stdev":None}
    m=statistics.fmean(xs)
    if n==1:return {"n":1,"mean":m,"low":None,"high":None,"stdev":None}
    sd=statistics.stdev(xs); h=T975.get(n-1,1.96)*sd/math.sqrt(n)
    return {"n":n,"mean":m,"low":m-h,"high":m+h,"stdev":sd}
def parse(path):
    # round-01-c4-serial.json / round-01-c4-native.json
    p=path.stem.split('-')
    if len(p)!=4 or p[0]!='round': return None
    return int(p[1]), int(p[2][1:]), p[3]
def tokens_equal(a,b):
    return a.get('output_tokens') == b.get('output_tokens')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('evidence_dir',type=Path); ap.add_argument('--json',type=Path); ap.add_argument('--markdown',type=Path); args=ap.parse_args()
    rec={}
    for path in sorted(args.evidence_dir.glob('round-*-c*-*.json')):
        key=parse(path)
        if not key: continue
        d=json.loads(path.read_text())
        if d.get('schema')!='air.benchmark.v5': raise SystemExit(f'unexpected schema in {path}: {d.get("schema")}')
        mode=key[2]; expected='baseline' if mode=='serial' else 'batch-reuse8'
        if d.get('decode_linear_tactic') != expected: raise SystemExit(f'decode tactic mismatch in {path}')
        rec[key]=d
    out={"schema":"air.prompt12.decode-batching-summary.v1","concurrency":{}}
    for c in sorted({k[1] for k in rec}):
        cell={"modes":{},"paired_native_vs_serial":{},"correctness":{"paired_outputs_equal":True,"mismatched_rounds":[]}}
        for mode in ('serial','native'):
            rows=sorted((r,d) for (r,cc,m),d in rec.items() if cc==c and m==mode)
            if not rows: continue
            agg=[float(d['summary']['aggregate_generated_tokens_per_second']) for _,d in rows]
            dec=[float(d['summary']['mean_decode_tokens_per_second']) for _,d in rows]
            ttft=[float(d['summary']['p50_ttft_ms']) for _,d in rows]
            total=[float(d['summary']['p50_total_ms']) for _,d in rows]
            batches=[float(d['summary'].get('native_decode_batches',0)) for _,d in rows]
            seqs=[float(d['summary'].get('native_decode_sequences',0)) for _,d in rows]
            cell['modes'][mode]={"aggregate_output_tokens_per_second":ci95(agg),"mean_decode_tokens_per_second":ci95(dec),"p50_ttft_ms":ci95(ttft),"p50_total_ms":ci95(total),"native_decode_batches":ci95(batches),"native_decode_sequences":ci95(seqs)}
        ra=[]; rd=[]; rt=[]
        for r in sorted({k[0] for k in rec if k[1]==c}):
            b=rec.get((r,c,'serial')); n=rec.get((r,c,'native'))
            if not b or not n: continue
            bs=b['summary']; ns=n['summary']
            ba=float(bs['aggregate_generated_tokens_per_second']); na=float(ns['aggregate_generated_tokens_per_second'])
            bd=float(bs['mean_decode_tokens_per_second']); nd=float(ns['mean_decode_tokens_per_second'])
            bt=float(bs['p50_total_ms']); nt=float(ns['p50_total_ms'])
            if ba>0: ra.append(na/ba)
            if bd>0: rd.append(nd/bd)
            if nt>0: rt.append(bt/nt)
            if not tokens_equal(b,n):
                cell['correctness']['paired_outputs_equal']=False; cell['correctness']['mismatched_rounds'].append(r)
        cell['paired_native_vs_serial']={"aggregate_output_ratio":ci95(ra),"mean_decode_ratio":ci95(rd),"total_speed_ratio":ci95(rt)}
        # Native mode must show real physical batches when c>1; serial must not.
        native_rows=[d for (r,cc,m),d in rec.items() if cc==c and m=='native']
        serial_rows=[d for (r,cc,m),d in rec.items() if cc==c and m=='serial']
        cell['physical_batch_observed']= any(int(d['summary'].get('native_decode_batches',0))>0 for d in native_rows)
        cell['serial_batch_observed']= any(int(d['summary'].get('native_decode_batches',0))>0 for d in serial_rows)
        out['concurrency'][str(c)]=cell
    rendered=json.dumps(out,indent=2,sort_keys=True)+'\n'
    if args.json: args.json.write_text(rendered)
    else: print(rendered)
    if args.markdown:
        lines=['# AIR Prompt 12 Decode Batching Summary','', 'Ratios above 1.0 favor native multi-sequence decode. Confidence intervals are paired two-sided 95% Student-t intervals.','', '| Concurrency | Serial agg tok/s | Native agg tok/s | Paired aggregate ratio | 95% CI | Physical batch seen | Exact outputs |','|---:|---:|---:|---:|---:|:---:|:---:|']
        for c,cell in out['concurrency'].items():
            s=cell['modes'].get('serial',{}).get('aggregate_output_tokens_per_second',{}); n=cell['modes'].get('native',{}).get('aggregate_output_tokens_per_second',{}); r=cell['paired_native_vs_serial']['aggregate_output_ratio']
            ci=f"[{r['low']:.4f}, {r['high']:.4f}]" if r['low'] is not None else 'n/a'
            lines.append(f"| {c} | {s.get('mean',0):.3f} | {n.get('mean',0):.3f} | {r['mean']:.4f} | {ci} | {'yes' if cell['physical_batch_observed'] else 'NO'} | {'yes' if cell['correctness']['paired_outputs_equal'] else 'NO'} |")
        args.markdown.write_text('\n'.join(lines)+'\n')
    bad=[]
    for c,cell in out['concurrency'].items():
        if int(c)>1 and (not cell['physical_batch_observed'] or cell['serial_batch_observed'] or not cell['correctness']['paired_outputs_equal']): bad.append(c)
    return 2 if bad else 0
if __name__=='__main__': raise SystemExit(main())

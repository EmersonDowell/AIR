#!/usr/bin/env python3
"""Analyze AIR Prompt 13 R3 dense-FP32-cuBLAS evidence from production air-bench reports."""
from __future__ import annotations
import argparse, json, math, statistics
from pathlib import Path

T975={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,18:2.101,19:2.093,20:2.086,21:2.080,22:2.074,23:2.069,24:2.064,25:2.060,26:2.056,27:2.052,28:2.048,29:2.045,30:2.042}

def ci(v):
    n=len(v)
    if not n:return {"n":0,"mean":None,"low":None,"high":None,"stdev":None}
    m=statistics.fmean(v)
    if n==1:return {"n":1,"mean":m,"low":None,"high":None,"stdev":None}
    sd=statistics.stdev(v); h=T975.get(n-1,1.96)*sd/math.sqrt(n)
    return {"n":n,"mean":m,"low":m-h,"high":m+h,"stdev":sd}

def load(path:Path):
    d=json.loads(path.read_text())
    if d.get('schema')!='air.benchmark.v9':
        raise SystemExit(f'unexpected schema in {path}: {d.get("schema")}')
    return d

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('evidence_dir',type=Path); ap.add_argument('--json',type=Path); ap.add_argument('--markdown',type=Path); a=ap.parse_args()
    rec={}
    for p in sorted(a.evidence_dir.glob('*.json')):
        parts=p.stem.split('-')
        if len(parts)<5 or parts[0] not in ('prefill','decode') or parts[1]!='round': continue
        lane=parts[0]; rnd=int(parts[2]); workload=parts[3]; tactic='-'.join(parts[4:])
        d=load(p)
        expected={'reuse8':'batch-reuse8','dense':'dense-f32-cublas'}[tactic]
        field='prefill_block_linear_tactic' if lane=='prefill' else 'decode_block_linear_tactic'
        if d.get(field)!=expected: raise SystemExit(f'tactic identity mismatch in {p}: {d.get(field)} != {expected}')
        rec[(lane,workload,tactic,rnd)]=d
    out={'schema':'air.prompt13.dense-f32-cublas-summary.v1','prefill':{},'decode':{}}
    for workload in sorted({k[1] for k in rec if k[0]=='prefill'}):
        cell={'tactics':{},'paired_dense_vs_reuse8':{}}
        for tactic in ('reuse8','dense'):
            ds=[d for (lane,w,t,r),d in sorted(rec.items()) if lane=='prefill' and w==workload and t==tactic]
            if not ds: continue
            cell['tactics'][tactic]={
                'prefill_tokens_per_second':ci([float(d['summary']['mean_prefill_tokens_per_second']) for d in ds]),
                'p50_ttft_ms':ci([float(d['summary']['p50_ttft_ms']) for d in ds]),
                'p50_total_ms':ci([float(d['summary']['p50_total_ms']) for d in ds]),
                'peak_device_bytes':ci([float(d['summary']['peak_device_bytes']) for d in ds]),
            }
        rp=[]; rt=[]; rm=[]
        rounds=sorted({r for (lane,w,t,r) in rec if lane=='prefill' and w==workload})
        for r in rounds:
            b=rec.get(('prefill',workload,'reuse8',r)); c=rec.get(('prefill',workload,'dense',r))
            if not b or not c: continue
            bp=float(b['summary']['mean_prefill_tokens_per_second']); cp=float(c['summary']['mean_prefill_tokens_per_second'])
            bt=float(b['summary']['p50_total_ms']); ct=float(c['summary']['p50_total_ms'])
            bm=float(b['summary']['peak_device_bytes']); cm=float(c['summary']['peak_device_bytes'])
            if bp>0: rp.append(cp/bp)
            if ct>0: rt.append(bt/ct)
            rm.append(cm-bm)
        cell['paired_dense_vs_reuse8']={'prefill_speed_ratio':ci(rp),'total_speed_ratio':ci(rt),'peak_device_delta_bytes':ci(rm)}
        out['prefill'][workload]=cell
    for workload in sorted({k[1] for k in rec if k[0]=='decode'}):
        cell={'tactics':{},'paired_dense_vs_reuse8':{},'exact_outputs':True,'mismatched_rounds':[]}
        for tactic in ('reuse8','dense'):
            ds=[d for (lane,w,t,r),d in sorted(rec.items()) if lane=='decode' and w==workload and t==tactic]
            if not ds: continue
            cell['tactics'][tactic]={
                'aggregate_output_tokens_per_second':ci([float(d['summary']['aggregate_generated_tokens_per_second']) for d in ds]),
                'requests_per_second':ci([float(d['summary']['requests_per_second']) for d in ds]),
                'p50_total_ms':ci([float(d['summary']['p50_total_ms']) for d in ds]),
                'p95_total_ms':ci([float(d['summary']['p95_total_ms']) for d in ds]),
                'native_decode_batches':ci([float(d['summary']['native_decode_batches']) for d in ds]),
                'peak_device_bytes':ci([float(d['summary']['peak_device_bytes']) for d in ds]),
            }
        ra=[]; rt=[]; rm=[]
        rounds=sorted({r for (lane,w,t,r) in rec if lane=='decode' and w==workload})
        for r in rounds:
            b=rec.get(('decode',workload,'reuse8',r)); c=rec.get(('decode',workload,'dense',r))
            if not b or not c: continue
            if b.get('output_tokens') != c.get('output_tokens'): cell['exact_outputs']=False; cell['mismatched_rounds'].append(r)
            ba=float(b['summary']['aggregate_generated_tokens_per_second']); ca=float(c['summary']['aggregate_generated_tokens_per_second'])
            bt=float(b['summary']['p50_total_ms']); ct=float(c['summary']['p50_total_ms'])
            bm=float(b['summary']['peak_device_bytes']); cm=float(c['summary']['peak_device_bytes'])
            if ba>0: ra.append(ca/ba)
            if ct>0: rt.append(bt/ct)
            rm.append(cm-bm)
        cell['paired_dense_vs_reuse8']={'aggregate_output_ratio':ci(ra),'total_speed_ratio':ci(rt),'peak_device_delta_bytes':ci(rm)}
        out['decode'][workload]=cell
    text=json.dumps(out,indent=2,sort_keys=True)+'\n'
    if a.json:a.json.write_text(text)
    else: print(text)
    if a.markdown:
        lines=['# AIR Prompt 13 R3 Dense FP32 cuBLAS Summary','','Ratios above 1 favor dense-f32-cublas. Confidence intervals are paired two-sided 95% Student-t intervals.','']
        for w,c in out['prefill'].items():
            p=c['paired_dense_vs_reuse8']['prefill_speed_ratio']; m=c['paired_dense_vs_reuse8']['peak_device_delta_bytes']
            lines += [f'## Prefill {w}','',f"dense/reuse8 prefill ratio: {p['mean']:.4f} [{p['low']:.4f}, {p['high']:.4f}]" if p['mean'] is not None else 'no paired evidence',f"peak-device delta: {m['mean']/1048576:.2f} MiB" if m['mean'] is not None else 'no memory evidence','']
        for w,c in out['decode'].items():
            p=c['paired_dense_vs_reuse8']['aggregate_output_ratio']; m=c['paired_dense_vs_reuse8']['peak_device_delta_bytes']
            lines += [f'## Decode {w}','',f"dense/reuse8 aggregate ratio: {p['mean']:.4f} [{p['low']:.4f}, {p['high']:.4f}]" if p['mean'] is not None else 'no paired evidence',f"Exact outputs: {'yes' if c['exact_outputs'] else 'NO'}",f"peak-device delta: {m['mean']/1048576:.2f} MiB" if m['mean'] is not None else 'no memory evidence','']
        a.markdown.write_text('\n'.join(lines)+'\n')
    return 0
if __name__=='__main__': raise SystemExit(main())

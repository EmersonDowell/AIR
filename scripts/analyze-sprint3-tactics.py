#!/usr/bin/env python3
"""Summarize AIR Prompt 13 Sprint-3 randomized tactic evidence.

Evidence only. Consumes air.benchmark.v10 reports emitted by the production
InferenceService path. Ratios >1 favor the candidate.
"""
from __future__ import annotations
import argparse, json, math, statistics
from pathlib import Path

T975={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,
11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,18:2.101,19:2.093,20:2.086,
21:2.080,22:2.074,23:2.069,24:2.064,25:2.060,26:2.056,27:2.052,28:2.048,29:2.045,30:2.042}

def ci95(xs):
    n=len(xs)
    if not n:return {"n":0,"mean":None,"low":None,"high":None,"stdev":None}
    m=statistics.fmean(xs)
    if n==1:return {"n":1,"mean":m,"low":None,"high":None,"stdev":None}
    sd=statistics.stdev(xs); t=T975.get(n-1,1.96); h=t*sd/math.sqrt(n)
    return {"n":n,"mean":m,"low":m-h,"high":m+h,"stdev":sd}

def parse(path):
    # prefill-round-01-p1042-reuse8.json / decode-round-01-c8-q5q8.json
    parts=path.stem.split('-')
    if len(parts)<5 or parts[1] != 'round': return None
    lane=parts[0]
    try:r=int(parts[2])
    except ValueError:return None
    workload=parts[3]; tactic='-'.join(parts[4:])
    return lane,r,workload,tactic

def metric(data,lane):
    s=data['summary']
    if lane=='prefill': return float(s['mean_prefill_tokens_per_second'])
    return float(s['aggregate_generated_tokens_per_second'])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('evidence_dir',type=Path); ap.add_argument('--json',type=Path); ap.add_argument('--markdown',type=Path)
    args=ap.parse_args(); records={}
    emitted_expected={'reuse8':'batch-reuse8','dense':'dense-f32-cublas','q5q8':'q5q8-dp4a-hybrid'}
    for p in sorted(args.evidence_dir.glob('*-round-*-*.json')):
        k=parse(p)
        if not k: continue
        lane,r,w,t=k; d=json.loads(p.read_text())
        if d.get('schema')!='air.benchmark.v10': raise SystemExit(f'unexpected schema in {p}: {d.get("schema")}')
        field='prefill_block_linear_tactic' if lane=='prefill' else 'decode_block_linear_tactic'
        exp=emitted_expected.get(t,t)
        if d.get(field)!=exp: raise SystemExit(f'tactic mismatch in {p}: {field}={d.get(field)} expected {exp}')
        if lane=='decode' and d.get('decode_output_linear_tactic')!='batch-reuse8': raise SystemExit(f'decode output tactic changed in {p}')
        records[(lane,w,t,r)]=d
    out={'schema':'air.prompt13.sprint3-tactic-summary.v1','lanes':{}}
    for lane,w in sorted({(k[0],k[1]) for k in records}):
        tactics=sorted({k[2] for k in records if k[0]==lane and k[1]==w})
        node={'tactics':{},'paired_vs_reuse8':{}}
        for t in tactics:
            rows=sorted((r,d) for (la,wo,ta,r),d in records.items() if la==lane and wo==w and ta==t)
            vals=[metric(d,lane) for _,d in rows]
            ss=[d['summary'] for _,d in rows]
            node['tactics'][t]={
                'primary_throughput':ci95(vals),
                'p50_ttft_ms':ci95([float(x['p50_ttft_ms']) for x in ss]),
                'p50_total_ms':ci95([float(x['p50_total_ms']) for x in ss]),
                'requests_per_second':ci95([float(x['requests_per_second']) for x in ss]),
                'prepared_artifact_bytes':ci95([float(x['prepared_artifact_bytes']) for x in ss]),
                'peak_device_bytes':ci95([float(x['peak_device_bytes']) for x in ss]),
                'peak_kv_bytes':ci95([float(x['peak_kv_bytes']) for x in ss]),
                'native_decode_batches':ci95([float(x['native_decode_batches']) for x in ss]),
            }
        if 'reuse8' in tactics:
            for t in tactics:
                if t=='reuse8':continue
                speed=[]; latency=[]; device_delta=[]; prepared_delta=[]
                rounds=sorted({r for (la,wo,ta,r) in records if la==lane and wo==w and ta==t})
                for r in rounds:
                    b=records.get((lane,w,'reuse8',r)); c=records.get((lane,w,t,r))
                    if not b or not c:continue
                    bv=metric(b,lane); cv=metric(c,lane)
                    if bv>0:speed.append(cv/bv)
                    bt=float(b['summary']['p50_total_ms']); ct=float(c['summary']['p50_total_ms'])
                    if ct>0:latency.append(bt/ct)
                    device_delta.append(float(c['summary']['peak_device_bytes'])-float(b['summary']['peak_device_bytes']))
                    prepared_delta.append(float(c['summary']['prepared_artifact_bytes'])-float(b['summary']['prepared_artifact_bytes']))
                node['paired_vs_reuse8'][t]={
                    'throughput_ratio':ci95(speed), 'total_speed_ratio':ci95(latency),
                    'peak_device_delta_bytes':ci95(device_delta), 'prepared_artifact_delta_bytes':ci95(prepared_delta)}
        out['lanes'][f'{lane}:{w}']=node
    rendered=json.dumps(out,indent=2,sort_keys=True)+'\n'
    if args.json:args.json.write_text(rendered)
    else:print(rendered)
    if args.markdown:
        lines=['# AIR Prompt 13 Sprint-3 Tactic Summary','', 'Ratios above 1.0 favor the candidate. Paired intervals are two-sided 95% Student-t intervals.','']
        for key,node in out['lanes'].items():
            lines += [f'## {key}','', '| Tactic | Throughput mean | 95% CI | Ratio vs reuse8 | 95% CI | Prepared bytes mean |','|---|---:|---:|---:|---:|---:|']
            for t,d in node['tactics'].items():
                q=d['primary_throughput']; ci='n/a' if q['low'] is None else f"[{q['low']:.3f}, {q['high']:.3f}]"
                rr=node['paired_vs_reuse8'].get(t,{}).get('throughput_ratio')
                if rr:
                    rt='n/a' if rr['mean'] is None else f"{rr['mean']:.4f}"; rci='n/a' if rr['low'] is None else f"[{rr['low']:.4f}, {rr['high']:.4f}]"
                else: rt=rci='baseline'
                pb=d['prepared_artifact_bytes']['mean']
                lines.append(f"| {t} | {q['mean']:.3f} | {ci} | {rt} | {rci} | {pb:.0f} |")
            lines.append('')
        args.markdown.write_text('\n'.join(lines)+'\n')
    return 0
if __name__=='__main__': raise SystemExit(main())

#!/usr/bin/env python3
"""AIR Prompt 14 Strategy Lab evidence reducer and manifest builder.

Consumes only production-path air.benchmark.v11 evidence plus strict gate exit files,
fingerprint JSON, cold preparation rounds, and optional product-path eviction probes.
It preserves a Pareto candidate set instead of selecting one global winner.
"""
from __future__ import annotations
import argparse, json, math, statistics, re
from pathlib import Path

T975={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,
11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,18:2.101,19:2.093,20:2.086,
21:2.080,22:2.074,23:2.069,24:2.064,25:2.060,26:2.056,27:2.052,28:2.048,29:2.045,30:2.042}

def stats(xs):
    xs=[float(x) for x in xs if math.isfinite(float(x))]
    n=len(xs)
    if not n:return dict(n=0,mean=0.0,stdev=0.0,half_width=0.0,low=0.0,high=0.0)
    m=statistics.fmean(xs)
    if n==1:return dict(n=1,mean=m,stdev=0.0,half_width=0.0,low=m,high=m)
    sd=statistics.stdev(xs); h=T975.get(n-1,1.96)*sd/math.sqrt(n)
    return dict(n=n,mean=m,stdev=sd,half_width=h,low=m-h,high=m+h)

def load_json(p:Path): return json.loads(p.read_text())

def reports(bench:Path,lane:str,workload:str,tactic:str):
    pat=f'{lane}-round-*-{workload}-{tactic}.json'
    return [load_json(p) for p in sorted(bench.glob(pat))]

def run_summary(rs):
    if not rs: raise SystemExit('missing benchmark evidence')
    for d in rs:
        if d.get('schema')!='air.benchmark.v11': raise SystemExit(f'unexpected benchmark schema {d.get("schema")}')
    ss=[d['summary'] for d in rs]
    return {
      'samples':len(rs),
      'prompt_tokens':int(ss[0]['prompt_tokens']),
      'p50_ttft_ms':stats([x['p50_ttft_ms'] for x in ss]),
      'p50_total_ms':stats([x['p50_total_ms'] for x in ss]),
      'prefill_tps':stats([x['mean_prefill_tokens_per_second'] for x in ss]),
      'decode_tps':stats([x['mean_decode_tokens_per_second'] for x in ss]),
      'aggregate_tps':stats([x['aggregate_generated_tokens_per_second'] for x in ss]),
      'prepared_bytes':int(max(x.get('prepared_artifact_bytes',0) for x in ss)),
      'peak_device_bytes':int(max(x.get('peak_device_bytes',0) for x in ss)),
      'peak_kv_bytes':int(max(x.get('peak_kv_bytes',0) for x in ss)),
      'native_decode_batches':int(min(x.get('native_decode_batches',0) for x in ss)),
    }

def strict_ok(correctness:Path,tactic:str):
    # all widths plus native decode qualification must pass
    for w in (1,8,64):
        p=correctness/f'{tactic}-p{w}.exit'
        if not p.exists() or p.read_text().strip()!='0': return False
    p=correctness/f'decode-{tactic}.exit'
    return p.exists() and p.read_text().strip()=='0'

def strategy(workload, sid, region, tactic, summary, prep, evict, seed, evidence_id, concurrent=False):
    dense=tactic=='dense'
    if concurrent:
        prefill='batch-reuse8'; decode='dense-f32-cublas' if dense else 'batch-reuse8'
    else:
        prefill=decode='dense-f32-cublas' if dense else 'batch-reuse8'
    return {
      'workload':workload,'strategy_id':sid,'strict_qualified':True,
      'region_min_prompt_tokens':region[0],'region_max_prompt_tokens':region[1],
      'region_min_active_sequences':region[2],'region_max_active_sequences':region[3],
      'backend':'cuda','prefill_quantum_tokens':32,
      'prefill_block_quantized_linear':prefill,'decode_block_quantized_linear':decode,
      'decode_output_quantized_linear':'batch-reuse8','prefill_attention':'online-softmax','decode_attention':'baseline','kv_page_tokens':16,
      'samples':summary['samples'],
      'p50_ttft_ms':summary['p50_ttft_ms']['mean'],'p50_ttft_confidence_half_width_ms':summary['p50_ttft_ms']['half_width'],
      'p50_total_ms':summary['p50_total_ms']['mean'],'p50_total_confidence_half_width_ms':summary['p50_total_ms']['half_width'],
      'mean_prefill_tokens_per_second':summary['prefill_tps']['mean'],'prefill_tokens_per_second_confidence_half_width':summary['prefill_tps']['half_width'],
      'mean_decode_tokens_per_second':summary['decode_tps']['mean'],'decode_tokens_per_second_confidence_half_width':summary['decode_tps']['half_width'],
      'aggregate_generated_tokens_per_second':summary['aggregate_tps']['mean'],'aggregate_generated_tokens_per_second_confidence_half_width':summary['aggregate_tps']['half_width'],
      'peak_kv_bytes':summary['peak_kv_bytes'],'peak_device_bytes':summary['peak_device_bytes'],'prepared_artifact_bytes':summary['prepared_bytes'],
      'preparation_measured':True,'preparation_ms_mean':prep['mean'],'preparation_ms_stddev':prep['stdev'],'preparation_ms_confidence_half_width':prep['half_width'],
      'eviction_measured':evict is not None,'eviction_ms_mean':0.0 if evict is None else evict['mean'],'eviction_ms_stddev':0.0 if evict is None else evict['stdev'],'eviction_ms_confidence_half_width':0.0 if evict is None else evict['half_width'],
      'evidence_seed':seed,'evidence_id':evidence_id,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('root',type=Path)
    ap.add_argument('--manifest-out',type=Path,required=True)
    ap.add_argument('--summary-json',type=Path,required=True)
    ap.add_argument('--summary-md',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=141011)
    ap.add_argument('--eviction-dir',type=Path)
    a=ap.parse_args(); root=a.root; bench=root/'bench'; correctness=root/'correctness'
    fp=load_json(root/'fingerprint.json')
    airver=fp['air_version']; model=fp['model_digest']; hardware=fp['hardware_digest']
    # correctness is a hard prerequisite for both live tactics
    for t in ('reuse8','dense'):
        if not strict_ok(correctness,t): raise SystemExit(f'{t} did not pass strict qualification')
    # cold dense preparation from fresh processes
    cold=[]; pbytes=[]
    for p in sorted((root/'cold-preparation').glob('dense-cold-*.json')):
        d=load_json(p); r=d['runs'][0]; cold.append(float(r['plan_preparation_ms'])); pbytes.append(int(d['summary'].get('prepared_artifact_bytes',0)))
    if not cold or min(cold)<=0: raise SystemExit('missing positive cold dense preparation evidence')
    prep_dense=stats(cold); dense_bytes=max(pbytes)
    prep_reuse=stats([0.0])
    # eviction evidence is only admitted when finalizing
    ev_dense=None
    if a.eviction_dir:
        ev=[]
        for p in sorted(a.eviction_dir.glob('eviction-*.json')):
            d=load_json(p); second=d['second']
            if second['strategy_id'].startswith('reuse8') and float(second['plan_eviction_ms'])>=0:
                ev.append(float(second['plan_eviction_ms']))
        if not ev or max(ev)<=0: raise SystemExit('missing positive product-path dense eviction evidence')
        ev_dense=stats(ev)
    ev_reuse=stats([0.0])  # reuse8 owns no optional prepared artifact; measured-zero semantics.
    lanes={}
    # exact measured regions, intentionally non-overlapping.
    for work in ('p54','p262','p1042'):
        tactics=('reuse8',) if work=='p54' else ('reuse8','dense')
        for t in tactics:
            rs=reports(bench,'prefill',work,t); sm=run_summary(rs); lanes[(work,t)]=sm
    for t in ('reuse8','dense'):
        lanes[('c8',t)]=run_summary(reports(bench,'decode','c8',t))
    # ensure dense prepared residency matches cold evidence
    for key,sm in lanes.items():
        if key[1]=='dense' and sm['prepared_bytes'] and sm['prepared_bytes']!=dense_bytes:
            raise SystemExit(f'dense prepared bytes disagree: {key} {sm["prepared_bytes"]} != {dense_bytes}')
    strategies=[]
    def add(work,t,workload,active=1,concurrent=False):
        sm=lanes[(work,t)]; pt=sm['prompt_tokens']; sid=f'{t}-{work}'
        strategies.append(strategy(workload,sid,(pt,pt,active,active),t,sm,
            prep_dense if t=='dense' else prep_reuse,
            ev_dense if t=='dense' else ev_reuse,
            a.seed,f'p14:{work}:{t}:seed-{a.seed}',concurrent))
    add('p54','reuse8','small')
    for w in ('p262','p1042'):
        add(w,'reuse8','medium'); add(w,'dense','medium')
    add('c8','reuse8','concurrent',8,True); add('c8','dense','concurrent',8,True)
    manifest={'schema_version':10,'air_version':airver,'model_digest':model,'hardware_digest':hardware,
              'manifest_id':f'air-p14-{a.seed}-'+('final' if ev_dense else 'bootstrap'),'strategies':strategies}
    a.manifest_out.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    out={'schema':'air.prompt14.strategy-evidence.v1','air_version':airver,'model_digest':model,'hardware_digest':hardware,
         'cold_dense_preparation':prep_dense,'dense_prepared_artifact_bytes':dense_bytes,
         'dense_eviction':ev_dense,'lanes':{f'{k[0]}:{k[1]}':v for k,v in lanes.items()},'manifest_id':manifest['manifest_id']}
    a.summary_json.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    lines=['# AIR Prompt 14 Strategy Lab Evidence','',f'- AIR: `{airver}`',f'- Dense prepared bytes: `{dense_bytes}`',
           f'- Cold dense preparation: {prep_dense["mean"]:.3f} ms, 95% CI [{prep_dense["low"]:.3f}, {prep_dense["high"]:.3f}]']
    if ev_dense: lines.append(f'- Dense eviction: {ev_dense["mean"]:.3f} ms, 95% CI [{ev_dense["low"]:.3f}, {ev_dense["high"]:.3f}]')
    else: lines.append('- Dense eviction: not yet measured (bootstrap manifest only)')
    lines += ['','| Lane | Tactic | Prefill tok/s | Decode tok/s | Aggregate tok/s | p50 total ms |','|---|---|---:|---:|---:|---:|']
    for (w,t),sm in lanes.items(): lines.append(f'| {w} | {t} | {sm["prefill_tps"]["mean"]:.3f} | {sm["decode_tps"]["mean"]:.3f} | {sm["aggregate_tps"]["mean"]:.3f} | {sm["p50_total_ms"]["mean"]:.3f} |')
    a.summary_md.write_text('\n'.join(lines)+'\n')

if __name__=='__main__': main()

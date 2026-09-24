#!/usr/bin/env python3
"""Prompt-15 Gate 9: separate process readiness, first reuse8/CUDA use, dense plan preparation, first dense residual, and hot dense."""
from __future__ import annotations
import argparse, contextlib, http.client, json, signal, subprocess, time
from pathlib import Path
from typing import Any


def http_json(port:int, method:str, path:str, payload:dict[str,Any]|None=None, timeout=300.0):
    c=http.client.HTTPConnection('127.0.0.1',port,timeout=timeout)
    body=None if payload is None else json.dumps(payload,separators=(',',':'))
    c.request(method,path,body=body,headers={} if body is None else {'Content-Type':'application/json'})
    r=c.getresponse(); raw=r.read(); status=r.status; c.close()
    return status,json.loads(raw.decode())


def wait_ready(port:int, proc:subprocess.Popen, started:float, timeout=120.0):
    last=''
    while time.perf_counter()-started<timeout:
        if proc.poll() is not None: raise RuntimeError(f'server exited rc={proc.returncode}')
        try:
            status,_=http_json(port,'GET','/health',timeout=2)
            if 200<=status<300: return (time.perf_counter()-started)*1000
        except Exception as e:last=str(e)
        time.sleep(.05)
    raise RuntimeError(f'server readiness timeout: {last}')


def stop(proc):
    if not proc or proc.poll() is not None:return
    with contextlib.suppress(ProcessLookupError):proc.send_signal(signal.SIGINT)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()


def request(port:int,prompt:str,process_started:float,max_tokens=2):
    payload={'prompt':prompt,'max_tokens':max_tokens,'temperature':0.0,'top_p':1.0,'top_k':0,'stream':False}
    req_start=time.perf_counter(); status,obj=http_json(port,'POST','/generate',payload,timeout=600); req_end=time.perf_counter()
    if status!=200: raise RuntimeError(f'HTTP {status}: {obj}')
    metrics=obj.get('metrics',{}) if isinstance(obj.get('metrics',{}),dict) else {}
    start_since=(req_start-process_started)*1000
    end_since=(req_end-process_started)*1000
    internal_ttft=float(metrics.get('ttft_ms',0.0) or 0.0)
    return {
        'request_start_since_process_ms':start_since,
        'request_end_since_process_ms':end_since,
        'external_total_ms':(req_end-req_start)*1000,
        'process_start_to_internal_ttft_ms':start_since+internal_ttft,
        'usage':obj.get('usage',{}),'metrics':metrics,'text':obj.get('text','')}


def run_server(args,port:int,label:str,requests:list[tuple[str,str]],out:Path):
    log=(out/f'{label}-server.txt').open('w')
    cmd=[args.air_server,'-m',args.model,'--backend','auto','--manifest',args.manifest,'--require-manifest',
         '--strategy-objective','maximum-throughput','--strategy-horizon-tokens',str(args.horizon),
         '--host','127.0.0.1','--port',str(port),'--max-active','8','--token-budget','256','--prefill-quantum','32',
         '--prefix-cache','0','--workers','4','--max-connections','16','--io-timeout','600']
    started=time.perf_counter();proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,text=True,start_new_session=True)
    result={'label':label,'command':cmd,'pid':proc.pid}
    try:
        result['startup_to_health_ms']=wait_ready(port,proc,started)
        rows=[]
        for req_label,prompt in requests:
            row=request(port,prompt,started);row['label']=req_label;rows.append(row)
        result['requests']=rows
        _,runtime=http_json(port,'GET','/runtime',timeout=30);result['runtime']=runtime
        result['process_wall_before_shutdown_ms']=(time.perf_counter()-started)*1000
    finally:stop(proc);log.close()
    return result


def get_req(run:dict,label:str)->dict:
    return next(r for r in run.get('requests',[]) if r.get('label')==label)


def derive(root:dict)->dict:
    fresh=root['fresh_reuse8']; warm=root['runtime_warmed_dense']
    first_reuse=get_req(fresh,'first-reuse8'); hot_reuse=get_req(fresh,'hot-reuse8')
    first_dense=get_req(warm,'first-dense-after-reuse8'); hot_dense=get_req(warm,'hot-dense')
    prep=float(first_dense.get('metrics',{}).get('plan_preparation_ms',0.0) or 0.0)
    cold_excess=float(first_dense['external_total_ms'])-float(hot_dense['external_total_ms'])
    return {
      'schema':'air.prompt15.startup-decomposition.v2',
      'process_startup_to_health_ms':fresh['startup_to_health_ms'],
      'first_cuda_reuse8_external_total_ms':first_reuse['external_total_ms'],
      'first_cuda_reuse8_process_start_to_internal_ttft_ms':first_reuse['process_start_to_internal_ttft_ms'],
      'hot_reuse8_external_total_ms':hot_reuse['external_total_ms'],
      'dense_after_cuda_warm_external_total_ms':first_dense['external_total_ms'],
      'dense_plan_preparation_ms':prep,
      'hot_dense_external_total_ms':hot_dense['external_total_ms'],
      'cold_dense_excess_over_hot_ms':cold_excess,
      'first_dense_non_plan_residual_ms':cold_excess-prep,
      'residual_note':'Observational residual only: cold-dense excess minus measured PreparedModel plan preparation. It can include cuBLAS/library first-use and other request-level cold effects; it is not labeled as pure library initialization.'
    }


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--air-server',default='air-server');ap.add_argument('-m','--model',required=True)
    ap.add_argument('--manifest',required=True);ap.add_argument('--small-prompt-file',required=True);ap.add_argument('--medium-prompt-file',required=True)
    ap.add_argument('--horizon',type=int,default=10000);ap.add_argument('--port',type=int,default=18740);ap.add_argument('--output-dir',required=True)
    a=ap.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    small=Path(a.small_prompt_file).read_text();medium=Path(a.medium_prompt_file).read_text()
    root={'schema':'air.prompt15.startup-transition.v2'}
    root['fresh_reuse8']=run_server(a,a.port,'fresh-reuse8',[('first-reuse8',small),('hot-reuse8',small)],out)
    root['runtime_warmed_dense']=run_server(a,a.port+1,'runtime-warmed-dense',[('reuse8-warm',small),('first-dense-after-reuse8',medium),('hot-dense',medium)],out)
    root['decomposition']=derive(root)
    (out/'startup-transition.json').write_text(json.dumps(root,indent=2)+'\n')
    (out/'startup-decomposition.json').write_text(json.dumps(root['decomposition'],indent=2)+'\n')
    return 0
if __name__=='__main__':raise SystemExit(main())

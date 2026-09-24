#!/usr/bin/env python3
"""Separate fresh-process startup, first-use execution, plan transition, and hot request evidence."""
from __future__ import annotations
import argparse, contextlib, http.client, json, os, signal, subprocess, time
from pathlib import Path
from typing import Any


def http_json(port:int, method:str, path:str, payload:dict[str,Any]|None=None, timeout=300.0):
    c=http.client.HTTPConnection('127.0.0.1',port,timeout=timeout)
    body=None if payload is None else json.dumps(payload,separators=(',',':'))
    c.request(method,path,body=body,headers={} if body is None else {'Content-Type':'application/json'})
    r=c.getresponse(); raw=r.read(); status=r.status; c.close()
    return status,json.loads(raw.decode())


def wait_ready(port:int, proc:subprocess.Popen, timeout=120.0):
    start=time.perf_counter(); last=''
    while time.perf_counter()-start<timeout:
        if proc.poll() is not None: raise RuntimeError(f'server exited rc={proc.returncode}')
        try:
            status,_=http_json(port,'GET','/health',timeout=2)
            if 200<=status<300: return (time.perf_counter()-start)*1000
        except Exception as e: last=str(e)
        time.sleep(.05)
    raise RuntimeError(f'server readiness timeout: {last}')


def stop(proc):
    if not proc or proc.poll() is not None:return
    with contextlib.suppress(ProcessLookupError): proc.send_signal(signal.SIGINT)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()


def request(port:int,prompt:str,max_tokens=2):
    payload={'prompt':prompt,'max_tokens':max_tokens,'temperature':0.0,'top_p':1.0,'top_k':0,'stream':False}
    t=time.perf_counter(); status,obj=http_json(port,'POST','/generate',payload,timeout=600); external=(time.perf_counter()-t)*1000
    if status!=200: raise RuntimeError(f'HTTP {status}: {obj}')
    return {'external_total_ms':external,'usage':obj.get('usage',{}),'metrics':obj.get('metrics',{}),'text':obj.get('text','')}


def run_server(args, port:int, label:str, requests:list[tuple[str,str]], out:Path):
    log=(out/f'{label}-server.txt').open('w')
    cmd=[args.air_server,'-m',args.model,'--backend','auto','--manifest',args.manifest,'--require-manifest',
         '--strategy-objective','maximum-throughput','--strategy-horizon-tokens',str(args.horizon),
         '--host','127.0.0.1','--port',str(port),'--max-active','8','--token-budget','256','--prefill-quantum','32',
         '--prefix-cache','0','--workers','4','--max-connections','16','--io-timeout','600']
    start=time.perf_counter(); proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,text=True,start_new_session=True)
    result={'label':label,'command':cmd,'pid':proc.pid}
    try:
        ready=wait_ready(port,proc); result['startup_to_health_ms']=ready
        rows=[]
        for req_label,prompt in requests:
            row=request(port,prompt)
            row['label']=req_label
            row['process_start_to_internal_ttft_ms']=ready+float(row.get('metrics',{}).get('ttft_ms',0.0))
            rows.append(row)
        result['requests']=rows
        _,runtime=http_json(port,'GET','/runtime',timeout=30)
        result['runtime']=runtime
        result['process_wall_before_shutdown_ms']=(time.perf_counter()-start)*1000
    finally:
        stop(proc); log.close()
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--air-server',default='air-server'); ap.add_argument('-m','--model',required=True)
    ap.add_argument('--manifest',required=True); ap.add_argument('--small-prompt-file',required=True); ap.add_argument('--medium-prompt-file',required=True)
    ap.add_argument('--horizon',type=int,default=10000); ap.add_argument('--port',type=int,default=18740); ap.add_argument('--output-dir',required=True)
    args=ap.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    small=Path(args.small_prompt_file).read_text(); medium=Path(args.medium_prompt_file).read_text()
    root={'schema':'air.prompt15.startup-transition.v1'}
    root['fresh_dense']=run_server(args,args.port,'fresh-dense',[('first-dense',medium),('hot-dense',medium)],out)
    root['runtime_warmed']=run_server(args,args.port+1,'runtime-warmed',[('reuse8-warm',small),('first-dense-after-reuse8',medium),('hot-dense',medium)],out)
    (out/'startup-transition.json').write_text(json.dumps(root,indent=2)+'\n')
    return 0

if __name__=='__main__': raise SystemExit(main())

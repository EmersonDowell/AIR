#!/usr/bin/env python3
"""Gate-10 fail-closed admission guard. Unknown comparator layouts are INADMISSIBLE, never guessed into a result."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any,Iterable


def walk(obj:Any,path=()):
    yield path,obj
    if isinstance(obj,dict):
        for k,v in obj.items():yield from walk(v,path+(str(k),))
    elif isinstance(obj,list):
        for i,v in enumerate(obj):yield from walk(v,path+(str(i),))


def json_docs(root:Path):
    for p in root.rglob('*.json'):
        try:yield p,json.loads(p.read_text())
        except Exception:continue


def find_key_values(docs,key:str):
    out=[]
    for p,obj in docs:
        for path,node in walk(obj):
            if isinstance(node,dict) and key in node:out.append((p,path+(key,),node[key]))
    return out


def explicit_prompt_pairs(docs):
    pairs=[]
    for p,obj in docs:
        for path,node in walk(obj):
            if not isinstance(node,dict):continue
            if 'air_prompt_tokens' in node and 'llama_prompt_tokens' in node:
                pairs.append((p,path,node['air_prompt_tokens'],node['llama_prompt_tokens']))
            elif isinstance(node.get('air'),dict) and isinstance(node.get('llama'),dict) and 'prompt_tokens' in node['air'] and 'prompt_tokens' in node['llama']:
                pairs.append((p,path,node['air']['prompt_tokens'],node['llama']['prompt_tokens']))
    return pairs


def order_evidence(docs):
    evidence=[]
    for p,obj in docs:
        for path,node in walk(obj):
            if isinstance(node,dict):
                for k,v in node.items():
                    if 'order' in k.lower() and (isinstance(v,(str,list))):
                        s=json.dumps(v).lower()
                        if 'air' in s and 'llama' in s:evidence.append((p,path+(k,),v))
    return evidence


def success_pairs(docs):
    pairs=[]
    for p,obj in docs:
        for path,node in walk(obj):
            if not isinstance(node,dict):continue
            a=node.get('air');l=node.get('llama')
            if isinstance(a,dict) and isinstance(l,dict):
                ak=next((k for k in ('successful_requests','success_count','successful') if isinstance(a.get(k),int)),None)
                lk=next((k for k in ('successful_requests','success_count','successful') if isinstance(l.get(k),int)),None)
                if ak and lk:pairs.append((p,path,a[ak],l[lk]))
    return pairs


def validate_lane(root:Path,expected_profile:str,min_rounds:int=5):
    docs=list(json_docs(root)); reasons=[]; details={"json_files":len(docs)}
    if not docs:return {'status':'INADMISSIBLE','reasons':['no JSON evidence in comparison lane'],'details':details}
    profiles=find_key_values(docs,'air_server_extra_args');details['profile_evidence']=[str(v) for _,_,v in profiles]
    if not any(str(v).strip()==expected_profile.strip() for _,_,v in profiles):reasons.append('fixed AIR tactic profile not proven by comparator JSON')
    rounds=find_key_values(docs,'rounds'); numeric=[int(v) for _,_,v in rounds if isinstance(v,int)];details['rounds_evidence']=numeric
    if not numeric or max(numeric)<min_rounds:reasons.append(f'at least {min_rounds} paired rounds not proven by comparator JSON')
    pp=explicit_prompt_pairs(docs);details['prompt_token_pairs']=[{'air':a,'llama':l} for _,_,a,l in pp]
    if not pp:reasons.append('AIR/llama prompt-token comparability is not explicitly recorded')
    elif any(a!=l for _,_,a,l in pp):reasons.append('one or more AIR/llama prompt-token counts differ')
    orders=order_evidence(docs);details['pair_order_records']=len(orders)
    if not orders:reasons.append('randomized AIR/llama pair ordering is not explicitly evidenced')
    sp=success_pairs(docs);details['success_pairs']=[{'air':a,'llama':l} for _,_,a,l in sp]
    if sp and any(a!=l for _,_,a,l in sp):reasons.append('AIR/llama successful request counts are asymmetric')
    # Absence of explicit success-pair fields is not silently converted to symmetry.
    if not sp:reasons.append('successful request-count symmetry is not explicitly evidenced')
    return {'status':'PASS' if not reasons else 'INADMISSIBLE','reasons':reasons,'details':details}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reuse8-dir',type=Path,required=True);ap.add_argument('--dense-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--min-rounds',type=int,default=5)
    a=ap.parse_args()
    reuse='--cuda-prefill-block-linear reuse8 --cuda-decode-block-linear reuse8 --cuda-decode-output-linear reuse8 --cuda-prefill-attention online-softmax'
    dense='--cuda-prefill-block-linear dense-f32-cublas --cuda-decode-block-linear dense-f32-cublas --cuda-decode-output-linear reuse8 --cuda-prefill-attention online-softmax'
    lanes={'reuse8':validate_lane(a.reuse8_dir,reuse,a.min_rounds),'dense':validate_lane(a.dense_dir,dense,a.min_rounds)}
    status='PASS' if all(x['status']=='PASS' for x in lanes.values()) else 'INADMISSIBLE'
    r={'schema':'air.prompt15.external-admission.v1','status':status,'lanes':lanes,'rule':'A comparison cell without explicit prompt-token comparability and pairing/success evidence is inadmissible, not averaged into competitive conclusions.'}
    a.output.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps(r,indent=2,sort_keys=True))
    return 0 if status=='PASS' else 3
if __name__=='__main__':raise SystemExit(main())

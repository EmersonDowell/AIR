#!/usr/bin/env python3
import argparse, concurrent.futures, http.client, json, time, urllib.error, urllib.parse, urllib.request


def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def post_json(url, payload, timeout=60):
    req=urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def completion(base, i, prompt):
    varied = prompt * (1 + (i % 4))
    payload={'prompt':varied, 'max_tokens':[8,16,24,32][i%4], 'temperature':0}
    return post_json(base+'/v1/completions', payload)


def disconnect(base, tokens, prompt):
    u=urllib.parse.urlparse(base)
    conn=http.client.HTTPConnection(u.hostname, u.port, timeout=15)
    body=json.dumps({'prompt':prompt, 'max_tokens':tokens, 'temperature':0, 'stream':True})
    conn.request('POST','/v1/completions',body,{'Content-Type':'application/json'})
    resp=conn.getresponse()
    chunk=resp.read(128)
    conn.close()
    return resp.status, len(chunk)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:18285')
    ap.add_argument('--requests', type=int, default=64)
    ap.add_argument('--concurrency', type=int, default=8)
    ap.add_argument('--disconnects', type=int, default=4)
    ap.add_argument('--prompt', default='Explain one useful fact about Lake Superior.')
    args=ap.parse_args()
    base=args.url.rstrip('/')
    initial=get_json(base+'/runtime')
    ok=0; errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs={ex.submit(completion,base,i,args.prompt):i for i in range(args.requests)}
        for fut,i in [(f,i) for f,i in futs.items()]:
            try:
                status,_=fut.result()
                if status==200: ok+=1
                else: errors.append(f'{i}:http:{status}')
            except Exception as e:
                errors.append(f'{i}:{type(e).__name__}:{e}')
    invalid_rejected=0
    for payload in ({}, {'prompt':'', 'max_tokens':1}, {'prompt':'x','max_tokens':999999999}):
        try:
            post_json(base+'/v1/completions', payload, timeout=10)
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500: invalid_rejected += 1
        except Exception as e:
            errors.append(f'invalid:{type(e).__name__}:{e}')
    disconnect_results=[]
    for _ in range(args.disconnects):
        try: disconnect_results.append(disconnect(base,256,args.prompt))
        except Exception as e: disconnect_results.append(('error',str(e)))
    idle=None
    for _ in range(300):
        idle=get_json(base+'/runtime')
        if idle.get('queued_requests')==0 and idle.get('active_requests')==0 and idle.get('capabilities',{}).get('admission_reserved_bytes')==0 and idle.get('current_kv_bytes')==0:
            break
        time.sleep(0.1)
    caps=idle.get('capabilities',{}) if idle else {}
    pool_alloc=caps.get('kv_pool_allocated_bytes',0); pool_free=caps.get('kv_pool_free_bytes',0)
    result={
        'requests':args.requests,'successes':ok,'errors':errors,'invalid_rejected':invalid_rejected,
        'disconnect_results':disconnect_results,'initial':initial,'final':idle,
        'resource_idle': bool(idle and idle.get('queued_requests')==0 and idle.get('active_requests')==0 and caps.get('admission_reserved_bytes')==0 and idle.get('current_kv_bytes')==0),
        'kv_pool_all_free': pool_alloc==pool_free,
    }
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if ok==args.requests and not errors and invalid_rejected==3 and result['resource_idle'] and result['kv_pool_all_free'] else 1

if __name__=='__main__':
    raise SystemExit(main())

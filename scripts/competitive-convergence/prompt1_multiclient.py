#!/usr/bin/env python3
import argparse, concurrent.futures, json, statistics, threading, time, urllib.request
from pathlib import Path

def post(url, prompt, max_tokens, delay_ms, barrier=None):
    if barrier is not None:
        barrier.wait()
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)
    body = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(url + "/generate", data=body,
                                 headers={"Content-Type":"application/json"}, method="POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            payload = json.loads(response.read().decode())
        wall_ms = (time.perf_counter() - start) * 1000.0
        return {"ok": True, "wall_ms": wall_ms, "response": payload}
    except Exception as e:
        return {"ok": False, "wall_ms": (time.perf_counter()-start)*1000.0, "error": str(e)}

def get_json(url, path):
    try:
        with urllib.request.urlopen(url + path, timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"error": str(e)}

def metrics(result):
    if not result.get("ok"): return {}
    return result.get("response",{}).get("metrics",{})

def independent(args):
    prompt = args.prompt.read_text()
    rounds = []
    for rnd in range(1, args.rounds + 1):
        barrier = threading.Barrier(args.concurrency)
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [
                ex.submit(post, args.url, prompt, args.tokens,
                          idx * args.stagger_ms, barrier)
                for idx in range(args.concurrency)
            ]
            results = [f.result() for f in futures]
        rounds.append({
            "round": rnd,
            "wall_ms": (time.perf_counter()-start)*1000.0,
            "requests": results,
        })
        time.sleep(args.cooldown_ms/1000.0)
    return rounds

def mixed(args):
    long_prompt = args.long_prompt.read_text()
    short_prompt = args.short_prompt.read_text()
    rounds = []
    for rnd in range(1, args.rounds + 1):
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.short_count + 1) as ex:
            started = time.perf_counter()
            long_f = ex.submit(post, args.url, long_prompt, args.tokens, 0.0, None)
            time.sleep(args.long_lead_ms/1000.0)
            barrier = threading.Barrier(args.short_count)
            short_fs = [
                ex.submit(post, args.url, short_prompt, args.tokens, 0.0, barrier)
                for _ in range(args.short_count)
            ]
            long_r = long_f.result()
            shorts = [f.result() for f in short_fs]
        rounds.append({
            "round": rnd,
            "wall_ms": (time.perf_counter()-started)*1000.0,
            "long": long_r,
            "short": shorts,
        })
        time.sleep(args.cooldown_ms/1000.0)
    return rounds

def summarize_independent(rounds):
    rows = [r for rd in rounds for r in rd["requests"] if r.get("ok")]
    ms = [metrics(r) for r in rows]
    return {
        "successful": len(rows),
        "strategy_counts": dict(__import__("collections").Counter(
            m.get("strategy_id","missing") for m in ms)),
        "planner_counts": dict(__import__("collections").Counter(
            m.get("planner_mode","missing") for m in ms)),
        "ttft_p50_ms": statistics.median([m.get("ttft_ms",0) for m in ms]) if ms else None,
        "total_p50_ms": statistics.median([m.get("total_ms",0) for m in ms]) if ms else None,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["independent","mixed"], required=True)
    ap.add_argument("--url", default="http://127.0.0.1:18810")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--cooldown-ms", type=float, default=100.0)
    ap.add_argument("--prompt", type=Path)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--stagger-ms", type=float, default=1.0)
    ap.add_argument("--long-prompt", type=Path)
    ap.add_argument("--short-prompt", type=Path)
    ap.add_argument("--short-count", type=int, default=7)
    ap.add_argument("--long-lead-ms", type=float, default=1.0)
    args = ap.parse_args()

    before = get_json(args.url, "/runtime")
    if args.mode == "independent":
        if not args.prompt: ap.error("--prompt required")
        rounds = independent(args)
        summary = summarize_independent(rounds)
    else:
        if not args.long_prompt or not args.short_prompt:
            ap.error("--long-prompt and --short-prompt required")
        rounds = mixed(args)
        summary = {}

    after = get_json(args.url, "/runtime")
    out = {
        "schema": "air.prompt1.multiclient.v1",
        "mode": args.mode,
        "runtime_before": before,
        "runtime_after": after,
        "rounds": rounds,
        "summary": summary,
    }
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

if __name__ == "__main__":
    main()

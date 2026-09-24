#!/usr/bin/env python3
"""Compare AIR teacher-forced next-token evidence against llama.cpp /completion.

The AIR report supplies exact numeric prompt and forced token IDs. llama.cpp is
queried with numeric-token prompts, so chat templates and tokenizer round trips
cannot alter the sequence being compared.
"""
import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path


def post_json(url, payload, timeout):
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def top_entries(response):
    probs = response.get("probs")
    if probs is None:
        probs = response.get("completion_probabilities")
    if not probs:
        return []
    first = probs[0]
    entries = first.get("top_logprobs") or first.get("top_probs") or []
    result = []
    for item in entries:
        value = item.get("logprob", item.get("prob"))
        result.append({
            "token": int(item["id"]),
            "score": float(value),
            "piece": item.get("token", ""),
        })
    return result


def margin(entries):
    if len(entries) < 2:
        return math.inf
    return float(entries[0]["score"] - entries[1]["score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("air_report")
    ap.add_argument("--url", default="http://127.0.0.1:18181")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if not 1 <= args.top_k <= 128:
        ap.error("--top-k must be in [1,128]")

    air = json.loads(Path(args.air_report).read_text())
    if air.get("schema") != "air.verification.v1":
        raise SystemExit("input is not an air.verification.v1 report")
    prompt = [int(x) for x in air["prompt_tokens"]]
    forced = [int(x) for x in air["teacher_forced_tokens"]]
    decisions = air["decisions"]
    if len(forced) != len(decisions):
        raise SystemExit("AIR teacher token/decision lengths differ")

    results = []
    first_mismatch = None
    exact = 0
    for index, air_decision in enumerate(decisions):
        numeric_prompt = prompt + forced[:index]
        payload = {
            "prompt": numeric_prompt,
            "n_predict": 1,
            "temperature": -1,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "cache_prompt": False,
            "return_tokens": True,
            "n_probs": args.top_k,
            "post_sampling_probs": False,
            "ignore_eos": True,
        }
        try:
            response = post_json(args.url.rstrip("/") + "/completion", payload, args.timeout)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SystemExit(f"llama.cpp request failed at decision {index}: {exc}") from exc
        tokens = response.get("tokens") or []
        if not tokens:
            raise SystemExit(f"llama.cpp returned no token at decision {index}: {response}")
        llama_token = int(tokens[0])
        llama_top = top_entries(response)
        air_top = int(air_decision["reference_top"])
        match = air_top == llama_token
        if match:
            exact += 1
        elif first_mismatch is None:
            first_mismatch = index
        result = {
            "index": index,
            "air_top": air_top,
            "llama_top": llama_token,
            "match": match,
            "air_top1_top2_margin": float(air_decision["reference_top1_top2_margin"]),
            "llama_top1_top2_margin": margin(llama_top),
            "air_top_k": air_decision["reference_top_k"],
            "llama_top_k": llama_top,
            "llama_timings": response.get("timings", {}),
        }
        results.append(result)
        status = "match" if match else "MISMATCH"
        print(f"decision {index}: AIR/llama={air_top}/{llama_token} "
              f"air_margin={result['air_top1_top2_margin']:.9g} "
              f"llama_margin={result['llama_top1_top2_margin']:.9g} {status}")

    output = {
        "schema": "air.external_verification.v1",
        "air_report": str(Path(args.air_report).resolve()),
        "llama_url": args.url,
        "top_k": args.top_k,
        "prompt_tokens": prompt,
        "teacher_forced_tokens": forced,
        "decisions": results,
        "summary": {
            "decisions": len(results),
            "exact_top1_matches": exact,
            "all_top1_match": exact == len(results),
            "first_top1_mismatch": first_mismatch,
        },
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(f"Report: {args.output}")
    if first_mismatch is None:
        print("External top-1 parity: yes")
    else:
        print(f"External top-1 parity: no; first mismatch at decision {first_mismatch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

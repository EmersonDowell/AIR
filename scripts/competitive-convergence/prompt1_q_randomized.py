#!/usr/bin/env python3
import argparse, json, random, subprocess, time
from pathlib import Path

def gpu():
    p=subprocess.run([
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.mem",
        "--format=csv,noheader,nounits"
    ],text=True,capture_output=True)
    return p.stdout.strip() if p.returncode==0 else ""

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bench",type=Path,required=True)
    ap.add_argument("--model",type=Path,required=True)
    ap.add_argument("--prompt-dir",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--rounds",type=int,default=5)
    ap.add_argument("--seed",type=int,default=1501)
    ap.add_argument("--cooldown",type=float,default=1.0)
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)

    cells=[(262,q) for q in (32,64,128)] + [(1042,q) for q in (32,64,128)]
    rng=random.Random(args.seed)
    records=[]
    for rnd in range(1,args.rounds+1):
        order=cells[:]
        rng.shuffle(order)
        for target,q in order:
            out=args.output_dir/f"p{target}-q{q}-r{rnd}.json"
            cmd=[
                str(args.bench),"-m",str(args.model),
                "--backend","cuda","--no-manifest",
                "--cuda-prefill-block-linear","reuse8",
                "--cuda-decode-block-linear","reuse8",
                "--cuda-decode-output-linear","reuse8",
                "--cuda-prefill-attention","online-softmax",
                "--prompt-file",str(args.prompt_dir/f"p{target}.txt"),
                "--tokens","8","--warmup","1","--runs","1","--concurrency","1",
                "--prefill-quantum",str(q),"--token-budget","512",
                "--output",str(out)
            ]
            before=gpu()
            start=time.perf_counter()
            p=subprocess.run(cmd,text=True,capture_output=True)
            elapsed=time.perf_counter()-start
            after=gpu()
            (args.output_dir/f"p{target}-q{q}-r{rnd}.txt").write_text(
                p.stdout+"\n--- stderr ---\n"+p.stderr)
            row={"round":rnd,"target":target,"q":q,"rc":p.returncode,
                 "elapsed_s":elapsed,"gpu_before":before,"gpu_after":after,
                 "path":str(out)}
            if p.returncode==0 and out.exists():
                j=json.loads(out.read_text())
                run=j.get("runs",[{}])[0]
                row.update({
                    "prefill_tps":run.get("prefill_tokens_per_second"),
                    "decode_tps":run.get("decode_tokens_per_second"),
                    "ttft_ms":run.get("ttft_ms"),
                    "total_ms":run.get("total_ms"),
                    "prefill_ms":run.get("prefill_ms"),
                })
            records.append(row)
            print(json.dumps(row,sort_keys=True),flush=True)
            time.sleep(args.cooldown)

    (args.output_dir/"q-randomized.json").write_text(
        json.dumps({"schema":"air.prompt1.q-randomized.v1",
                    "seed":args.seed,"rounds":args.rounds,"records":records},
                   indent=2,sort_keys=True)+"\n")
if __name__=="__main__":
    main()

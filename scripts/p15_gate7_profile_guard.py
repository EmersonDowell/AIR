#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,re
from pathlib import Path
REQUIRED={
"--cuda-prefill-block-linear":"batch-reuse8",
"--cuda-decode-block-linear":"batch-reuse8",
"--cuda-decode-output-linear":"batch-reuse8",
"--cuda-prefill-attention":"online-softmax",
"--cuda-decode-attention":"baseline",
}
def validate_collector(text):
    errors=[]; m=re.search(r"# Gate 7:.*?(?=# Gate 8:)",text,re.S)
    if not m:return {"schema":"air.prompt15.gate7-profile.v2","status":"FAIL","errors":["Gate 7 block not found"]}
    block=m.group(0); calls=[x for x in re.findall(r'\"\$VERIFY\".*?(?=\n\s*(?:rc=|if |\"\$VERIFY\"|#|$))',block,re.S) if "--generate" in x]
    if len(calls)<2: errors.append(f"expected main verification and trace verification commands, found {len(calls)}")
    for i,cmd in enumerate(calls):
        normalized=" ".join(line.strip().rstrip("\\") for line in cmd.splitlines())
        for flag,val in REQUIRED.items():
            if f"{flag} {val}" not in normalized: errors.append(f"Gate7 verify command {i} does not pin {flag}={val}")
    return {"schema":"air.prompt15.gate7-profile.v2","status":"PASS" if not errors else "FAIL","required":REQUIRED,"verify_commands":len(calls),"errors":errors}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--collector",type=Path,required=True); ap.add_argument("--verify-help",type=Path); ap.add_argument("--output",type=Path); a=ap.parse_args()
    r=validate_collector(a.collector.read_text())
    if a.verify_help:
        h=a.verify_help.read_text(errors="replace"); missing=[f for f in REQUIRED if f not in h]; r["verify_help_missing_flags"]=missing
        if missing:r["errors"].append("air-verify --help does not advertise: "+", ".join(missing)); r["status"]="FAIL"
    s=json.dumps(r,indent=2,sort_keys=True)+"\n"
    if a.output:a.output.write_text(s)
    print(s,end=""); return 0 if r["status"]=="PASS" else 2
if __name__=="__main__":raise SystemExit(main())

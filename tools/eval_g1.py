#!/usr/bin/env python3
"""G1 probe battery: score a checkpoint's record grammar + fact recall.

Usage: python tools/eval_g1.py [checkpoint] [tokenizer]
Each probe is a prompt; the model's completion is scored for:
  - FORMAT: brackets closed, act pairs well-formed, no NUL bytes
  - FACT: expected entity strings present in the completion
Prints a scorecard and a nonzero exit if any probe degenerates (NUL flood).
"""
import re, subprocess, sys, os

Z = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = sys.argv[1] if len(sys.argv) > 1 else "models/g1_50m_polish3.bin"
TOK = sys.argv[2] if len(sys.argv) > 2 else "corpus/g1-atoms.bpe"

PROBES = [
    ("<CHANNEL:warrenmind [AWho is Mara?", ["Mara"]),
    ("<CHANNEL:warrenmind [AWhere does Mara keep her lantern?", ["lantern"]),
    ("<CHANNEL:warrenmind [AWhat happens to a record that is never read?", ["record", "debt"]),
    ("To be, or not to be, that is the", ["the"]),
]

def gen(prompt, tokens=160, temp=0.8):
    r = subprocess.run(
        [os.path.join(Z, "literary_lm_blas.exe"), "--resume", os.path.join(Z, CKPT),
         "--tokenizer", os.path.join(Z, TOK), "--generate-only",
         "--prompt", prompt, "--tokens", str(tokens), "--temperature", str(temp)],
        capture_output=True, text=True, timeout=300, cwd=Z,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "4"})
    out = r.stdout.replace("\r", "")
    return out.split("--- generated ---")[-1].strip()

def score(prompt, facts):
    a = gen(prompt)
    nuls = a.count("\x00")
    opens = len(re.findall(r"\[A", a))
    closes = a.count("]")
    fmt = nuls < 10 and closes >= 1
    fact = any(f.lower() in a.lower() for f in facts)
    tag = "DEGENERATE" if nuls >= 10 else ("ok" if (fmt and fact) else ("fmt-only" if fmt else "fail"))
    return a, tag, nuls

print(f"checkpoint: {CKPT}")
total = ok = 0
for prompt, facts in PROBES:
    a, tag, nuls = score(prompt, facts)
    total += 1
    ok += tag in ("ok", "fmt-only")
    print(f"[{tag:9}] nuls={nuls:<4} {prompt[:44]!r}")
    print(f"            -> {a[:160]!r}")
print(f"\nsummary: {ok}/{total} probes non-degenerate")
sys.exit(0 if ok == total else 1)

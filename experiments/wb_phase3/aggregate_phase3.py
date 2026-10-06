#!/usr/bin/env python3
"""Warren Braid Phase 3 -- aggregate per-world eval CSV into the report table."""
import csv
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def agg(rows, keyfn):
    """Token-weighted mean nll / accuracy over rows grouped by keyfn(row)."""
    out = {}
    groups = defaultdict(lambda: [0.0, 0.0, 0])  # nll_sum, top1_sum, tokens
    for r in rows:
        k = keyfn(r)
        n = int(r["tokens"])
        g = groups[k]
        g[0] += float(r["nll_mean"]) * n
        g[1] += float(r["top1_acc"]) * n
        g[2] += n
    for k, (s, t, n) in groups.items():
        out[k] = (s / n, t / n, n, len(groups) and sum(1 for r in rows if keyfn(r) == k))
    return out


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "eval_per_world.csv")
    rows = load(csv_path)
    for arm in ("armA", "armB"):
        for mode in ("text", "cond"):
            sel = [r for r in rows if r["label"] == arm and r["mode"] == mode]
            if not sel:
                continue
            nll = sum(float(r["nll_mean"]) * int(r["tokens"]) for r in sel)
            tok = sum(int(r["tokens"]) for r in sel)
            acc = sum(float(r["top1_acc"]) * int(r["tokens"]) for r in sel)
            cn = sum(int(r["copy_tokens"]) for r in sel)
            cc = sum(float(r["copy_nll_mean"]) * int(r["copy_tokens"]) for r in sel)
            cacc = sum(float(r["copy_top1_acc"]) * int(r["copy_tokens"]) for r in sel)
            nn = sum(int(r["novel_tokens"]) for r in sel)
            nc = sum(float(r["novel_nll_mean"]) * int(r["novel_tokens"]) for r in sel)
            nacc = sum(float(r["novel_top1_acc"]) * int(r["novel_tokens"]) for r in sel)
            print("%s %s: worlds=%d tokens=%d nll=%.4f top1=%.4f | "
                  "copy n=%d nll=%.4f top1=%.4f | novel n=%d nll=%.4f top1=%.4f"
                  % (arm, mode, len(sel), tok, nll / tok, acc / tok,
                     cn, cc / max(cn, 1), cacc / max(cn, 1),
                     nn, nc / max(nn, 1), nacc / max(nn, 1)))


if __name__ == "__main__":
    main()

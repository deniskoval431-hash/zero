#!/usr/bin/env python3
"""Converter v2: crownless core_pairs -> question-shaped G1 act records.

Genre fix over v1: the [A slot holds a real QUESTION built from the record's
fields, the [Z>A slot holds the event text — so the model learns dialog
adjacency (Q -> A), not just record-opening.

Templates by role pattern; fallback composes from kind + where + who.
Doubt doctrine: confidence < 40 gets ' ~?' after the closing bracket.
"""
import json, random, sys

DOUBT_BELOW = 40

def build_question(d, who, where, what):
    kind = (d.get("kind") or "EVENT").lower().replace("_", " ")
    if who and where and what:
        return random.Random(d.get("id", "")).choice([
            f"What happened with {what} in {where}?",
            f"Who {kind} in {where}?",
        ])
    if who and where:
        return f"What did {who} do in {where}?"
    if where and what:
        return f"What became of {what} at {where}?"
    if who:
        return f"What did {who} do?"
    if where:
        return f"What happened at {where}?"
    return f"What was the {kind}?"

def record_from(d):
    out = (d.get("output") or "").strip().rstrip(".")
    if not out:
        return None
    fields = d.get("fields", [])
    who = next((f["text"] for f in fields if f.get("role") == 1), None)
    where = next((f["text"] for f in fields if f.get("role") == 3), None)
    what = next((f["text"] for f in fields if f.get("role") == 4), None)
    q = build_question(d, who, where, what)
    conf = d.get("confidence") or 50
    doubt = " ~?" if conf < DOUBT_BELOW else ""
    return f"[A{q}][Z>A{out}.]{doubt}\n"

def main():
    src, dst = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 99991
    rows = open(src, encoding="utf-8").readlines()
    random.Random(seed).shuffle(rows)
    n = 0
    with open(dst, "w", encoding="utf-8", newline="\n") as g:
        for line in rows:
            if n >= limit:
                break
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            r = record_from(d)
            if r:
                g.write(r)
                n += 1
    print(f"wrote {n} question-shaped act records -> {dst}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert crownless core_pairs JSONL into G1 act-marker records.

Record format (one event per record, act grammar):
  [AKIND ~ place ~ actor][Z>Aoutput text][~? if confidence low]

- '~' and '?' are atomic bytes in the G1 vocab.
- Confidence < DOUBT_BELOW gets a doubt mark: ' ~?' inside the closing bracket,
  per the doubt-mark doctrine (a record states its own uncertainty).
- Output: UTF-8 text stream, later tokenized by bpe_tokenizer with the G1 vocab.
"""
import json, random, sys

DOUBT_BELOW = 40
FIELDS_ROLES = {1: "who", 3: "where", 4: "what"}

def record_from(d, place_of):
    kind = d.get("kind", "EVENT")
    conf = d.get("confidence") or 50
    out = (d.get("output") or "").strip().rstrip(".")
    if not out:
        return None
    # subject from fields: prefer who, else where
    who = next((f["text"] for f in d.get("fields", []) if f.get("role") == 1), None)
    where = next((f["text"] for f in d.get("fields", []) if f.get("role") == 3), None)
    head = " ~ ".join(x for x in (kind, where, who) if x)
    doubt = " ~?" if conf < DOUBT_BELOW else ""
    return f"[A{head}][Z>A{out}.]{doubt}\n"

def main():
    src, dst = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 99991
    rows = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            rows.append(line)
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
            r = record_from(d, None)
            if r:
                g.write(r)
                n += 1
    print(f"wrote {n} act records -> {dst}")

if __name__ == "__main__":
    main()

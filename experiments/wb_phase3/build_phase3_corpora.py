#!/usr/bin/env python3
"""
Warren Braid Phase 3 -- build the two training corpora + eval files.

Arm A (baseline):   the 800 train worlds' TEXT records only.
Arm B (conditioned): per world, the FULL wbv1 symbolic file's lines followed
                     by that world's text record (world-aligned pairing).
Validation:         100 val worlds' text records, one file per world for
                    per-world eval, plus one combined file.

The corpora carry a three-line header comment in the corpus files' style of
plain prose so the tokenizer sees only corpus content; records are separated
by a blank line.  No split of `src/`, `spec/`, `recon/` is touched.
"""
import json
import os
import sys

CORPUS_ROOT = "C:/Users/agent/warrenbraid/corpus"
RELEASES = ["release-a", "release-b", "release-c", "release-d"]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(OUT_DIR, "eval_val_worlds")


def world_text(release, world_id):
    """Return the text field of data/text.jsonl for one world id."""
    path = os.path.join(CORPUS_ROOT, release, "data", "text.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["world_id"] == world_id:
                return rec["text"]
    raise KeyError(world_id)


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    train, val = [], []
    for release in RELEASES:
        manifest = json.load(open(
            os.path.join(CORPUS_ROOT, release, "manifest.json"),
            "r", encoding="utf-8"))
        for w in manifest["worlds"]:
            wid = w["world_id"]
            split = w["split"]
            text = world_text(release, wid)
            if split == "train":
                train.append((release, wid, text))
            elif split == "validation":
                val.append((release, wid, text))
            # split == test is deliberately excluded per the brief.
    print("train worlds: %d, validation worlds: %d" % (len(train), len(val)))
    assert len(train) == 800, "expected 800 train worlds, got %d" % len(train)
    assert len(val) == 100, "expected 100 validation worlds, got %d" % len(val)

    # ---- Arm A: text only -------------------------------------------------
    with open(os.path.join(OUT_DIR, "corpus_A.txt"), "w",
              encoding="utf-8", newline="\n") as out:
        for _, _, text in train:
            out.write(text.strip() + "\n\n")

    # ---- Arm B: full wbv1 lines, then the world's text --------------------
    with open(os.path.join(OUT_DIR, "corpus_B.txt"), "w",
              encoding="utf-8", newline="\n") as out:
        for release, wid, text in train:
            wbv1_path = os.path.join(CORPUS_ROOT, release, "wbv1", wid + ".wbv1")
            with open(wbv1_path, "r", encoding="utf-8") as f:
                symbolic = f.read().rstrip("\n")
            out.write(symbolic + "\n" + text.strip() + "\n\n")

    # ---- Validation: per-world text files + combined -----------------------
    with open(os.path.join(OUT_DIR, "val_text_all.txt"), "w",
              encoding="utf-8", newline="\n") as allout:
        for release, wid, text in val:
            body = text.strip() + "\n"
            with open(os.path.join(EVAL_DIR, wid + ".txt"), "w",
                      encoding="utf-8", newline="\n") as f:
                f.write(body)
            allout.write(body + "\n")
    # per-world eval list for the driver
    with open(os.path.join(OUT_DIR, "val_worlds.csv"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write("release,world_id\n")
        for release, wid, _ in val:
            f.write("%s,%s\n" % (release, wid))
    print("wrote corpus_A.txt, corpus_B.txt, val_text_all.txt, %d per-world files"
          % len(val))


if __name__ == "__main__":
    main()

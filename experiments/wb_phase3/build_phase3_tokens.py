#!/usr/bin/env python3
"""
Warren Braid Phase 3 -- token-level artifacts.

Context is 512 (the g1/mara literary preset).  A FULL wbv1 file is 3.4K-53K
byte-tokens (median 28.8K), so a full symbolic file + text record can never
sit inside one 512-token window on this trainer.  DEVIATION (flagged in the
report, never silent): the symbolic prefix of each Arm B record is the LONGEST
suffix of the wbv1 file cut on a line boundary that fits the window budget,
followed by the TARGET token and the first TEXT_SEG bytes of the world's text.

Channel record layout (uint16 LE token stream, ids per channel_protocol.h):
  1 (START) 'W'  <symbolic tail bytes>  6 (TARGET)  <text bytes>  4 5
The trainer's channel_loss_mask computes loss ONLY from TARGET onward, so
Arm B training = text continuation conditioned on the symbolic prefix.
"""
import json
import os
import struct

CONTEXT = 512
TEXT_SEG = 128
CONTROL = 4  # START + style + TARGET + END_MESSAGE + END_RECORD = 1+1+1+1+1

CORPUS_ROOT = "C:/Users/agent/warrenbraid/corpus"
RELEASES = ["release-a", "release-b", "release-c", "release-d"]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

PROTO_START, PROTO_TARGET, PROTO_END_MESSAGE, PROTO_END_RECORD = 1, 6, 4, 5
STYLE = ord("W")


def world_text(release, world_id):
    path = os.path.join(CORPUS_ROOT, release, "data", "text.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["world_id"] == world_id:
                return rec["text"].encode("utf-8")
    raise KeyError(world_id)


def wbv1_tail(release, world_id, budget):
    """Longest line-aligned suffix of the wbv1 file within `budget` bytes."""
    path = os.path.join(CORPUS_ROOT, release, "wbv1", world_id + ".wbv1")
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().rstrip("\n").split("\n")
    sym_budget = budget - TEXT_SEG - CONTROL - 1  # -1: newline joining prefix/text
    tail, total = [], 0
    for line in reversed(lines):
        chunk = len(line.encode("utf-8")) + 1  # + newline
        if total + chunk > sym_budget:
            break
        tail.append(line)
        total += chunk
    tail.reverse()
    return ("\n".join(tail) + "\n").encode("utf-8")


def record(release, world_id, text):
    sym = wbv1_tail(release, world_id, CONTEXT)
    body = bytearray()
    body.append(PROTO_START)
    body.append(STYLE)
    body += sym
    body.append(PROTO_TARGET)
    body += text[:TEXT_SEG]
    body.append(PROTO_END_MESSAGE)
    body.append(PROTO_END_RECORD)
    assert len(body) <= CONTEXT + 1, (world_id, len(body))
    return body


def write_tokens(path, data):
    with open(path, "wb") as f:
        f.write(struct.pack("<%dH" % len(data), *data))


def main():
    val_dir = os.path.join(OUT_DIR, "eval_val_tok")
    cond_dir = os.path.join(OUT_DIR, "eval_cond_tok")
    for d in (val_dir, cond_dir):
        os.makedirs(d, exist_ok=True)

    train, val = [], []
    for release in RELEASES:
        manifest = json.load(open(os.path.join(CORPUS_ROOT, release,
                                               "manifest.json"),
                                  "r", encoding="utf-8"))
        for w in manifest["worlds"]:
            if w["split"] in ("train", "validation"):
                (train if w["split"] == "train" else val).append(
                    (release, w["world_id"]))
    print("train %d val %d" % (len(train), len(val)))

    # Arm B channel training file
    out = bytearray()
    for release, wid in train:
        out += record(release, wid, world_text(release, wid))
    write_tokens(os.path.join(OUT_DIR, "channel_B.tok"), out)
    print("channel_B.tok: %d tokens, %d records" % (len(out), len(train)))

    # eval files: text-only and conditioned, per validation world.
    # Cond files are padded to CONTEXT+1 tokens: the trainer evaluates full
    # context windows only, and the padding sits after the record's
    # END_MESSAGE token, where the channel loss mask is inactive and the
    # causal attention cannot let it influence the symbolic->text segment.
    for release, wid in val:
        text = world_text(release, wid)
        write_tokens(os.path.join(val_dir, wid + ".text.tok"), text)
        r = bytearray(record(release, wid, text))
        r += b"\n" * (CONTEXT + 1 - len(r))
        write_tokens(os.path.join(cond_dir, wid + ".cond.tok"), r)
    print("wrote %d text eval + %d cond eval token files" % (len(val), len(val)))


if __name__ == "__main__":
    main()

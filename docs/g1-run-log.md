# G1 run log — first 50M train + polish collapse (2026-09-29)

## Run 1: base (models/g1_50m_20k_snapshot.bin)
- Config: preset literary (context 512, dim 256, heads 8, layers 6, ff 1056, rotary),
  vocab 133 (G1 atomic act-markers <CHANNEL:, <SUMMARY>, <END>, [Z>A, [A]),
  corpus: shakespeare 5.35M + blake + crowley tokens, atom records 11K,
  channel-protocol records weight 8. 20,000 steps, constant lr 3e-4, batch 1.
- Result: 5,457 s, val 0.508 (18K) -> 0.502 (19K) -> 0.492 (20K). Near-saturation
  at constant lr. Final sample spoke record language: "Eternity is the Law...
  friction of dissenting parts... the mycelium of minds]<END>".
- Trajectory of probes: 5K character soup w/ verse shape -> 10K format compliance
  (model closes [A...? bracket unprompted) -> 16K real words + register shift.

## Run 2: polish (models/g1_50m_polish.bin) — FAILED, collapse (confirmed)
- Config: resume 20K, lr 1e-4, channel-record weight 8 -> 24 (~50% stream share).
- BUG in the run command: --steps on --resume means ADDITIONAL updates (docs say
  so); passing 25000 scheduled 45K total, not 25K. Killed at ~26.7K.
- Result: model degenerates to a NUL attractor — after ANY off-record prompt
  ([A probe, literary prompt) it emits endless token-0 bytes. Confirmed at both
  step 23K and 26K checkpoints.
- MISLEADING signal: val loss fell to 0.127 while the model was degenerate.
  Cause: the validation split is re-weighted with the training corpus, so at
  weight 24 it is dominated by repetitive record templates the model
  memorized. Val loss across runs with different corpus weights is NOT
  comparable. Lesson: keep the evaluation set and its weighting FIXED.
- ROOT CAUSE FOUND (second postmortem): the polish runs omitted --tokenizer.
  A --resume restores architecture + weights but NOT the tokenizer; without
  --tokenizer the uint16 token files are read as raw bytes — exactly 2x the
  token count (11,482,902 vs 5,802,468), half of them NUL. The model was
  trained on a ~50% NUL stream: the NUL attractor was fed, not emergent.
  RETRACTED: "curriculum breaking point at ~50% record share" — weight 24
  was never validly tested. Fix: always re-state --tokenizer on resume, and
  check the printed corpus token count against expectation before training.

## Lessons (the record)
0. A model can post a beautiful val loss and still be broken — validate on a
   fixed, unweighted split, and always probe generation before believing the
   curve.
1. Steps are the wrong lever near saturation; curriculum and lr schedule are the
   levers. But curriculum has a breaking point: at 4.85M params, ~50% share of
   the old-format channel protocol destroys the model even at lr 1e-4.
2. Mixed-format corpora (old channel-protocol control tokens vs new atom
   tokens) fight when heavily weighted — re-balance gradually (weight 12), not
   in one step.
3. Snapshot doctrine: keeping the 20K checkpoint separate before any polish
   turned a model-destroying run into a cheap lesson.
4. OpenBLAS backend matters 30x: the repo shipped libopenblas.dll without
   cblas.h; without -DUSE_OPENBLAS the trainer silently used portable-C
   (71 tok/s vs 2100 tok/s).
5. Saturation verdict at 20K: more steps at constant lr = flat curve. The gain
   is in record share (carefully), lr decay, and eventually the 1024-context
   arm on bigger hardware.


## Run 3: polish3 (models/g1_50m_polish3.bin) — SUCCESS
- Config: resume 20K, --tokenizer STATED (the fix), channel weight 12, lr 1e-4,
  +5,000 steps (25K total), 1,421 s.
- Val 0.356 at close (val split re-weighted at w12 — compare within-run trend,
  not across weights; base at w8 read 0.492).
- Final probes — the record grammar SURFACED:
  [AWho is Mara? -> "Mara Venn dwells of ... a prediction of dissenting parts
  holding their ground.] [A>ZWhat is a prediction in the warren?]
  [Z>A] => A prediction is not a promise. Claims are ..."
  Model recalls the record entity (Mara Venn), self-sustains [A]/[Z>A act
  pairs, and quotes record doctrine ("A prediction is not a promise").
  Literary probe stays competent.
- Verdict: format + facts fusion proven at 4.85M params. G1 gate passed.


## Run 4: acts (models/g1_50m_acts.bin) — curriculum genre finding
- Config: resume 20K, --tokenizer stated, +2.99M act-record tokens (30K core_pairs
  as [AKIND ~ where ~ who][Z>Aevent]), channel w8, lr 1e-4, +5,000 steps (25K).
- Val 0.490 (val split now act-dominated — not comparable to polish3's 0.356).
- Finding: the act genre mattered more than act volume. Event-style records
  ([A KIND ~ place]) taught record-opening, not Q->A dialog: single-turn Mara
  recall decayed (model re-opens the channel header instead of answering),
  while world knowledge grew — turn 3 of the thread test fused the lantern
  fact with acts material ("the horses starve") in one in-format answer.
- Keepers: polish3 = conversation model; acts = world-knowledge model.
- Next: converter v2 — question-shaped [A slots built from core_pair fields,
  so volume and genre match.

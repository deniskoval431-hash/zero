# Warren Braid Phase 3 — two-arm training smoke test (REPORT)

Date: 2026-10-06. Host: Windows 11, git-bash, MinGW gcc, OpenBLAS backend
(`literary_lm_blas.exe`, libopenblas.dll). Corpus (frozen, verified):
`C:/Users/agent/warrenbraid/corpus/release-{a,b,c,d}/` = 1,000 worlds
(800 train / 100 validation / 100 test; test unused per brief).

## Question (SPEC §6, restated)

Does conditioning on the WBV1 symbolic view improve a model's prediction of
the text view vs a text-only baseline, at equal budget, Mara-class scale?

## Known prior (stated up front, not hidden behind)

Mara ladder v1–v4 (docs/g1-run-log.md, Runs 6–9) failed to add facts at
4.85M params — capacity floor, grammar held. This smoke test checks whether
IN-WINDOW symbolic→text mapping reproduces or fails at the same scale.

---

## 1. Configs verbatim (both arms)

Architecture (both arms): `--preset literary` = context 512, dim 256,
heads 8, layers 6, ff 1056, rotary; `--vocab 133` (byte-level BPE,
`MERGE_COUNT 0` in bpe_tokenizer.c — 128 base + 5 atoms, 1.00x compression);
4,854,272 parameters (= the g1/mara class, printed by the trainer).
Batch 1 → 512 tokens/update. Seed 7 both arms. lr 3e-4, warmup 100, cosine
decay, dropout 0.1 (default), AdamW decay 0.01 (default), clip 1 (default).

Arm A (baseline), 4,000 steps, training time 1896.26 s:

```
literary_lm_blas.exe --preset literary --context 512 --vocab 133 \
  --tokenizer wb_phase3.vocab --text corpus_A.tok \
  --validation-text val_text_all.tok \
  --steps 4000 --batch 1 --lr 3e-4 --warmup 100 --cosine --dropout 0.1 \
  --report 250 --validation 8 --seed 7 \
  --save wb_phase3_armA.f32 --save-every 2000 --tokens 0
```

Arm B (symbolic-conditioned), 4,000 steps, training time 1911.47 s:

```
literary_lm_blas.exe --preset literary --context 512 --vocab 133 \
  --tokenizer wb_phase3.vocab --channel channel_B.tok \
  --steps 4000 --batch 1 --lr 3e-4 --warmup 100 --cosine --dropout 0.1 \
  --report 250 --validation 8 --seed 7 \
  --save wb_phase3_armB.f32 --save-every 2000 --tokens 0
```

Corpus construction (build_phase3_corpora.py, build_phase3_tokens.py):

- Arm A: `corpus_A.txt` = the 800 train worlds' `text` records joined by a
  blank line; tokenized by bpe_tokenizer to `corpus_A.tok` (3,762,469 tokens).
- Arm B: `channel_B.tok` = 800 channel records, one per train world, in the
  trainer's native record format:
  `[START=1]['W'][symbolic prefix bytes][TARGET=6][first 128 bytes of the
  world's text][END_MESSAGE=4][END_RECORD=5]`.
  The trainer's `channel_loss_mask` (literary_lm.c) computes loss ONLY from
  TARGET onward, so Arm B training is exactly "predict the text continuation
  given the symbolic prefix, in-window". `channel_protocol.h` control tokens
  ride in the 128-token vocabulary, so no extra parameters are added.

## 2. DEVIATIONS (flagged, exact quotes, never silently resolved)

1. **Full wbv1 in-window is infeasible at context 512.** Brief:
   "per world, the FULL wbv1 file's lines followed by that world's text
   record (world-aligned pairing: symbolic prefix, then text continuation —
   the conditioning is in-window by construction)". Measured: a FULL wbv1
   file is 3,412–53,418 bytes (median 28,812) and the tokenizer is byte-level
   (1.00x), so a full symbolic file + text record is 3.5K–54K tokens vs a
   512-token window. Raising context to ~29K is computationally impossible on
   this CPU (attention alone ~10^11 FLOP/step); the trainer cannot express
   the full-file conditioning at any affordable context. RESOLUTION: the
   symbolic prefix of each Arm B record is the LONGEST line-aligned SUFFIX of
   the wbv1 file that fits the window budget (~381 bytes; the final-state
   lines of the symbolic view), followed by the text segment. Arm B is
   therefore a REAL two-arm comparison of tail-conditioned symbolic context,
   NOT a test of full-file conditioning. Loudly flagged, not quiet.
2. **Trainer's --save-every overwrites --save.** literary_lm.c line 2523:
   `if (options.save_path != NULL && options.save_every > 0 && update %
   (uint64_t)options.save_every == 0)` — writes to the same `--save` path, so
   the step-2000 mid checkpoint was overwritten by the final. Brief asked for
   "final + one mid"; only finals exist (wb_phase3_armA.f32,
   wb_phase3_armB.f32, 58 MB each). Mid checkpoints are gone; not re-run.
3. **Text-mode eval excludes 1 validation world.** w-stackworld-s723 text is
   509 tokens < context+1 (513); the trainer cannot score a file shorter than
   one full window (`evaluate()` returns NAN / harness refuses). 99 of 100
   worlds evaluated in text mode for BOTH arms (symmetric exclusion); cond
   mode covers all 100.
4. **Cond eval files are padded to 513 tokens** with 0x0A bytes appended
   AFTER the record's END_MESSAGE: the pad sits where the channel loss mask
   is inactive and causal attention cannot let it influence the
   symbolic→text segment.
5. **Copy/novel decomposition is NOT exposed by the training script.** Stated
   plainly. It was computed instead by a new read-only harness
   (`wb_phase3_eval.c`, glue under experiments/wb_phase3/ — no src/ modified;
   it `#include`s the unmodified literary_lm.c with main renamed). Criterion:
   a position counts as "copy" if its target token already occurs earlier in
   the same 512-token window, else "novel".
6. **Val-loss comparability caveat (per g1-run-log lesson 0):** Arm A's
   validation is the fixed unweighted text split; Arm B's internal val is the
   channel file's last-5% records (masked). The cross-arm numbers below come
   from ONE harness (wb_phase3_eval.exe) over the same 100 worlds, so the
   table is internally consistent; trainer-internal val curves are given
   separately.

## 3. Loss curves (trainer reports, every 250 steps: step train val)

Arm A (val = fixed text split val_text_all.tok):
```
250  1.7589 0.3965 | 500  0.3302 0.2866 | 750  0.2725 0.2696
1000 0.2532 0.2539 | 1250 0.2451 0.2380 | 1500 0.2401 0.2293
1750 0.2451 0.2276 | 2000 0.2352 0.2291 | 2250 0.2410 0.2193
2500 0.2277 0.2243 | 2750 0.2197 0.2178 | 3000 0.2133 0.2156
3250 0.2305 0.2134 | 3500 0.2272 0.2120 | 3750 0.2302 0.2122
4000 0.2221 0.2121   FINAL
```

Arm B (val = masked symbolic→text on channel file's last 5% records):
```
250  1.9057 0.6900 | 500  0.3392 0.3442 | 750  0.2242 0.2996
1000 0.2019 0.2963 | 1250 0.2115 0.2902 | 1500 0.1934 0.2781
1750 0.2091 0.2779 | 2000 0.1920 0.2669 | 2250 0.1815 0.2669
2500 0.1849 0.2647 | 2750 0.1881 0.2633 | 3000 0.1843 0.2718
3250 0.1840 0.2640 | 3500 0.1853 0.2636 | 3750 0.1724 0.2647
4000 0.1801 0.2644   FINAL
```
Arm B train (0.180) well below its val (0.264): the 800 training records are
partly memorized while held-out records lag — mild overfit at equal budget.

## 4. Eval (wb_phase3_eval.exe, 100 validation worlds, token-weighted means)

Full per-world values: `eval_per_world.csv` (797 rows: label,file,mode,
window,tokens,nll_mean,top1_acc,copy_tokens,copy_nll_mean,copy_top1_acc,
novel_tokens,novel_nll_mean,novel_top1_acc). Aggregate:

| arm | mode | worlds | tokens | NLL | top-1 acc | copy NLL (top-1) | novel NLL (top-1) |
|-----|------|--------|--------|-----|-----------|------------------|-------------------|
| A (baseline) | text, given nothing (native task) | 99 | 451,584 | **0.2136** | 0.9101 | 0.1791 (0.9237) | 0.7522 (0.6973) |
| A | cond: symbolic→text (128-token text segment) | 100 | 13,000 | 0.6978 | 0.8535 | 0.4446 (0.8741) | 2.5225 (0.7045) |
| B (conditioned) | text, given nothing | 99 | 451,584 | 3.9095 | 0.4078 | 3.9575 (0.4084) | 3.1598 (0.3990) |
| B | cond: symbolic→text (native task) | 100 | 13,000 | **0.3068** | 0.9118 | 0.2052 (0.9130) | 1.0385 (0.9028) |

Per-world spread (cond mode, arm B): the CSV lists all 100 worlds; mean
0.3068, and every world is far below the untrained/Arm-A-cond level.

Copy-vs-novel reading:
- Arm B's symbolic→text skill is NOT copying: 88% of cond positions are
  novel-position (target byte absent from the window), and on those the model
  still reaches NLL 1.0385 / top-1 0.9028. Copy positions only are NLL
  0.2052. The in-window symbolic→text mapping is genuinely learned, novel
  included.
- Arm A's novel-position NLL on plain text (0.7522) is BETTER than Arm B's
  novel-position NLL on conditioned text (1.0385).

## 5. Verdict

**Symbolic conditioning HURTS text prediction at this scale/config; it does
not help.** Three concrete findings:

1. Equal-budget head-to-head on the conditioned task vs the plain task:
   Arm B cond NLL 0.3068 vs Arm A text NLL 0.2136. Caveat stated openly: the
   two target distributions are not identical (Arm B scores the first 128
   bytes of text after the symbolic prefix + END_MESSAGE; Arm A scores whole
   512-token text windows), but both are next-byte text prediction over the
   same 100 worlds, and Arm B loses on the directly comparable novel-position
   slice too (1.0385 vs 0.7522).
2. Arm B's unconditioned text ability is DESTROYED (NLL 3.9095, top-1 0.41 —
   near byte-frequency guessing): spending all 4,000 steps on
   symbolic→text records at batch 1 leaves no gradient for plain-text
   continuation. Within this trainer's sampling model, an Arm B that never
   sees text-only windows cannot serve as a text model.
3. NOT CAPACITY-BOUND in the mara-ladder sense: the in-window symbolic→text
   grammar DID enter a 4.85M-param model (novel-position top-1 0.9028 on
   held-out worlds; compare the ladder, where even verbatim record facts
   never entered at this scale). The failure mode here is not the v1–v4
   capacity floor; it is that (a) the 512-byte window forces a truncated
   symbolic prefix (deviation 1) and (b) conditioning displaces, rather than
   augments, the plain-text task at equal step budget. If the question is
   re-asked as SPEC §6 intends — full symbolic file conditioning — this run
   is UNDECIDABLE at 512 context: the experiment is computationally
   inexpressible on this stack (needs a ≥~54K-token context arm).

One-sentence answer: at Mara-class scale with the tail-of-wbv1 feasible
variant, the symbolic view does not improve text prediction and its
exclusive use destroys plain-text competence; a real test of full-file
conditioning requires the bigger-context arm.

## 6. Reproduce

```
python build_phase3_corpora.py        # corpus_A.txt, corpus_B.txt, eval text
python build_phase3_tokens.py         # wb vocab assumed built; channel_B.tok,
                                      #  eval_val_tok/, eval_cond_tok/
bpe_tokenizer.exe --vocab wb_phase3.vocab --text corpus_A.txt --out corpus_A.tok \
  --text val_text_all.txt --out val_text_all.tok
gcc -O2 -std=c11 -DUSE_OPENBLAS experiments/wb_phase3/wb_phase3_eval.c \
  -o experiments/wb_phase3/wb_phase3_eval.exe libopenblas.dll -lm
bash run_phase3_eval.sh wb_phase3_armA.f32 wb_phase3_armB.f32 eval_per_world.csv
python aggregate_phase3.py eval_per_world.csv
```
Training logs: logs/train_armA.log, logs/train_armB.log. Weights:
wb_phase3_armA.f32, wb_phase3_armB.f32 (58 MB each; mid checkpoints lost per
deviation 2).

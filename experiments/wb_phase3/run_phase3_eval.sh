#!/bin/bash
# Warren Braid Phase 3 -- run per-world evaluation for both arms, both modes.
# Usage: run_phase3_eval.sh <armA-model.f32> <armB-model.f32> <out.csv>
set -euo pipefail
cd "$(dirname "$0")"

ARM_A="$1"
ARM_B="$2"
OUT="$3"
EVAL="./wb_phase3_eval.exe"
TOKARGS=(--vocab 133 --tokenizer wb_phase3.vocab)

echo "label,file,mode,window,tokens,nll_mean,top1_acc,copy_tokens,copy_nll_mean,copy_top1_acc,novel_tokens,novel_nll_mean,novel_top1_acc" > "$OUT"

for arm in A B; do
  if [ "$arm" = "A" ]; then model="$ARM_A"; else model="$ARM_B"; fi
  for f in eval_val_tok/*.text.tok; do
    # text mode needs > context (512) tokens; token files are uint16 (2
    # bytes/token), so skip files with fewer than 513 tokens (recorded in
    # the report as excluded)
    if [ "$(( $(stat -c %s "$f") / 2 ))" -lt 513 ]; then
      echo "skip $f ($(stat -c %s "$f") bytes < 514)" >&2
      continue
    fi
    "$EVAL" "${TOKARGS[@]}" --model "$model" --file "$f" --mode text --label "arm$arm" >> "$OUT"
  done
  for f in eval_cond_tok/*.cond.tok; do
    "$EVAL" "${TOKARGS[@]}" --model "$model" --file "$f" --mode cond --label "arm$arm" >> "$OUT"
  done
done
echo "wrote $OUT"

#!/usr/bin/env bash
# Retrain one voice on its (already-built) curated corpus and
# merge every epoch at penalty 1.10.  Usage: run_2h_retrain.sh <prefix> <token> <corpus_dir>
set -u
PREFIX="$1"; TOKEN="$2"; CORPUS="$3"
FT=/home/telltale/xtts_ft
TRAINPY=/home/telltale/anaconda3/envs/orpheus_train/bin/python
OO=/mnt/c/Users/tellt/Projects/orpheus-finetune/orpheus_owen.py
LOG=/mnt/c/tmp/${PREFIX}_train.log
log() { echo "[$PREFIX $(date +%H:%M:%S)] $*" >> "$LOG"; }

OUT="$FT/${PREFIX}_out"
rm -rf "$OUT"; mkdir -p "$OUT"
log "=== TRAIN on $CORPUS ==="
"$TRAINPY" "$OO" --source-name "$TOKEN" --recut-dir "$CORPUS" \
  --out-base "$OUT" --mask-prompt-loss --no-dedup --lr-schedule constant_with_warmup \
  --epochs 8 --stop-overtrain --overtrain-patience 2 train >> "$LOG" 2>&1
log "train exit=$?"

LORA="$OUT/orpheus_${TOKEN}_lora"
cd /mnt/c/Users/tellt/Projects/orpheus-finetune
for CKPT in $(ls -d "$LORA"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
  N=$(basename "$CKPT" | grep -oE "[0-9]+")
  log "merge ${PREFIX}_ep${N}"
  bash prep_voice.sh "$CKPT" "${PREFIX}_ep${N}" "$PREFIX ckpt-$N" 21.5 1.10 "$TOKEN" >> "$LOG" 2>&1
done
log "${PREFIX}_CHAIN_DONE"

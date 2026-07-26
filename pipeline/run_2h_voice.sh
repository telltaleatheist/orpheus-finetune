#!/usr/bin/env bash
# One voice through the PROVEN curated-2h pipeline (deathstalker-validated
# 2026-07-20, Owen: "very good"): census -> brightest ~480 clips (narrow
# spread) -> E-recipe build (own bed) -> train (tight leash: the 2h curve
# runs away hard past the bottom; late-dip null) -> merge every epoch @1.10.
#
# Usage: run_2h_voice.sh <src_corpus> <dst_corpus> <bed> <prefix> <token>
set -u
SRC="$1"; DST="$2"; BED="$3"; PREFIX="$4"; TOKEN="$5"
FT=/home/telltale/xtts_ft
TRAINPY=/home/telltale/anaconda3/envs/orpheus_train/bin/python
TTSPY=/home/telltale/anaconda3/envs/orpheus_tts/bin/python
OO=/mnt/c/Users/tellt/Projects/orpheus-finetune/orpheus_owen.py
LOG=/mnt/c/tmp/${PREFIX}_train.log
log() { echo "[$PREFIX $(date +%H:%M:%S)] $*" >> "$LOG"; }

log "=== CENSUS $SRC ==="
"$TTSPY" /home/telltale/brightness_census.py "$SRC" "/home/telltale/${PREFIX}_bright" >> "$LOG" 2>&1 || { log "CENSUS FAILED"; exit 1; }

log "=== BUILD curated-2h -> $DST ==="
"$TTSPY" /home/telltale/build_2h_corpus.py "$SRC" "$DST" "$BED" "/home/telltale/${PREFIX}_bright" >> "$LOG" 2>&1 || { log "BUILD FAILED"; exit 1; }

OUT="$FT/${PREFIX}_out"
rm -rf "$OUT"; mkdir -p "$OUT"
log "=== TRAIN (epochs<=8, patience 2) ==="
"$TRAINPY" "$OO" --source-name "$TOKEN" --recut-dir "$DST" \
  --out-base "$OUT" --mask-prompt-loss --no-dedup --lr-schedule constant_with_warmup \
  --epochs 8 --stop-overtrain --overtrain-patience 2 train >> "$LOG" 2>&1
log "train exit=$?"

LORA="$OUT/orpheus_${TOKEN}_lora"
cd /mnt/c/Users/tellt/Projects/orpheus-finetune
for CKPT in $(ls -d "$LORA"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
  N=$(basename "$CKPT" | grep -oE "[0-9]+")
  log "merge ${PREFIX}_ep${N}"
  bash prep_voice.sh "$CKPT" "${PREFIX}_ep${N}" "$PREFIX curated-2h ckpt-$N" 21.5 1.10 "$TOKEN" >> "$LOG" 2>&1
done
log "${PREFIX}_CHAIN_DONE"

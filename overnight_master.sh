#!/usr/bin/env bash
# ============================================================================
# overnight_master.sh — fully autonomous overnight Orpheus+RVC pipeline (WSL).
#
# Order: deathstalker -> mistborn -> owen (pre-staged datasets), then THIRD REICH
# (this script waits for the Mac's VTT, cuts it here in WSL, stages, trains — so
# it needs no help while you sleep), then the RVC training chain.
#
# * Each training waits for a FREE GPU (<3GB used); if busy it waits 5 min.
# * Each dataset step waits (polls) for its inputs so launch timing is forgiving.
# * All checkpoints kept (stop-overtrain+2); keepers picked by ear in the morning.
# * Launched once, detached — makes no further permission requests:
#     wsl.exe -d Ubuntu bash -c "nohup bash /mnt/c/Users/tellt/Projects/orpheus-finetune/overnight_master.sh >> /home/telltale/xtts_ft/overnight_master.log 2>&1 &"
# ============================================================================
set -uo pipefail
PY=/home/telltale/anaconda3/envs/orpheus_train/bin/python
OO=/mnt/c/Users/tellt/Projects/orpheus-finetune/orpheus_owen.py
CUT=/mnt/c/Users/tellt/Projects/orpheus-finetune/cut_audiobook.py
FT=/home/telltale/xtts_ft
LOG=$FT/overnight_master.log
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1

log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
gpu_free(){ local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); [ -n "$u" ] && [ "$u" -lt 3000 ]; }

train_voice(){  # $1 voice  $2 dataset_dir  $3 dataset_wait_max_s(default 6h)
  local v=$1 ds=$2 wmax=${3:-21600} waited=0
  if [ -f "$FT/.done_$v" ]; then log "[$v] already completed (marker) — SKIP"; return 0; fi
  while [ ! -f "$ds/metadata_train.csv" ]; do
    [ "$waited" -eq 0 ] && log "[$v] waiting for dataset $ds ..."
    sleep 120; waited=$((waited+120))
    [ "$waited" -ge "$wmax" ] && { log "[$v] dataset never staged (${wmax}s) — SKIP"; return 1; }
  done
  until gpu_free; do log "[$v] GPU busy — waiting 5min"; sleep 300; done
  log "[$v] ===== TRAIN START ($(( $(wc -l < "$ds/metadata_train.csv") - 1 )) train clips) ====="
  cd "$FT"
  if "$PY" "$OO" --source-name "$v" --recut-dir "$ds" --out-base "$FT" \
       --mask-prompt-loss --no-dedup --lr-schedule constant_with_warmup \
       --stop-overtrain --overtrain-patience 2 train >> "$LOG" 2>&1; then
    log "[$v] ===== TRAIN DONE ====="
    touch "$FT/.done_$v"
  else
    log "[$v] !!!!! TRAIN FAILED (exit $?) — continuing"
  fi
  sleep 30
}

# Best-effort: after a voice trains, merge its settle+1 checkpoint with production
# caps and render a ~200-word dialogue sample via the BookForge CLI (WSL->Windows).
# Any failure here is logged and ignored — it must never break the training chain.
sample_voice(){  # $1 voice
  local v=$1 best bestn next ckpt tid
  if [ -f "/mnt/e/Shared/overnight_samples/${v}_sample.wav" ]; then log "[$v/sample] sample exists — SKIP"; return 0; fi
  best=$(grep "BEST epoch checkpoint = $FT/orpheus_${v}_lora/checkpoint-" "$LOG" | tail -1 | grep -oE "checkpoint-[0-9]+" | head -1)
  if [ -z "$best" ]; then log "[$v/sample] no BEST checkpoint in log — skip"; return 0; fi
  bestn=${best#checkpoint-}
  # settle+1 = the next kept checkpoint after the eval-best; fall back to best if none
  next=$(ls -d "$FT/orpheus_${v}_lora"/checkpoint-* 2>/dev/null | sed 's#.*/##;s#checkpoint-##' | sort -n | awk -v b="$bestn" '$1>b{print;exit}')
  ckpt="$FT/orpheus_${v}_lora/checkpoint-${next:-$bestn}"
  tid="${v}_sample"
  log "[$v/sample] keeper $(basename "$ckpt") -> prep + generate (best-effort)"
  ( bash /mnt/c/Users/tellt/Projects/orpheus-finetune/prep_voice.sh "$ckpt" "$tid" "$v sample" >> "$LOG" 2>&1 \
    && powershell.exe -NoProfile -ExecutionPolicy Bypass \
         -File 'C:\Users\tellt\Projects\orpheus-finetune\gen_sample.ps1' \
         -Voice "$tid" \
         -InputTxt 'E:\Shared\overnight_samples\dialogue_prompt.txt' \
         -OutWav "E:\\Shared\\overnight_samples\\${v}_sample.wav" >> "$LOG" 2>&1 \
    && log "[$v/sample] wrote E:\\Shared\\overnight_samples\\${v}_sample.wav" ) \
    || log "[$v/sample] FAILED (ignored)"
}

log "=================== overnight_master START ==================="

# ---- 1. pre-staged voices (cut on Windows while awake) ----
train_voice deathstalker "$FT/deathstalker_mm"; sample_voice deathstalker
train_voice mistborn     "$FT/mistborn_full";  sample_voice mistborn
train_voice owen         "$FT/owen_full";       sample_voice owen

# ---- 2. Third Reich: wait for the Mac's VTT, cut here, stage, train ----
TR_AUDIO="/mnt/e/training/thirdreich/source/third reich FULL merged (dehummed).flac"
TR_VTT="/mnt/e/training/thirdreich/source/third reich FULL merged (dehummed).vtt"
TR_EPUB="/mnt/e/Shared/BookForge/projects/The_Coming_of_the_Third_Reich_-_Richard_J_Evans_(2004)/archive/The Coming of the Third Reich. Evans, Richard J. (2004).epub"
TR_DS="$FT/thirdreich_full"
waited=0
while [ ! -f "$TR_VTT" ]; do
  [ "$waited" -eq 0 ] && log "[thirdreich] waiting for Mac VTT: $TR_VTT"
  sleep 300; waited=$((waited+300))
  [ "$waited" -ge 43200 ] && { log "[thirdreich] VTT never arrived (12h) — SKIP"; break; }
done
if [ -f "$TR_VTT" ]; then
  log "[thirdreich] VTT landed — cutting in WSL"
  rm -rf "$TR_DS" "${TR_DS}_hf"
  if "$PY" "$CUT" --vtt "$TR_VTT" --audio "$TR_AUDIO" --epub "$TR_EPUB" \
       --out-dir "$TR_DS" --source-name thirdreich \
       --target-minutes 0 --max-clip 20 --trail-cap 0.1 --mix "2:3-10,3:10-20" >> "$LOG" 2>&1 \
     && "$PY" "$OO" --source-name thirdreich --dataset-dir "$TR_DS" --output-dir "${TR_DS}_hf" build >> "$LOG" 2>&1; then
    log "[thirdreich] cut+staged $(ls "$TR_DS/wavs" 2>/dev/null | wc -l) clips"
    train_voice thirdreich "$TR_DS"; sample_voice thirdreich
  else
    log "[thirdreich] !!!!! cut/build FAILED — skipping"
  fi
fi

# ---- 3. RVC training chain (GPU guaranteed free now) ----
log "=================== all Orpheus done — launching RVC ==================="
RVC_WSL=/mnt/c/Users/tellt/Projects/ultimate-rvc/bookforge_train/run_all_rvc.ps1
if [ -f "$RVC_WSL" ]; then
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\tellt\Projects\ultimate-rvc\bookforge_train\run_all_rvc.ps1' >> "$LOG" 2>&1
  log "=================== RVC returned (exit $?) ==================="
else
  log "RVC script missing at $RVC_WSL — skipping"
fi
log "=================== EVERYTHING COMPLETE ==================="

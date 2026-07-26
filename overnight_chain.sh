#!/usr/bin/env bash
# ============================================================================
# overnight_chain.sh — unattended Orpheus training chain.
#
# Trains a QUEUE of voices in order (deathstalker FIRST, then the rest), each from
# a pre-staged dataset dir (<wsl_ds>/metadata_train.csv). Runs unattended:
#   * For each voice it WAITS (polling) until its dataset is staged — so the cut/
#     stage of later voices can still be in progress when this launches.
#   * Before each run it checks the GPU is FREE (<3GB used); if something else is
#     on the GPU it waits 5 min and re-checks (per Owen's spec).
#   * On a clean run it keeps all epoch checkpoints (stop-overtrain+2) and loads
#     the eval-best; keeper (settle / settle+1) is picked by ear afterward.
#
# Launch detached so it survives the controlling session:
#   wsl.exe -d Ubuntu bash -c "nohup bash /mnt/c/Users/tellt/Projects/orpheus-finetune/overnight_chain.sh >> /home/telltale/xtts_ft/overnight_chain.log 2>&1 &"
# ============================================================================
set -uo pipefail
OO=/mnt/c/Users/tellt/Projects/orpheus-finetune/orpheus_owen.py
PY=/home/telltale/anaconda3/envs/orpheus_train/bin/python
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
cd /home/telltale/xtts_ft
LOG=/home/telltale/xtts_ft/overnight_chain.log

# Ordered queue — "voice:dataset_dir". deathstalker first; rest any order.
QUEUE=(
  "deathstalker:/home/telltale/xtts_ft/deathstalker_mm"
  "mistborn:/home/telltale/xtts_ft/mistborn_full"
  "thirdreich:/home/telltale/xtts_ft/thirdreich_full"
  "owen:/home/telltale/xtts_ft/owen_full"
)

DATASET_WAIT_MAX=$((14*3600))   # give a slow align/cut up to 14h to produce a dataset
GPU_FREE_MB=3000                 # GPU counts as free below this (idle ~1-2GB)

log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

gpu_free(){
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -n "$used" ] && [ "$used" -lt "$GPU_FREE_MB" ]
}

log "=========== overnight_chain start ==========="
for entry in "${QUEUE[@]}"; do
  vid="${entry%%:*}"; ds="${entry#*:}"

  # 1. wait for the dataset to be staged
  waited=0
  while [ ! -f "$ds/metadata_train.csv" ]; do
    [ "$waited" -eq 0 ] && log "[$vid] waiting for dataset $ds ..."
    sleep 300; waited=$((waited+300))
    if [ "$waited" -ge "$DATASET_WAIT_MAX" ]; then
      log "[$vid] dataset never appeared after $((DATASET_WAIT_MAX/3600))h — SKIP"; break
    fi
  done
  [ -f "$ds/metadata_train.csv" ] || continue
  n=$(( $(wc -l < "$ds/metadata_train.csv") - 1 ))
  log "[$vid] dataset ready ($n train clips)"

  # 2. wait for a free GPU
  until gpu_free; do log "[$vid] GPU busy — waiting 5min"; sleep 300; done

  # 3. train
  log "[$vid] ===== TRAIN START ====="
  if "$PY" "$OO" --source-name "$vid" --recut-dir "$ds" --out-base /home/telltale/xtts_ft \
       --mask-prompt-loss --no-dedup --lr-schedule constant_with_warmup \
       --stop-overtrain --overtrain-patience 2 train >> "$LOG" 2>&1; then
    log "[$vid] ===== TRAIN DONE ====="
  else
    log "[$vid] !!!!! TRAIN FAILED (exit $?) — continuing to next voice"
  fi
  sleep 30   # let the GPU settle before the next
done
log "=========== all Orpheus jobs COMPLETE ==========="

# ---- Chain the RVC training (Windows) once every Orpheus job has exited. The RVC
#      script assumes a free GPU at start — guaranteed here since the Orpheus loop
#      above ran to completion and released the GPU. It trains Owen -> Mistborn ->
#      Third Reich (preprocess -> features -> 300ep train -> build .index -> deploy),
#      waiting for the GPU to clear between models, and logs to its own
#      bookforge_train\logs\overnight_status.txt. ----
RVC_PS1_WSL='/mnt/c/Users/tellt/Projects/ultimate-rvc/bookforge_train/run_all_rvc.ps1'
RVC_PS1_WIN='C:\Users\tellt\Projects\ultimate-rvc\bookforge_train\run_all_rvc.ps1'
if [ -f "$RVC_PS1_WSL" ]; then
  log "=========== launching RVC training chain ==========="
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$RVC_PS1_WIN" >> "$LOG" 2>&1
  log "=========== RVC training chain returned (exit $?) ==========="
else
  log "RVC script not found at $RVC_PS1_WSL — skipping RVC chain"
fi
log "=========== EVERYTHING COMPLETE ==========="

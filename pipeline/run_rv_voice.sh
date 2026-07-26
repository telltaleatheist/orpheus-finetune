#!/usr/bin/env bash
# ============================================================================
# run_rv_voice.sh — the full raw-verbatim chain for ONE voice:
#   train + merge every epoch  ->  arm the EOS boost on every merged epoch
#   ->  greedy EOS gate the last N epochs.
#
# Generalized from the 2026-07-21 rv_train_chain.sh, which lived only in WSL
# home and drove ds_rv1 / mb_rv1 / tr_rv1. That script was never tracked, so
# the runs it produced could not be reproduced after cleanup deleted the
# checkpoints — hence this one is in the repo.
#
#   bash pipeline/run_rv_voice.sh <prefix> <token> <corpus_dir> [gate_epochs]
#
# e.g. bash pipeline/run_rv_voice.sh tr_rv2 thirdreich \
#          /home/telltale/xtts_ft/thirdreich_rv2h 3
#
# Run in WSL. Holds external-gpu-job.lock for the whole chain so BookForge's
# global process sweeps do not kill the trainer mid-run.
# ============================================================================
set -u
if [ $# -lt 3 ]; then
  echo "usage: run_rv_voice.sh <prefix> <token> <corpus_dir> [gate_epochs]" >&2
  exit 2
fi
PREFIX="$1"; TOKEN="$2"; CORPUS="$3"; GATE_EPOCHS="${4:-3}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FT=/home/telltale/xtts_ft
OM=/home/telltale/orpheus-models
GATEPY=/home/telltale/anaconda3/envs/orpheus_tts/bin/python
LOCK=/mnt/c/Users/tellt/AppData/Roaming/BookForge/external-gpu-job.lock
LOG=/mnt/c/tmp/${PREFIX}_chain.log
log() { echo "[$PREFIX $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

for required in "$CORPUS/wavs" "$CORPUS/metadata_train.csv" "$CORPUS/metadata_eval.csv"; do
  [ -e "$required" ] || { echo "missing corpus part: $required" >&2; exit 1; }
done
[ -x "$GATEPY" ] || { echo "gate python not found: $GATEPY" >&2; exit 1; }

# The lock makes BookForge skip its process sweeps; a stale one would mask a
# real job, so it is removed on every exit path including Ctrl-C.
trap 'rm -f "$LOCK"' EXIT
echo "$PREFIX raw-verbatim train+gate chain, started $(date -Is)" > "$LOCK"

log "=== TRAIN $PREFIX on $CORPUS ($(ls "$CORPUS/wavs" | wc -l) clips) ==="
bash "$REPO/pipeline/run_2h_retrain.sh" "$PREFIX" "$TOKEN" "$CORPUS"
log "train+merge exit=$?"

EPOCHS=$(ls -d "$OM/${PREFIX}"_ep* 2>/dev/null | wc -l)
log "merged+registered epochs: $EPOCHS"
if [ "$EPOCHS" -eq 0 ]; then
  echo "FAILED $(date -Is) - no merged epochs" > "/mnt/c/tmp/${PREFIX}_chain_done.marker"
  exit 1
fi

# prep_voice.sh already registered each epoch with maxCharsPerSec/repPenalty;
# the EOS boost is the one cap it does not write, so arm it here. A voice that
# reaches production without it runs unguarded — that is the thirdreich bug.
rm -rf "$FT/${PREFIX}"_ep*_merged     # staging duplicates of the serving copies
PFX="$PREFIX" python3 - <<'PYEOF' | tee -a "$LOG"
import json, os
path = "/home/telltale/orpheus-models/models.json"
document = json.load(open(path))
prefix = os.environ["PFX"]
boosted = []
for model in document.get("models", []):
    if model.get("id", "").startswith(prefix + "_ep"):
        vllm = model.setdefault("backends", {}).setdefault("vllm", {})
        vllm["eosBoost"] = 8
        vllm["eosBoostStart"] = 2.0
        boosted.append(model["id"])
json.dump(document, open(path, "w"), indent=2)
print("armed eosBoost 8 @ 2.0 on:", ", ".join(boosted) or "(nothing)")
PYEOF

# Greedy gate at production settings (pen 1.10 + boost 8 @ 2.0). Deterministic:
# a pass is a property of the checkpoint, not sampling luck. Ship only PASS.
for MODEL_DIR in $(ls -d "$OM/${PREFIX}"_ep* 2>/dev/null | sort -V | tail -"$GATE_EPOCHS"); do
  log "gate $(basename "$MODEL_DIR")"
  "$GATEPY" "$REPO/eos_gate.py" "$MODEL_DIR" "$TOKEN" "$CORPUS" 20 1.10 8 2.0 2>&1 | tee -a "$LOG"
done

echo "DONE $(date -Is)" > "/mnt/c/tmp/${PREFIX}_chain_done.marker"
log "=== ${PREFIX} CHAIN DONE — review EOS_GATE_ lines above, then ear-check ==="
grep -E "^EOS_GATE_|^gate " "$LOG" | tail -20

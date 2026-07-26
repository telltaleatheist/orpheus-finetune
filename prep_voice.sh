#!/usr/bin/env bash
# ============================================================================
# prep_voice.sh — LOCAL prep of a trained Orpheus checkpoint so the BookForge
# CLI can generate from it, configured EXACTLY like production.
#
# Two steps the BookForge CLI does NOT do itself (it is generation-only, and
# resolves --voice from an already-merged, already-registered models.json entry):
#   1. MERGE the LoRA checkpoint -> a 16-bit standalone model (vLLM can't load a
#      bare LoRA adapter).
#   2. REGISTER it in the local models dir + models.json, INCLUDING the per-voice
#      runaway caps production uses (backends.vllm.repPenalty + maxCharsPerSec).
#      Omitting those caps lets Orpheus run away (repeat/gibberish) — the exact
#      bug the old throwaway audition script had.
#
# This is the LOCAL-ONLY test path (no HF push) — use before deploy_voice.sh
# ships a validated keeper. Run in WSL (env orpheus_train).
#
#   bash prep_voice.sh <checkpoint_dir> <voice_id> [label] [maxCharsPerSec] [repPenalty] [token]
#
# Defaults match the deployed deathstalker caps (21.5 / 1.15). Pass a voice_id
# that is NOT a live deployed id (e.g. deathstalker_v6_ep3) so the real voice is
# untouched.
#
# ⚠ TOKEN: the registered token is used VERBATIM as the prompt speaker prefix
# (orpheus.py "<token>: <text>"). It MUST equal the speaker_name the model was
# TRAINED on, or the mismatched part gets SPOKEN ALOUD (a deathstalker_mm1590
# test voice literally said "1590" at every sentence start). For epoch-test
# voice_ids with suffixes, ALWAYS pass the trained speaker name as arg 6. e.g.
#   bash prep_voice.sh /home/telltale/xtts_ft/orpheus_deathstalker_lora/checkpoint-465 \
#        deathstalker_v6_ep3 "Deathstalker v6 ep3 (test)" 21.5 1.15 deathstalker
# ============================================================================
set -euo pipefail
export PATH=/home/telltale/anaconda3/bin:/usr/bin:/bin
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=/home/telltale/anaconda3/envs/orpheus_train/bin/python

CKPT="${1:?usage: prep_voice.sh <checkpoint_dir> <voice_id> [label] [maxCharsPerSec] [repPenalty] [token]}"
VID="${2:?need a voice_id, e.g. deathstalker_v6_ep3}"
LABEL="${3:-$VID}"
MAXCPS="${4:-21.5}"
REPPEN="${5:-1.15}"
TOKEN="${6:-$VID}"
if [ "$TOKEN" = "$VID" ] && [[ "$VID" == *_* ]]; then
  echo "⚠ WARNING: token defaulting to voice_id '$VID', which looks like a suffixed"
  echo "  test id. The token must equal the TRAINED speaker_name or the suffix gets"
  echo "  SPOKEN. If this model was trained as e.g. 'deathstalker', pass it as arg 6."
fi

MODELS_DIR=/home/telltale/orpheus-models
MERGED="/home/telltale/xtts_ft/${VID}_merged"
DEST="$MODELS_DIR/$VID"
MJSON="$MODELS_DIR/models.json"
[ -d "$CKPT" ] || { echo "ERROR: no checkpoint at $CKPT"; exit 1; }

echo "===== [1/2] MERGE $CKPT -> 16-bit ====="
rm -rf "$MERGED"
ORPHEUS_GPU_MEM_UTIL=0.5 "$PY" - <<PYEOF
from unsloth import FastLanguageModel
m, t = FastLanguageModel.from_pretrained(
    model_name="$CKPT", max_seq_length=2048, dtype=None, load_in_4bit=False)
m.save_pretrained_merged("$MERGED", t, save_method="merged_16bit")
print("MERGED_OK")
PYEOF
[ -f "$MERGED/config.json" ] || { echo "ERROR: merge produced no config.json"; exit 1; }

echo "===== [2/2] REGISTER '$VID' locally (caps $MAXCPS / $REPPEN, no HF) ====="
cp "$MJSON" "$MJSON.bak"
rsync -a --delete --exclude='.cache' --exclude='*.lock' "$MERGED/" "$DEST/"
MJSON="$MJSON" VID="$VID" LABEL="$LABEL" MAXCPS="$MAXCPS" REPPEN="$REPPEN" TOKEN="$TOKEN" "$PY" - <<'PY'
import json, os
p = os.environ['MJSON']; d = json.load(open(p))
vid = os.environ['VID']
entry = {"id": vid, "label": os.environ['LABEL'], "token": os.environ['TOKEN'], "dir": vid,
         "format": "hf", "sampleRate": 24000,
         "source": {"type": "hf", "ref": "owenmorgan/deathstalker-orpheus-3b"},
         "license": "apache-2.0", "addedAt": "2026-07-16",
         "maxCharsPerSec": float(os.environ['MAXCPS']),
         "backends": {"vllm": {"repPenalty": float(os.environ['REPPEN'])}}}
models = d.setdefault("models", [])
for m in models:
    if m.get("id") == vid:
        m.update(entry); break
else:
    models.append(entry)
json.dump(d, open(p, "w"), indent=2)
print(f"  registered {vid}: token={entry['token']} maxCharsPerSec={entry['maxCharsPerSec']} repPenalty={entry['backends']['vllm']['repPenalty']}")
PY
echo "PREP_${VID}_DONE"

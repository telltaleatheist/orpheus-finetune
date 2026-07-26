#!/bin/bash
# ============================================================================
# deploy_voice.sh — publish a trained Orpheus voice so BookForge can use it.
#
# A merged voice (`<out-base>/orpheus_<voice>_merged`) becomes usable TWO ways
# and we do BOTH:
#   1. LOCAL  — rsync the merged model into the local models dir + upsert
#               models.json. Works immediately on THIS machine (Windows/WSL).
#   2. HF     — upload to owenmorgan/<voice>-orpheus-3b, tagged
#               `bookforge-orpheus-voice`, so it shows in BookForge's download
#               catalog AND is pullable on the Mac.
# Then it prints a ready-to-paste prompt for Claude-on-Mac (step 3).
#
# Run in WSL (env orpheus_train has huggingface_hub):
#   bash deploy_voice.sh <voice> "<Label>" [merged_dir]
# e.g.
#   bash deploy_voice.sh mistborn   "Mistborn (Michael Kramer)"
#   bash deploy_voice.sh thirdreich "Third Reich (Sean Pratt)"
#   bash deploy_voice.sh owen       "Owen Morgan"
#
# merged_dir defaults to /home/telltale/xtts_ft/<voice>_v2/orpheus_<voice>_merged
# (the path train_all_v2.sh / orpheus_owen.py write to). Override for old layouts.
# ============================================================================
set -euo pipefail

VOICE="${1:?usage: deploy_voice.sh <voice> \"<Label>\" [merged_dir]}"
LABEL="${2:?need a display label, e.g. \"Mistborn (Michael Kramer)\"}"
MERGED="${3:-/home/telltale/xtts_ft/${VOICE}_v2/orpheus_${VOICE}_merged}"

MODELS_DIR=/home/telltale/orpheus-models
DEST="$MODELS_DIR/$VOICE"
MJSON="$MODELS_DIR/models.json"

# HF repo name is normally owenmorgan/<voice>-orpheus-3b, but a few voices have a
# legacy/canonical repo name that ISN'T just the token. Keep these EXACT — the catalog
# (electron/orpheus-hf-catalog.ts DEFAULT_ORPHEUS_SOURCES) + the Mac reference these refs.
case "$VOICE" in
  owen) REPO="owenmorgan/owen-morgan-orpheus-3b" ;;   # NOT owen-orpheus-3b
  *)    REPO="owenmorgan/${VOICE}-orpheus-3b" ;;
esac
TODAY="$(date '+%Y-%m-%d')"
PY=/home/telltale/anaconda3/envs/orpheus_train/bin/python
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -f "$MERGED/config.json" ] || { echo "ERROR: no merged model at $MERGED"; exit 1; }

# --- token: env -> WSL hf cache -> Windows hf cache -> Downloads file --------
resolve_token() {
  for src in "${HF_TOKEN:-}" \
             "$(cat ~/.cache/huggingface/token 2>/dev/null || true)" \
             "$(cat /mnt/c/Users/tellt/.cache/huggingface/token 2>/dev/null || true)" \
             "$(grep -o 'hf_[A-Za-z0-9_]\{20,\}' /mnt/c/Users/tellt/Downloads/bookforge-hf-token.txt 2>/dev/null || true)"; do
    tok="$(printf '%s' "$src" | grep -o 'hf_[A-Za-z0-9_]\{20,\}' || true)"
    [ -n "$tok" ] && { printf '%s' "$tok"; return 0; }
  done
  echo "ERROR: no HF token (set HF_TOKEN, or put it in ~/.cache/huggingface/token)" >&2
  exit 1
}

echo "======================================================================"
echo "Deploying voice='$VOICE' label='$LABEL'"
echo "  merged : $MERGED"
echo "  local  : $DEST"
echo "  hf repo: $REPO"
echo "======================================================================"

# --- 1. LOCAL install --------------------------------------------------------
echo "[1/3] local install (rsync + manifest)..."
cp "$MJSON" "$MJSON.bak"
rsync -a --delete --exclude='.cache' --exclude='*.lock' "$MERGED/" "$DEST/"
VOICE="$VOICE" LABEL="$LABEL" REPO="$REPO" TODAY="$TODAY" MJSON="$MJSON" "$PY" - <<'PY'
import json, os
p = os.environ['MJSON']
d = json.load(open(p))
vid = os.environ['VOICE']
entry = {"id": vid, "label": os.environ['LABEL'], "token": vid, "dir": vid,
         "format": "hf", "sampleRate": 24000,
         "source": {"type": "hf", "ref": os.environ['REPO']},
         "license": "apache-2.0", "addedAt": os.environ['TODAY']}
models = d.setdefault("models", [])
for i, m in enumerate(models):
    if m.get("id") == vid:
        # preserve any extra keys already present, overwrite the ones we manage
        m.update(entry); break
else:
    models.append(entry)
json.dump(d, open(p, "w"), indent=2)
print("  manifest upserted:", vid, "->", os.environ['LABEL'])
PY

# --- 2. HF publish -----------------------------------------------------------
echo "[2/3] uploading to HuggingFace ($REPO, private, ~6.6GB)..."
export HF_TOKEN="$(resolve_token)"
PYTHONIOENCODING=utf-8 "$PY" "$REPO_DIR/upload_to_hf.py" \
    --model-dir "$DEST" --repo "$REPO" --voice-token "$VOICE" --label "$LABEL"

# --- 3. Mac prompt -----------------------------------------------------------
cat <<EOF

[3/3] DONE. Local + HF both updated.

Paste this to Claude-on-Mac to pull the new model:
----------------------------------------------------------------------
Re-pull the Orpheus voice '$VOICE' — it was retrained. Download
$REPO into the Mac's orpheusModelsDir/$VOICE (overwrite the old files),
keep/upsert its models.json entry (token '$VOICE', label '$LABEL',
format per the Mac's backend — MLX reads these HF safetensors directly,
so format:"hf" unless the Mac path needs an MLX conversion). Confirm the
old files were replaced, then render a one-line test with speak.
----------------------------------------------------------------------
EOF

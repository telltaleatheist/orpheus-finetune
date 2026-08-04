#!/bin/sh
# Train the Headline metadata adapters on the Mac Studio, sequentially:
# description first, then tags. One Qwen3-14B-4bit base, one adapter per task.
#
# Launch DETACHED so it survives an ssh disconnect, and hold off sleep for the
# duration:
#
#   nohup caffeinate -is /Volumes/Callisto/Projects/orpheus-finetune/configs/mac/run_metadata_adapters.sh \
#       > "$HOME/train_runner.log" 2>&1 &
#
# `caffeinate -i` prevents idle sleep, `-s` prevents system sleep on AC. Without
# it a long run dies quietly the first time the machine idles out.
#
# Logs: ~/train_description_v2.log, ~/train_tags_v2.log, ~/train_runner.log.
# Adapters + per-checkpoint weights land under
# /Volumes/Callisto/training/titles/mlx-adapters/{description,tags}-v2/.
#
# There is no early stopping in MLX (see docs/MAC_TRAINING.md) — these runs go to
# 3 epochs on purpose and the winner is picked afterwards by val loss from the
# NNNNNNN_adapters.safetensors checkpoints, not by taking the final weights.

set -u

# Absolute paths throughout: a non-interactive ssh session gets no PATH, and
# conda's shell function does not exist here. Call the env's binaries directly.
ENV="/opt/homebrew/Caskroom/miniconda/base/envs/finetune"
REPO="/Volumes/Callisto/Projects/orpheus-finetune"
LORA="$ENV/bin/mlx_lm.lora"

[ -x "$LORA" ] || { echo "FATAL: no mlx_lm.lora at $LORA"; exit 1; }

run() {
    task="$1"
    cfg="$REPO/configs/mac/qwen3_${task}_mlx.yaml"
    log="$HOME/train_${task}_v2.log"
    [ -f "$cfg" ] || { echo "FATAL: no config at $cfg"; exit 1; }
    echo "=== $task starting $(date -u '+%Y-%m-%dT%H:%M:%SZ') -> $log"
    "$LORA" -c "$cfg" > "$log" 2>&1
    rc=$?
    echo "=== $task finished $(date -u '+%Y-%m-%dT%H:%M:%SZ') rc=$rc"
    return $rc
}

run description || { echo "description FAILED — not starting tags"; exit 1; }
run tags        || { echo "tags FAILED"; exit 1; }

echo "=== both adapters done $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

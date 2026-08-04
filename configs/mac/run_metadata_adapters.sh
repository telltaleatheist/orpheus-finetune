#!/bin/sh
# Train a Headline metadata adapter on the Mac Studio. One Qwen3-32B-4bit base,
# one adapter per task; this script trains ONE task per invocation.
#
#   ./run_metadata_adapters.sh description
#   ./run_metadata_adapters.sh tags
#
# Launch DETACHED so it survives an ssh disconnect, and hold off sleep for the
# duration:
#
#   nohup caffeinate -is /Volumes/Callisto/Projects/orpheus-finetune/configs/mac/run_metadata_adapters.sh \
#       description > "$HOME/train_runner.log" 2>&1 &
#
# `caffeinate -i` prevents idle sleep, `-s` prevents system sleep on AC. Without
# it a long run dies quietly the first time the machine idles out.
#
# ONE TASK PER INVOCATION, DELIBERATELY. The two runs are NOT chained. 32B on
# this machine is a test, not the production path (that stays 14B on the CUDA
# rig), and description alone is many hours — Owen decides whether tags follows
# here or the whole pass moves to the rig once description's numbers are in.
# Chaining them would spend a second overnight on that decision before it exists.
#
# THE MACHINE MUST BE OTHERWISE IDLE. 32B-4bit is ~18.5 GB of weights against a
# ~48 GB wired ceiling; a TTS render or a browser alongside it puts the system
# into swap and everything crawls. macOS degrades, it does not OOM cleanly.
#
# Logs: ~/train_<task>_v2.log, plus whatever you redirect the launcher itself to.
# Adapters + per-checkpoint weights land under
# /Volumes/Callisto/training/titles/mlx-adapters/<task>-v2/.
#
# There is no early stopping in MLX (see docs/MAC_TRAINING.md) — these runs go to
# 3 epochs on purpose and the winner is picked afterwards by val loss from the
# NNNNNNN_adapters.safetensors checkpoints, not by taking the final weights.

set -u

if [ $# -ne 1 ]; then
    echo "usage: $0 <description|tags>"
    exit 2
fi
task="$1"

# Absolute paths throughout: a non-interactive ssh session gets no PATH, and
# conda's shell function does not exist here. Call the env's binaries directly.
ENV="/opt/homebrew/Caskroom/miniconda/base/envs/finetune"
REPO="/Volumes/Callisto/Projects/orpheus-finetune"
LORA="$ENV/bin/mlx_lm.lora"
cfg="$REPO/configs/mac/qwen3_${task}_mlx.yaml"
log="$HOME/train_${task}_v2.log"

[ -x "$LORA" ] || { echo "FATAL: no mlx_lm.lora at $LORA"; exit 1; }
[ -f "$cfg" ]  || { echo "FATAL: no config at $cfg"; exit 1; }

echo "=== $task starting $(date -u '+%Y-%m-%dT%H:%M:%SZ') -> $log"
"$LORA" -c "$cfg" > "$log" 2>&1
rc=$?
echo "=== $task finished $(date -u '+%Y-%m-%dT%H:%M:%SZ') rc=$rc"
exit $rc

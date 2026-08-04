#!/bin/sh
# Train one Headline adapter on the Mac Studio. One Qwen3-32B-4bit base, one
# adapter per task; the argument names the config, `qwen3_<task>_mlx.yaml`.
#
#   ./run_metadata_adapters.sh title32b      # the 32B test run
#   ./run_metadata_adapters.sh description
#   ./run_metadata_adapters.sh tags
#
# Launch DETACHED so it survives an ssh disconnect, and hold off sleep for the
# duration. TWO NON-OBVIOUS THINGS, both learned the hard way on 2026-08-04:
#
#   nohup /bin/sh /Volumes/Callisto/Projects/orpheus-finetune/configs/mac/run_metadata_adapters.sh \
#       title32b > "$HOME/train_runner.log" 2>&1 &
#   TRAINPID=$!
#   nohup caffeinate -is -w $TRAINPID > /dev/null 2>&1 &
#
# 1. `/bin/sh <script>`, not `./script`. /Volumes/Callisto is mounted `noowners`
#    (it is APFS, despite what older notes say), and executing a file on it
#    directly fails with EPERM — "Operation not permitted" — even with the exec
#    bit set and visible in `ls`. Handing the path to sh reads the file instead
#    of exec'ing it, which is permitted.
#
# 2. `caffeinate -w <pid>` ALONGSIDE the run, never `caffeinate <command>`
#    wrapping it. Wrapping fails: the child cannot read /Volumes/Callisto at all
#    and dies with EPERM before it starts, while the identical command run
#    without caffeinate works. The likely cause is TCC attributing the child's
#    file access to caffeinate, which has no permission for the external volume;
#    what is certain is the behaviour. `-w` watches a pid we already have and
#    sidesteps the whole question.
#
# `caffeinate -i` prevents idle sleep, `-s` prevents system sleep on AC. Without
# it a long run dies quietly the first time the machine idles out.
#
# ONE TASK PER INVOCATION, DELIBERATELY. Runs are NOT chained. 32B on this
# machine is a test, not the production path (that stays on the CUDA rig), and a
# single task is many hours — Owen decides what runs next once the first set of
# numbers exists. Chaining would spend a second overnight on that decision before
# there is anything to decide with. It also means a wedged run never takes the
# next one down with it.
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
    echo "usage: $0 <title32b|description|tags>"
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

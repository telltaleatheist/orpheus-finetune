#!/usr/bin/env bash
# bench.sh <voice_id> <minutes> — steady-state chunks/min for one Orpheus voice on
# a fixed book, through the real BookForge CLI audiobook path at --tier fast.
#
# MEASUREMENT TRAP (hit 2026-07-25): the worker writes sentence flacs in BATCHES OF
# 64, so any short window's delta is a multiple of 64 and a 3-minute sample carries
# ~±33% quantization error. Three consecutive 3-min windows on the same run read
# 64.0 / 64.0 / 85.3. Use a LONG window (>= 8 min) so batch boundaries average out,
# and report the cumulative delta, not a per-window average.
#
# Also: progress files are .flac (not .wav), and the live state file is
# session-state.json (hyphen) holding only PREP fields — counting flacs in
# chapters/sentences/ is the only unambiguous progress signal.
set -u
VOICE="$1"; MINUTES="${2:-9}"
BOOK="E:/Shared/BookForge/projects/God_s_People_Christian_Nationalism_In_The_Third_Reich_-_Owen_Morgan_(2026)"
LOG="/mnt/c/tmp/bench_${VOICE}.log"

cd /mnt/c/Users/tellt/Projects/bookforge || exit 1
PYTHONIOENCODING=utf-8 /mnt/c/Users/tellt/AppData/Local/Programs/Python/Python311/python.exe \
  cli/bookforge-tts.py --audiobook --project "$BOOK" \
  --voice "$VOICE" --tier fast --fresh > "$LOG" 2>&1 &
CLI=$!

# Wait for generation to start: at least 2 batches on disk so we are past warmup.
SID=""; D=""
for i in $(seq 1 80); do
  sleep 15
  SID=$(pgrep -af "worker.py --session" | grep -oE "session [a-f0-9-]{36}" | head -1 | awk '{print $2}')
  [ -z "$SID" ] && continue
  D=$(ls -d /home/telltale/ebook2audiobook/tmp/ebook-"$SID"/*/chapters/sentences 2>/dev/null | head -1)
  [ -n "$D" ] && [ "$(ls "$D"/*.flac 2>/dev/null | wc -l)" -ge 128 ] && break
done
if [ -z "$D" ]; then echo "$VOICE: generation never started"; kill $CLI 2>/dev/null; exit 1; fi

a=$(ls "$D"/*.flac 2>/dev/null | wc -l); t0=$(date +%s)
sleep $(( MINUTES * 60 ))
b=$(ls "$D"/*.flac 2>/dev/null | wc -l); t1=$(date +%s)
dt=$(( t1 - t0 ))
python3 -c "print(f'{\"$VOICE\":<18s} {$a} -> {$b} chunks in {$dt}s = {($b-$a)/($dt/60):6.1f} chunks/min')"

# TEARDOWN. Two bugs cost a whole benchmark run on 2026-07-25:
#
# 1. Sleeping a few seconds after the TERM is not enough. BookForge's GPU preflight
#    refuses to start a new job while ANY guest ebook2audiobook process survives, and
#    waits only 60s before erroring out — so the next voice in a sweep died with
#    "A previous TTS worker is still running inside WSL". WAIT for actual exit.
# 2. `pkill -f bench.sh` inside bench.sh matches its OWN command line and kills the
#    script (exit 15). Bracket the first character so the pattern cannot self-match.
#
# TERM only, never KILL: SIGKILL on a guest process holding the GPU wedges CUDA until
# the distro is restarted.
kill "$CLI" 2>/dev/null
pkill -TERM -f "ebook2audiobook/[a-z_]*\.py" 2>/dev/null
for i in $(seq 1 36); do
  n=$(pgrep -cf "ebook2audiobook/[a-z_]*\.py" 2>/dev/null || echo 0)
  [ "$n" = "0" ] && break
  sleep 5
done
n=$(pgrep -cf "ebook2audiobook/[a-z_]*\.py" 2>/dev/null || echo 0)
if [ "$n" != "0" ]; then
  # MEASURED 07-25: a worker mid-generation does NOT always exit on SIGTERM within
  # 180s — vLLM sits in a CUDA op. Escalating with SIGKILL is FORBIDDEN (a killed
  # guest process holding the GPU wedges CUDA until the distro restarts), and the
  # sanctioned escalation `wsl -t <distro>` can only be issued from the HOST, not
  # from inside the guest where this script runs.
  # So: fail loudly and STOP. Starting the next voice anyway just burns 10 minutes
  # producing a run that dies on GPU preflight ("A previous TTS worker is still
  # running inside WSL") — which is exactly what happened before this guard existed.
  echo "  FATAL: $n guest proc(s) survived TERM+180s. From the HOST run:"
  echo "         wsl.exe -t Ubuntu     # sanctioned escalation, never SIGKILL"
  echo "         then re-run the remaining voices."
  exit 3
fi
sleep 5

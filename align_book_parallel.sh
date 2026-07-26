#!/usr/bin/env bash
# align_book_parallel.sh — parallel epub-align for a book split into per-chapter
# (or per-fragment) audio+text units. Reusable across voices (deathstalker,
# God's People, Third Reich, ...).
#
# WHY PARALLEL: bookforge-tts --generate-sentences is CPU-bound on epub matching +
# drift self-check, with only brief GPU bursts for whisperx transcription. On a
# 20-core / 24 GB-GPU box the GPU sits at ~5-9% during a serial run, so aligning
# several chapters at once is a large wall-clock win at no accuracy cost (each
# chapter aligns independently).
#
# WHEN IT'S SAFE (see memory orpheus-align-parallelization):
#   * The source is SPLIT into per-chapter audio+text units (glob of wavs). A
#     single monolithic whole-book wav cannot be parallelized this way.
#   * NO concurrent WSL vLLM render / other GPU job. The real OOM risk is HOST RAM,
#     not GPU: each align worker peaks ~1.7 GB at the 60s --chunk-s default, so
#     CONC=4 ~= 7 GB host commit; stacked on a 12 GB WSL render that killed a 32 GB
#     box (see epub-align-quadratic-memory-and-cli-concurrency-oom). Check first:
#         powershell "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\""
#   * GPU has room: each whisperx model ~1.5 GB VRAM; CONC=4 ~= 6 GB.
#   Rule of thumb: CONC = min(4, free_gpu_gib/2, free_host_gib/2, cores/4).
#
# ============================ HOW TO STOP CLEANLY =============================
# This loop DISPATCHES workers. Killing a worker alone makes the loop launch the
# next one ("respawn"). Also: Git Bash launches a two-layer tree (bin/bash ->
# detached usr/bin/bash), so a TaskStop on the wrapper leaves the real script bash
# alive and dispatching. The reliable off switch is the STOP flag:
#
#     touch "<out-dir>/STOP"
#
# The loop checks STOP before each dispatch and exits without starting new work;
# in-flight chapters finish on their own and never respawn (a chapter's VTT now
# exists -> SKIP). To abort in-flight work too: drop STOP FIRST, then kill workers.
# A hard-kill PID for the whole tree is written to <out-dir>/align_par.loop.pid.
# =============================================================================
#
# Usage:
#   align_book_parallel.sh --epub BOOK.epub --out OUTDIR [--conc N] \
#       [--report] --wav "ch 0.wav" --wav "ch 1.wav" ...
#   align_book_parallel.sh --epub BOOK.epub --out OUTDIR --wav-dir DIR   # globs DIR/*.wav
#
set -uo pipefail
CONC=2      # SAFE default. Each chapter's TRANSCRIPTION stage spawns its own
            # multi-worker WhisperModel pool, so conc chapters => conc pools of
            # host-RAM spike (NOT the 1.7 GB align figure). conc=4 with only ~12 GB
            # avail OOM'd 2 chapters' slices (mkl_malloc) — see the OOM memory.
            # Raise to 3-4 ONLY with >~20 GB avail host RAM.
REPORT="--report"
EPUB=""; OUT=""; WAVDIR=""; WAVS=()
BF="C:/Users/tellt/Projects/bookforge"

while [ $# -gt 0 ]; do
  case "$1" in
    --epub) EPUB="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --conc) CONC="$2"; shift 2;;
    --wav) WAVS+=("$2"); shift 2;;
    --wav-dir) WAVDIR="$2"; shift 2;;
    --no-report) REPORT=""; shift;;
    --report) REPORT="--report"; shift;;
    --bookforge) BF="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$EPUB" ] && [ -n "$OUT" ] || { echo "need --epub and --out" >&2; exit 2; }
if [ -n "$WAVDIR" ]; then for w in "$WAVDIR"/*.wav; do WAVS+=("$w"); done; fi
[ "${#WAVS[@]}" -gt 0 ] || { echo "no wavs (use --wav or --wav-dir)" >&2; exit 2; }

cd "$BF"
STOP="$OUT/STOP"
mkdir -p "$OUT"
rm -f "$STOP"
echo "$$" > "$OUT/align_par.loop.pid"
echo "align_book_parallel: ${#WAVS[@]} wavs, conc=$CONC, out=$OUT"

align_one() {
  local wav="$1"
  local base; base=$(basename "$wav" .wav)
  local safe; safe=$(echo "$base" | tr ' /' '__')
  local out="$OUT/${safe}.vtt"
  local log="$OUT/${safe}.log"
  echo ">>> START $safe ($(date +%H:%M:%S))"
  python cli/bookforge-tts.py --generate-sentences \
      --audio "$wav" --epub "$EPUB" --out "$out" $REPORT \
      > "$log" 2>&1
  local rc=$?
  # DO NOT let a failed run count as "done" (SKIP-if-VTT-exists would keep garbage).
  # A transcription-pool OOM (mkl_malloc) drops slices -> huge no-match holes.
  if grep -qE 'completed WITH FAILURES|mkl_malloc: failed to allocate' "$log" 2>/dev/null; then
    mv "$out" "${out}.FAILED_oom" 2>/dev/null
    echo "!!! FAILED (OOM/slices) $safe -> quarantined VTT; will re-run next pass. Lower --conc."
  fi
  echo "<<< DONE  $safe rc=$rc ($(date +%H:%M:%S))"
}

pids=()
for wav in "${WAVS[@]}"; do
  if [ -f "$STOP" ]; then echo "STOP flag set -> no new dispatch"; break; fi
  base=$(basename "$wav" .wav); safe=$(echo "$base" | tr ' /' '__')
  if [ -f "$OUT/${safe}.vtt" ]; then echo ">>> SKIP (done): $safe"; continue; fi
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do
    [ -f "$STOP" ] && break
    sleep 3
  done
  [ -f "$STOP" ] && { echo "STOP flag set -> no new dispatch"; break; }
  align_one "$wav" &
  pids+=("$!")
  echo "    dispatched $safe pid=$! (running: $(jobs -rp | wc -l)/$CONC)"
  sleep 2   # stagger GPU model loads
done

echo "--- waiting on ${#pids[@]} dispatched jobs ---"
wait
rm -f "$OUT/align_par.loop.pid"
if [ -f "$STOP" ]; then echo "STOPPED EARLY (flag)"; else echo "ALL ALIGNMENTS COMPLETE"; fi

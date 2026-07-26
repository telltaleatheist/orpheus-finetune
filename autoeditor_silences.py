#!/usr/bin/env python
"""Auto-editor-driven silence map for cut_audiobook.

Runs auto-editor's per-frame loudness analysis over the book, thresholds it into
silence intervals, and writes JSON [[start_sec, end_sec], ...]. cut_audiobook loads
this via --silence-map and snaps every cut to these REAL silences (instead of librosa),
so no clip ends mid-word. Owen trusts auto-editor's silence detection for exactly this.

Usage: autoeditor_silences.py <audio> <out.json> [--thr 0.03] [--min-sil 0.10]
"""
import sys, os, subprocess, json, argparse, numpy as np, soundfile as sf

PY = r"C:\Users\tellt\Miniforge3\envs\bookforge-urvc\python.exe"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out")
    ap.add_argument("--thr", type=float, default=0.03,
                    help="loudness below this (0-1, auto-editor normalized) = silence")
    ap.add_argument("--min-sil", type=float, default=0.10,
                    help="minimum silence duration (s) to record")
    a = ap.parse_args()

    dur = sf.info(a.audio).duration           # FLAC/valid file -> correct duration
    cache = a.out + ".levels.npy"
    if os.path.exists(cache):
        v = np.load(cache)
        print(f"[ae] loaded cached levels {cache} ({v.size} frames)", flush=True)
    else:
        print(f"[ae] running auto-editor levels on {dur/3600:.2f}h ...", flush=True)
        p = subprocess.run([PY, "-m", "auto_editor", "levels", a.audio, "--edit", "audio"],
                           capture_output=True, text=True)
        if p.returncode != 0:
            print("[ae] STDERR:", p.stderr[-800:]); sys.exit(1)
        vals = []; started = False
        for line in p.stdout.splitlines():
            s = line.strip()
            if not s: continue
            if s.startswith("@"):
                started = True if s == "@start" else started
                continue
            if started:
                try: vals.append(float(s))
                except ValueError: pass
        v = np.asarray(vals, dtype=np.float32)
        if v.size == 0:
            print("[ae] no level values parsed"); sys.exit(1)
        np.save(cache, v)
    fps = 30.0  # auto-editor timebase is EXACTLY 30 fps for audio-only input.
    # NEVER compute fps = v.size/dur: auto-editor drops the trailing partial chunk,
    # so v.size < dur*30 by ~1-2s of frames; dividing smears that deficit across the
    # whole timeline as a linear stretch (+1.5s drift by the end of an 11.9h book),
    # which snapped cut boundaries into mid-sentence -> clips missing opening words.
    expected = dur * 30.0
    if abs(v.size - expected) > 90:  # >3s discrepancy = something else is wrong
        print(f"[ae] FATAL: frame count {v.size} vs expected {expected:.0f} "
              f"differs by {abs(v.size-expected)/30.0:.1f}s - not just trailing-chunk loss")
        sys.exit(1)
    print(f"[ae] frames {v.size}, expected {expected:.0f} "
          f"(trailing deficit {(expected-v.size)/30.0:.2f}s, harmless)", flush=True)
    pct = {p: float(np.percentile(v, p)) for p in (1, 5, 10, 25, 50, 75, 90)}
    print(f"[ae] {v.size} frames, {fps:.3f} fps  loudness "
          + " ".join(f"p{k}={pct[k]:.4f}" for k in (1,5,10,25,50,75,90))
          + f" max={v.max():.4f}", flush=True)
    for t in (0.008, 0.012, 0.02, 0.03):
        print(f"    thr={t}: {100*np.mean(v<t):.1f}% frames below", flush=True)

    # silence runs: consecutive frames below thr
    sil = v < a.thr
    intervals = []
    i = 0; n = len(sil)
    while i < n:
        if sil[i]:
            j = i
            while j < n and sil[j]: j += 1
            s0, s1 = i / fps, j / fps
            if s1 - s0 >= a.min_sil:
                intervals.append([round(s0, 4), round(s1, 4)])
            i = j
        else:
            i += 1
    total_sil = sum(b - aa for aa, b in intervals)
    print(f"[ae] {len(intervals)} silences >= {a.min_sil}s, total {total_sil/60:.1f} min "
          f"({100*total_sil/dur:.1f}% of book)", flush=True)
    json.dump({"fps": fps, "thr": a.thr, "duration": dur, "silences": intervals},
              open(a.out, "w"))
    print(f"[ae] wrote {a.out}", flush=True)

if __name__ == "__main__":
    main()

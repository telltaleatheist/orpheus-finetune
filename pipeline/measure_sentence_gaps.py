# -*- coding: utf-8 -*-
"""Measure a rendered voice's SELF-PRODUCED pauses from a BookForge sentence cache.

What is in each cached file (orpheus.py _save_audio, after the 2026-07-11 no-fallback
change that REMOVED the trailing-silence trim):

    [lead_gap] + [model audio, tail INTACT] + [trail_gap]

trail_gap comes from _classify_gap's floor, which is 0.6 s unless ORPHEUS_SENTENCE_GAP
is exported (BookForge only forwards it when set on its own process, so 0.6 in practice).
So:

    model_tail        = trailing_silence_in_file - BAKED_GAP
    internal_gap      = a silence run BETWEEN sentences inside a packed chunk. Nothing
                        touches these, so they are the model's natural inter-sentence
                        pause -- the number the assembly inject has to reproduce at a
                        chunk-to-chunk join.

Usage: measure_gaps.py <sentences_dir> <label> [baked_gap] [sample_n]
"""
import os, sys, io, json
import numpy as np, soundfile as sf
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SIL_DB = -55.0        # < this = true silence (same threshold as the corpus tail trim)
MIN_RUN = 0.12        # ignore sub-120ms dips: those are stop consonants, not pauses
FRAME = 0.010

def analyse(path):
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x.mean(axis=1)
    n = max(1, int(sr * FRAME))
    pad = (-len(x)) % n
    if pad: x = np.concatenate([x, np.zeros(pad, dtype='float32')])
    f = x.reshape(-1, n)
    rms = np.sqrt((f ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    sil = db <= SIL_DB
    if sil.all():
        return None
    first, last = int(np.argmin(sil)), len(sil) - 1 - int(np.argmin(sil[::-1]))
    lead = first * FRAME
    trail = (len(sil) - 1 - last) * FRAME
    internal = []
    i = first
    while i <= last:
        if sil[i]:
            j = i
            while j <= last and sil[j]:
                j += 1
            run = (j - i) * FRAME
            if run >= MIN_RUN:
                internal.append(run)
            i = j
        else:
            i += 1
    return lead, trail, internal, len(x) / sr

def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else float('nan')

d, label = sys.argv[1], sys.argv[2]
baked = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
want = int(sys.argv[4]) if len(sys.argv) > 4 else 400

files = sorted(f for f in os.listdir(d) if f.endswith(('.flac', '.wav')))
step = max(1, len(files) // want)
files = files[::step][:want]

leads, trails, internals, durs = [], [], [], []
skipped = 0
for f in files:
    r = analyse(os.path.join(d, f))
    if r is None:
        skipped += 1; continue
    l, t, ig, dur = r
    leads.append(l); trails.append(t); internals.extend(ig); durs.append(dur)

model_tails = [t - baked for t in trails]
print(f"\n===== {label} =====")
print(f"clips analysed {len(trails)} (of {len(os.listdir(d))} in cache), silent/skipped {skipped}")
print(f"baked trail_gap assumed {baked:.2f}s\n")
print(f"  clip duration      median {np.median(durs):6.2f}s")
print(f"  leading silence    median {np.median(leads):6.3f}s  p90 {pct(leads,90):6.3f}")
print(f"  trailing IN FILE   median {np.median(trails):6.3f}s  p90 {pct(trails,90):6.3f}")
print(f"  -> MODEL'S OWN TAIL median {np.median(model_tails):6.3f}s  p10 {pct(model_tails,10):6.3f}  p90 {pct(model_tails,90):6.3f}")
print(f"  INTERNAL sentence gaps: n={len(internals)}  median {np.median(internals) if internals else float('nan'):6.3f}s"
      f"  p25 {pct(internals,25):6.3f}  p75 {pct(internals,75):6.3f}  p90 {pct(internals,90):6.3f}")
print(f"     (internal gaps per clip: {len(internals)/max(1,len(trails)):.2f})")

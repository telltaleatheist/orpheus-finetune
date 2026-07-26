#!/usr/bin/env python
"""Internal-pause census of a gap-0 render: every silence is the MODEL's own.
Reports pause count/min and duration stats. Threshold -50 dBFS (tails are
~-50..-70 rendered hiss, speech gaps are deeper than speech)."""
import sys
import numpy as np, soundfile as sf

for p in sys.argv[1:]:
    y, sr = sf.read(p)
    if y.ndim > 1:
        y = y.mean(1)
    w = int(0.02 * sr)
    m = len(y) // w
    rms = np.sqrt(np.mean(y[:m * w].reshape(m, w) ** 2, axis=1))
    sil = rms < 10 ** (-50 / 20)
    runs, i = [], 0
    while i < m:
        if sil[i]:
            j = i
            while j < m and sil[j]:
                j += 1
            if i > 0 and j < m:
                runs.append((j - i) * 0.02)
            i = j
        else:
            i += 1
    r = np.array([x for x in runs if x >= 0.08])
    dur = len(y) / sr
    name = p.split("/")[-1].split(chr(92))[-1]
    if len(r):
        print(f"{name:38s} {dur:5.0f}s | pauses>=80ms: {len(r):3d} ({len(r)/(dur/60):4.1f}/min) | "
              f"median {np.median(r)*1000:4.0f}ms p90 {np.percentile(r,90)*1000:4.0f}ms max {r.max()*1000:4.0f}ms")
    else:
        print(f"{name:38s} {dur:5.0f}s | NO internal pauses >= 80ms — fully run together")

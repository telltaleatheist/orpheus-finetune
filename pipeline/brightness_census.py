#!/usr/bin/env python
"""Per-clip brightness census of a corpus. Confirms/kills the oscillation
mechanism: if per-clip presence-band energy has a wide or bimodal spread,
the model learned that spread and samples from it chunk by chunk.

Score per clip: (3.5-8 kHz mean dB) - (300-2000 Hz mean dB) — a level-immune
brightness ratio. Also RMS dynamics (prosody proxy: std of speech-window RMS).
Prints distribution stats + decile histogram.
"""
import csv, os, sys
import numpy as np, soundfile as sf

src = sys.argv[1]
rows = []
for split in ("train", "eval"):
    p = os.path.join(src, f"metadata_{split}.csv")
    if os.path.exists(p):
        rows += list(csv.reader(open(p), delimiter="|"))[1:]

scores = []
for k, r in enumerate(rows):
    y, sr = sf.read(os.path.join(src, r[0]))
    if y.ndim > 1:
        y = y.mean(1)
    y = y.astype(np.float32)
    NF = 4096
    w = np.hanning(NF)
    m = (len(y) - NF) // NF + 1
    ps = np.zeros(NF // 2 + 1)
    for j in range(m):
        s = y[j * NF: j * NF + NF] * w
        ps += np.abs(np.fft.rfft(s)) ** 2
    db = 10 * np.log10(ps / m + 1e-20)
    f = np.fft.rfftfreq(NF, 1 / sr)
    lo1, hi1 = np.searchsorted(f, 300), np.searchsorted(f, 2000)
    lo2, hi2 = np.searchsorted(f, 3500), np.searchsorted(f, 8000)
    bright = db[lo2:hi2].mean() - db[lo1:hi1].mean()
    # RMS dynamics on speech windows (50ms, above -40dB)
    wlen = int(0.05 * sr)
    mm = len(y) // wlen
    rms = np.sqrt(np.mean(y[:mm * wlen].reshape(mm, wlen) ** 2, axis=1))
    sp = 20 * np.log10(rms[rms > 10 ** (-40 / 20)] + 1e-12)
    dyn = float(np.std(sp)) if len(sp) > 4 else 0.0
    scores.append((r[0], bright, dyn, len(y) / sr))
    if (k + 1) % 500 == 0:
        print(f"... {k+1}/{len(rows)}", flush=True)

b = np.array([s[1] for s in scores])
print(f"\nclips: {len(b)}")
print(f"brightness ratio: mean {b.mean():.1f} dB | std {b.std():.1f} dB | "
      f"p10 {np.percentile(b,10):.1f} | p50 {np.median(b):.1f} | p90 {np.percentile(b,90):.1f} | "
      f"spread p90-p10 = {np.percentile(b,90)-np.percentile(b,10):.1f} dB")
hist, edges = np.histogram(b, bins=12)
for h, e in zip(hist, edges):
    print(f"  {e:6.1f} dB: {'#' * (h * 60 // max(hist))} {h}")
# chronological drift: mean brightness per 10% slice of the book
order = np.array([s[1] for s in scores])
n = len(order)
print("\nchronological drift (mean brightness per decile of corpus order):")
for d in range(10):
    seg = order[d * n // 10:(d + 1) * n // 10]
    print(f"  decile {d}: {seg.mean():6.1f} dB")
prefix = sys.argv[2] if len(sys.argv) > 2 else "/home/telltale/brightness"
np.save(prefix + "_scores.npy", np.array([(s[1], s[2], s[3]) for s in scores]))
with open(prefix + "_names.txt", "w") as fh:
    fh.write("\n".join(s[0] for s in scores))
print("CENSUS_DONE")

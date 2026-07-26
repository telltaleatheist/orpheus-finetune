#!/usr/bin/env python
"""Spectral tilt + level comparison: is v7 muffled vs v5, and by how much where."""
import sys
import numpy as np, soundfile as sf

for p in sys.argv[1:]:
    y, sr = sf.read(p)
    if y.ndim > 1:
        y = y.mean(1)
    y = y.astype(np.float32)
    rms = 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12)
    NF = 8192
    w = np.hanning(NF)
    m = (len(y) - NF) // (NF // 2) + 1
    ps = np.zeros(NF // 2 + 1)
    for j in range(m):
        s = y[j * NF // 2: j * NF // 2 + NF] * w
        ps += np.abs(np.fft.rfft(s)) ** 2
    db = 10 * np.log10(ps / m + 1e-20)
    f = np.fft.rfftfreq(NF, 1 / sr)
    bands = [(300, 1000), (1000, 2000), (2000, 3500), (3500, 5000), (5000, 6500), (6500, 8000), (8000, 10000)]
    out = []
    for lo, hi in bands:
        a, b = np.searchsorted(f, lo), np.searchsorted(f, hi)
        out.append(f"{lo//100}-{hi//100}:{db[a:b].mean():6.1f}")
    print(f"{p.split(chr(92))[-1].split('/')[-1]:42s} rms {rms:6.1f} dBFS | " + " ".join(out))

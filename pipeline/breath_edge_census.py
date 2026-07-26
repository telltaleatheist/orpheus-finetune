#!/usr/bin/env python
"""Do training clips END mid-breath? Census of clip-edge energy.

A silence-snapped cutter treats a -40dB breath as silence, so cuts can land
mid-breath — the model then learns 'speech, half-breath, room tone, stop' and
reproduces whispery ghost-breaths in every pause (Owen's report, all 3 voices).

Measures the final 300ms (and first 300ms) of each SOURCE clip: RMS in the
breath band (400-4000 Hz). Buckets: DEAD (<-55dB), BREATHY (-55..-30 — the
suspect zone), HOT (>-30, speech runs to the edge).
Usage: breath_edge_census.py <corpus_dir> [n]
"""
import csv, os, sys
import numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt

src = sys.argv[1]
n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 300
rows = []
for split in ("train", "eval"):
    p = os.path.join(src, f"metadata_{split}.csv")
    if os.path.exists(p):
        rows += [r[0] for r in list(csv.reader(open(p), delimiter="|"))[1:]]
rows = rows[::max(1, len(rows) // n_sample)][:n_sample]

sos = None
buckets = {"end": {"dead": 0, "breathy": 0, "hot": 0},
           "start": {"dead": 0, "breathy": 0, "hot": 0}}
breathy_ex = []
for rel in rows:
    y, sr = sf.read(os.path.join(src, rel))
    if y.ndim > 1:
        y = y.mean(1)
    y = y.astype(np.float32)
    if sos is None:
        sos = butter(4, [400, 4000], "bandpass", fs=sr, output="sos")
    n300 = int(0.3 * sr)
    for edge, seg in (("end", y[-n300:]), ("start", y[:n300])):
        b = sosfilt(sos, seg)
        db = 20 * np.log10(np.sqrt(np.mean(b ** 2)) + 1e-12)
        if db < -55:
            buckets[edge]["dead"] += 1
        elif db < -30:
            buckets[edge]["breathy"] += 1
            if edge == "end" and len(breathy_ex) < 5:
                breathy_ex.append((os.path.basename(rel), round(db, 1)))
        else:
            buckets[edge]["hot"] += 1

n = len(rows)
for edge in ("end", "start"):
    b = buckets[edge]
    print(f"{edge:5s}: dead {b['dead']*100//n}% | BREATHY(-55..-30) {b['breathy']*100//n}% | hot {b['hot']*100//n}%  (n={n})")
print("breathy-end examples:", breathy_ex)
print("BREATH_CENSUS_DONE")

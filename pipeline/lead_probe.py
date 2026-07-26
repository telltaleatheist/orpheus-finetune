#!/usr/bin/env python
"""What ACTUALLY precedes speech onset in corpus clips?
Per clip: onset = first 10ms window with full-band RMS > -30dBFS.
Reports: lead duration before onset, and the peak 10ms level in the lead
(after the fade region). Distinguishes 'clip starts with an inhale'
(lead >60ms with -50..-32dB content) from 'quiet speech start'.
Usage: lead_probe.py <corpus_dir> [n]"""
import csv, os, sys
import numpy as np, soundfile as sf

src = sys.argv[1]
n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 300
rows = []
for split in ("train", "eval"):
    p = os.path.join(src, f"metadata_{split}.csv")
    if os.path.exists(p):
        rows += [r[0] for r in list(csv.reader(open(p), delimiter="|"))[1:]]
rows = rows[::max(1, len(rows) // n_sample)][:n_sample]

leads, lead_peaks, n_breathy_lead = [], [], 0
for rel in rows:
    y, sr = sf.read(os.path.join(src, rel))
    if y.ndim > 1:
        y = y.mean(1)
    w = int(0.01 * sr)
    m = len(y) // w
    rms = np.sqrt(np.mean(y[:m * w].reshape(m, w) ** 2, axis=1))
    db = 20 * np.log10(rms + 1e-12)
    on = np.where(db > -30)[0]
    if len(on) == 0:
        continue
    lead_ms = on[0] * 10
    leads.append(lead_ms)
    if on[0] > 0:
        peak = float(db[:on[0]].max())
        lead_peaks.append(peak)
        # an inhale = >=60ms of lead whose peak sits in the breath zone
        if lead_ms >= 60 and -50 <= peak <= -32:
            n_breathy_lead += 1

leads = np.array(leads)
print(f"{src.rstrip('/').split('/')[-1]}: n={len(leads)}")
print(f"  lead-before-onset ms: p10 {np.percentile(leads,10):.0f}  median {np.median(leads):.0f}  p90 {np.percentile(leads,90):.0f}  max {leads.max():.0f}")
print(f"  clips with BREATHY lead (>=60ms, peak -50..-32dB): {n_breathy_lead} ({n_breathy_lead*100//max(1,len(leads))}%)")
if lead_peaks:
    lp = np.array(lead_peaks)
    print(f"  lead peak dB: median {np.median(lp):.1f}  p90 {np.percentile(lp,90):.1f}")
print("LEAD_PROBE_DONE")

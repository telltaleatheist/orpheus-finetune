#!/usr/bin/env python
"""Hunt the 'quiet ringing over the voice' in a finished audiobook.

Streams the file in segments via ffmpeg, and in every 4s SPEECH window
(RMS > -35 dBFS) looks for narrowband spectral peaks 800-11000 Hz that stand
>= 10 dB above the local spectral median (median over +/-400 Hz, excluding
+/-60 Hz around the peak). A real 'ringing' should show up as a frequency
CLUSTER that recurs across many windows; codec/formant noise scatters.

Usage: ring_scan.py <audiofile> [start_s] [dur_s]
Prints: per-cluster stats + exemplar timestamps, plus a per-10-min density
map so 'periodic, not every section' becomes visible.
"""
import subprocess, sys, os
import numpy as np

SR = 22050
WIN = 4.0
RMS_GATE = 10 ** (-35 / 20)
PROM_DB = 10.0
FMIN, FMAX = 800, 11000

path = sys.argv[1]
start = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
# total duration
dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries",
    "format=duration", "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip())
if len(sys.argv) > 3:
    dur = min(dur - start, float(sys.argv[3]))
else:
    dur = dur - start

SEG = 600.0  # decode 10 min at a time
hits = []    # (abs_time, freq, prom_db)
nwin = 0

t = start
while t < start + dur - 1:
    seglen = min(SEG, start + dur - t)
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-t", str(seglen),
                        "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                       capture_output=True)
    y = np.frombuffer(p.stdout, dtype=np.float32)
    n = int(WIN * SR)
    for k in range(len(y) // n):
        seg = y[k * n:(k + 1) * n]
        rms = np.sqrt(np.mean(seg ** 2))
        if rms < RMS_GATE:
            continue
        nwin += 1
        # Welch: 8192-pt hann, 50% overlap
        NF = 8192
        w = np.hanning(NF)
        m = (len(seg) - NF) // (NF // 2) + 1
        ps = np.zeros(NF // 2 + 1)
        for j in range(m):
            s = seg[j * NF // 2: j * NF // 2 + NF] * w
            ps += np.abs(np.fft.rfft(s)) ** 2
        ps /= m
        db = 10 * np.log10(ps + 1e-20)
        freqs = np.fft.rfftfreq(NF, 1 / SR)
        lo, hi = np.searchsorted(freqs, FMIN), np.searchsorted(freqs, FMAX)
        bw = int(400 / (SR / NF))   # bins in 400 Hz
        ex = int(60 / (SR / NF))    # exclusion +/-60 Hz
        band = db[lo:hi]
        # local median via strided windows (cheap approximation: percentile filter)
        for i in range(bw, len(band) - bw):
            v = band[i]
            if v < db[lo:hi].max() - 40:
                continue
            neigh = np.concatenate([band[i - bw:i - ex], band[i + ex:i + bw]])
            med = np.median(neigh)
            if v - med >= PROM_DB and v == band[max(0, i - ex):i + ex + 1].max():
                hits.append((t + k * WIN, freqs[lo + i], v - med))
    t += seglen
    print(f"... scanned to {t/60:.0f} min, {len(hits)} hits", flush=True)

hits = np.array(hits) if hits else np.zeros((0, 3))
print(f"\nspeech windows: {nwin}, tonal hits: {len(hits)} ({100*len(set(hits[:,0]))/max(nwin,1):.1f}% of windows)")
if len(hits):
    # cluster by frequency (100 Hz bins)
    fb = (hits[:, 1] // 100 * 100).astype(int)
    uniq, cnt = np.unique(fb, return_counts=True)
    order = np.argsort(-cnt)
    print("\ntop frequency clusters:")
    for i in order[:12]:
        sel = hits[fb == uniq[i]]
        ts = sel[np.argsort(-sel[:, 2])][:4, 0]
        print(f"  {uniq[i]:5d}-{uniq[i]+100} Hz: {cnt[i]:4d} hits | median prom {np.median(sel[:,2]):.1f} dB | exemplars @ "
              + ", ".join(f"{x/60:.1f}min" for x in ts))
    # density per 10 min
    print("\nhit density per 10-min block (hits/min):")
    tb = (hits[:, 0] // 600).astype(int)
    for b in range(int(start // 600), int((start + dur) // 600) + 1):
        c = (tb == b).sum()
        bar = "#" * min(60, int(c / 10))
        print(f"  {b*10:4d}-{b*10+10:4d} min: {c/10:5.1f} {bar}")
print("RING_SCAN_DONE")

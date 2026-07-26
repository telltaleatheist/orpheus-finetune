#!/usr/bin/env python
"""Tonal-artifact scanner for Orpheus training data (the denoise blind spot).

Broadband denoise (mel-band roformer) is MASKING: it removes hiss/hum but NOT
narrowband tones — AAC "birdies", electronic whines, hum harmonics. Those survive
cleaning and get learned into the voice, where they surface as a faint ringing
over the narration (Owen heard exactly this in the mistborn/Hellworld renders).

Detection: a real tone is a spectral peak that is (a) NARROW (a few bins wide),
(b) PROMINENT vs its local spectral neighbourhood, and (c) PERSISTENT at a stable
frequency across many frames. Speech harmonics fail (c) — pitch moves constantly;
sibilance fails (a) — it is broadband. So we score, per frequency bin, the
fraction of frames where that bin stands proud of its neighbours.

Usage: tonal_scan.py <dir-of-wavs | wav> [--n 60] [--prom 8] [--persist 0.30]
"""
import sys, os, glob, argparse, random
import numpy as np, soundfile as sf


def tone_report(path, prom_db, persist, fmin=200.0):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(1)
    fl, hop = 2048, 512
    if len(y) < fl * 4:
        return []
    win = np.hanning(fl)
    frames = np.array([y[i:i+fl]*win for i in range(0, len(y)-fl, hop)])
    S = np.abs(np.fft.rfft(frames, axis=1))
    f = np.fft.rfftfreq(fl, 1/sr)
    S_db = 20*np.log10(S + 1e-12)

    # local neighbourhood median: +-12 bins, excluding the +-2 bins around centre
    k = 12
    pad = np.pad(S_db, ((0, 0), (k, k)), mode='edge')
    nb = np.stack([pad[:, i:i+S_db.shape[1]] for i in list(range(0, k-2)) + list(range(k+3, 2*k+1))], axis=0)
    local = np.median(nb, axis=0)
    prominence = S_db - local                      # dB above local spectrum

    # only consider frames with actual signal (tones matter under speech)
    energy = S.sum(1)
    active = energy > np.percentile(energy, 40)
    if active.sum() < 10:
        return []
    P = prominence[active]

    hits = (P > prom_db)
    frac = hits.mean(0)                            # per-bin persistence
    out = []
    band = f >= fmin
    idx = np.where((frac > persist) & band)[0]
    # collapse adjacent bins into one tone
    for i in idx:
        if out and i - out[-1][3] <= 2:
            if frac[i] > out[-1][1]:
                out[-1] = (f[i], frac[i], float(np.median(P[:, i][hits[:, i]])), i)
            continue
        out.append((f[i], frac[i], float(np.median(P[:, i][hits[:, i]])), i))
    return [(a, b, c) for a, b, c, _ in out]


def scan(target, n, prom_db, persist, label, fmin=200.0):
    files = ([target] if target.lower().endswith(('.wav', '.flac'))
             else sorted(glob.glob(os.path.join(target, '**', '*.wav'), recursive=True)))
    if not files:
        print(f"{label}: no audio found at {target}")
        return
    random.seed(11)
    if len(files) > n:
        files = random.sample(files, n)
    agg = {}
    per_clip = 0
    for p in files:
        tones = tone_report(p, prom_db, persist, fmin)
        if tones:
            per_clip += 1
        for freq, fr, pr in tones:
            key = round(freq / 25) * 25            # 25 Hz buckets
            e = agg.setdefault(key, [0, 0.0, 0.0])
            e[0] += 1
            e[1] = max(e[1], fr)
            e[2] = max(e[2], pr)
    print(f"\n=== {label}  ({len(files)} clips scanned, prom>{prom_db}dB, persist>{persist:.0%})")
    print(f"    clips containing >=1 persistent tone: {per_clip}/{len(files)} ({100*per_clip/len(files):.0f}%)")
    if not agg:
        print("    NO persistent tones detected — clean")
        return
    rows = sorted(agg.items(), key=lambda kv: -kv[1][0])[:12]
    print(f"    {'freq':>8s} {'clips':>6s} {'max-persist':>12s} {'max-prom':>9s}")
    for freq, (cnt, fr, pr) in rows:
        print(f"    {freq:7.0f}Hz {cnt:6d} {100*fr:11.0f}% {pr:8.1f}dB")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('targets', nargs='+', help='dir|file, optionally path=Label')
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--prom', type=float, default=8.0)
    ap.add_argument('--persist', type=float, default=0.30)
    ap.add_argument('--fmin', type=float, default=200.0,
                    help='ignore tones below this (voice fundamentals/low harmonics live '
                         'under ~600Hz; use 1000 to isolate true artifact tones)')
    a = ap.parse_args()
    for t in a.targets:
        path, _, label = t.partition('=')
        scan(path, a.n, a.prom, a.persist, label or os.path.basename(path.rstrip('/\\')), a.fmin)

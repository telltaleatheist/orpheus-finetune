#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Restore the treble a denoiser removed, matched to the MEASURED deficit.

Adobe Enhance cleans the noise floor but de-brightens: measured on mistborn part1 it
removed ~1.5 dB below 1 kHz (mostly just level) and 7-9 dB above 3 kHz. A generic
high shelf would be a guess; this measures the average SPEECH-band spectrum of the
processed corpus against the untouched one and builds the exact inverse.

`--strength` scales the correction. 1.0 restores the original tilt exactly, which also
re-amplifies whatever HF noise the denoiser removed; below 1.0 keeps part of the
cleaning benefit. The broadband offset (the low-band deficit, which is level not tone)
is removed first so this only ever changes TONE, not loudness.

Zero-phase linear FIR via firwin2 — no phase smearing on the training audio.

Usage: eq_restore.py <processed_corpus> <reference_corpus> <out_corpus> [--strength 0.8]
"""
import argparse, csv, io, os, shutil, sys
import numpy as np, soundfile as sf
from scipy.signal import firwin2, fftconvolve
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

ap = argparse.ArgumentParser()
ap.add_argument('processed'); ap.add_argument('reference'); ap.add_argument('out')
ap.add_argument('--strength', type=float, default=0.8)
ap.add_argument('--max-boost', type=float, default=12.0, help='dB ceiling, safety')
ap.add_argument('--ntaps', type=int, default=1023)
a = ap.parse_args()

NFFT = 2048
def avg_speech_spec(d, names):
    acc, n, sr = None, 0, None
    for c in names:
        x, sr = sf.read(os.path.join(d, 'wavs', c), dtype='float32')
        if len(x) < NFFT * 4: continue
        f = x[:len(x)//NFFT*NFFT].reshape(-1, NFFT)
        rms = np.sqrt((f**2).mean(axis=1) + 1e-12)
        sp = f[20*np.log10(rms+1e-12) > -30]
        if not len(sp): continue
        S = (np.abs(np.fft.rfft(sp*np.hanning(NFFT), axis=1))**2).mean(axis=0)
        acc = S if acc is None else acc + S; n += 1
    return acc/n, sr

names = sorted(os.listdir(os.path.join(a.processed, 'wavs')))[:80]
sp_, sr = avg_speech_spec(a.processed, names)
sr_, _ = avg_speech_spec(a.reference, names)
fq = np.fft.rfftfreq(NFFT, 1/sr)
deficit = 10*np.log10((sr_+1e-20)/(sp_+1e-20))

# smooth, then remove the broadband offset (level, not tone) using the 80-800 Hz mean
k = 9
sm = np.convolve(deficit, np.ones(k)/k, mode='same')
base = sm[(fq >= 80) & (fq < 800)].mean()
curve = np.clip((sm - base) * a.strength, 0, a.max_boost)
curve[fq < 300] = 0.0

freqs = np.concatenate([[0.0], fq[1:-1], [sr/2]])
gains = np.concatenate([[1.0], 10**(curve[1:-1]/20), [10**(curve[-1]/20)]])
taps = firwin2(a.ntaps, freqs/(sr/2), gains)

print(f'strength {a.strength}  |  correction applied (dB):')
for lo, hi in [(300,1000),(1000,2000),(2000,3000),(3000,4000),(4000,6000),(6000,8000),(8000,11500)]:
    s = (fq>=lo)&(fq<hi)
    print(f'   {lo:>5}-{hi:<6} {curve[s].mean():+5.2f}')

os.makedirs(os.path.join(a.out, 'wavs'), exist_ok=True)
allnames = sorted(os.listdir(os.path.join(a.processed, 'wavs')))
peak = 0.0
for c in allnames:
    x, s2 = sf.read(os.path.join(a.processed, 'wavs', c), dtype='float32')
    y = fftconvolve(x, taps, mode='same').astype('float32')
    peak = max(peak, float(np.abs(y).max()))
    sf.write(os.path.join(a.out, 'wavs', c), np.clip(y, -1.0, 1.0), s2, subtype='PCM_16')
for fn in ('metadata_train.csv', 'metadata_eval.csv'):
    p = os.path.join(a.processed, fn)
    if os.path.isfile(p): shutil.copy2(p, os.path.join(a.out, fn))
print(f'\nwrote {len(allnames)} clips -> {a.out}   (peak {20*np.log10(peak+1e-12):+.2f} dBFS)')

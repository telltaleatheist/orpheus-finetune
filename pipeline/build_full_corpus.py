#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-split several Adobe-processed parts and assemble ONE training corpus.

Each part is cut at the sample offsets recorded in split_map.json (x the rate ratio,
since Enhance returns 48 kHz), decimated back to 24 kHz with resample_poly, and written
under its ORIGINAL clip name so the metadata text still matches.

Verifies per part that the processed file's duration matches the original before
cutting anything — a mismatch means the timeline moved and the offsets are invalid.

Usage:
  build_full_corpus.py <adobe_test_dir> <out_corpus> <part_no>:<wav> [<part_no>:<wav> ...]
"""
import csv, io, os, sys
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import json
D, OUT, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
if not specs: sys.exit('need at least one <part_no>:<wav>')
m = json.load(open(os.path.join(D, 'split_map.json'), encoding='utf-8'))
SR = m['sample_rate']
os.makedirs(os.path.join(OUT, 'wavs'), exist_ok=True)

text = {}
for fn in ('metadata_train.csv', 'metadata_eval.csv'):
    p = os.path.join(m['corpus'], fn)
    if os.path.isfile(p):
        for r in csv.reader(open(p, encoding='utf-8'), delimiter='|'):
            if len(r) >= 3 and r[0] != 'audio_file':
                text[os.path.basename(r[0])] = (r[1], r[2])

written = []
for spec in specs:
    pno, wav = spec.split(':', 1)
    part = next(p for p in m['parts'] if p['part'] == int(pno))
    info = sf.info(wav)
    ratio = info.samplerate // SR
    exp = part['frames'] * ratio
    drift = (info.frames - exp) / info.samplerate
    status = 'OK' if abs(drift) < 0.05 else 'TIMELINE MOVED'
    print(f"part{pno}: {info.samplerate} Hz (x{ratio})  {info.frames} frames vs {exp} expected  "
          f"drift {drift*1000:+.1f} ms  [{status}]")
    if status != 'OK':
        sys.exit(f'  REFUSING: part{pno} duration changed by {drift:.3f}s — offsets are no longer valid')
    audio, _ = sf.read(wav, dtype='float32', always_2d=True)
    for e in part['entries']:
        seg = audio[e['start_sample']*ratio : e['end_sample']*ratio].mean(axis=1)
        if ratio > 1: seg = resample_poly(seg, 1, ratio).astype('float32')
        seg = np.pad(seg, (0, max(0, e['frames']-len(seg))))[:e['frames']]
        sf.write(os.path.join(OUT, 'wavs', e['clip']), seg, SR, subtype=m['subtype'])
        written.append(e['clip'])
    print(f"   -> {len(part['entries'])} clips")

ev = [c for c in written if c.startswith('e_')]
tr = [c for c in written if not c.startswith('e_')]
if len(ev) < 8:                       # too few held out — carve a deterministic 10%
    allc = sorted(written); ev = [c for i, c in enumerate(allc) if i % 10 == 4]
    tr = [c for c in allc if c not in set(ev)]
for split, fn in ((sorted(tr), 'metadata_train.csv'), (sorted(ev), 'metadata_eval.csv')):
    with open(os.path.join(OUT, fn), 'w', encoding='utf-8', newline='') as f:
        f.write('audio_file|text|speaker_name\n')
        for c in split:
            if c in text: f.write(f'wavs/{c}|{text[c][0]}|{text[c][1]}\n')
dur = sum(sf.info(os.path.join(OUT, 'wavs', c)).frames for c in written)/SR
print(f"\ncorpus {OUT}: {len(written)} clips, {dur/3600:.2f} h, {len(tr)} train / {len(ev)} eval")

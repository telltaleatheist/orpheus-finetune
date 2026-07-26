#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-split an Adobe-processed part back into clips, and build a twin CONTROL corpus.

Adobe Enhance returns the file at 48 kHz with the timeline preserved (verified: -1.2 ms
of drift across 30 minutes, i.e. none), so a clip's bounds are just its recorded 24 kHz
sample offsets x2. Decimation back to 24 kHz is an exact /2, done with resample_poly so
it is properly anti-aliased rather than naive dropping.

Writes TWO corpora with identical filenames and identical metadata, differing only in
the audio: the processed one and an untouched control cut from the ORIGINAL part. That
is what makes the quicktrain causal — one variable, everything else byte-identical.

Usage: resplit_adobe_parts.py <adobe_test_dir> <part_no> <processed_wav> <out_prefix>
"""
import csv, io, json, os, sys
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

D, PART, PROC, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
m = json.load(open(os.path.join(D, 'split_map.json'), encoding='utf-8'))
SR = m['sample_rate']
part = next(p for p in m['parts'] if p['part'] == PART)

proc, psr = sf.read(PROC, dtype='float32', always_2d=True)
orig, osr = sf.read(os.path.join(D, part['file']), dtype='float32', always_2d=True)
assert osr == SR, f'original rate {osr} != {SR}'
ratio = psr // SR
assert psr == SR * ratio, f'processed rate {psr} is not an integer multiple of {SR}'
print(f'part{PART}: {len(part["entries"])} clips   processed {psr} Hz (x{ratio})   '
      f'{len(proc)/psr:.3f}s vs original {len(orig)/osr:.3f}s')

corpora = {f'{OUT}_adobe': ('proc', proc, psr), f'{OUT}_control': ('orig', orig, osr)}
for path, (tag, audio, sr) in corpora.items():
    os.makedirs(os.path.join(path, 'wavs'), exist_ok=True)
rows = {'train': [], 'eval': []}
short = 0
for e in part['entries']:
    for path, (tag, audio, sr) in corpora.items():
        s = e['start_sample'] * (sr // SR)
        t = e['end_sample'] * (sr // SR)
        seg = audio[s:t].mean(axis=1)
        if sr != SR:
            seg = resample_poly(seg, 1, sr // SR).astype('float32')
        want = e['frames']
        if len(seg) < want: seg = np.pad(seg, (0, want - len(seg)))
        seg = seg[:want]
        sf.write(os.path.join(path, 'wavs', e['clip']), seg, SR, subtype=m['subtype'])
    if e['split'] in rows: rows[e['split']].append(e['clip'])
    else: short += 1

# metadata: same text, same split, for BOTH corpora
text = {}
for fn in ('metadata_train.csv', 'metadata_eval.csv'):
    p = os.path.join(m['corpus'], fn)
    if os.path.isfile(p):
        for r in csv.reader(open(p, encoding='utf-8'), delimiter='|'):
            if len(r) >= 3: text[os.path.basename(r[0])] = (r[1], r[2])
for path in corpora:
    for split, fn in (('train', 'metadata_train.csv'), ('eval', 'metadata_eval.csv')):
        with open(os.path.join(path, fn), 'w', encoding='utf-8', newline='') as f:
            f.write('audio_file|text|speaker_name\n')
            for c in rows[split]:
                if c in text:
                    f.write(f'wavs/{c}|{text[c][0]}|{text[c][1]}\n')
    print(f'  {os.path.basename(path)}: {len(rows["train"])} train, {len(rows["eval"])} eval')
if short: print(f'  ({short} clip(s) not in either metadata file - audio written, not listed)')

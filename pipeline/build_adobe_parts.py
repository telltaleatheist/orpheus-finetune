#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Concatenate a training corpus into Adobe-sized parts, with an exact re-split map.

The clips are joined AS THEY ARE — no guard tones, no inserted silence. The natural
inter-clip silence already at every join is the landmark, and auto-editor's silence
map over the part is what we compare before and after Adobe: if the silences land in
the same places, the sample offsets below are still exact and the re-split is lossless.

Every clip's offset is recorded in SAMPLES (not seconds) so the re-split is exact
rather than rounded. Also records each clip's own md5 so a returned part can be proved
to line up clip-for-clip.

Usage: build_adobe_parts.py <corpus_dir> <out_dir> [--minutes 30]
"""
import argparse, csv, hashlib, io, json, os, re, sys
import numpy as np, soundfile as sf
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

ap = argparse.ArgumentParser()
ap.add_argument('corpus'); ap.add_argument('out'); ap.add_argument('--minutes', type=float, default=30.0)
ap.add_argument('--prefix', default='',
                help="output filename stem; defaults to the corpus folder name")
ap.add_argument('--parts', type=int, default=0,
                help='force EXACTLY this many parts (with --max-total, drops the excess uniformly)')
ap.add_argument('--max-total', type=float, default=0.0,
                help='minutes; corpora larger than this are thinned UNIFORMLY across book order '
                     'so the loss is spread instead of amputating the ending')
a = ap.parse_args()

split_of = {}
for fn, tag in (('metadata_train.csv', 'train'), ('metadata_eval.csv', 'eval')):
    p = os.path.join(a.corpus, fn)
    if not os.path.isfile(p): continue
    with open(p, encoding='utf-8') as f:
        r = csv.reader(f, delimiter='|')
        head = next(r, None)
        for row in r:
            if row: split_of[os.path.basename(row[0])] = tag

wav_dir = os.path.join(a.corpus, 'wavs')
# Order by the NUMERIC clip id, not the filename: eval clips carry an `e_` prefix, so a
# plain sort sweeps every one of them to the front and the first part stops being a book
# slice at all (this silently happened to mistborn part1).
def booknum(name):
    mm = re.search(r'(\d+)\.wav$', name)
    return int(mm.group(1)) if mm else 0
clips = sorted((f for f in os.listdir(wav_dir) if f.lower().endswith('.wav')), key=booknum)
if not clips: sys.exit('no clips found')
durs = {c: sf.info(os.path.join(wav_dir, c)).frames for c in clips}
if a.max_total > 0:
    sr0 = sf.info(os.path.join(wav_dir, clips[0])).samplerate
    total = sum(durs.values()) / sr0 / 60
    if total > a.max_total:
        drop_frac = 1 - a.max_total / total
        step = max(2, int(round(1 / drop_frac)))
        dropped = [c for i, c in enumerate(clips) if i % step == step - 1]
        clips = [c for c in clips if c not in set(dropped)]
        kept = sum(durs[c] for c in clips) / sr0 / 60
        print(f'thinned to fit {a.max_total:g} min: dropped every {step}th clip '
              f'({len(dropped)} clips, {total-kept:.2f} min) -> {kept:.2f} min')
if a.parts > 0:
    sr0 = sf.info(os.path.join(wav_dir, clips[0])).samplerate
    LIMIT_OVERRIDE = int(sum(durs[c] for c in clips) / a.parts) + 1
else:
    LIMIT_OVERRIDE = None
info0 = sf.info(os.path.join(wav_dir, clips[0]))
SR, CH, ST = info0.samplerate, info0.channels, info0.subtype
LIMIT = LIMIT_OVERRIDE if LIMIT_OVERRIDE else int(a.minutes * 60 * SR)
os.makedirs(a.out, exist_ok=True)
print(f'{len(clips)} clips  {SR}Hz {CH}ch {ST}   target {a.minutes:g} min/part')

if a.parts > 0:
    # Split on cumulative targets rather than a fixed cap: a cap plus rounding leaves a
    # 2-clip stub part, which is useless to submit and skews the count.
    total_n = sum(durs[c] for c in clips)
    parts, cur, cur_n, done = [], [], 0, 0
    for c in clips:
        cur.append(c); cur_n += durs[c]; done += durs[c]
        if len(parts) < a.parts - 1 and done >= (len(parts) + 1) * total_n / a.parts:
            parts.append(cur); cur, cur_n = [], 0
    if cur: parts.append(cur)
else:
    parts, cur, cur_n = [], [], 0
    for c in clips:
        n = durs[c]
        if cur and cur_n + n > LIMIT:
            parts.append(cur); cur, cur_n = [], 0
        cur.append(c); cur_n += n
    if cur: parts.append(cur)

STEM = a.prefix or os.path.basename(os.path.normpath(a.corpus))
manifest = {'corpus': os.path.abspath(a.corpus), 'sample_rate': SR, 'channels': CH,
            'subtype': ST, 'guard_samples': 0, 'parts': []}
for pi, names in enumerate(parts, 1):
    buf, entries, off = [], [], 0
    for c in names:
        d, sr = sf.read(os.path.join(wav_dir, c), dtype='int16', always_2d=True)
        assert sr == SR, f'{c} sample-rate mismatch {sr}'
        h = hashlib.md5(d.tobytes()).hexdigest()[:16]
        entries.append({'clip': c, 'split': split_of.get(c, 'unlisted'),
                        'start_sample': off, 'end_sample': off + len(d),
                        'frames': len(d), 'seconds': round(len(d) / SR, 4), 'md5_16': h})
        buf.append(d); off += len(d)
    audio = np.concatenate(buf, axis=0)
    name = f'{STEM}_part{pi}.wav'
    sf.write(os.path.join(a.out, name), audio, SR, subtype=ST)
    manifest['parts'].append({'part': pi, 'file': name, 'clips': len(names),
                              'frames': int(len(audio)), 'seconds': round(len(audio)/SR, 3),
                              'entries': entries})
    print(f'  part{pi}: {len(names):>3} clips  {len(audio)/SR/60:6.2f} min  -> {name}')

json.dump(manifest, open(os.path.join(a.out, 'split_map.json'), 'w', encoding='utf-8'), indent=1)
with open(os.path.join(a.out, 'split_map.txt'), 'w', encoding='utf-8') as f:
    f.write('Re-split map. Offsets are SAMPLES at %d Hz. Clips were joined with NO gap.\n' % SR)
    f.write('Verify a returned part by: total frames match, and auto-editor silences align.\n\n')
    for p in manifest['parts']:
        f.write(f"=== {p['file']}  {p['clips']} clips  {p['seconds']:.3f}s ===\n")
        for e in p['entries']:
            f.write(f"  {e['start_sample']:>10} .. {e['end_sample']:>10}  {e['seconds']:7.3f}s  "
                    f"{e['split']:<8} {e['clip']}\n")
print(f"\nwrote split_map.json / split_map.txt  ({sum(len(p['entries']) for p in manifest['parts'])} clips mapped)")

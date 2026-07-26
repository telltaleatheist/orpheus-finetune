#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""epoch_battery.py — render EVERY epoch of a run, measure it, and rank by distance
to the TRAINING CORPUS.

Why this exists. The gates it consolidates each measure a model against ITSELF, and
that hid two real failures on 2026-07-25:

  * the EOS gate passes on `stopped=True`. A model that emits 10 s of silence and THEN
    stops is a pass. The silence mass is visible in the token counts the gate already
    prints (1.3-1.5x expected) and simply was not part of the criterion.
  * loop_gate's BLOAT verdict is relative to the probe set's OWN median, so a model
    that is uniformly slow drags the median with it and reads clean. That same model
    rendered dialogue at 8.9 ch/s against a normal ~14.

The fix is to make the CORPUS the reference. It is the ground truth the model was asked
to imitate, so every metric is reported as a distance from it and "best epoch" means
closest to the narrator, not best on some absolute scale.

  # stage 1 (WSL, env orpheus_tts — holds the GPU)
  epoch_battery.py render <models_root> <prefix> <token> <corpus> <out_root> [n]

  # stage 2 (anywhere)
  epoch_battery.py report <out_root> <corpus>
"""
import csv
import glob
import io
import json
import os
import re
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace',
                              line_buffering=True)

EOA, BASE, MAX_TOKENS = 128258, 128266, 3700
SIL_DB, MIN_RUN, FRAME = -55.0, 0.12, 0.010

# Denominators for the distance score: roughly "how much of this is audible".
# 2 dB of tilt, 1 ch/s of pace, 100 ms of rhythm are each about one JND here.
W_TILT, W_RATE, W_GAP = 2.0, 1.0, 0.1


def analyse(path):
    """Every per-clip number the battery needs, from one read of the file."""
    import soundfile as sf
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x.mean(axis=1)
    n = max(1, int(sr * FRAME))
    x = np.concatenate([x, np.zeros((-len(x)) % n, dtype='float32')])
    f = x.reshape(-1, n)
    db = 20 * np.log10(np.sqrt((f ** 2).mean(axis=1)) + 1e-12)
    sil = db <= SIL_DB
    if sil.all():
        return None
    first = int(np.argmin(sil))
    last = len(sil) - 1 - int(np.argmin(sil[::-1]))
    internal, i = [], first
    while i <= last:
        if sil[i]:
            j = i
            while j <= last and sil[j]:
                j += 1
            if (j - i) * FRAME >= MIN_RUN:
                internal.append((j - i) * FRAME)
            i = j
        else:
            i += 1
    speech = db > -30
    tilt = float('nan')
    if speech.sum() >= 5:
        S = np.abs(np.fft.rfft(f[speech] * np.hanning(n), axis=1)) ** 2
        fq = np.fft.rfftfreq(n, 1 / sr)
        lo = S[:, (fq >= 200) & (fq < 2000)].sum(axis=1)
        hi = S[:, (fq >= 4000) & (fq < 9000)].sum(axis=1)
        tilt = float(np.median(10 * np.log10(hi / (lo + 1e-12) + 1e-12)))
    return dict(dur=len(x) / sr, tail=(len(sil) - 1 - last) * FRAME, internal=internal,
                max_sil=max(internal) if internal else 0.0,
                speech_frac=float(speech.mean()), tilt=tilt)


def summarise(rows, texts=None):
    if not rows:
        return {}
    ig = [g for r in rows for g in r['internal']]
    out = dict(
        n=len(rows),
        tail=float(np.median([r['tail'] for r in rows])),
        tail_p90=float(np.percentile([r['tail'] for r in rows], 90)),
        internal=float(np.median(ig)) if ig else float('nan'),
        max_sil=float(max(r['max_sil'] for r in rows)),
        speech_frac=float(np.median([r['speech_frac'] for r in rows])) * 100,
        tilt=float(np.median([r['tilt'] for r in rows])),
        tilt_spread=float(np.std([r['tilt'] for r in rows])),
    )
    if texts:
        rates = [len(t) / r['dur'] for t, r in zip(texts, rows) if r['dur'] > 0]
        out['chars_per_sec'] = float(np.median(rates)) if rates else float('nan')
    return out


def corpus_reference(corpus, limit=120):
    """The narrator's own numbers — the target every epoch is scored against."""
    text_of = {}
    for fn in ('metadata_train.csv', 'metadata_eval.csv'):
        p = os.path.join(corpus, fn)
        if os.path.isfile(p):
            for r in csv.reader(open(p, encoding='utf-8'), delimiter='|'):
                if len(r) >= 2 and r[0] != 'audio_file':
                    text_of[os.path.basename(r[0])] = r[1]
    rows, texts = [], []
    for c in sorted(os.listdir(os.path.join(corpus, 'wavs')))[:limit]:
        a = analyse(os.path.join(corpus, 'wavs', c))
        if a:
            rows.append(a)
            texts.append(text_of.get(c, ''))
    return summarise(rows, texts)


def probe_texts(corpus, n_chunks):
    rows = []
    for split in ('eval', 'train'):
        p = os.path.join(corpus, 'metadata_%s.csv' % split)
        if os.path.exists(p):
            rows += [r[1] for r in list(csv.reader(open(p, encoding='utf-8'),
                                                   delimiter='|'))[1:]]
    texts = [t for t in rows if 200 <= len(t) <= 360][:n_chunks]
    if len(texts) < n_chunks:
        texts += [t for t in rows if 120 <= len(t) < 200][:n_chunks - len(texts)]
    return texts


def render(argv):
    import soundfile as sf
    import torch
    from snac import SNAC
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    root, prefix, token, corpus, out_root = argv[:5]
    n_chunks = int(argv[5]) if len(argv) > 5 else 20
    eps = sorted(glob.glob(os.path.join(root, prefix + '_ep*')),
                 key=lambda p: int(re.search(r'_ep(\d+)$', p).group(1)))
    if not eps:
        raise SystemExit('no epochs matching %s_ep* in %s' % (prefix, root))
    print('%d epochs: %s' % (len(eps), ', '.join(os.path.basename(e) for e in eps)))

    texts = probe_texts(corpus, n_chunks)
    if not texts:
        raise SystemExit('no chunk-sized texts in %s' % corpus)
    print('%d probe chunks, IDENTICAL across every epoch' % len(texts))

    snac = SNAC.from_pretrained('hubertsiuzdak/snac_24khz').eval()
    for ep in eps:
        name = os.path.basename(ep)
        out = os.path.join(out_root, name)
        if os.path.exists(os.path.join(out, 'manifest.json')):
            print('  %s: already rendered, skipping' % name)
            continue
        os.makedirs(out, exist_ok=True)
        tok = AutoTokenizer.from_pretrained(ep)
        llm = LLM(model=ep, dtype='bfloat16', gpu_memory_utilization=0.54,
                  max_model_len=4096, enforce_eager=False)

        def sampling_for(n_chars):
            expected = max(300.0, n_chars / 18.4 * 84)
            start = 2.0 * expected

            def _boost(ids, logits):
                k = len(ids)
                if k > start:
                    logits[EOA] += 8.0 * min(4.0, 1.0 + (k - start) / expected)
                return logits

            return SamplingParams(temperature=0.6, top_p=0.8, seed=3407,
                                  repetition_penalty=1.10, max_tokens=MAX_TOKENS,
                                  stop_token_ids=[EOA], logits_processors=[_boost])

        prompts = [TokensPrompt(prompt_token_ids=[128259]
                                + list(tok('%s: %s' % (token, t)).input_ids)
                                + [128009, 128260, 128261, 128257]) for t in texts]
        outs = llm.generate(prompts, [sampling_for(len(t)) for t in texts], use_tqdm=False)

        man = []
        for i, (t, o) in enumerate(zip(texts, outs)):
            ids = list(o.outputs[0].token_ids)
            au = [v for v in ids if BASE <= v < BASE + 4096 * 7]
            au = au[:len(au) - len(au) % 7]
            wav = None
            if len(au) >= 7 * 12:
                c = [v - BASE for v in au]
                l1, l2, l3, ok = [], [], [], True
                for k in range(len(c) // 7):
                    if any(not (p * 4096 <= c[7 * k + p] < (p + 1) * 4096) for p in range(7)):
                        ok = False
                        break
                    l1.append(c[7 * k])
                    l2.append(c[7 * k + 1] - 4096)
                    l3.append(c[7 * k + 2] - 2 * 4096)
                    l3.append(c[7 * k + 3] - 3 * 4096)
                    l2.append(c[7 * k + 4] - 4 * 4096)
                    l3.append(c[7 * k + 5] - 5 * 4096)
                    l3.append(c[7 * k + 6] - 6 * 4096)
                if ok:
                    with torch.inference_mode():
                        wav = snac.decode([torch.tensor(v)[None]
                                           for v in (l1, l2, l3)])[0, 0].numpy()
            if wav is not None:
                sf.write(os.path.join(out, '%03d.wav' % i), wav, 24000)
            man.append(dict(wav=('%03d.wav' % i) if wav is not None else None, text=t,
                            tokens=len(ids), expected=max(300.0, len(t) / 18.4 * 84),
                            hit_cap=len(ids) >= MAX_TOKENS, stopped=EOA in ids))
        json.dump(man, open(os.path.join(out, 'manifest.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False)
        print('  %s: rendered %d/%d' % (name, sum(1 for m in man if m['wav']), len(man)))
        del llm
        torch.cuda.empty_cache()


def report(argv):
    out_root, corpus = argv[0], argv[1]
    ref = corpus_reference(corpus)
    print('CORPUS REFERENCE  (%s)' % corpus)
    print('  tail %.2fs | internal %.2fs | %.1f ch/s | tilt %.1f dB | speech %.1f%%\n'
          % (ref['tail'], ref['internal'], ref.get('chars_per_sec', float('nan')),
             ref['tilt'], ref['speech_frac']))

    eps = sorted(glob.glob(os.path.join(out_root, '*_ep*')),
                 key=lambda p: int(re.search(r'_ep(\d+)$', p).group(1)))
    results = []
    for ep in eps:
        mp = os.path.join(ep, 'manifest.json')
        if not os.path.isfile(mp):
            continue
        man = json.load(open(mp, encoding='utf-8'))
        # Other tools write a manifest.json here too (loop_gate uses a different shape).
        # Skip loudly rather than guessing at a foreign schema.
        if not (isinstance(man, list) and man and isinstance(man[0], dict)
                and 'expected' in man[0]):
            print('  skip %s: manifest is not an epoch_battery render'
                  % os.path.basename(ep))
            continue
        rows, texts = [], []
        for m in man:
            if not m['wav']:
                continue
            a = analyse(os.path.join(ep, m['wav']))
            if a:
                rows.append(a)
                texts.append(m['text'])
        s = summarise(rows, texts)
        if not s:
            continue
        ov = [m['tokens'] / m['expected'] for m in man if m['expected']]
        s['overrun'] = float(np.median(ov)) if ov else float('nan')
        s['overrun_p90'] = float(np.percentile(ov, 90)) if ov else float('nan')
        s['no_stop'] = sum(1 for m in man if not m['stopped'])
        s['cap_hits'] = sum(1 for m in man if m['hit_cap'])
        s['decode_fail'] = sum(1 for m in man if not m['wav'])
        s['name'] = os.path.basename(ep)
        s['distance'] = (abs(s['tilt'] - ref['tilt']) / W_TILT
                         + abs(s.get('chars_per_sec', 0) - ref.get('chars_per_sec', 0)) / W_RATE
                         + abs(s['internal'] - ref['internal']) / W_GAP
                         + abs(s['tail'] - ref['tail']) / W_GAP)
        results.append(s)

    hdr = ('%-16s%7s%7s%8s%7s%7s%8s%8s%9s%7s%7s%8s'
           % ('epoch', 'tail', 'p90', 'intern', 'ch/s', 'tilt', 'spread', 'maxSil',
              'overrun', 'p90', 'noStop', 'dist'))
    print(hdr)
    print('-' * len(hdr))
    for s in results:
        flag = ''
        if s['max_sil'] >= 3.0 or s['overrun_p90'] >= 1.6:
            flag = '  <<< SILENCE RISK'
        if s['no_stop'] or s['cap_hits']:
            flag = '  <<< RUNAWAY'
        print('%-16s%7.2f%7.2f%8.2f%7.1f%7.1f%8.2f%8.2f%9.2f%7.2f%7d%8.1f%s'
              % (s['name'], s['tail'], s['tail_p90'], s['internal'],
                 s.get('chars_per_sec', float('nan')), s['tilt'], s['tilt_spread'],
                 s['max_sil'], s['overrun'], s['overrun_p90'], s['no_stop'],
                 s['distance'], flag))
    print('\n%-16s%7.2f%7s%8.2f%7.1f%7.1f   <- CORPUS (the target)'
          % ('', ref['tail'], '', ref['internal'],
             ref.get('chars_per_sec', float('nan')), ref['tilt']))

    safe = [s for s in results if not s['no_stop'] and not s['cap_hits']
            and s['max_sil'] < 3.0 and s['overrun_p90'] < 1.6]
    print('\nsafe epochs (no runaway, no silence risk): %s'
          % (', '.join(s['name'] for s in safe) or 'NONE'))
    if safe:
        best = min(safe, key=lambda s: s['distance'])
        print('CLOSEST TO THE NARRATOR: %s  (distance %.1f)' % (best['name'], best['distance']))
        print('  tilt %+.1f vs %+.1f | %.1f vs %.1f ch/s | internal %.2f vs %.2fs | tail %.2f vs %.2fs'
              % (best['tilt'], ref['tilt'], best.get('chars_per_sec', 0),
                 ref.get('chars_per_sec', 0), best['internal'], ref['internal'],
                 best['tail'], ref['tail']))
    print('\nNOTE: distance is MEASURABLE fidelity only. It cannot hear character voices,')
    print('      breathiness or the SNAC ring — gate those separately, and the ear ranks last.')
    json.dump(dict(reference=ref, epochs=results),
              open(os.path.join(out_root, 'battery.json'), 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'render':
        render(sys.argv[2:])
    elif mode == 'report':
        report(sys.argv[2:])
    else:
        raise SystemExit(__doc__)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ASR-vs-text QC — find rendered chunks whose AUDIO doesn't say what the TEXT says.

Because the expected text is KNOWN, this is a diff, not a transcription job. That's
what makes it strong: it returns exact chunk indices and a score, instead of an hour
of transcript to read.

WHISPER SETTINGS — these matter more than the model size
--------------------------------------------------------
Whisper is a language model, so it has no raw-phonetic mode; it always smooths toward
fluent English. Every default below exists to REPAIR bad audio, which is exactly the
thing we are trying to detect, so all of them are turned off:

  condition_on_previous_text=False  The big one. On (the default) each window is
                                    conditioned on prior output, so Whisper repairs
                                    garbled audio from context and gibberish comes
                                    back as clean prose. Off, garbage stays garbage.
  temperature=[0.0]                 No fallback ladder. The default re-rolls segments
                                    it judges low-quality at rising temperature, which
                                    invents fluent text over exactly the bad audio.
  beam_size=1                       Greedy. Beam search optimises for the most probable
                                    SENTENCE — it prefers plausible over faithful.
  initial_prompt=None               Any prompt biases the output toward its own wording.

THE THREE SIGNALS
-----------------
  similarity        difflib ratio, ASR vs expected (letters+digits, lowercased).
                    LOW  = the audio does not match the script.
                    *** autojunk=False is REQUIRED. *** With the default, difflib treats
                    characters appearing in >1% of a sequence as junk once the string
                    passes 200 chars, and the ratio collapses to ~0 for perfectly good
                    matches. This cost a wrong diagnosis on 2026-07-25.
  compression_ratio Whisper's own per-segment metric. HIGH = the transcript repeats
                    itself, i.e. the model looped. A direct repetition detector:
                    a looping chunk scored 3.56 against a ~1.5 baseline.
  avg_logprob       Acoustic confidence. LOW = it heard something unlike speech.

Usage
-----
  asr_text_qc.py <sentences_dir> <session-state.json> --range 0-240
  asr_text_qc.py <sentences_dir> <session-state.json> --indices 121,229,95
  ... [--model small.en] [--device cpu] [--json out.json] [--worst N]

session-state.json is e2a's, whose `chapter_sentences` (list of per-chapter lists)
flattens to exactly the sentence-file indices.
"""
import argparse, difflib, json, os, re, sys, io, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

SML = re.compile(r'\[(?:break|pause|music|sfx|silence)(?::[^\]]+)?\]', re.I)


def norm(s: str) -> str:
    """Lowercase, drop e2a SML tags and punctuation — compare words, not typography."""
    return ' '.join(re.sub(r'[^a-z0-9 ]+', ' ', SML.sub(' ', s or '').lower()).split())


def load_expected(state_path: str):
    st = json.load(open(state_path, encoding='utf-8'))
    return [s for ch in st['chapter_sentences'] for s in ch]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('sentences_dir')
    ap.add_argument('state_json')
    ap.add_argument('--range', dest='rng')
    ap.add_argument('--indices')
    ap.add_argument('--model', default='small.en')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--compute-type', default='int8')
    ap.add_argument('--json', dest='out_json')
    ap.add_argument('--worst', type=int, default=15)
    ap.add_argument('--min-chars', type=int, default=60,
                    help='skip very short chunks; their scores are noise')
    a = ap.parse_args()

    expected = load_expected(a.state_json)
    if a.indices:
        idx = [int(x) for x in a.indices.split(',')]
    elif a.rng:
        lo, hi = a.rng.split('-')
        idx = list(range(int(lo), int(hi) + 1))
    else:
        idx = list(range(len(expected)))

    from faster_whisper import WhisperModel
    print(f'loading {a.model} ({a.device}/{a.compute_type}) ...')
    m = WhisperModel(a.model, device=a.device, compute_type=a.compute_type, cpu_threads=8)

    rows, t0 = [], time.time()
    for n, i in enumerate(idx):
        p = os.path.join(a.sentences_dir, f'{i}.flac')
        if not os.path.isfile(p):
            p = os.path.join(a.sentences_dir, f'{i}.wav')
        if not os.path.isfile(p) or i >= len(expected):
            continue
        if len(expected[i]) < a.min_chars:
            continue
        segs, _ = m.transcribe(p, language='en', beam_size=1, temperature=[0.0],
                               condition_on_previous_text=False, without_timestamps=True)
        segs = list(segs)
        asr = ' '.join(s.text for s in segs).strip()
        rows.append(dict(
            i=i, asr=asr, exp=expected[i],
            sim=difflib.SequenceMatcher(None, norm(expected[i]), norm(asr), autojunk=False).ratio(),
            cr=max([s.compression_ratio for s in segs], default=0.0),
            lp=min([s.avg_logprob for s in segs], default=0.0),
        ))
        if n and n % 20 == 0:
            print(f'  {n}/{len(idx)}  ({time.time()-t0:.0f}s)')

    if not rows:
        print('no chunks matched')
        return 1
    if a.out_json:
        json.dump(rows, open(a.out_json, 'w', encoding='utf-8'), ensure_ascii=False)

    med = sorted(r['sim'] for r in rows)[len(rows) // 2]
    print(f'\n{len(rows)} chunks in {time.time()-t0:.0f}s   median similarity {med:.3f}')

    print(f'\n{"="*78}\nWORST {a.worst} BY TEXT MATCH\n{"="*78}')
    for r in sorted(rows, key=lambda r: r['sim'])[:a.worst]:
        print(f"\n[{r['i']}]  similarity {r['sim']:.2f}   compression {r['cr']:.2f}   logprob {r['lp']:.2f}")
        print(f"   TEXT : {r['exp']}")
        print(f"   HEARD: {r['asr']}")

    loops = [r for r in rows if r['cr'] > 2.0]
    if loops:
        print(f'\n{"="*78}\nREPETITION (compression_ratio > 2.0)\n{"="*78}')
        for r in sorted(loops, key=lambda r: -r['cr']):
            print(f"\n[{r['i']}]  compression {r['cr']:.2f}   similarity {r['sim']:.2f}")
            print(f"   TEXT : {r['exp']}")
            print(f"   HEARD: {r['asr']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

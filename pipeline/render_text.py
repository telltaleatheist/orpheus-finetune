#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render arbitrary text with a voice, chunked and joined the way ASSEMBLY would.

Same prompt construction, sampling params and EOS boost as loop_gate.py's render, so
what you hear is the production path rather than a bespoke demo path. Text is packed
greedily to <=max-chars at sentence boundaries (e2a's rule), and chunks are joined
with the voice's tuned sentenceGap so chunk seams land exactly where a real book
would put them.

Usage: render_text.py <model_dir> <token> <text_file> <out.wav> [--gap 0.0] [--max-chars 350]
"""
import argparse, os, re, sys
import numpy as np, soundfile as sf, torch
from snac import SNAC
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt

EOA, BASE, MAX_TOKENS = 128258, 128266, 3700
ap = argparse.ArgumentParser()
ap.add_argument('model'); ap.add_argument('token'); ap.add_argument('text'); ap.add_argument('out')
ap.add_argument('--gap', type=float, default=0.0)
ap.add_argument('--max-chars', type=int, default=350)
ap.add_argument('--rep', type=float, default=1.10)
ap.add_argument('--boost', type=float, default=8.0)
ap.add_argument('--boost-start', type=float, default=2.0)
a = ap.parse_args()

raw = open(a.text, encoding='utf-8').read().strip()
sents = re.findall(r'[^.!?]*[.!?]["”’\']*\s*', raw) or [raw]
chunks, cur = [], ''
for s in sents:
    if cur and len(cur) + len(s) > a.max_chars:
        chunks.append(cur.strip()); cur = s
    else:
        cur += s
if cur.strip(): chunks.append(cur.strip())
print(f'{len(chunks)} chunks: ' + ', '.join(str(len(c)) for c in chunks))

tok = AutoTokenizer.from_pretrained(a.model)
llm = LLM(model=a.model, dtype='bfloat16', gpu_memory_utilization=0.54,
          max_model_len=4096, enforce_eager=False)
snac = SNAC.from_pretrained('hubertsiuzdak/snac_24khz').eval()

def prompt_ids(t):
    return [128259] + list(tok(f'{a.token}: {t}').input_ids) + [128009, 128260, 128261, 128257]

def sampling_for(n_chars):
    expected = max(300.0, n_chars / 18.4 * 84); start = a.boost_start * expected
    def _boost(ids, logits):
        n = len(ids)
        if n > start: logits[EOA] += a.boost * min(4.0, 1.0 + (n - start) / expected)
        return logits
    return SamplingParams(temperature=0.6, top_p=0.8, seed=3407, repetition_penalty=a.rep,
                          max_tokens=MAX_TOKENS, stop_token_ids=[EOA],
                          logits_processors=[_boost] if a.boost > 0 else None)

outs = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids(c)) for c in chunks],
                    [sampling_for(len(c)) for c in chunks], use_tqdm=False)

def decode(ids):
    au = [t for t in ids if BASE <= t < BASE + 4096 * 7]
    au = au[:len(au) - len(au) % 7]
    if len(au) < 7 * 12: return None
    c = [t - BASE for t in au]; l1, l2, l3 = [], [], []
    for i in range(len(c) // 7):
        if any(not (p * 4096 <= c[7*i+p] < (p+1)*4096) for p in range(7)): return None
        l1.append(c[7*i]); l2.append(c[7*i+1]-4096); l3.append(c[7*i+2]-2*4096)
        l3.append(c[7*i+3]-3*4096); l2.append(c[7*i+4]-4*4096)
        l3.append(c[7*i+5]-5*4096); l3.append(c[7*i+6]-6*4096)
    with torch.inference_mode():
        return snac.decode([torch.tensor(x)[None] for x in (l1, l2, l3)])[0, 0].numpy()

parts = []
for i, o in enumerate(outs):
    w = decode(list(o.outputs[0].token_ids))
    if w is None: print(f'  chunk {i}: DECODE FAILED'); continue
    parts.append(w)
    if a.gap > 0: parts.append(np.zeros(int(24000 * a.gap), dtype='float32'))
    print(f'  chunk {i}: {len(w)/24000:.2f}s  ({len(chunks[i])} chars)')
audio = np.concatenate(parts)
sf.write(a.out, audio, 24000)
print(f'wrote {a.out}  {len(audio)/24000:.1f}s')

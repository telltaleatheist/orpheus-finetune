#!/usr/bin/env python
"""RING GATE: greedy-decode a few real chunks to AUDIO and scan 5-12 kHz for
the SNAC tonal comb (the KA/Ghostworld '8.4 kHz quiet ringing', 2026-07-19).

voice_diff.py gates EOS margin on token COUNTS; this gate listens to the
actual waveform. Greedy decode = deterministic, so a clean pass is a property
of the checkpoint, not luck.

PASS: no frequency bin 5-12 kHz is tonal (>=8 dB over local median) in more
than 1/3 of chunks. FAIL prints the offending comb.

Usage: ring_probe.py <model_dir> <voice_token> <session-state.json>
"""
import sys, json, itertools
import numpy as np

MODEL, TOKEN, STATE = sys.argv[1], sys.argv[2], sys.argv[3]
MAX_TOKENS, EOA, BASE = 3700, 128258, 128266
N_CHUNKS = 6

d = json.load(open(STATE))
cs = d["chapter_sentences"]
flat = list(itertools.chain.from_iterable(cs)) if cs and isinstance(cs[0], list) else cs
idxs = [i for i, s in enumerate(flat) if 280 <= len(s) < 352][:N_CHUNKS]

from vllm import LLM, SamplingParams, TokensPrompt
from transformers import AutoTokenizer
import torch
from snac import SNAC

tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.54,
          max_model_len=4096, enforce_eager=False)
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()


def pids(text):
    body = tok(f"{TOKEN}: {text}").input_ids
    return [128259] + list(body) + [128009, 128260, 128261, 128257]


prompts = [TokensPrompt(prompt_token_ids=pids(flat[i])) for i in idxs]
sp = SamplingParams(max_tokens=MAX_TOKENS, stop_token_ids=[EOA],
                    temperature=0.0, repetition_penalty=1.15)
outs = llm.generate(prompts, sp, use_tqdm=False)


def decode_audio(token_ids):
    a = [t for t in token_ids if BASE <= t < BASE + 4096 * 7]
    a = a[:len(a) - len(a) % 7]
    if len(a) < 7 * 12:
        return None
    cl = [t - BASE for t in a]
    l1, l2, l3 = [], [], []
    for i in range(len(cl) // 7):
        if any(not (p * 4096 <= cl[7 * i + p] < (p + 1) * 4096) for p in range(7)):
            return None  # slot corruption — counts as fail elsewhere, skip here
        l1.append(cl[7 * i])
        l2.append(cl[7 * i + 1] - 4096)
        l3.append(cl[7 * i + 2] - 2 * 4096)
        l3.append(cl[7 * i + 3] - 3 * 4096)
        l2.append(cl[7 * i + 4] - 4 * 4096)
        l3.append(cl[7 * i + 5] - 5 * 4096)
        l3.append(cl[7 * i + 6] - 6 * 4096)
    with torch.inference_mode():
        codes = [torch.tensor(x)[None] for x in (l1, l2, l3)]
        return snac.decode(codes)[0, 0].numpy()


def comb(y, sr=24000, prom=8):
    NF = 8192
    w = np.hanning(NF)
    m = (len(y) - NF) // (NF // 2) + 1
    if m < 4:
        return []
    ps = np.zeros(NF // 2 + 1)
    for j in range(m):
        s = y[j * NF // 2: j * NF // 2 + NF] * w
        ps += np.abs(np.fft.rfft(s)) ** 2
    db = 10 * np.log10(ps / m + 1e-20)
    f = np.fft.rfftfreq(NF, 1 / sr)
    lo, hi = np.searchsorted(f, 5000), np.searchsorted(f, 11800)
    band = db[lo:hi]
    bw = int(400 / (sr / NF)); ex = int(60 / (sr / NF))
    out = []
    for i in range(bw, len(band) - bw):
        v = band[i]
        neigh = np.concatenate([band[i - bw:i - ex], band[i + ex:i + bw]])
        if v - np.median(neigh) >= prom and v == band[max(0, i - ex):i + ex + 1].max():
            out.append((round(f[lo + i]), round(v - np.median(neigh), 1)))
    return out


per_chunk = []
for o in outs:
    y = decode_audio(list(o.outputs[0].token_ids))
    per_chunk.append(comb(y) if y is not None else None)

valid = [c for c in per_chunk if c is not None]
freqs = [t[0] for c in valid for t in c]
# a comb line = same 250Hz bin tonal in > 1/3 of decodable chunks
bad = []
if freqs:
    fb = (np.array(freqs) // 250 * 250).astype(int)
    u, cnt = np.unique(fb, return_counts=True)
    bad = [(int(a), int(b)) for a, b in zip(u, cnt) if b > len(valid) / 3]
verdict = "FAIL" if bad or not valid else "PASS"
print(f"RING RESULT voice={TOKEN} model={MODEL.rstrip('/').split('/')[-1]}: {verdict} | "
      f"chunks decoded {len(valid)}/{len(per_chunk)} | per-chunk tones: {per_chunk} | "
      f"comb lines: {bad or 'none'}", flush=True)

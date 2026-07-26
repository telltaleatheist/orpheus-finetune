"""Runaway probe: render known-long chunks on vLLM with a RAISED token ceiling and a
rep-penalty ladder; for each render dump token stats, tail-frame loop analysis, and the
FULL decoded wav (never truncated) so a human can hear what the model emits instead of EOS.

Run INSIDE WSL:  conda run -n orpheus_tts python /mnt/c/.../probe_runaway.py
Prompt framing / token layout copied verbatim from e2a orpheus.py (_format_prompt_ids,
_redistribute_codes): prompt = [128259] + tok("{voice}: {text}").input_ids + [128009,
128260, 128261, 128257]; EOS=128258; audio tokens 128266+pos*4096, 7/frame; SNAC 24k.
Sampling matches production: temp 0.6, top_p 0.8, min_p 0, stop=[128258].
"""
import json
import os
import sys
import time

MODEL = "/home/telltale/xtts_ft/orpheus_deathstalker_merged_ep5"
VOICE = "deathstalker"
OUTDIR = "/home/telltale/probe_runaway"
# Durable copy of the 12 longest chunks from the 2026-07-13 test session (the live
# session dir gets deleted by the CLI's scratch cleanup).
CHUNKS_JSON = "/mnt/c/Users/tellt/Projects/orpheus-finetune/probe_chunks.json"
MAX_TOKENS = 6000          # ABOVE the 3700 production cap: does EOS ever arrive late?
# Round 2 (2026-07-14): bracket the sweet spot. Round 1: 1.0→6/6, 1.05→3/6, 1.1→2/6,
# 1.3→0/6 runaway — but 1.3 produced one early-EOS truncation (36 ch/s).
REP_LADDER = [1.15, 1.2, 1.25]
N_CHUNKS = 12              # all saved chunks for tighter stats
EOS = 128258
AUDIO_BASE = 128266
SR = 24000

os.makedirs(OUTDIR, exist_ok=True)

chunks = json.load(open(CHUNKS_JSON))   # {"idx": "already-cleaned chunk text"}
picks = sorted(chunks, key=lambda k: -len(chunks[k]))[:N_CHUNKS]
sents = chunks
def clean(s):
    return s  # texts in probe_chunks.json are pre-cleaned
print(f"probing chunks {picks} (lens {[len(chunks[k]) for k in picks]})", flush=True)

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt
import numpy as np
import torch
from snac import SNAC

tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(model=MODEL, max_model_len=8192, gpu_memory_utilization=0.55, dtype="bfloat16")
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to("cuda")

def prompt_ids(text):
    body = tok(f"{VOICE}: {text}").input_ids
    return [128259] + list(body) + [128009, 128260, 128261, 128257]

def frames_of(tokens):
    at = [t for t in tokens if AUDIO_BASE <= t < AUDIO_BASE + 4096 * 7]
    at = at[: (len(at) // 7) * 7]
    return [tuple(at[i:i+7]) for i in range(0, len(at), 7)]

def decode(tokens):
    fr = frames_of(tokens)
    if not fr:
        return np.zeros(1, dtype=np.float32)
    l1, l2, l3 = [], [], []
    for f in fr:
        c = [f[p] - AUDIO_BASE - p * 4096 for p in range(7)]
        if any(x < 0 or x >= 4096 for x in c):
            raise SystemExit(f"out-of-slot code in frame {f} — misaligned stream, aborting probe")
        l1.append(c[0]); l2.extend([c[1], c[4]]); l3.extend([c[2], c[3], c[5], c[6]])
    codes = [torch.tensor(l, dtype=torch.long, device="cuda").unsqueeze(0) for l in (l1, l2, l3)]
    with torch.no_grad():
        return snac.decode(codes).squeeze().cpu().numpy()

def tail_report(tokens):
    fr = frames_of(tokens)
    tail = fr[-70:]                     # last ~0.8s worth of frames... (70 frames ~ 6s @ ~12 fps)
    uniq = len(set(tail))
    # longest immediate-repeat run in the tail
    run, best = 1, 1
    for a, b in zip(tail, tail[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return {"frames_total": len(fr), "tail70_unique": uniq, "tail70_longest_repeat_run": best}

def trailing_silence_sec(audio, thresh_ratio=0.01):
    peak = float(np.abs(audio).max()) or 1.0
    idx = np.where(np.abs(audio) >= peak * thresh_ratio)[0]
    if len(idx) == 0:
        return len(audio) / SR
    return (len(audio) - 1 - int(idx[-1])) / SR

results = []
for rep in REP_LADDER:
    sp = SamplingParams(temperature=0.6, top_p=0.8, min_p=0.0, repetition_penalty=rep,
                        max_tokens=MAX_TOKENS, stop_token_ids=[EOS], seed=0)
    prompts = [TokensPrompt(prompt_token_ids=prompt_ids(clean(sents[i]))) for i in picks]
    t0 = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.time() - t0
    for i, out in zip(picks, outs):
        toks = list(out.outputs[0].token_ids)
        eos_at = toks.index(EOS) if EOS in toks else None
        body = toks if eos_at is None else toks[:eos_at]
        audio = decode(body)
        tail_sil = trailing_silence_sec(audio)
        rec = {"chunk": i, "rep": rep, "chars": len(clean(sents[i])),
               "n_tokens": len(toks), "eos_at": eos_at,
               "hit_3700_cap": eos_at is None or eos_at >= 3700,
               "audio_sec": round(len(audio) / SR, 2),
               "trailing_silence_sec": round(tail_sil, 2),
               **tail_report(body)}
        results.append(rec)
        tag = "RUNAWAY" if eos_at is None else ("LATE-EOS" if eos_at >= 3700 else "ok")
        print(f"[rep={rep}] chunk {i} ({rec['chars']}ch): {tag} tokens={len(toks)} eos_at={eos_at} "
              f"audio={rec['audio_sec']}s tail_silence={rec['trailing_silence_sec']}s "
              f"tail70_uniq={rec['tail70_unique']} rep_run={rec['tail70_longest_repeat_run']}", flush=True)
        if tag != "ok":
            import soundfile as sf
            wav = os.path.join(OUTDIR, f"chunk{i}_rep{rep}_{tag}.wav")
            sf.write(wav, audio, SR)
            print(f"    saved {wav}", flush=True)

with open(os.path.join(OUTDIR, "probe_results.json"), "w") as f:
    json.dump(results, f, indent=1)
print("DONE — results in", os.path.join(OUTDIR, "probe_results.json"), flush=True)

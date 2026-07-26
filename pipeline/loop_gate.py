#!/usr/bin/env python3
"""loop_gate.py — does the voice REPEAT itself, or silently drop text?

WHY THIS EXISTS (2026-07-24). Every guard in the stack watches for audio that is
too SHORT for its text: e2a's chars-per-second truncation guard, and eos_gate.py
(which only asks "did it stop before the token cap?"). A model that loops back
and re-speaks a phrase produces audio that is too LONG — so it passes all of
them and ships duplicated sentences into the audiobook. That is exactly how a
broken thirdreich shipped in July and survived an audit on 2026-07-24 that
passed it on termination alone.

Three failures, one pass:
  LOOP       a word n-gram repeats in the transcript (the model re-spoke text)
  DROPPED    the transcript covers too little of the source text (early EOS)
  BLOATED    the clip is far longer than its text warrants (added 07-25) — dead air
             or non-repeating filler, which is invisible to BOTH checks above

Runs SAMPLED at production settings by default, not greedy: looping is a
sampled-path failure and a greedy probe will not reproduce it. A fixed seed
keeps it reproducible.

Two stages, because vLLM and faster-whisper live in different environments:

  # stage 1 (WSL, env orpheus_tts — holds the GPU)
  loop_gate.py render <model_dir> <token> <corpus_dir> <out_dir> [n] [rep] [boost] [start]

  # stage 2 (any env with faster_whisper; BookForge's e2a-env has it)
  loop_gate.py check <out_dir> [whisper_model]

PASS = zero LOOP and zero DROPPED.
"""
import csv
import glob
import json
import os
import re
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else ""
EOA = 128258
BASE = 128266
MAX_TOKENS = 3700
WS = re.compile(r"[^a-z0-9 ]")
NGRAM = 6
COVERAGE_FLOOR = 0.85       # transcript must cover >=85% of the source words
# Seconds-of-audio per source character, relative to this probe set's own median.
# Measured over 80 clips (tr_tt1 ep140/280/420/560): every clean clip came in under
# 1.3x, and the two real defects were 1.94x and 1.80x. 1.5x sits in the empty gap
# between those populations. Relative to the probe set's OWN median, so it travels
# across voices with different speaking rates without retuning.
BLOAT_CEILING = 1.5


# e2a's TTS-prep EXPANDS numbers to words ("one thousand nine hundred twenty
# eight") and Whisper collapses them back to digits ("1928"). A word-level
# comparison therefore scores a full, correct reading as ~15% missing on any
# chunk containing a number — measured 2026-07-24, it false-flagged every such
# clip in all three tr_rv2 epochs identically. Numbers are not where dropped
# text hides, so drop numeric tokens from BOTH sides (symmetric, so it cannot
# hide a real drop) and measure coverage on the words that matter.
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion", "trillion",
}


def normalize(text):
    return [w for w in WS.sub("", text.lower()).split()
            if not w.isdigit() and w not in NUMBER_WORDS]


# --------------------------------------------------------------------------
def render(argv):
    import numpy as np
    import soundfile as sf
    import torch
    from snac import SNAC
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    model_dir, token, corpus, out_dir = argv[0], argv[1], argv[2], argv[3]
    n_chunks = int(argv[4]) if len(argv) > 4 else 20
    rep_pen = float(argv[5]) if len(argv) > 5 else 1.10
    boost = float(argv[6]) if len(argv) > 6 else 8.0
    boost_start = float(argv[7]) if len(argv) > 7 else 2.0
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for split in ("eval", "train"):
        path = os.path.join(corpus, f"metadata_{split}.csv")
        if os.path.exists(path):
            rows += [r[1] for r in list(csv.reader(open(path, encoding="utf-8"),
                                                   delimiter="|"))[1:]]
    texts = [t for t in rows if 200 <= len(t) <= 360][:n_chunks]
    if len(texts) < n_chunks:
        texts += [t for t in rows if 120 <= len(t) < 200][:n_chunks - len(texts)]
    if not texts:
        raise SystemExit(f"no chunk-sized texts in {corpus}")

    tok = AutoTokenizer.from_pretrained(model_dir)
    llm = LLM(model=model_dir, dtype="bfloat16", gpu_memory_utilization=0.54,
              max_model_len=4096, enforce_eager=False)
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

    def prompt_ids(text):
        body = tok(f"{token}: {text}").input_ids
        return [128259] + list(body) + [128009, 128260, 128261, 128257]

    def eos_processor(n_chars):
        """Mirror of e2a orpheus.py _eos_boost_processor."""
        if boost <= 0:
            return None
        expected = max(300.0, n_chars / 18.4 * 84)
        start = boost_start * expected

        def _boost(token_ids, logits):
            n = len(token_ids)
            if n > start:
                logits[EOA] += boost * min(4.0, 1.0 + (n - start) / expected)
            return logits
        return _boost

    def sampling_for(n_chars):
        proc = eos_processor(n_chars)
        return SamplingParams(temperature=0.6, top_p=0.8, seed=3407,
                              repetition_penalty=rep_pen, max_tokens=MAX_TOKENS,
                              stop_token_ids=[EOA],
                              logits_processors=[proc] if proc else None)

    outputs = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids(t)) for t in texts],
                           [sampling_for(len(t)) for t in texts], use_tqdm=False)

    def decode_audio(token_ids):
        audio = [t for t in token_ids if BASE <= t < BASE + 4096 * 7]
        audio = audio[:len(audio) - len(audio) % 7]
        if len(audio) < 7 * 12:
            return None
        codes = [t - BASE for t in audio]
        l1, l2, l3 = [], [], []
        for i in range(len(codes) // 7):
            if any(not (p * 4096 <= codes[7 * i + p] < (p + 1) * 4096) for p in range(7)):
                return None                      # slot corruption
            l1.append(codes[7 * i])
            l2.append(codes[7 * i + 1] - 4096)
            l3.append(codes[7 * i + 2] - 2 * 4096)
            l3.append(codes[7 * i + 3] - 3 * 4096)
            l2.append(codes[7 * i + 4] - 4 * 4096)
            l3.append(codes[7 * i + 5] - 5 * 4096)
            l3.append(codes[7 * i + 6] - 6 * 4096)
        with torch.inference_mode():
            tensors = [torch.tensor(x)[None] for x in (l1, l2, l3)]
            return snac.decode(tensors)[0, 0].numpy()

    manifest = []
    for i, (text, out) in enumerate(zip(texts, outputs)):
        generated = out.outputs[0]
        wav = decode_audio(list(generated.token_ids))
        name = f"{i:03d}.wav"
        if wav is None:
            manifest.append({"wav": None, "text": text, "tokens": len(generated.token_ids),
                             "note": "decode failed (slot corruption or too short)"})
            continue
        sf.write(os.path.join(out_dir, name), wav, 24000)
        manifest.append({"wav": name, "text": text, "tokens": len(generated.token_ids),
                         "seconds": round(len(wav) / 24000, 2)})
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"model": model_dir, "token": token, "rep_penalty": rep_pen,
                   "eos_boost": boost, "eos_boost_start": boost_start,
                   "sampling": "temperature 0.6 / top_p 0.8 / seed 3407",
                   "clips": manifest}, handle, indent=2)
    print(f"[loop_gate] rendered {sum(1 for c in manifest if c['wav'])}/{len(manifest)} "
          f"clips -> {out_dir}")


# --------------------------------------------------------------------------
def check(argv):
    import difflib
    from faster_whisper import WhisperModel

    out_dir = argv[0]
    whisper_model = argv[1] if len(argv) > 1 else "medium.en"
    manifest = json.load(open(os.path.join(out_dir, "manifest.json"), encoding="utf-8"))
    model = WhisperModel(whisper_model, device="cuda", compute_type="float16")

    # BLOAT reference: seconds of audio per source character, median over the probe
    # set. A clip far above it is emitting audio the text does not account for —
    # dead air, or filler that never repeats as an n-gram.
    rendered = [c for c in manifest["clips"] if c["wav"] and len(c["text"]) > 0]
    sec_per_char = sorted(c["seconds"] / len(c["text"]) for c in rendered)
    bloat_ref = sec_per_char[len(sec_per_char) // 2] if sec_per_char else None

    loops, dropped, failed, bloated, ok = [], [], [], [], 0
    print(f"\n=== LOOP / COMPLETENESS GATE ({manifest['sampling']}) ===")
    for i, clip in enumerate(manifest["clips"]):
        if not clip["wav"]:
            failed.append(i)
            print(f"[FAIL] #{i:03d} {clip.get('note')}")
            continue
        segments, _ = model.transcribe(os.path.join(out_dir, clip["wav"]),
                                       language="en", beam_size=5)
        spoken = " ".join(s.text.strip() for s in segments).strip()
        said = normalize(spoken)
        want = normalize(clip["text"])

        grams = [" ".join(said[j:j + NGRAM]) for j in range(len(said) - NGRAM + 1)]
        repeated = {g for g in grams if grams.count(g) > 1}
        coverage = difflib.SequenceMatcher(None, want, said).ratio()

        # Duration bloat, the third failure. Added 2026-07-25 after tr_tt1_ep140 #013
        # rendered 38.9 s / 3193 tokens for a text its own ep420 sibling read in
        # 17.4 s / 1429 — 2.2x too long, with 15.6 s of trailing dead air. It passed
        # BOTH existing checks: coverage was 93.1% (above the floor, so not DROPPED)
        # and the dead air repeats no n-gram (so not LOOP). A clip can therefore
        # balloon arbitrarily and be called clean, which is precisely the class of
        # defect this gate exists to catch.
        bloat = (clip["seconds"] / len(clip["text"]) / bloat_ref
                 if bloat_ref and len(clip["text"]) else 1.0)

        verdict = "ok  "
        if repeated:
            verdict = "LOOP"
            loops.append(i)
        elif coverage < COVERAGE_FLOOR:
            verdict = "DROP"
            dropped.append(i)
        elif bloat > BLOAT_CEILING:
            verdict = "BLOAT"
            bloated.append(i)
        else:
            ok += 1
        print(f"[{verdict}] #{i:03d} {clip['seconds']:5.1f}s tokens={clip['tokens']:4d} "
              f"coverage={coverage*100:5.1f}% bloat={bloat:4.2f}x"
              + (f'  repeated: "{sorted(repeated, key=len)[-1][:60]}"' if repeated else ""))

    total = len(manifest["clips"])
    print(f"\n  clean {ok}/{total} | LOOP {len(loops)} {loops} | "
          f"DROPPED {len(dropped)} {dropped} | BLOATED {len(bloated)} {bloated} | "
          f"render-fail {len(failed)} {failed}")
    passed = not loops and not dropped and not bloated and not failed
    print(f"LOOP_GATE_{'PASS' if passed else 'FAIL'}_OF_{total}")


if MODE == "render":
    render(sys.argv[2:])
elif MODE == "check":
    check(sys.argv[2:])
else:
    raise SystemExit(__doc__)

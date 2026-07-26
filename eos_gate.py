#!/usr/bin/env python
"""EOS gate for a merged Orpheus voice (2026-07-21, no-bed validation).

Greedy-decodes N real chunk-sized texts (deterministic — a pass is a property
of the checkpoint, not sampling luck) and reports, per chunk:
  - stopped: did generation emit EOS before the token ceiling?
  - tokens emitted vs a chars-based expectation (runaway = far over)
PASS = every chunk stops naturally and no chunk exceeds 2x expected tokens.

Prompt framing copied from e2a orpheus.py / probe_runaway.py verbatim.
Usage (inside WSL):
  eos_gate.py <merged_model_dir> <voice_token> <corpus_dir> [n_chunks] [rep_pen] [eos_boost] [boost_start]

This is the MANDATORY pre-deploy gate — ship only a checkpoint that returns
EOS_GATE_PASS. It lived in C:\\tmp until 2026-07-24 and is tracked here now
because tr_rv1's gate-clean epochs were lost to a temp-dir cleanup.
"""
import csv, os, sys
import numpy as np

MODEL, TOKEN, CORPUS = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 20
MAX_TOKENS = 3700
EOS = 128258

rows = []
for split in ("eval", "train"):
    p = os.path.join(CORPUS, f"metadata_{split}.csv")
    if os.path.exists(p):
        rows += [r[1] for r in list(csv.reader(open(p), delimiter="|"))[1:]]
texts = [t for t in rows if 200 <= len(t) <= 360][:N]
if len(texts) < N:
    texts += [t for t in rows if 120 <= len(t) < 200][:N - len(texts)]
assert texts, "no chunk-sized texts found"

from vllm import LLM, SamplingParams, TokensPrompt
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.54,
          max_model_len=4096, enforce_eager=False)


def pids(text):
    body = tok(f"{TOKEN}: {text}").input_ids
    return [128259] + list(body) + [128009, 128260, 128261, 128257]


REP_PEN = float(sys.argv[5]) if len(sys.argv) > 5 else 1.1
EOS_BOOST = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
BOOST_START = float(sys.argv[7]) if len(sys.argv) > 7 else 1.2


def make_boost(n_chars):
    """Mirror of e2a orpheus.py _eos_boost_processor."""
    if EOS_BOOST <= 0:
        return None
    expected = max(300.0, n_chars / 18.4 * 84)
    start = BOOST_START * expected

    def _boost(token_ids, logits):
        n = len(token_ids)
        if n > start:
            logits[EOS] += EOS_BOOST * min(4.0, 1.0 + (n - start) / expected)
        return logits
    return _boost


def sp_for(n_chars):
    proc = make_boost(n_chars)
    return SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, stop_token_ids=[EOS],
                          repetition_penalty=REP_PEN,
                          logits_processors=[proc] if proc else None)
prompts = [TokensPrompt(prompt_token_ids=pids(t)) for t in texts]
outs = llm.generate(prompts, [sp_for(len(t)) for t in texts])
if EOS_BOOST > 0:
    print(f"(EOS boost active: +{EOS_BOOST} from {BOOST_START}x expected)", flush=True)

fails = 0
print("\n=== EOS GATE (greedy, deterministic) ===", flush=True)
for t, o in zip(texts, outs):
    gen = o.outputs[0]
    n_tok = len(gen.token_ids)
    stopped = (gen.finish_reason == "stop") or (n_tok and gen.token_ids[-1] == EOS)
    # ~86 SNAC tokens/sec of audio; expect roughly chars/18 sec of speech
    expect = max(300, int(len(t) / 18 * 86 * 1.6))
    runaway = n_tok >= min(MAX_TOKENS, 2 * expect)
    flag = "OK " if (stopped and not runaway) else "FAIL"
    if flag == "FAIL":
        fails += 1
    print(f"[{flag}] chars={len(t):3d} tokens={n_tok:4d} expect<~{expect} stopped={stopped}", flush=True)

print(f"\nEOS_GATE_{'PASS' if fails == 0 else f'FAIL_{fails}'}_OF_{len(texts)}", flush=True)

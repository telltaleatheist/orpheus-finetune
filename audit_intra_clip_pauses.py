"""Audit training clips for MID-SENTENCE internal pauses (2026-07-14).

Why: the ep5 CoD deathstalker inserts spurious intra-phrase pauses (e.g. "after
... all") identically on vLLM and MLX -> learned from data ("training-data
silence becomes output pauses"). The cut pipeline caps internal pauses at 2s,
which is only correct BETWEEN sentences.

Heuristic (no alignment needed):
  - internal silence = a run of samples below 1% of clip peak, >= MIN_PAUSE s,
    not touching the first/last 0.3s of the clip
  - expected long-pause budget = (number of sentence terminators in the text) - 1
  - a clip whose internal-pause count EXCEEDS its budget has at least one
    mid-sentence pause -> flagged (worst offenders listed for ear-check)

Usage:  python audit_intra_clip_pauses.py [dataset_dir]
        dataset_dir must contain wavs/ + metadata_train.csv + metadata_eval.csv
        (default: the pt01-04 CoD cut)
"""
import csv
import os
import re
import struct
import sys
import wave

DEFAULT_DIR = r"E:\training\deathstalker\source\celebration of discipline\clips\deathstalker\pt01-04"
MIN_PAUSE = 0.40      # seconds of internal silence that counts as a pause
EDGE = 0.30           # ignore this much at each end (lead-in / tail, already audited)
SILENCE_RATIO = 0.01  # threshold as fraction of clip peak (matches trim_audio spirit)

root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR

rows = []
for name in ("metadata_train.csv", "metadata_eval.csv"):
    p = os.path.join(root, name)
    with open(p, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="|")
        header = next(r)
        rows += [(row[0], row[1]) for row in r if len(row) >= 2]
print(f"{len(rows)} metadata rows in {root}")

def internal_pauses(path):
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        n = w.getnframes()
        data = struct.unpack(f"<{n}h", w.readframes(n))
    peak = max(abs(min(data)), abs(max(data))) or 1
    th = peak * SILENCE_RATIO
    edge = int(EDGE * rate)
    min_run = int(MIN_PAUSE * rate)
    pauses = []
    run_start = None
    for i in range(edge, n - edge):
        if abs(data[i]) < th:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= min_run:
                pauses.append(((i - run_start) / rate, run_start / rate))
            run_start = None
    if run_start is not None and (n - edge) - run_start >= min_run:
        pauses.append((((n - edge) - run_start) / rate, run_start / rate))
    return pauses

flagged = []
total_pauses = 0
for rel, text in rows:
    wav_path = os.path.join(root, rel)
    if not os.path.isfile(wav_path):
        continue
    pauses = internal_pauses(wav_path)
    total_pauses += len(pauses)
    # sentence terminators (., !, ?, ...) — em-dash/colon pauses are legitimate
    # narration too, so this budget is deliberately generous
    budget = max(0, len(re.findall(r"[.!?](?:\s|$|['\"])", text)) - 1)
    if len(pauses) > budget:
        excess = len(pauses) - budget
        worst = max(p[0] for p in pauses)
        flagged.append((excess, worst, rel, len(pauses), budget, text[:90]))

flagged.sort(reverse=True)
print(f"clips with internal pauses >= {MIN_PAUSE}s: {total_pauses} pauses total")
print(f"FLAGGED (pauses exceed sentence budget): {len(flagged)}/{len(rows)}"
      f"  ({100*len(flagged)/max(1,len(rows)):.1f}%)")
print("\nworst 25 (excess, longest-pause-s, file, pauses, budget, text):")
for excess, worst, rel, np_, budget, text in flagged[:25]:
    print(f"  +{excess} {worst:4.2f}s {rel} ({np_} vs {budget}) {text!r}")

#!/usr/bin/env python3
"""render_pause_report.py — pause census of RENDERED clips, internal AND trailing.

WHY (2026-07-25): `pause_meter.py` deliberately EXCLUDES the run at the very end of
a clip (`if i > 0 and j < m`), because it was written to measure the pauses a model
puts *between* sentences. But the trailing pause is the one that costs throughput:
it is emitted on every generated chunk, and generation is KV-limited so clip length
costs quadratically. To test whether trimming the training tails moved the model, we
have to measure precisely the number pause_meter throws away.

Reports, per model directory of rendered wavs:
  INTERNAL pauses  count/min, median, p90, max   (the model's own sentence gaps)
  TRAILING pause   median, p90, max              (the per-chunk cost)
  clip duration    median, and the speech/silence split

Threshold −50 dBFS, matching pause_meter, so internal numbers stay comparable to it.
Rendered "silence" is ~−50..−70 dBFS hiss, never digital zero, so a deeper threshold
would miss real pauses.

Compare models on IDENTICAL texts or the numbers are not an A/B — loop_gate.py draws
its probe texts deterministically from the corpus it is given, so pass the SAME corpus
to every render arm even when the models were trained on different ones.

Usage:
  render_pause_report.py <dir_of_wavs> [<dir_of_wavs> ...]

NO FALLBACKS: a directory with no wavs is reported as such, not skipped silently.
"""
import glob
import os
import sys

import numpy as np
import soundfile as sf

WIN = 0.02
SIL_DB = -50.0
MIN_PAUSE = 0.08          # below this is co-articulation, not a pause


def runs_of_silence(y, sr):
    """(internal_pause_seconds, trailing_pause_seconds, speech_seconds)."""
    w = int(WIN * sr)
    m = len(y) // w
    if m == 0:
        return [], 0.0, 0.0
    rms = np.sqrt(np.mean(y[:m * w].reshape(m, w) ** 2, axis=1))
    sil = rms < 10 ** (SIL_DB / 20)

    internal, trailing = [], 0.0
    i = 0
    while i < m:
        if not sil[i]:
            i += 1
            continue
        j = i
        while j < m and sil[j]:
            j += 1
        length = (j - i) * WIN
        if j >= m:
            trailing = length          # reaches the end of the clip
        elif i > 0:
            internal.append(length)    # a genuine gap between speech
        i = j
    return [p for p in internal if p >= MIN_PAUSE], trailing, float((~sil).sum()) * WIN


def report(d):
    wavs = sorted(glob.glob(os.path.join(d, "*.wav")))
    name = os.path.basename(d.rstrip("/\\"))
    if not wavs:
        print(f"{name:22s} NO WAVS in {d}")
        return
    internal, trailing, durs, speech = [], [], [], []
    for p in wavs:
        y, sr = sf.read(p)
        if y.ndim > 1:
            y = y.mean(1)
        ins, tail, spk = runs_of_silence(y, sr)
        internal.append(ins)
        trailing.append(tail)
        durs.append(len(y) / sr)
        speech.append(spk)

    flat = np.array([p for ins in internal for p in ins])
    tr = np.array(trailing)
    du = np.array(durs)
    sp = np.array(speech)
    total_min = du.sum() / 60.0
    per_min = len(flat) / total_min if total_min > 0 else 0.0

    if len(flat):
        i_med, i_p90, i_max = np.median(flat), np.percentile(flat, 90), flat.max()
    else:
        i_med = i_p90 = i_max = 0.0

    print(f"{name:22s} n={len(wavs):3d} "
          f"| INTERNAL {per_min:5.1f}/min med {i_med*1000:4.0f}ms p90 {i_p90*1000:5.0f}ms max {i_max*1000:5.0f}ms "
          f"| TRAILING med {np.median(tr):4.2f}s p90 {np.percentile(tr,90):4.2f}s max {tr.max():4.2f}s "
          f"| clip med {np.median(du):5.2f}s speech {sp.sum()/du.sum()*100:4.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for d in sys.argv[1:]:
        if not os.path.isdir(d):
            sys.exit(f"not a directory: {d}")
        report(d)

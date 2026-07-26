#!/usr/bin/env python3
"""cleanliness_census.py — per-clip CLEANLINESS scoring, to pair with brightness.

WHY (2026-07-24): the curated-2h selection has always ranked on brightness and
dynamics. That is the right quality lever for a uniformly-clean master, but when
cutting from a RAW book (the go-forward default — see VOICE_TRAINING_PIPELINE.md
principle 2) the source is NOT uniformly clean: hum level drifts between source
parts, some stretches carry more hiss, and a few clips clip or drop out. Ranking
on brightness alone will happily pick a bright-but-noisy stretch.

This scores each clip on four independent defects, all measured in the clip's
own QUIET regions (so narration level does not confound them):

  floor_db     quiet-region RMS. Lower is cleaner — but NOT too low: a floor
               below ~-100 dBFS means the pauses are digitally dead, which is
               the SNAC silence attractor we are specifically avoiding.
  hum_db       excess at the mains line (and harmonics) over a local median.
  hiss_db      quiet-region energy above 6 kHz, relative to the clip's speech.
  bad_frac     clipped samples (|x| >= 0.999) plus exactly-zero runs (dropouts).

Output: `<out_prefix>_scores.npy` + `<out_prefix>_names.txt` containing ONLY the
clips that pass the cleanliness gates, in the SAME (bright, dyn, dur) format
brightness_census.py writes — so `build_2h_corpus.py` consumes it unchanged and
still does the final top-N-by-brightness pick. Selection stays one decision;
this just removes the dirty candidates first.

Usage:
  cleanliness_census.py <corpus_dir> <bright_prefix> <out_prefix> [mains_hz]
                        [--restrict <metadata.csv>]

--restrict narrows the candidate pool to the clips listed in a metadata CSV before
any cleanliness scoring — the composable way to combine selection criteria without
a second selection authority. Chiefly `metadata_narration.csv` from
`clipforge narration`, so the clean pool is also character-voice-free (principle 9:
Orpheus has no speaker map, so character voices in the corpus fire at random).
"""
import csv
import os
import sys

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
RESTRICT = None
if "--restrict" in sys.argv:
    i = sys.argv.index("--restrict")
    if i + 1 >= len(sys.argv):
        raise SystemExit("--restrict needs a metadata CSV path")
    RESTRICT = sys.argv[i + 1]
    argv = [a for a in argv if a != RESTRICT]

CORPUS = argv[0]
BRIGHT_PREFIX = argv[1]
OUT_PREFIX = argv[2]
MAINS = float(argv[3]) if len(argv) > 3 else 120.0

# A floor BELOW this is dead-pause territory, not cleanliness — reject it too.
DEAD_FLOOR_DB = -100.0
QUIET_PCT = 20          # bottom N% of 20 ms frames are "quiet"
FFT = 1 << 14


def clip_metrics(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if mono.size < rate // 4:
        return None
    step = max(1, int(rate * 0.02))
    frames = mono[:len(mono) // step * step].reshape(-1, step)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    quiet = frames[rms <= np.percentile(rms, QUIET_PCT)].ravel()
    speech = frames[rms >= np.percentile(rms, 80)].ravel()
    if quiet.size < FFT // 4 or speech.size == 0:
        return None

    floor_db = 20 * np.log10(np.sqrt((quiet ** 2).mean()) + 1e-12)
    speech_db = 20 * np.log10(np.sqrt((speech ** 2).mean()) + 1e-12)

    buf = np.zeros(FFT, dtype=np.float64)
    take = min(FFT, quiet.size)
    buf[:take] = quiet[:take]
    spec = 20 * np.log10(np.abs(np.fft.rfft(buf * np.hanning(FFT))) + 1e-12)
    freq = np.fft.rfftfreq(FFT, 1 / rate)
    local = median_filter(spec, size=201)
    excess = spec - local

    hum_db = 0.0
    for harmonic in (1, 2, 3):
        target = MAINS * harmonic
        if target >= freq[-1]:
            break
        near = np.abs(freq - target) <= 8.0
        if near.any():
            hum_db = max(hum_db, float(excess[near].max()))

    high = freq >= 6000
    mid = (freq >= 300) & (freq < 6000)
    if not high.any() or not mid.any():
        raise SystemExit(f"{path}: sample rate {rate} too low to measure hiss tilt")
    hiss_db = float(spec[high].mean() - spec[mid].mean())

    clipped = float((np.abs(mono) >= 0.999).mean())
    # A true dropout is a LONG run of zeros. Scattered exact zeros are just 16-bit
    # quantization of a very quiet floor (a -83 dBFS floor sits ~13 dB above the
    # PCM_16 LSB, so ~6% of its samples round to zero) — counting those as a
    # defect rejects every clip of a legitimately quiet raw master.
    #
    # The run threshold was 5ms and that was still far too tight. MEASURED on Alloy
    # of Law, whose floor is -90.9 dBFS (quieter than thirdreich's -81.9, so more of
    # it falls under the LSB), over 120 clips — fraction of clip in qualifying runs,
    # and clips exceeding the 0.1% reject line:
    #     >=5ms   median 0.240%   66/120 clips rejected
    #     >=20ms  median 0.049%   13/120
    #     >=50ms  median 0.032%    4/120
    #     >=100ms median 0.000%    0/120
    # At 5ms this gate threw away 1546 of 2652 clips (58%) of a corpus whose SNAC
    # pause-frame concentration is 0.3% — i.e. IDENTICAL to mm_narr2h, the source
    # behind a shipped, working model. The audio was fine; the metric was wrong.
    # 100ms is 2400 consecutive samples below the LSB: that cannot be quantization,
    # it is real dead air. SNAC pause concentration (pause_frame_conc.py) remains the
    # authority on whether a floor is alive; this gate only catches missing audio.
    zero = np.abs(mono) == 0
    edges = np.diff(np.concatenate(([0], zero.view(np.int8), [0])))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    runs = ends - starts
    dropout = float(runs[runs >= int(rate * 0.100)].sum() / len(mono)) if runs.size else 0.0
    return floor_db, hum_db, hiss_db, clipped, dropout, speech_db - floor_db


names = open(BRIGHT_PREFIX + "_names.txt").read().splitlines()
bright_scores = np.load(BRIGHT_PREFIX + "_scores.npy")
assert len(names) == len(bright_scores), f"{len(names)} names vs {len(bright_scores)} scores"

allowed = None
if RESTRICT:
    path = RESTRICT if os.path.isabs(RESTRICT) else os.path.join(CORPUS, RESTRICT)
    if not os.path.exists(path):
        raise SystemExit(f"--restrict metadata not found: {path}")
    with open(path, encoding="utf-8") as handle:
        allowed = {r[0] for r in list(csv.reader(handle, delimiter="|"))[1:] if r}
    print(f"restricted to {len(allowed)} clips from {os.path.basename(path)}")

rows, keep_index = [], []
for i, name in enumerate(names):
    if allowed is not None and name not in allowed:
        continue
    metrics = clip_metrics(os.path.join(CORPUS, name))
    if metrics is None:
        continue
    rows.append(metrics)
    keep_index.append(i)
    if len(rows) % 500 == 0:
        print(f"  scored {len(rows)}/{len(names)}", flush=True)

data = np.array(rows)
keep_index = np.array(keep_index)
floor_db, hum_db, hiss_db, clipped, dropout, snr_db = data.T

print(f"\n=== cleanliness census: {len(data)} clips ===")
for label, values, unit in (("quiet floor", floor_db, "dBFS"), ("hum excess", hum_db, "dB"),
                            ("hiss tilt", hiss_db, "dB"), ("SNR", snr_db, "dB")):
    print(f"  {label:12s} p10={np.percentile(values,10):8.2f} median={np.median(values):8.2f} "
          f"p90={np.percentile(values,90):8.2f} {unit}")
print(f"  clipped frac    max={clipped.max():.5f}   clips>0.1% = {int((clipped>0.001).sum())}")
print(f"  dropout (zero runs >=100ms) max={dropout.max():.5f}   clips>0.1% = {int((dropout>0.001).sum())}")

# Gates: drop the dirtiest decile on each independent defect, plus hard rejects.
gates = {
    "dead pauses (floor too low)": floor_db < DEAD_FLOOR_DB,
    "noisiest decile (floor)": floor_db > np.percentile(floor_db, 90),
    "hummiest decile": hum_db > np.percentile(hum_db, 90),
    "hissiest decile": hiss_db > np.percentile(hiss_db, 90),
    "clipping": clipped > 0.001,
    "dropout (dead-air runs)": dropout > 0.001,
    "worst SNR decile": snr_db < np.percentile(snr_db, 10),
}
reject = np.zeros(len(data), dtype=bool)
for label, mask in gates.items():
    print(f"  reject {label:28s}: {int(mask.sum()):5d}")
    reject |= mask
keep = ~reject
print(f"\n  KEPT {int(keep.sum())} of {len(data)} clips "
      f"({keep.sum()/len(data)*100:.1f}%) as the clean pool")

kept_names = [names[i] for i in keep_index[keep]]
kept_scores = bright_scores[keep_index[keep]]
np.save(OUT_PREFIX + "_scores.npy", kept_scores)
with open(OUT_PREFIX + "_names.txt", "w") as handle:
    handle.write("\n".join(kept_names) + "\n")
hours = kept_scores[:, 2].sum() / 3600
print(f"  wrote {OUT_PREFIX}_names.txt / _scores.npy  ({hours:.2f} h of clean candidates)")
if hours < 2.5:
    print("  ** WARNING: clean pool is thin; loosen a gate or accept a smaller corpus **")

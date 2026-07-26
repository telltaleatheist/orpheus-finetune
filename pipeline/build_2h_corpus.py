#!/usr/bin/env python
"""Curated bright-2h deathstalker corpus (Owen's call, 2026-07-20).

Census measured an 8.0 dB p90-p10 per-clip brightness spread in mm8_v2 — the
model learns the spread and samples from it chunk-by-chunk (the muffled/sharp
oscillation Owen heard in v8). Selection narrows the distribution to its
bright end: consistency AND brightness in one move.

Select: rank by brightness ratio, cap outliers at p99.5 (anomaly guard), drop
clips in the bottom dynamics decile (monotone), take top ~480 (~2 h), hold out
~40 for eval. Then the locked E recipe: bed at -65 random offsets + tails.
"""
import csv, os, sys
import numpy as np
import soundfile as sf

FT = "/home/telltale/xtts_ft"
# args: SRC_DIR DST_DIR BED_PATH SCORES_PREFIX [--no-bed] [--no-tails]
# --no-bed/--no-tails (2026-07-21 whisper-sprint verdict): the bed itself
# teaches the model to emit hiss codes in pauses, which SNAC renders as
# ghostly syllable-babble (w1 no-bed = clean; w2 bed+silent-tails = whispers;
# w3 synthetic bed = audible hiss). BED_PATH is still required positionally
# even with --no-bed so invocations stay explicit.
NO_BED = "--no-bed" in sys.argv
NO_TAILS = "--no-tails" in sys.argv
# --silent-tails: punctuation-scaled tails of DIGITAL SILENCE instead of bed
# (the StyleTTS2-style stop cue; ~1.6% silence mass at 0.2-0.4s/clip — far
# below the 21-31% that bred the silence attractor).
SILENT_TAILS = "--silent-tails" in sys.argv
if SILENT_TAILS and NO_TAILS:
    sys.exit("--silent-tails and --no-tails are mutually exclusive")
# --verbatim (2026-07-21 raw-verbatim pivot, TESTING_BIBLE go-forward recipe):
# selection ONLY. Clips are copied bit-identical from the cutter's output —
# no breath-safe edges, no bed, no tails. Natural pauses/tails/floors stay
# exactly as narrated (the w4 verdict: the model drops unpredictable noise;
# EOS margin comes from the logit boost, not acoustics).
VERBATIM = "--verbatim" in sys.argv
if VERBATIM:
    NO_BED = NO_TAILS = True
    SILENT_TAILS = False
# CORPUS SIZE IS A PARAMETER, NOT A CONSTANT (2026-07-24). It used to be a
# hardcoded 480 clips / 40 eval, which is a CLIP-COUNT target — so the hours landed
# wherever the book's clip lengths fell: mm_narr2h 1.99h, mistborn_narr2h 1.90h,
# thirdreich_trip2h 1.63h off the same 480. Owen's floor is 2.0 HOURS (he trained on
# 8h before and does not want to go below 2), so duration is the thing to target.
#
# Cost of hitting 2.0h instead of stopping at 480 clips — MEASURED on the thirdreich
# clean pool (4034 eligible clips, 15.4h deep):
#     480 clips = 1.63h, mean -15.1 dB, spread 2.0
#     574 clips = 2.00h, mean -15.2 dB, spread 2.1
#     700 clips = 2.47h, mean -15.5 dB, spread 2.2
# i.e. 0.1 dB of brightness for +0.37h. Reaching deeper only costs real quality when
# the pool is shallow; at 8x the target it is free.
#
# ⚠ Clip count also sets the STEP count (batch 1 x accum 4 => clips/4 steps per
# epoch). 440 train clips = 110 steps/epoch, which is where the 8-epoch/patience-2
# recipe and every "epNNN = epoch N" reading were calibrated. Changing the count
# MOVES the settle epoch and the checkpoint numbering — record the count with the
# corpus so a later gate result can be interpreted.


def _flag_value(name, cast, default=None):
    """Pull `--name value` out of argv. Returns default when absent."""
    if name not in sys.argv:
        return default
    i = sys.argv.index(name)
    if i + 1 >= len(sys.argv):
        sys.exit(f"{name} needs a value")
    return cast(sys.argv[i + 1])


TARGET_HOURS = _flag_value("--hours", float)
N_TARGET = _flag_value("--clips", int)
EVAL_FRAC = _flag_value("--eval-frac", float, 1.0 / 12.0)   # 40/480 = the old ratio
N_EVAL = _flag_value("--eval", int)
if TARGET_HOURS is None and N_TARGET is None:
    sys.exit("specify --hours <H> (duration target, preferred) or --clips <N>; "
             "there is no default corpus size — see the comment above")

# Strip flags AND their values so the positional args still line up.
_flag_vals = set()
for _f in ("--hours", "--clips", "--eval", "--eval-frac"):
    if _f in sys.argv:
        _i = sys.argv.index(_f)
        if _i + 1 < len(sys.argv):
            _flag_vals.add(_i + 1)
argv = [a for i, a in enumerate(sys.argv) if not a.startswith("--") and i not in _flag_vals]
if len(argv) < 5:
    sys.exit("usage: build_2h_corpus.py <src_corpus> <dst_corpus> <bed.wav> <scores_prefix> "
             "(--hours H | --clips N) [--eval N | --eval-frac F] [--verbatim]")
SRC, DST, BED_PATH, SCORES_PREFIX = argv[1], argv[2], argv[3], argv[4]
LEVEL = 10 ** (-65 / 20)
TAIL_FULL, TAIL_HALF = 0.40, 0.20
CEIL_S = 19.9
rng = np.random.default_rng(11)

scores = np.load(SCORES_PREFIX + "_scores.npy")   # (bright, dyn, dur) per clip
names = open(SCORES_PREFIX + "_names.txt").read().splitlines()
assert len(names) == len(scores), f"{len(names)} names vs {len(scores)} scores"

bright, dyn, dur = scores[:, 0], scores[:, 1], scores[:, 2]
cap = np.percentile(bright, 99.5)
dyn_floor = np.percentile(dyn, 10)
eligible = (bright <= cap) & (dyn > dyn_floor)
order = [i for i in np.argsort(-bright) if eligible[i]]
if TARGET_HOURS is not None:
    # Duration target: take the brightest clips until the FLOOR is reached, then
    # stop. Never undershoot — 2.0h is Owen's minimum, so the last clip crosses it.
    need = TARGET_HOURS * 3600.0
    total, picked = 0.0, []
    for i in order:
        picked.append(i)
        total += dur[i]
        if total >= need:
            break
    if total < need:
        sys.exit(f"pool holds only {total/3600:.2f} h of eligible clips — cannot reach "
                 f"--hours {TARGET_HOURS}. Loosen a cleanliness gate or cut more source.")
else:
    picked = order[:N_TARGET]
if N_EVAL is None:
    N_EVAL = max(1, round(len(picked) * EVAL_FRAC))
if N_EVAL >= len(picked):
    sys.exit(f"--eval {N_EVAL} leaves no training clips out of {len(picked)}")
b = bright[picked]
print(f"selected {len(picked)} clips, {dur[picked].sum()/3600:.2f} h "
      f"(target {'--hours ' + str(TARGET_HOURS) if TARGET_HOURS is not None else '--clips ' + str(N_TARGET)}), "
      f"eval {N_EVAL} | brightness mean {b.mean():.1f} p10 {np.percentile(b,10):.1f} "
      f"p90 {np.percentile(b,90):.1f} (spread {np.percentile(b,90)-np.percentile(b,10):.1f} dB vs corpus 8.0) "
      f"| steps/epoch at batch1xaccum4: {(len(picked)-N_EVAL)//4}",
      flush=True)

meta = {}
for split in ("train", "eval"):
    for r in list(csv.reader(open(f"{SRC}/metadata_{split}.csv"), delimiter="|"))[1:]:
        meta[r[0]] = (r[1], r[2])

need_bed = (not NO_BED) or (not NO_TAILS and not SILENT_TAILS)
if need_bed:
    bed, bsr = sf.read(BED_PATH)
    if bed.ndim > 1:
        bed = bed.mean(1)
    bed = bed.astype(np.float32)
    assert bsr == 24000


def bed_slice(n):
    if n >= len(bed):
        parts, need = [], n
        while need > 0:
            off = rng.integers(0, len(bed) // 2)
            take = min(need, len(bed) - off)
            parts.append(bed[off:off + take])
            need -= take
        return np.concatenate(parts)[:n] * LEVEL
    off = rng.integers(0, len(bed) - n)
    return bed[off:off + n] * LEVEL


def tail_len(text):
    t = text.rstrip()
    if t.endswith(('"', "'", "”", "’")):
        t = t[:-1].rstrip()
    return TAIL_HALF if t.endswith((",", ";", ":", "—", "-")) else TAIL_FULL


def breath_safe_edges(y, sr):
    """Trim the clip's kept trailing pause down to 200ms past the last real
    speech, and the lead-in to 150ms before the first — UNLESS the lead
    contains breath-band audio, in which case cut to 40ms before onset with a
    15ms fade (leading-inhale rule).

    WHY trailing (measured 2026-07-20): 95-100% of clips END with -55..-30 dB
    breath-band content — the cutter keeps each sentence's full trailing pause,
    and the narrator's pre-next-sentence inhale lives inside it. With pause
    tails appended, models learned 'speech -> ghost breath -> room tone ->
    stop' and whisper ghost-breaths appeared in every rendered pause.

    WHY leading (measured 2026-07-20, mb_2hb postmortem): the trailing fix
    alone left mistborn/AoL with 94% of clip STARTS carrying -55..-30 dB
    breath-band energy (the narrator's inhale before each phrase; Audible
    processing had tightened these in Marked Man, so ds was 48% and fine).
    The model learned 'utterance begins with a breath' and rendered a ghost
    whisper before sentences. Census gate: corpus starts should be majority
    hot, breathy well under ~50%.
    """
    w = int(0.02 * sr)
    m = len(y) // w
    if m < 10:
        return y
    rms = np.sqrt(np.mean(y[:m * w].reshape(m, w) ** 2, axis=1))
    speech = np.where(rms > 10 ** (-30 / 20))[0]
    if len(speech) == 0:
        return y
    end = min(len(y), (speech[-1] + 1) * w + int(0.20 * sr))

    # Leading edge: does the pre-onset region hold breath-band energy?
    db = 20 * np.log10(rms + 1e-12)
    lead_lo, lead_hi = max(0, speech[0] - int(0.30 / 0.02)), speech[0]
    lead_breathy = np.any((db[lead_lo:lead_hi] > -55) & (db[lead_lo:lead_hi] < -30))
    if lead_breathy:
        start = max(0, speech[0] * w - int(0.04 * sr))
        fade_in = int(0.015 * sr)
    else:
        start = max(0, speech[0] * w - int(0.15 * sr))
        fade_in = int(0.05 * sr)

    y = y[start:end].copy()
    f_out = int(0.05 * sr)
    if len(y) > fade_in + f_out:
        y[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
        y[-f_out:] *= np.linspace(1, 0, f_out, dtype=np.float32)
    return y


os.makedirs(f"{DST}/wavs", exist_ok=True)
picked_names = [names[i] for i in picked]
rng.shuffle(picked_names)
splits = {"eval": picked_names[:N_EVAL], "train": picked_names[N_EVAL:]}
for split, rows in splits.items():
    out = [["audio_file", "text", "speaker_name"]]
    added = 0.0
    for rel in rows:
        y, sr = sf.read(os.path.join(SRC, rel))
        if y.ndim > 1:
            y = y.mean(1)
        assert sr == 24000, f"{rel} is {sr}"
        y = y.astype(np.float32)
        if not VERBATIM:
            y = breath_safe_edges(y, sr)
        if not NO_BED:
            y = y + bed_slice(len(y))
        text, spk = meta[rel]
        tl = 0.0 if NO_TAILS else tail_len(text)
        room = CEIL_S - len(y) / sr
        if room <= 0.05:
            tl = 0.0
        elif tl > room:
            tl = room
        if tl > 0:
            tail = (np.zeros(int(tl * sr), dtype=np.float32) if SILENT_TAILS
                    else bed_slice(int(tl * sr)))
            y = np.concatenate([y, tail])
            added += tl
        np.clip(y, -1.0, 1.0, out=y)
        fn = ("e_" if split == "eval" else "") + os.path.basename(rel)
        sf.write(os.path.join(DST, "wavs", fn), y, sr)
        out.append([f"wavs/{fn}", text, spk])
    with open(f"{DST}/metadata_{split}.csv", "w", newline="") as f:
        csv.writer(f, delimiter="|").writerows(out)
    print(f"[{split}] {len(out)-1} clips, +{added/60:.1f} min tails", flush=True)
print("BUILD_2H_DONE", flush=True)

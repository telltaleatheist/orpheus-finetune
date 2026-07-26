#!/usr/bin/env python3
"""trim_tails.py — shorten each clip's TRAILING SILENCE to a target, breath-safely.

WHY (Owen, 2026-07-25): a narrator's natural end-of-sentence pause is trained in
verbatim and reproduced on every generated chunk. Pratt's is 1.51s where Rohan's is
0.90s, so the thirdreich model emits ~0.6s more silence per chunk than deathstalker.
Because generation is KV-limited, longer clips cost throughput QUADRATICALLY — which
is most of the ~30 vs ~70 chunks/min gap. Trimming the tail to the proven corpus's
length recovers it without touching speech.

THE HARD CONSTRAINT: never cut inside a breath. A half-cut inhale is the documented
disaster — it trains ghost-breaths into every rendered pause. Bands follow the
pipeline's own definition:
    speech  > -30 dBFS
    breath  -55 .. -30 dBFS   <- never cut here
    silence < -55 dBFS        <- only ever cut here

The cut point is the first SILENCE frame at or after the target offset, so it can
only ever land in true silence and can only ever be >= the target (never shorter).
A clip whose tail holds no silence frame at/after the target is LEFT UNTOUCHED and
reported — never trimmed on a guess.

MEASURED on thirdreich_trip2h before writing this (574 clips): 763 breath runs, onset
median 0.00s / p90 0.16s after speech end, duration median 0.10s — i.e. Pratt's
breaths are the trailing exhale and all finish well before 0.90s. Breath runs
straddling a 0.90s cut: 0.

Usage:
  trim_tails.py <src_corpus> <dst_corpus> [--target 0.90] [--fade 0.020] [--dry-run]

NO FALLBACKS: a missing source, an unreadable clip, or a clip with no safe cut point
is reported explicitly. Nothing is silently approximated.
"""
import argparse
import csv
import os
import shutil
import sys

import numpy as np
import soundfile as sf

FRAME = 0.010
SPEECH_DB, BREATH_DB = -30.0, -55.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--target", type=float, default=None,
                    help="CLAMP mode: keep exactly this much trailing silence, seconds. "
                         "Flattens every tail to one length — see --median-to.")
    ap.add_argument("--median-to", type=float, default=None,
                    help="SCALE mode (preferred): scale every tail by a constant factor so the "
                         "corpus MEDIAN tail lands here, preserving natural variation. "
                         "deathstalker reference: 0.98 (median) with 1.43 p90.")
    ap.add_argument("--fade", type=float, default=0.020,
                    help="fade-out applied at the cut so it cannot click")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if (a.target is None) == (a.median_to is None):
        sys.exit("specify exactly one of --median-to <s> (scale, preferred) or --target <s> (clamp)")
    if not os.path.isdir(a.src):
        sys.exit(f"source corpus not found: {a.src}")
    wav_dir = os.path.join(a.src, "wavs")
    if not os.path.isdir(wav_dir):
        sys.exit(f"no wavs/ under {a.src}")

    def tail_frames(mono, sr):
        """Frames of trailing pause after the last speech frame, as dB, or None."""
        step = max(1, int(sr * FRAME))
        fr = mono[:len(mono) // step * step].reshape(-1, step)
        db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-12)
        speech = db > SPEECH_DB
        if not speech.any():
            return None, None, step
        last = int(np.nonzero(speech)[0][-1])
        return last, db[last + 1:], step

    names = [n for n in sorted(os.listdir(wav_dir)) if n.lower().endswith(".wav")]

    # SCALE mode needs one pass to learn the corpus's own median tail, so the factor
    # is derived from data rather than assumed. CLAMP mode needs no pass.
    factor = None
    if a.median_to is not None:
        tails = []
        for n in names:
            y, sr = sf.read(os.path.join(wav_dir, n), dtype="float32", always_2d=True)
            _, tail, _ = tail_frames(y.mean(axis=1), sr)
            if tail is not None:
                tails.append(len(tail) * FRAME)
        if not tails:
            sys.exit("no clip had measurable speech — cannot derive a scale factor")
        current = float(np.median(tails))
        factor = a.median_to / current
        print(f"scale mode: corpus median tail {current:.2f}s -> target {a.median_to:.2f}s "
              f"(factor {factor:.3f}); variation preserved")
        if factor >= 1.0:
            print("  factor >= 1.0 — tails are already at or below target; nothing to shorten")

    if not a.dry_run:
        os.makedirs(os.path.join(a.dst, "wavs"), exist_ok=True)

    kept_before = kept_after = 0.0
    trimmed = untouched = 0
    no_safe = []
    for name in names:
        path = os.path.join(wav_dir, name)
        y, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = y.mean(axis=1)
        kept_before += len(mono) / sr
        last, tail, step = tail_frames(mono, sr)
        if last is None:
            no_safe.append((name, "no speech"))
            if not a.dry_run:
                shutil.copy2(path, os.path.join(a.dst, "wavs", name))
            kept_after += len(mono) / sr
            untouched += 1
            continue
        # Per-clip goal: a constant in CLAMP mode; this clip's own tail scaled in
        # SCALE mode, so a naturally long pause stays proportionally longer.
        goal = a.target if a.target is not None else (len(tail) * FRAME) * factor
        target_idx = int(round(goal / FRAME))
        # Only ever cut in TRUE SILENCE, and never before the target.
        safe = None
        for i in range(min(target_idx, len(tail)), len(tail)):
            if tail[i] <= BREATH_DB:
                safe = i
                break
        if safe is None:
            no_safe.append((name, f"no silence frame at/after {goal:.2f}s in a {len(tail)*FRAME:.2f}s tail"))
            if not a.dry_run:
                shutil.copy2(path, os.path.join(a.dst, "wavs", name))
            kept_after += len(mono) / sr
            untouched += 1
            continue
        cut = (last + 1 + safe + 1) * step        # end of the chosen silence frame
        cut = min(cut, len(mono))
        out = mono[:cut].copy()
        f = int(sr * a.fade)
        if f > 0 and len(out) > f:
            out[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
        kept_after += len(out) / sr
        trimmed += 1
        if not a.dry_run:
            sf.write(os.path.join(a.dst, "wavs", name), out, sr,
                     subtype=sf.info(path).subtype)

    for split in ("train", "eval"):
        s = os.path.join(a.src, f"metadata_{split}.csv")
        if os.path.exists(s) and not a.dry_run:
            shutil.copy2(s, os.path.join(a.dst, f"metadata_{split}.csv"))

    mode = f"clamp {a.target}s" if a.target is not None else f"scale to median {a.median_to}s (x{factor:.3f})"
    print(f"trim mode: {mode}  (fade {a.fade*1000:.0f}ms)")
    print(f"  trimmed   : {trimmed} clips")
    print(f"  untouched : {untouched} clips" + ("  <- inspect these" if untouched else ""))
    for n, why in no_safe[:10]:
        print(f"      {n}: {why}")
    print(f"  duration  : {kept_before/3600:.3f} h -> {kept_after/3600:.3f} h "
          f"({(kept_before-kept_after)/60:.1f} min removed)")
    if a.dry_run:
        print("  (dry run — nothing written)")


if __name__ == "__main__":
    main()

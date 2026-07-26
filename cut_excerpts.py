#!/usr/bin/env python3
r"""cut_excerpts.py — cut the Deathstalker GraphicAudio training excerpts into an
Orpheus-ready dataset of LONG clips, from the book-anchored word-timestamped
transcripts (align_excerpts.py output).

WHY LONG: the deployed voice truncates long text; clips cut at sentence length
teach short-utterance prosody only. This cutter merges toward --target (default
18 s) and caps at --max-clip (default 20.0 s = the measured SNAC-token budget for
max_seq_length=2048 in orpheus_owen.py train). Going past 20 s requires raising
max_seq_length — the cutter will let you (--max-clip), but says so loudly.

REUSES orpheus_owen.py's validated recut machinery (do not reimplement):
  _silence_gaps / _quietest_point  cut ONLY at sustained silence troughs
  RECUT_STRONG_SIL (0.28 s)        breath-safe: only strong gaps become cuts
  _edge_silent                     both clip edges must actually be quiet
  _normalize_pauses                natural pauses <=2 s kept EXACTLY; structural
                                   gaps capped; edges tightened to LEAD/TRAIL pad

ADDS over cmd_recut (which assumes one contiguous chapter and aligned.json):
  * words come from <stem>.words.json (book text with punctuation + whisper-only
    stretches tagged) instead of aligned.json;
  * SFX keep-out windows are HARD WALLS: the timeline is partitioned into allowed
    intervals between excluded windows; no clip may straddle or touch one;
  * digit tokens are expanded to spoken words (narrator says "ten", book prints
    "10" — Orpheus spec wants normalized text); leftover digits are reported;
  * every clip row records whisper_frac (fraction of its words that are
    whisper-only, i.e. unverified ASR text) so Owen can filter at train time.

Outputs (a NEW dataset dir; never touches the source audio):
  <out>/wavs/<stem>_########.wav      24000 Hz mono (Orpheus/SNAC rate)
  <out>/metadata_train.csv, metadata_eval.csv   audio_file|text|speaker_name
  <out>/clips.json                    provenance: source file, start/end sec,
                                      duration, whisper_frac, text
  <out>/CUT_REPORT.md                 totals, duration histogram, digit leftovers

Usage (env needs numpy/librosa/soundfile — `xtts` conda env works):
  python cut_excerpts.py --words "<transcripts>/deathstalker honor.words.json" \
      --wav "<matched>/deathstalker honor.wav" [--words ... --wav ... pairs] \
      --out-dir "<...>/dataset_v1" --source-name deathstalker
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import orpheus_owen as oo                                   # noqa: E402
from align_excerpts import num_to_words                     # noqa: E402

SR = oo.TARGET_SAMPLE_RATE          # 24000 — cut directly at Orpheus rate
MIN_CLIP = oo.MIN_CLIP_SECONDS      # 1.5


def expand_digits(text):
    """Expand bare digit tokens (0-9999) to spoken words, preserving attached
    punctuation. Returns (new_text, leftover_digit_tokens)."""
    out = []
    leftovers = []
    for tok in text.split():
        m = re.match(r"^(\W*)(\d{1,4})(\W*)$", tok)
        if m and num_to_words(int(m.group(2))):
            words = num_to_words(int(m.group(2)))
            words[0] = m.group(1) + words[0]
            words[-1] = words[-1] + m.group(3)
            out.extend(words)
            continue
        if any(c.isdigit() for c in tok):
            leftovers.append(tok)
        out.append(tok)
    return " ".join(out), leftovers


def allowed_intervals(total_sec, keepouts):
    """Complement of the SFX keep-out windows over [0, total_sec]."""
    iv = []
    cur = 0.0
    for s, e in sorted(keepouts):
        if s > cur:
            iv.append((cur, s))
        cur = max(cur, e)
    if cur < total_sec:
        iv.append((cur, total_sec))
    return iv


def make_mix_sampler(mix_spec, rng):
    """Parse "w:lo-hi,w:lo-hi,..." into a per-clip target sampler. Each new clip
    draws a bucket by weight, then a uniform target inside it — this is how the
    duration-MIXED distribution from TRAINING_CLIP_LENGTH_RESEARCH.md is realized
    (EOS must be seen at every scale; uniform-long or uniform-short both fail)."""
    buckets = []
    for part in mix_spec.split(","):
        w, rng_part = part.split(":")
        lo, hi = rng_part.split("-")
        buckets.append((float(w), float(lo), float(hi)))
    total = sum(w for w, _, _ in buckets)
    weights = [w / total for w, _, _ in buckets]

    def sample():
        r = rng.random()
        acc = 0.0
        for wgt, (_, lo, hi) in zip(weights, buckets):
            acc += wgt
            if r <= acc:
                return rng.uniform(lo, hi)
        return rng.uniform(buckets[-1][1], buckets[-1][2])
    return sample


def cut_file(stem, wav_path, words_json, target_sampler, max_clip, rows,
             clips_meta, out_wavs, np, librosa, sf, gain_db=0.0):
    data = json.load(open(words_json, encoding="utf-8"))
    words = [(w["word"], w["start"], w["end"], w["source"])
             for w in data["words"]]
    keepouts = [(r["start"], r["end"])
                for r in (data["sfx_keepout_regions"] or [])]

    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    if gain_db:
        # STATIC per-source gain to hit -20 LUFS (README hard rule #4: never
        # per-clip loudnorm — its dynamic gain bakes amplitude wobble in).
        y = y * (10.0 ** (gain_db / 20.0))
        peak = float(np.max(np.abs(y)))
        if peak >= 1.0:
            raise SystemExit(f"{stem}: gain {gain_db:+.1f} dB clips "
                             f"(peak {peak:.3f}) — lower the gain")
    total_sec = len(y) / SR
    intervals = allowed_intervals(total_sec, keepouts)
    print(f"[cut] {stem}: {total_sec/60:.1f} min, {len(keepouts)} keep-out "
          f"windows -> {len(intervals)} allowed intervals")

    gaps_all = oo._silence_gaps(y, np, librosa)
    kept = 0
    emitted = 0.0
    dropped_edge = 0
    digit_leftovers = []

    def words_in(a, b):
        """Words whose midpoint falls in [a,b) samples; returns (text, wfrac)."""
        sel = [(w, src) for (w, ws, we, src) in words
               if a <= int(((ws + we) / 2) * SR) < b]
        if not sel:
            return "", 0.0
        txt = re.sub(r"\s+", " ", " ".join(w for w, _ in sel)).strip()
        wfrac = sum(1 for _, s in sel if s == "whisper") / len(sel)
        return txt, wfrac

    for iv_start, iv_end in intervals:
        a0, b0 = int(iv_start * SR), int(iv_end * SR)
        if (b0 - a0) / SR < MIN_CLIP:
            continue
        gaps = [(g0, g1) for g0, g1 in gaps_all if g0 >= a0 and g1 <= b0]
        strong = [oo._quietest_point(y, g0, g1, np) for g0, g1 in gaps
                  if (g1 - g0) / SR >= oo.RECUT_STRONG_SIL]
        bounds = sorted(set([a0] + strong + [b0]))

        raw = []
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            txt, _ = words_in(a, b)
            if txt:
                raw.append([a, b])

        # merge ADJACENT spans toward a per-clip target, cap max_clip. Adjacent
        # only — the clip stays one contiguous audio region (real pauses
        # included); non-adjacent audio is NEVER spliced.
        merged = []
        cur_target = target_sampler()
        for a, b in raw:
            if merged and (merged[-1][1] - merged[-1][0]) / SR < cur_target and \
               (b - merged[-1][0]) / SR <= max_clip:
                merged[-1][1] = b
            else:
                merged.append([a, b])
                cur_target = target_sampler()

        def split_long(a, b):
            if (b - a) / SR <= max_clip:
                return [(a, b)]
            interior = [(g0, g1) for g0, g1 in gaps if a < g0 and g1 < b]
            if not interior:
                return [(a, b)]
            g0, g1 = max(interior, key=lambda g: g[1] - g[0])
            mid = oo._quietest_point(y, g0, g1, np)
            if mid <= a or mid >= b:
                return [(a, b)]
            return split_long(a, mid) + split_long(mid, b)

        final = []
        for a, b in merged:
            final.extend(split_long(a, b))

        for a, b in final:
            dur = (b - a) / SR
            if dur < MIN_CLIP or dur > max_clip:
                continue
            if not oo._edge_silent(y, a, b, np):
                dropped_edge += 1
                continue
            txt, wfrac = words_in(a, b)
            if len(txt) < 3:
                continue
            txt, leftovers = expand_digits(txt)
            digit_leftovers.extend(leftovers)
            clip = oo._normalize_pauses(y[a:b], np, librosa)
            cdur = len(clip) / SR
            if cdur < MIN_CLIP:
                continue
            safe = re.sub(r"\s+", "_", stem)
            fn = f"{safe}_{str(kept).zfill(8)}.wav"
            sf.write(str(out_wavs / fn), clip, SR)
            rows.append((f"wavs/{fn}", txt))
            clips_meta.append({
                "file": f"wavs/{fn}", "source": stem,
                "src_start": round(a / SR, 3), "src_end": round(b / SR, 3),
                "duration": round(cdur, 2),
                "whisper_frac": round(wfrac, 3),
                "text": txt,
            })
            kept += 1
            emitted += cdur

    print(f"[cut] {stem}: {kept} clips, {emitted/60:.1f} min "
          f"(edge-gate dropped {dropped_edge})")
    return kept, emitted, digit_leftovers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", action="append", required=True,
                    help="<stem>.words.json from align_excerpts.py (repeatable)")
    ap.add_argument("--wav", action="append", required=True,
                    help="matching source wav, same order as --words")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source-name", required=True,
                    help="speaker_name column / Orpheus voice name")
    ap.add_argument("--target", type=float, default=18.0,
                    help="merge clips toward this many seconds (uniform mode)")
    ap.add_argument("--mix", default=None,
                    help="duration-MIXED mode: 'w:lo-hi,w:lo-hi,...' — each clip "
                         "draws a bucket by weight then a uniform target in it "
                         "(e.g. '1:3-10,1.5:10-25,0.5:25-38'); overrides --target")
    ap.add_argument("--seed", type=int, default=3407,
                    help="RNG seed for --mix bucket draws (reproducible cuts)")
    ap.add_argument("--max-clip", type=float, default=oo.MAX_CLIP_SECONDS,
                    help=f"hard cap (default {oo.MAX_CLIP_SECONDS} = the "
                         f"max_seq_length token budget in orpheus_owen.py)")
    ap.add_argument("--eval-frac", type=float, default=0.15)
    ap.add_argument("--gain-db", type=float, action="append", default=None,
                    help="static gain per --wav (repeat per file, or give once "
                         "for all): bring each source to -20 LUFS")
    args = ap.parse_args()
    if len(args.words) != len(args.wav):
        raise SystemExit("--words and --wav must be paired (same count/order)")
    gains = args.gain_db or [0.0]
    if len(gains) == 1:
        gains = gains * len(args.wav)
    if len(gains) != len(args.wav):
        raise SystemExit("--gain-db count must be 1 or match --wav count")
    if args.max_clip > oo.MAX_CLIP_SECONDS:
        print(f"[cut] WARNING: --max-clip {args.max_clip}s exceeds the "
              f"{oo.MAX_CLIP_SECONDS}s token budget for max_seq_length=2048; "
              f"raise max_seq_length in orpheus_owen.py train accordingly")

    import random
    import numpy as np
    import librosa
    import soundfile as sf

    if args.mix:
        sampler = make_mix_sampler(args.mix, random.Random(args.seed))
    else:
        sampler = lambda: args.target   # noqa: E731 — uniform mode

    out = Path(args.out_dir)
    out_wavs = out / "wavs"
    out_wavs.mkdir(parents=True, exist_ok=True)

    rows = []           # (rel_wav, text)
    clips_meta = []
    all_leftovers = []
    for wj, wav, g in zip(args.words, args.wav, gains):
        stem = Path(wj).name.replace(".words.json", "")
        _, _, leftovers = cut_file(stem, wav, wj, sampler, args.max_clip,
                                   rows, clips_meta, out_wavs, np, librosa, sf,
                                   gain_db=g)
        all_leftovers.extend(leftovers)

    if not rows:
        raise SystemExit("no clips emitted — check inputs")

    # evenly-spread eval split (variety beats a contiguous tail-slice)
    n_eval = max(1, int(round(len(rows) * args.eval_frac)))
    eval_idx = set(int(round(x)) for x in
                   np.linspace(0, len(rows) - 1, n_eval))
    def write_csv(path, idxs):
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("audio_file|text|speaker_name\n")
            for i in idxs:
                rel, txt = rows[i]
                f.write(f"{rel}|{txt.replace('|', '/')}|{args.source_name}\n")
    write_csv(out / "metadata_train.csv",
              [i for i in range(len(rows)) if i not in eval_idx])
    write_csv(out / "metadata_eval.csv", sorted(eval_idx))
    json.dump(clips_meta, open(out / "clips.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    durs = [c["duration"] for c in clips_meta]
    wfracs = [c["whisper_frac"] for c in clips_meta]
    hist = {}
    for d in durs:
        b = f"{int(d // 5) * 5}-{int(d // 5) * 5 + 5}s"
        hist[b] = hist.get(b, 0) + 1
    lines = [
        "# Cut report", "",
        f"- Clips: **{len(durs)}**, total **{sum(durs)/60:.1f} min**",
        f"- Duration: min {min(durs):.1f}s / median "
        f"{sorted(durs)[len(durs)//2]:.1f}s / max {max(durs):.1f}s "
        f"({'mix ' + args.mix if args.mix else f'target {args.target}s'}, "
        f"cap {args.max_clip}s, seed {args.seed})",
        f"- Duration histogram: " + ", ".join(
            f"{k}: {v}" for k, v in sorted(hist.items())),
        f"- Sample rate: {SR} Hz mono (Orpheus/SNAC)",
        f"- Whisper-only text: {sum(1 for w in wfracs if w > 0)} clips contain "
        f"ASR-sourced words; {sum(1 for w in wfracs if w >= 0.5)} are "
        f"majority-ASR (filter via clips.json whisper_frac if desired)",
        f"- Train/eval: {len(rows)-n_eval}/{n_eval}",
        f"- Unexpanded digit tokens: "
        f"{sorted(set(all_leftovers)) if all_leftovers else 'none'}",
        "", "SFX keep-out windows were hard walls: no clip overlaps one.",
    ]
    (out / "CUT_REPORT.md").write_text("\n".join(lines) + "\n",
                                       encoding="utf-8")
    print(f"[cut] dataset written: {out} "
          f"({len(durs)} clips, {sum(durs)/60:.1f} min)")


if __name__ == "__main__":
    main()

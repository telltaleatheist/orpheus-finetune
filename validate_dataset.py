#!/usr/bin/env python3
r"""validate_dataset.py — Stage-6 sanity check on a recut Orpheus dataset.

Confirms the spec the model will faithfully learn: 24 kHz mono, in-bounds durations,
SILENT clip edges (the proxy for "no half-cut words/breaths"), and coherent text.
Flags clips whose text contains digit tokens (Whisper wrote "1937" not spoken form).
Exports a sample of clips to a Windows-visible dir for the human ear-check.

Run in env `orpheus_ft`. Read-only except for the exported sample copies.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def read_rows(ds_dir: Path):
    rows = []
    for name in ("metadata_train.csv", "metadata_eval.csv"):
        with (ds_dir / name).open(encoding="utf-8") as fh:
            r = csv.reader(fh, delimiter="|")
            next(r)
            for row in r:
                if row:
                    rows.append((name.split("_")[1].split(".")[0], row[0], row[1]))
    return rows  # (split, rel_wav, text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds-dir", required=True)
    ap.add_argument("--sample-out", default=None, help="copy sample clips here for ear-check")
    ap.add_argument("--sample-n", type=int, default=8)
    ap.add_argument("--edge-ms", type=float, default=15.0)
    ap.add_argument("--edge-rms-max", type=float, default=0.02)
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf

    ds = Path(args.ds_dir)
    rows = read_rows(ds)
    print(f"[validate] {len(rows)} clips in {ds}")

    durs, bad_sr, bad_ch, bad_edge, digit_clips = [], [], [], [], []
    by_src = {}
    for split, rel, text in rows:
        wav = ds / rel
        src = rel.replace("wavs/", "").rsplit("_", 1)[0]   # e.g. hq_vid1
        by_src.setdefault(src, []).append((rel, text))
        info = sf.info(str(wav))
        secs = info.frames / info.samplerate
        durs.append(secs)
        if info.samplerate != 24000:
            bad_sr.append(rel)
        if info.channels != 1:
            bad_ch.append(rel)
        # edge silence: read head+tail windows
        w = int(args.edge_ms / 1000 * info.samplerate)
        y, _ = sf.read(str(wav), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        head_rms = float(np.sqrt(np.mean(y[:w] ** 2))) if len(y) >= w else 1.0
        tail_rms = float(np.sqrt(np.mean(y[-w:] ** 2))) if len(y) >= w else 1.0
        if head_rms > args.edge_rms_max or tail_rms > args.edge_rms_max:
            bad_edge.append((rel, round(head_rms, 3), round(tail_rms, 3)))
        if any(ch.isdigit() for ch in text):
            digit_clips.append(rel)

    d = np.array(durs)
    print("\n=== DURATION ===")
    print(f"  count {len(d)}  total {d.sum()/3600:.2f} h")
    print(f"  min {d.min():.2f}s  median {np.median(d):.2f}s  mean {d.mean():.2f}s  max {d.max():.2f}s")
    print("\n=== PER SOURCE ===")
    for src in sorted(by_src):
        print(f"  {src:12s} {len(by_src[src]):4d} clips")
    print("\n=== SPEC CONFORMANCE ===")
    print(f"  non-24k .............. {len(bad_sr)}")
    print(f"  non-mono ............. {len(bad_ch)}")
    print(f"  noisy edges (>{args.edge_rms_max} rms) .. {len(bad_edge)}  (proxy for half-cut words)")
    for rel, h, t in bad_edge[:5]:
        print(f"      {rel}  head={h} tail={t}")
    print(f"  clips w/ digit tokens  {len(digit_clips)}  "
          f"(Whisper digits != spoken form; review if many)")

    print("\n=== TEXT SAMPLE (first words of 8 clips) ===")
    for split, rel, text in rows[:8]:
        print(f"  [{rel.replace('wavs/','')}] {text[:90]}")

    if args.sample_out:
        out = Path(args.sample_out)
        out.mkdir(parents=True, exist_ok=True)
        # spread the sample across sources
        picks = []
        srcs = sorted(by_src)
        i = 0
        while len(picks) < args.sample_n and any(by_src.values()):
            s = srcs[i % len(srcs)]
            if by_src[s]:
                picks.append(by_src[s].pop(len(by_src[s]) // 2))
            i += 1
        for rel, text in picks:
            shutil.copy(ds / rel, out / Path(rel).name)
        (out / "_texts.txt").write_text(
            "\n".join(f"{Path(r).name}\t{t}" for r, t in picks), encoding="utf-8")
        print(f"\n[validate] copied {len(picks)} sample clips -> {out}  (texts in _texts.txt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

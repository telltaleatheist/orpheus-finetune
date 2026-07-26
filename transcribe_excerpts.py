#!/usr/bin/env python3
r"""transcribe_excerpts.py — faster-whisper word-timestamp transcription of the
Deathstalker GraphicAudio training excerpts (honor / legacy), for the book-anchored
alignment step (align_excerpts.py).

Whisper here is NOT the truth (align_excerpts.py anchors it to the epub); it only
provides the word timeline. VAD is OFF so no spoken words are dropped, and
word_timestamps=True. Run in the `whisperx` conda env (faster_whisper 1.2.1 +
ctranslate2 4.8.1 CUDA). ctranslate2 uses cuDNN/cuBLAS, not torch — safe to run
standalone (never in the same process as a torch model).

Output (per input <stem>.wav): <out-dir>/<stem>.whisper.json
  [{"word","start","end","prob"}]  raw word stream, ordered.

Usage:
  python transcribe_excerpts.py --in-file "<wav>" [...] --out-dir <dir>
                                [--model large-v3] [--device cuda]
                                [--compute-type float16]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _add_cuda_dll_dirs() -> None:
    """On Windows, faster-whisper's ctranslate2 needs the env's bundled
    nvidia cuDNN/cuBLAS DLLs on the DLL search path."""
    if os.name != "nt":
        return
    base = Path(sys.executable).parent  # env root (python.exe lives here)
    for sub in ("Library/bin",):
        p = base / sub
        if p.is_dir():
            os.add_dll_directory(str(p))
    site = base / "Lib" / "site-packages" / "nvidia"
    if site.is_dir():
        for d in site.rglob("bin"):
            if d.is_dir():
                try:
                    os.add_dll_directory(str(d))
                except OSError:
                    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-file", action="append", required=True,
                    help="source .wav (repeatable)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--language", default="en")
    ap.add_argument("--beam-size", type=int, default=5)
    args = ap.parse_args()

    _add_cuda_dll_dirs()
    from faster_whisper import WhisperModel

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[whisper] loading {args.model} on {args.device} ({args.compute_type})",
          flush=True)
    model = WhisperModel(args.model, device=args.device,
                         compute_type=args.compute_type)

    for wav in args.in_file:
        stem = Path(wav).stem
        out = out_dir / f"{stem}.whisper.json"
        if out.exists():
            print(f"[whisper] skip {stem} (exists)")
            continue
        print(f"[whisper] transcribing {stem} ...", flush=True)
        segments, info = model.transcribe(
            wav,
            language=args.language,
            word_timestamps=True,
            vad_filter=False,             # want full continuous coverage
            beam_size=args.beam_size,
            condition_on_previous_text=True,
        )
        words: list[dict] = []
        nseg = 0
        for seg in segments:              # generator: work happens here
            nseg += 1
            for w in (seg.words or []):
                tok = (w.word or "").strip()
                if not tok:
                    continue
                words.append({
                    "word": tok,
                    "start": round(float(w.start), 3),
                    "end": round(float(w.end), 3),
                    "prob": round(float(w.probability), 3),
                })
            if nseg % 100 == 0:
                print(f"    {stem}: {nseg} segs, {len(words)} words, "
                      f"t={seg.end:.0f}s", flush=True)
        out.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
        dur = words[-1]["end"] if words else 0.0
        print(f"[whisper] {stem}: {len(words)} words / {nseg} segs / "
              f"{dur/60:.1f} min -> {out.name}", flush=True)

    print("[whisper] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

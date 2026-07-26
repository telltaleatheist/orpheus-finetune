#!/usr/bin/env python3
r"""transcribe_whisper.py — faster-whisper word timestamps -> aligned.json (recut input).

For cloning voices that have NO reliable source text (e.g. YouTube audiobook narrations
where we don't have the matching ebook edition). Unlike the Owen pipeline — which aligns
epub-truth text to Whisper timing — here **Whisper IS the truth**. That's the pragmatic
ASR fallback from the curate-tts-dataset skill: accept Whisper's name/number errors
(Orpheus only imprints timbre, so minor text noise is tolerable), and let the silence-snapped
`orpheus_owen.py recut` place the actual cut points (never at these word timestamps).

Emits, per input <stem>.wav:
  <out>/<stem>.whisper.json : [{"word","start","end","prob"}]      (raw, for QA)
  <out>/<stem>.aligned.json : [{"w","start","end","corr":false}]   (the recut input format)

Run in WSL env `xttstrainer` (has faster_whisper + CUDA). faster-whisper needs the env's
bundled cuDNN/cuBLAS on LD_LIBRARY_PATH (the known cuDNN gotcha) — the run_*.sh wrapper
sets it. CPU fallback: --device cpu --compute-type int8 (much slower).

Cut points are decided later by recut's silence detection, so VAD is OFF here: we want
full, continuous word coverage of the narration, not VAD-dropped spans.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="faster-whisper -> aligned.json (recut input)")
    ap.add_argument("--in-dir", required=True, help="dir of source .wav narrations")
    ap.add_argument("--out-dir", required=True, help="dir to write *.whisper.json / *.aligned.json")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16", help="cuda: float16; cpu: int8")
    ap.add_argument("--language", default="en")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--only", default=None, help="only process stems containing this substring")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(glob.glob(str(in_dir / "*.wav")))
    if args.only:
        wavs = [w for w in wavs if args.only in Path(w).stem]
    if not wavs:
        print(f"[whisper] no wavs in {in_dir}")
        return 1

    print(f"[whisper] loading {args.model} on {args.device} ({args.compute_type})", flush=True)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    for wav in wavs:
        stem = Path(wav).stem
        out_aligned = out_dir / f"{stem}.aligned.json"
        if out_aligned.exists():
            print(f"[whisper] skip {stem} (already transcribed)")
            continue
        print(f"[whisper] transcribing {stem} ...", flush=True)
        segments, info = model.transcribe(
            wav,
            language=args.language,
            word_timestamps=True,
            vad_filter=False,                  # narration is continuous; want full coverage
            beam_size=args.beam_size,
            condition_on_previous_text=True,
        )
        words_raw: list[dict] = []
        words_aligned: list[dict] = []
        nseg = 0
        for seg in segments:                   # generator: transcription happens here
            nseg += 1
            for w in (seg.words or []):
                tok = (w.word or "").strip()
                if not tok:
                    continue
                s, e = round(float(w.start), 3), round(float(w.end), 3)
                words_raw.append({"word": tok, "start": s, "end": e,
                                  "prob": round(float(w.probability), 3)})
                words_aligned.append({"w": tok, "start": s, "end": e, "corr": False})
            if nseg % 100 == 0:
                print(f"    {stem}: {nseg} segs, {len(words_aligned)} words, t={seg.end:.0f}s",
                      flush=True)

        (out_dir / f"{stem}.whisper.json").write_text(
            json.dumps(words_raw, ensure_ascii=False))
        out_aligned.write_text(json.dumps(words_aligned, ensure_ascii=False))

        # QA flag: tokens containing digits won't match spoken form (Whisper writes
        # "1990" not "nineteen ninety"). Rare in fiction; surfaced so we can decide.
        digits = [w["w"] for w in words_aligned if any(ch.isdigit() for ch in w["w"])]
        dur = words_aligned[-1]["end"] if words_aligned else 0.0
        note = (f"  [WARN {len(digits)} digit-tokens, e.g. {digits[:8]}]" if digits else "")
        print(f"[whisper] {stem}: {len(words_aligned)} words / {nseg} segs / "
              f"{dur/60:.1f} min{note}", flush=True)

    print("[whisper] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

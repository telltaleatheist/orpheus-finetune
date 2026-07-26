"""Collapse YouTube auto-caption VTTs into plain transcript text.

YouTube's rolling captions repeat each line across consecutive cues and wrap
words in <c>/timestamp tags. This strips tags, dedupes the rolling repeats,
and writes one .txt per video, capped so a multi-hour stream still yields a
summarizable chunk.

Input : E:/training/titles/source/captions/<videoId>.en.vtt
Output: E:/training/titles/source/captions/text/<videoId>.txt   (resumable)
"""

from __future__ import annotations

import re
from pathlib import Path

CAPTIONS_DIR = Path("E:/training/titles/source/captions")
TEXT_DIR = CAPTIONS_DIR / "text"
MAX_CHARS = 30000  # ~7.5k tokens; plenty to capture what a video is about

TAG_RE = re.compile(r"<[^>]+>")
TIMESTAMP_LINE_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3} --> ")


def vtt_to_text(vtt: str) -> str:
    lines: list[str] = []
    for raw_line in vtt.splitlines():
        line = raw_line.strip()
        if (not line or line == "WEBVTT" or TIMESTAMP_LINE_RE.match(line)
                or line.startswith(("Kind:", "Language:", "NOTE", "align:"))):
            continue
        text = TAG_RE.sub("", line).strip()
        if not text:
            continue
        if lines and (text == lines[-1] or lines[-1].endswith(text)):
            continue
        lines.append(text)
    return " ".join(lines)[:MAX_CHARS]


def main() -> None:
    TEXT_DIR.mkdir(exist_ok=True)
    converted = skipped = empty = 0
    for vtt_path in sorted(CAPTIONS_DIR.glob("*.en.vtt")):
        video_id = vtt_path.name.removesuffix(".en.vtt")
        out_path = TEXT_DIR / f"{video_id}.txt"
        if out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
            continue
        text = vtt_to_text(vtt_path.read_text(encoding="utf-8"))
        if len(text) < 200:
            empty += 1
            continue
        out_path.write_text(text, encoding="utf-8")
        converted += 1
    print(f"converted {converted}, already done {skipped}, too-short {empty}")


if __name__ == "__main__":
    main()

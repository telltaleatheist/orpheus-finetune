"""Audit the backfill summaries for agent-written junk.

A summary agent that cannot read a transcript is supposed to skip the video. Some
instead wrote a placeholder file ("transcript appears to be missing..."), which both
marks the video done and would feed meta-text into the dataset as if it described the
video. This finds those, plus summaries that are too short or that leaked meta-commentary.

--delete removes the offenders so the videos re-enter the summary-wave todo list.
"""

from __future__ import annotations

import argparse
import re

from config import TITLES_ROOT

SUMMARIES = TITLES_ROOT / "source/captions/summaries"
TEXT = TITLES_ROOT / "source/captions/text"

# A short summary is only suspicious when the transcript had plenty to work with:
# a 2-line description of a 40-second clip is correct, of a 30 KB stream it is a failure.
MIN_CHARS = 150
SUBSTANTIAL_TRANSCRIPT_BYTES = 3000

# Meta-commentary about the transcript/file rather than about the video's subject.
JUNK_RE = re.compile(
    r"transcript (file )?(appears to be |is )?(missing|empty|corrupt|unavailable|unreadable)"
    r"|insufficient content"
    r"|unable to (read|access|generate|process)"
    r"|no (usable |coherent )?(content|transcript) (found|available)"
    r"|cannot (be )?(read|access|summari[sz]e)"
    r"|(file|transcript) (not found|does not exist)"
    r"|i (was )?(could not|couldn't|cannot|can't)",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()

    junk: list[tuple[str, str, int]] = []
    for path in sorted(SUMMARIES.glob("*.txt")):
        body = path.read_text(encoding="utf-8").strip()
        source_chars = (TEXT / path.name).stat().st_size if (TEXT / path.name).exists() else 0
        if JUNK_RE.search(body):
            junk.append((path.stem, "meta-commentary", source_chars))
        elif len(body) < MIN_CHARS and source_chars >= SUBSTANTIAL_TRANSCRIPT_BYTES:
            junk.append((path.stem, f"too short ({len(body)} chars)", source_chars))

    for video_id, reason, source_chars in junk:
        print(f"{video_id}\t{reason}\ttranscript={source_chars}B")
    print(f"\n{len(junk)} junk of {len(list(SUMMARIES.glob('*.txt')))} summaries")

    if args.delete:
        for video_id, _, _ in junk:
            (SUMMARIES / f"{video_id}.txt").unlink()
        print(f"deleted {len(junk)}; they re-enter the todo list")


if __name__ == "__main__":
    main()

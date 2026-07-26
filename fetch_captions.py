"""
Resumable YouTube auto-caption sweep.

Reads E:\\training\\titles\\build\\backfill_needed.json (array of objects with
videoId/channel/format/publishedAt/title), and for each videoId fetches the
English auto-generated captions VTT via yt-dlp into
E:\\training\\titles\\source\\captions\\<videoId>.en.vtt.

Resumable: any videoId that already has a non-empty .en.vtt is skipped.
State is rewritten after every video to sweep_state.json, and a log line is
appended after every video to sweep.log.

Python 3.11, stdlib only. All file I/O uses encoding="utf-8" explicitly.
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

INPUT_JSON = Path(r"E:\training\titles\build\backfill_needed.json")
OUT_DIR = Path(r"E:\training\titles\source\captions")
STATE_PATH = OUT_DIR / "sweep_state.json"
LOG_PATH = OUT_DIR / "sweep.log"

YT_DLP = r"C:\Users\tellt\AppData\Local\Programs\Python\Python311\Scripts\yt-dlp.exe"

PER_VIDEO_TIMEOUT_S = 120
SLEEP_BETWEEN_S = 2
CONSEC_FAIL_SOFT_LIMIT = 5
CONSEC_FAIL_SOFT_SLEEP_S = 60
CONSEC_FAIL_HARD_LIMIT = 25


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_line(line: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_state(done: int, failed: list, remaining: int, last_video_id: str) -> None:
    state = {
        "done": done,
        "failed": failed,
        "remaining": remaining,
        "lastVideoId": last_video_id,
        "updatedAt": utc_now_iso(),
    }
    tmp_path = STATE_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp_path.replace(STATE_PATH)


def load_backfill_needed() -> list:
    if not INPUT_JSON.exists():
        sys.exit(f"FATAL: input file does not exist: {INPUT_JSON}")
    try:
        with open(INPUT_JSON, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"FATAL: input file is not valid JSON: {INPUT_JSON} ({e})")
    if not isinstance(data, list):
        sys.exit(f"FATAL: input JSON root is not a list: {INPUT_JSON}")
    for i, item in enumerate(data):
        for field in ("videoId", "channel", "format", "publishedAt", "title"):
            if field not in item:
                sys.exit(
                    f"FATAL: item {i} in {INPUT_JSON} is missing required field "
                    f"'{field}': {item}"
                )
    return data


def existing_vtt_ok(video_id: str) -> bool:
    vtt_path = OUT_DIR / f"{video_id}.en.vtt"
    return vtt_path.exists() and vtt_path.stat().st_size > 0


def fetch_one(video_id: str) -> tuple:
    """Returns (ok: bool, error: str or None)."""
    url = f"https://youtu.be/{video_id}"
    out_template = str(OUT_DIR / "%(id)s")
    cmd = [
        YT_DLP,
        "--skip-download",
        "--write-auto-subs",
        "--sub-langs",
        "en",
        "--sub-format",
        "vtt",
        "-o",
        out_template,
        url,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=PER_VIDEO_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"

    if result.returncode != 0:
        stderr = result.stderr or ""
        return False, stderr[-300:] if stderr else f"nonzero exit {result.returncode}"

    if not existing_vtt_ok(video_id):
        return False, "no-vtt-produced"

    return True, None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    items = load_backfill_needed()
    items_sorted = sorted(items, key=lambda x: x["publishedAt"])

    done = 0
    failed = []
    consecutive_failures = 0
    total = len(items_sorted)

    for idx, item in enumerate(items_sorted):
        video_id = item["videoId"]
        channel = item["channel"]
        title = item["title"]

        if existing_vtt_ok(video_id):
            done += 1
            log_line(f"SKIP {video_id} already-present")
            remaining = total - (idx + 1)
            write_state(done, failed, remaining, video_id)
            continue

        ok, error = fetch_one(video_id)

        if ok:
            done += 1
            consecutive_failures = 0
            log_line(f"OK {video_id}")
        else:
            consecutive_failures += 1
            record = {
                "videoId": video_id,
                "channel": channel,
                "title": title,
                "error": error,
            }
            failed.append(record)
            short_reason = (error or "").replace("\n", " ")[:150]
            log_line(f"FAIL {video_id} {short_reason}")

        remaining = total - (idx + 1)
        write_state(done, failed, remaining, video_id)

        if consecutive_failures >= CONSEC_FAIL_HARD_LIMIT:
            log_line(
                f"ABORT: {consecutive_failures} consecutive failures "
                f"(hard limit {CONSEC_FAIL_HARD_LIMIT}); stopping sweep."
            )
            write_state(done, failed, remaining, video_id)
            sys.exit(2)

        if consecutive_failures >= CONSEC_FAIL_SOFT_LIMIT:
            log_line(
                f"THROTTLE: {consecutive_failures} consecutive failures; "
                f"sleeping {CONSEC_FAIL_SOFT_SLEEP_S}s"
            )
            time.sleep(CONSEC_FAIL_SOFT_SLEEP_S)

        time.sleep(SLEEP_BETWEEN_S)

    log_line(f"DONE sweep complete: {done} succeeded, {len(failed)} failed")


if __name__ == "__main__":
    main()

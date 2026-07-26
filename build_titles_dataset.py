"""Build the qwen3_titles chat-JSONL dataset from the ContentStudio analytics corpus.

Input  ($ORPHEUS_TITLES_ROOT/source): channels.json + <channelId>/{videos,verdicts,descriptions}.json
Output ($ORPHEUS_TITLES_ROOT/build):  train.jsonl / eval.jsonl (chronological split),
                                    stats.json, backfill_needed.json

Each example:
  system: fixed title-writing doctrine
  user:   channel + format + performance-tier target + cleaned description content
  assistant: the actually-published title

Tier = within-(channel,format) percentile of lifetime CTR, only where lifetime
impressions >= MIN_IMPRESSIONS; everything else is tier "unrated" (style-only
signal, no fabricated performance label). At inference always request top-decile.

Descriptions are reduced to their content portion: URL/promo boilerplate lines are
dropped and a leading title-echo line is stripped (verbatim title in the input
would teach copy-through). Videos whose content portion is too short to describe
the video go to backfill_needed.json (candidates for transcript-based summaries)
and are excluded from training.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from config import TITLES_ROOT

SOURCE_DIR = TITLES_ROOT / "source"
BUILD_DIR = TITLES_ROOT / "build"

MIN_IMPRESSIONS = 1000        # below this, lifetime CTR is noise -> tier "unrated"
MIN_CONTENT_CHARS = 60        # cleaned description shorter than this = junk input
MAX_CONTENT_CHARS = 2200      # keep prompts comfortably inside max_seq_length 2048
EVAL_FRACTION = 0.08          # newest slice held out, chronological

CHANNEL_SLUGS = {
    "UCgIi12EA6BQ8HKL8QUccsOQ": "telltale",
    "UCo6JSNp6SuUKf-yiaBQReNA": "fireside",
    "UCOB86WpguzlOEs4z93iZ7kA": "unfiltered",
}

# Shorts, long-form and livestreams are different products on different algorithms, and
# the corpus shows it: median title length 45 / 53 / 60 chars, median CTR 2.6% / 4.6% / 4.7%,
# and shorts lean on questions where long-form leans on ALL-CAPS emphasis. One doctrine for
# all three would train shorts to pad toward a long-form length their own winners don't use
# ("To Carly", "Why I don't debate"). Tiers are already computed within (channel, format) so
# the CTR scales never mix; this makes the STYLE rules match too.
SYSTEM_PROMPT_BASE = (
    "You write YouTube titles for Owen Morgan's channels (telltale, fireside, "
    "unfiltered). Given a description of a video, write one title. Name names; "
    "plain concrete language, no corporate phrasing; be the prosecutor, not the "
    "journalist - state what happened and why it matters, don't hedge. Specificity "
    "plus an open loop beats vague drama."
)

FORMAT_RULES = {
    "long": (
        " This is a long-form upload: the hook lands inside the first 45 characters and "
        "the whole title runs 45-70 characters."
    ),
    "short": (
        " This is a YouTube Short, which is browsed and swiped rather than searched: keep "
        "it tight, roughly 25-60 characters, one idea only. A blunt statement or a direct "
        "question both work; do not pad it to long-form length."
    ),
    "live": (
        " This is a livestream, often covering several topics: 45-90 characters, and it is "
        "fine to name two or three subjects the stream covers."
    ),
}


def system_prompt_for(video_format: str) -> str:
    return SYSTEM_PROMPT_BASE + FORMAT_RULES[video_format]


# 55% of Shorts titles carry a "#shorts" tag. It is 100% predictable from the format, so
# training the model to emit it teaches nothing and risks emitting it only half the time.
# Strip it from the target; whatever consumes the model appends it when format == short.
SHORTS_TAG_RE = re.compile(r"\s*#shorts?\b", re.IGNORECASE)

# A line is boilerplate when it is a link, a social/promo label, or a plug.
BOILERPLATE_RE = re.compile(
    r"(https?://|www\.)"
    r"|^\s*(become a (youtube |channel )?member|patreon|twitter|twitch|facebook"
    r"|discord|tiktok|instagram|telltale( fireside chat| unfiltered| reads| u)?"
    r"|owen'?s? fireside chat|social media|subscribe( to the email list)?"
    r"|email list|get my book|check out my book|check out my books"
    r"|broadcasted live on twitch|merch|paypal|donate|business inquiries|voicemail)\b",
    re.IGNORECASE,
)

TIMESTAMP_LINE_RE = re.compile(r"^\s*\d{1,2}:\d{2}")

# Trailing episode/series-position markers ("| P3", "- Part 2", "| Podcast 276 Clip",
# bare "Part 4") are upload bookkeeping, not title craft — strip them from targets.
# Named series tags ("| Delusion Roundup") are deliberate style and stay.
EPISODE_MARKER_RE = re.compile(
    r"\s*(?:[|\-–]\s*)?(?:(?:P(?:ar)?t\.?|Podcast|Episode|Ep\.?|#)\s*\d+[^|]*|P\d+)\s*$",
    re.IGNORECASE,
)


def clean_title(title: str) -> str:
    title = SHORTS_TAG_RE.sub("", title)
    while True:
        stripped = EPISODE_MARKER_RE.sub("", title).rstrip(" |-–")
        if stripped == title or not stripped:
            return title.strip()
        title = stripped


def clean_description(description: str, title: str) -> str:
    """Reduce a published description to the portion that describes the video."""
    kept: list[str] = []
    for line in description.splitlines():
        if BOILERPLATE_RE.search(line):
            continue
        kept.append(line.rstrip())
    # Collapse blank runs left behind by removed blocks.
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    # Strip a leading title echo (older uploads repeat the title as line one).
    first, _, rest = text.partition("\n")
    normalized_first = re.sub(r"\W+", " ", first).strip().lower()
    normalized_title = re.sub(r"\W+", " ", title).strip().lower()
    if normalized_first and normalized_title and (
        normalized_first == normalized_title
        or normalized_first.startswith(normalized_title)
        or normalized_title.startswith(normalized_first)
    ):
        text = rest.strip()
    return text[:MAX_CONTENT_CHARS].strip()


def tier_for(percentile: float) -> str:
    if percentile >= 90:
        return "top-decile"
    if percentile >= 70:
        return "strong"
    if percentile >= 30:
        return "typical"
    return "weak"


def load_channel(channel_id: str) -> list[dict]:
    chan_dir = SOURCE_DIR / channel_id
    videos = {v["videoId"]: v for v in json.loads((chan_dir / "videos.json").read_text(encoding="utf-8"))}
    verdicts = json.loads((chan_dir / "verdicts.json").read_text(encoding="utf-8"))
    descriptions = json.loads((chan_dir / "descriptions.json").read_text(encoding="utf-8"))["fetched"]
    rows = []
    for verdict in verdicts:
        video = videos[verdict["videoId"]]
        rows.append({
            "videoId": verdict["videoId"],
            "channel": CHANNEL_SLUGS[channel_id],
            "format": video["format"],
            "publishedAt": video["publishedAt"],
            "titles": verdict["titles"],
            "lifetime": verdict["lifetime"],
            "snippet": descriptions.get(verdict["videoId"]),
        })
    return rows


def main() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for channel_id in CHANNEL_SLUGS:
        rows.extend(load_channel(channel_id))

    stats = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "totalVideos": len(rows),
        "skipped": {"noSnippet": 0, "noTitle": 0, "junkContent": 0},
        "tiers": {},
        "byChannelFormat": {},
        "renamedVideosUsingCurrentTitle": 0,
    }
    backfill: list[dict] = []

    # Within-(channel,format) CTR percentiles over the rated population.
    groups: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        lifetime = row["lifetime"]
        rated = (
            lifetime is not None
            and lifetime.get("ctr") is not None
            and (lifetime.get("impressions") or 0) >= MIN_IMPRESSIONS
        )
        row["ratedCtr"] = lifetime["ctr"] if rated else None
        if rated:
            groups.setdefault((row["channel"], row["format"]), []).append(lifetime["ctr"])
    for values in groups.values():
        values.sort()

    examples: list[dict] = []
    for row in rows:
        if row["snippet"] is None:
            stats["skipped"]["noSnippet"] += 1
            continue
        if not row["titles"]:
            stats["skipped"]["noTitle"] += 1
            continue
        title = clean_title(row["titles"][-1])  # current title; lifetime metrics accrued mostly here
        if len(row["titles"]) > 1:
            stats["renamedVideosUsingCurrentTitle"] += 1
        content = clean_description(row["snippet"]["description"], title)
        if len(content) < MIN_CONTENT_CHARS:
            stats["skipped"]["junkContent"] += 1
            backfill.append({
                "videoId": row["videoId"],
                "channel": row["channel"],
                "format": row["format"],
                "publishedAt": row["publishedAt"],
                "title": title,
                "cleanedContentChars": len(content),
            })
            continue

        if row["ratedCtr"] is None:
            tier = "unrated"
        else:
            population = groups[(row["channel"], row["format"])]
            below = sum(1 for value in population if value < row["ratedCtr"])
            tier = tier_for(100.0 * below / len(population))
        stats["tiers"][tier] = stats["tiers"].get(tier, 0) + 1
        key = f"{row['channel']}/{row['format']}"
        stats["byChannelFormat"][key] = stats["byChannelFormat"].get(key, 0) + 1

        user = (
            f"channel: {row['channel']}\n"
            f"format: {row['format']}\n"
            f"target: {tier}\n\n"
            f"Video:\n{content}"
        )
        examples.append({
            "publishedAt": row["publishedAt"],
            "messages": [
                {"role": "system", "content": system_prompt_for(row["format"])},
                {"role": "user", "content": user},
                {"role": "assistant", "content": title},
            ],
        })

    examples.sort(key=lambda item: item["publishedAt"])
    eval_count = max(1, round(len(examples) * EVAL_FRACTION))
    train, eval_ = examples[:-eval_count], examples[-eval_count:]
    stats["train"] = len(train)
    stats["eval"] = len(eval_)
    stats["evalCutoffPublishedAt"] = eval_[0]["publishedAt"]

    for name, split in (("train.jsonl", train), ("eval.jsonl", eval_)):
        with (BUILD_DIR / name).open("w", encoding="utf-8") as handle:
            for example in split:
                handle.write(json.dumps({"messages": example["messages"]},
                                        ensure_ascii=False) + "\n")
    (BUILD_DIR / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    (BUILD_DIR / "backfill_needed.json").write_text(json.dumps(backfill, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))
    print(f"wrote {BUILD_DIR / 'train.jsonl'} and eval.jsonl; "
          f"{len(backfill)} videos queued in backfill_needed.json")


if __name__ == "__main__":
    main()

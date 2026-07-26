"""Fetch published snippet (title/description/tags) for every video in the
ContentStudio analytics store, via YouTube Data API videos.list with the API key.

Runs on the Mac. Output: <analytics>/<channelId>/descriptions.json
mapping videoId -> {title, description, tags, fetchedAt}.
Videos the API does not return (private/deleted) are listed in missing[].
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = Path.home() / "Library/Application Support/ContentStudio"
ANALYTICS = BASE / "analytics"

oauth = json.loads((BASE / "youtube-oauth.json").read_text())
api_key = oauth["youtubeApiKey"]
if not api_key:
    raise SystemExit("youtubeApiKey is empty in youtube-oauth.json")

channels = [d for d in ANALYTICS.iterdir() if d.is_dir() and d.name.startswith("UC")]
if not channels:
    raise SystemExit(f"no channel dirs under {ANALYTICS}")

for chan_dir in channels:
    videos = json.loads((chan_dir / "videos.json").read_text())
    ids = [v["videoId"] for v in videos]
    out: dict[str, dict] = {}
    missing: list[str] = []
    for start in range(0, len(ids), 50):
        batch = ids[start:start + 50]
        params = urllib.parse.urlencode({
            "part": "snippet",
            "id": ",".join(batch),
            "maxResults": 50,
            "key": api_key,
        })
        url = f"https://www.googleapis.com/youtube/v3/videos?{params}"
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = json.load(resp)
                break
            except Exception as error:  # noqa: BLE001 - retry then surface
                if attempt == 3:
                    raise SystemExit(f"{chan_dir.name} batch at {start}: {error}")
                time.sleep(2 * (attempt + 1))
        returned = {item["id"]: item["snippet"] for item in data.get("items", [])}
        now = datetime.now(timezone.utc).isoformat()
        for vid in batch:
            snip = returned.get(vid)
            if snip is None:
                missing.append(vid)
                continue
            out[vid] = {
                "title": snip.get("title"),
                "description": snip.get("description"),
                "tags": snip.get("tags", []),
                "fetchedAt": now,
            }
        print(f"{chan_dir.name}: {min(start + 50, len(ids))}/{len(ids)}",
              file=sys.stderr)
    result = {"fetched": out, "missing": missing}
    (chan_dir / "descriptions.json").write_text(json.dumps(result, indent=1))
    print(f"{chan_dir.name}: {len(out)} fetched, {len(missing)} missing "
          f"-> {chan_dir / 'descriptions.json'}")

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


CHANNEL_ID = "UCBvkQFBskQR9NeOoDYR8ckA"
CHANNEL_NAME = "견자희"
MAX_RESULTS = 3
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "youtube-latest.json"


def require_api_key() -> str:
    api_key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY 환경 변수가 비어 있습니다.")
    return api_key


def fetch_json(base_url: str, params: dict[str, str]) -> dict:
    query = urllib.parse.urlencode(params)
    url = f"{base_url}?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "gyeonjahee-fansite-updater/1.0",
        },
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        body = response.read().decode(charset)
        return json.loads(body)


def fetch_uploads_playlist_id(api_key: str) -> str:
    payload = fetch_json(
        "https://www.googleapis.com/youtube/v3/channels",
        {
            "part": "contentDetails",
            "id": CHANNEL_ID,
            "key": api_key,
        },
    )

    items = payload.get("items") or []
    if not items:
        raise RuntimeError("채널 정보를 찾지 못했습니다.")

    playlist_id = (
        items[0]
        .get("contentDetails", {})
        .get("relatedPlaylists", {})
        .get("uploads", "")
        .strip()
    )
    if not playlist_id:
        raise RuntimeError("업로드 재생목록 ID를 찾지 못했습니다.")
    return playlist_id


def normalize_item(item: dict) -> dict:
    snippet = item.get("snippet", {})
    resource = snippet.get("resourceId", {})
    thumbnails = snippet.get("thumbnails", {})
    video_id = resource.get("videoId", "").strip()

    thumbnail = (
        thumbnails.get("maxres")
        or thumbnails.get("standard")
        or thumbnails.get("high")
        or thumbnails.get("medium")
        or thumbnails.get("default")
        or {}
    ).get("url", "")

    if not video_id:
        return {}

    return {
        "video_id": video_id,
        "title": snippet.get("title", "").strip(),
        "description": snippet.get("description", "").strip(),
        "published_at": snippet.get("publishedAt"),
        "thumbnail": thumbnail,
        "url": f"https://www.youtube.com/watch?v={video_id}",
    }


def fetch_latest_uploads(api_key: str, playlist_id: str) -> list[dict]:
    payload = fetch_json(
        "https://www.googleapis.com/youtube/v3/playlistItems",
        {
            "part": "snippet",
            "playlistId": playlist_id,
            "maxResults": str(MAX_RESULTS),
            "key": api_key,
        },
    )

    items = payload.get("items") or []
    normalized = [normalize_item(item) for item in items]
    return [item for item in normalized if item]


def build_payload(items: list[dict]) -> dict:
    return {
        "channel_id": CHANNEL_ID,
        "channel_name": CHANNEL_NAME,
        "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "items": items,
    }


def write_payload(payload: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    try:
        api_key = require_api_key()
        playlist_id = fetch_uploads_playlist_id(api_key)
        items = fetch_latest_uploads(api_key, playlist_id)
        payload = build_payload(items)
        write_payload(payload)
        print(f"Updated {OUTPUT_PATH} with {len(items)} items.")
        return 0
    except Exception as exc:  # pragma: no cover - CLI entry point
        print(f"Failed to update latest YouTube videos: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

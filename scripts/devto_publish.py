#!/usr/bin/env python3
"""Publish a Markdown file with frontmatter to DEV.to via the API.

Usage:
    DEV_TO_KEY=... python scripts/devto_publish.py path/to/post.md

The Markdown file must start with a YAML frontmatter block:

    ---
    title: "..."
    published: true
    description: "..."
    tags: foo, bar, baz
    ---

The script POSTs to https://dev.to/api/articles. Returns the article URL.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    if not text.startswith("---\n"):
        raise SystemExit("file must start with --- frontmatter block")
    rest = text[4:]
    end = rest.find("\n---\n")
    if end == -1:
        raise SystemExit("frontmatter block not closed with ---")
    fm_text = rest[:end]
    body = rest[end + 5 :]
    fm: dict[str, object] = {}
    for line in fm_text.splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip().strip('"')
        if value.lower() in ("true", "false"):
            fm[key.strip()] = value.lower() == "true"
        else:
            fm[key.strip()] = value
    return fm, body


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: devto_publish.py path/to/post.md")
    key = os.environ.get("DEV_TO_KEY") or os.environ.get("DEV_TO_API_KEY")
    if not key:
        raise SystemExit("DEV_TO_KEY env var not set")
    path = sys.argv[1]
    with open(path) as f:
        text = f.read()
    fm, body = parse_frontmatter(text)
    tags = [t.strip() for t in str(fm.get("tags", "")).split(",") if t.strip()]
    payload = {
        "article": {
            "title": fm["title"],
            "body_markdown": text,  # DEV.to accepts the full doc with frontmatter
            "published": bool(fm.get("published", False)),
            "tags": tags[:4],  # DEV.to caps at 4 tags
        }
    }
    if "description" in fm:
        payload["article"]["description"] = fm["description"]
    req = urllib.request.Request(
        "https://dev.to/api/articles",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "api-key": key,
            "content-type": "application/json",
            "accept": "application/vnd.forem.api-v1+json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        msg = e.read().decode()
        raise SystemExit(f"DEV.to API error {e.code}: {msg}")
    print(f"published: {data.get('url') or data.get('canonical_url')}")
    print(f"id: {data.get('id')}")
    print(f"published flag: {data.get('published')}")


if __name__ == "__main__":
    main()

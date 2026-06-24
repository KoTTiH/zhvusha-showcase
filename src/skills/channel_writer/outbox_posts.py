"""Helpers for channel post files emitted by the morning workspace session."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_FRONTMATTER = "---"


def load_channel_post_file(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load a morning outbox channel post with optional YAML frontmatter."""
    raw_text = Path(path).read_text(encoding="utf-8")
    return load_channel_post_text(raw_text)


def load_channel_post_text(text: str) -> tuple[dict[str, Any], str]:
    """Parse optional frontmatter while keeping plain legacy files valid."""
    if not text.startswith(_FRONTMATTER + "\n"):
        return {}, text.rstrip() + "\n"

    rest = text[len(_FRONTMATTER) + 1 :]
    marker = "\n" + _FRONTMATTER
    end = rest.find(marker)
    if end < 0:
        return {}, text.rstrip() + "\n"

    raw_yaml = rest[:end]
    body = rest[end + len(marker) :].lstrip("\n")
    raw = yaml.safe_load(raw_yaml) or {}
    if not isinstance(raw, dict):
        raw = {}
    return raw, body.rstrip() + "\n"


def save_channel_post_file(
    path: str | Path,
    raw: dict[str, Any],
    body: str,
) -> None:
    """Persist a channel post with frontmatter."""
    Path(path).write_text(
        _render_frontmatter(raw) + "\n" + body.rstrip() + "\n",
        encoding="utf-8",
    )


def channel_post_title(*, raw: dict[str, Any], body: str, fallback: str) -> str:
    """Best-effort title for visual jobs from a free-form channel post."""
    raw_title = str(raw.get("title", "")).strip()
    if raw_title:
        return raw_title
    for line in body.splitlines():
        title = line.strip(" #\t")
        if title:
            return title[:140]
    return fallback


def _render_frontmatter(data: dict[str, Any]) -> str:
    return (
        _FRONTMATTER
        + "\n"
        + yaml.safe_dump(data, allow_unicode=True, sort_keys=False).rstrip()
        + "\n"
        + _FRONTMATTER
        + "\n"
    )

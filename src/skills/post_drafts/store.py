"""Workspace store for channel post drafts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path

    from src.skills.post_drafts.models import PostDraft

_FRONTMATTER = "---"


def drafts_dir(workspace_root: Path) -> Path:
    return workspace_root / "channel" / "drafts"


def list_draft_files(workspace_root: Path) -> list[Path]:
    root = drafts_dir(workspace_root)
    if not root.exists():
        return []
    files = [path for path in root.iterdir() if path.suffix == ".md" and path.is_file()]
    files.sort(key=lambda path: path.stat().st_mtime)
    return files


def find_draft_path(workspace_root: Path, slug: str) -> Path | None:
    root = drafts_dir(workspace_root)
    if not root.exists():
        return None
    matches = [
        path
        for path in root.iterdir()
        if path.suffix == ".md" and path.is_file() and path.stem.endswith(f"-{slug}")
    ]
    if matches:
        return matches[0]
    exact = root / f"{slug}.md"
    return exact if exact.is_file() else None


def write_post_draft(workspace_root: Path, draft: PostDraft) -> Path:
    root = drafts_dir(workspace_root)
    root.mkdir(parents=True, exist_ok=True)
    filename = f"{draft.created_at.astimezone(UTC).date().isoformat()}-{draft.slug}.md"
    path = root / filename
    save_draft_raw(path, _draft_frontmatter(draft), draft.text)
    return path


def load_post_draft(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith(_FRONTMATTER + "\n"):
        raise ValueError(f"post draft {path} is missing YAML frontmatter")
    rest = text[len(_FRONTMATTER) + 1 :]
    marker = "\n" + _FRONTMATTER
    end = rest.find(marker)
    if end < 0:
        raise ValueError(f"post draft {path} has unterminated YAML frontmatter")
    raw_text = rest[:end]
    body = rest[end + len(marker) :].lstrip("\n")
    raw = yaml.safe_load(raw_text) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"post draft {path} frontmatter is not a YAML mapping")
    return raw, body.rstrip() + "\n"


def save_draft_raw(path: Path, data: dict[str, Any], body: str) -> None:
    path.write_text(
        _render_frontmatter(data) + "\n" + body.rstrip() + "\n",
        encoding="utf-8",
    )


def mark_draft_published(path: Path, *, message_id: int) -> None:
    raw, body = load_post_draft(path)
    raw["status"] = "published"
    raw["published_at"] = datetime.now(tz=UTC).isoformat()
    raw["message_id"] = message_id
    save_draft_raw(path, raw, body)


def mark_draft_publish_result(
    path: Path,
    *,
    status: str,
    message_id: int,
    media: dict[str, Any] | None = None,
    error: str = "",
) -> None:
    raw, body = load_post_draft(path)
    raw["status"] = status
    raw["published_at"] = datetime.now(tz=UTC).isoformat()
    raw["message_id"] = message_id
    if media is not None:
        raw["media"] = media
    if error:
        raw["publish_error"] = error
    save_draft_raw(path, raw, body)


def _draft_frontmatter(draft: PostDraft) -> dict[str, Any]:
    data: dict[str, Any] = {
        "slug": draft.slug,
        "title": draft.title,
        "created_at": draft.created_at.isoformat(),
        "status": draft.status,
        "source_cluster": draft.source_cluster,
        "pillar_alignment": draft.pillar_alignment,
    }
    if draft.message_id is not None:
        data["message_id"] = draft.message_id
    if draft.visual is not None:
        data["visual"] = draft.visual
    if draft.style is not None:
        data["style"] = draft.style
    return data


def _render_frontmatter(data: dict[str, Any]) -> str:
    return (
        _FRONTMATTER
        + "\n"
        + yaml.safe_dump(data, allow_unicode=True, sort_keys=False).rstrip()
        + "\n"
        + _FRONTMATTER
        + "\n"
    )

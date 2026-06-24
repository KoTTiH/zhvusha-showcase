"""Style checks for channel post drafts before publication."""

from __future__ import annotations

import re
from typing import Any

_SERVICE_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"никита\s*:.+|"
    r"новый\s+план|"
    r"служебный\s+план|"
    r"✅\s*опубликовано.+|"
    r"опубликовано\s+в\s+канал.+"
    r")\s*$",
    re.IGNORECASE,
)


def clean_draft_text(text: str) -> tuple[str, tuple[str, ...]]:
    """Strip service-only headings without touching the author's body."""
    lines = text.splitlines()
    notes: list[str] = []
    while lines and _SERVICE_HEADING_RE.match(lines[0]):
        notes.append("service_heading")
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    cleaned = "\n".join(lines).strip()
    return cleaned, tuple(notes)


def check_post_style(
    text: str,
    *,
    extra_notes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return frontmatter-safe style metadata for a draft."""
    warnings: list[str] = []
    warnings.extend(extra_notes)

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if any(len(paragraph) > 900 for paragraph in paragraphs):
        warnings.append("wall_of_text")
    if text.count("\n\n") < 2 and len(text) > 1200:
        warnings.append("wall_of_text")

    unique_warnings = list(dict.fromkeys(warnings))
    return {
        "status": "needs_review" if unique_warnings else "ok",
        "warnings": unique_warnings,
    }

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

import structlog
import yaml

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger()


async def save_published_post(
    workspace_root: Path,
    text: str,
    message_id: int,
    *,
    post_date: date | None = None,
    visual: dict[str, Any] | None = None,
    media: dict[str, Any] | None = None,
) -> Path:
    """Save a published channel post to the workspace archive."""
    if post_date is None:
        post_date = datetime.now(tz=UTC).date()

    posts_dir = workspace_root / "channel" / "posts"
    posts_dir.mkdir(parents=True, exist_ok=True)

    date_str = post_date.isoformat()
    sequence = len(list(posts_dir.glob(f"{date_str}_*.md"))) + 1
    filename = f"{date_str}_{sequence}.md"
    file_path = posts_dir / filename

    optional_frontmatter: dict[str, Any] = {}
    if visual is not None:
        optional_frontmatter["visual"] = visual
    if media is not None:
        optional_frontmatter["media"] = media
    optional_yaml = ""
    if optional_frontmatter:
        optional_yaml = (
            yaml.safe_dump(
                optional_frontmatter,
                allow_unicode=True,
                sort_keys=False,
            ).rstrip()
            + "\n"
        )

    content = (
        f"---\n"
        f"date: {date_str}\n"
        f"message_id: {message_id}\n"
        f"reactions: 0\n"
        f"{optional_yaml}"
        f"---\n\n"
        f"{text}\n"
    )
    file_path.write_text(content, encoding="utf-8")

    logger.info("channel_post_archived", path=str(file_path), message_id=message_id)
    return file_path

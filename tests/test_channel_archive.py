"""Tests for channel post archiving."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from src.skills.channel_writer.archive import save_published_post


async def test_save_creates_file(tmp_path: Path) -> None:
    post_date = date(2026, 4, 1)
    path = await save_published_post(
        workspace_root=tmp_path,
        text="Hello channel!",
        message_id=42,
        post_date=post_date,
    )

    assert path.exists()
    assert path.name == "2026-04-01_1.md"
    content = path.read_text(encoding="utf-8")
    assert "date: 2026-04-01" in content
    assert "message_id: 42" in content
    assert "reactions: 0" in content
    assert "Hello channel!" in content


async def test_save_increments_sequence(tmp_path: Path) -> None:
    post_date = date(2026, 4, 1)

    p1 = await save_published_post(
        workspace_root=tmp_path,
        text="First post",
        message_id=10,
        post_date=post_date,
    )
    p2 = await save_published_post(
        workspace_root=tmp_path,
        text="Second post",
        message_id=11,
        post_date=post_date,
    )

    assert p1.name == "2026-04-01_1.md"
    assert p2.name == "2026-04-01_2.md"
    assert p1.exists()
    assert p2.exists()


async def test_save_creates_directory_if_missing(tmp_path: Path) -> None:
    # No channel/posts/ dir pre-created
    path = await save_published_post(
        workspace_root=tmp_path,
        text="Post text",
        message_id=1,
        post_date=date(2026, 4, 1),
    )

    assert path.exists()
    assert (tmp_path / "channel" / "posts").is_dir()

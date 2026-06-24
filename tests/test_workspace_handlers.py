from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.skills.workspace_session.handlers import (
    build_outbox_keyboard,
    handle_approve,
    handle_skip,
    set_outbox_root,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def outbox_root(tmp_path: Path) -> Path:
    root = tmp_path / "outbox"
    (root / "channel_posts").mkdir(parents=True)
    (root / "kwork_drafts").mkdir(parents=True)
    (root / "reports").mkdir(parents=True)
    (root / ".processed").mkdir(parents=True)
    return root


def test_build_outbox_keyboard():
    kb = build_outbox_keyboard("channel_posts", "2026-03-31.md")
    buttons = kb.inline_keyboard[0]
    assert len(buttons) == 2
    assert "ws:approve:channel_posts:2026-03-31.md" in buttons[0].callback_data
    assert "ws:skip:channel_posts:2026-03-31.md" in buttons[1].callback_data


def test_build_outbox_keyboard_kwork():
    kb = build_outbox_keyboard("kwork_drafts", "project_42.md")
    buttons = kb.inline_keyboard[0]
    assert "kwork_drafts" in buttons[0].callback_data
    assert "project_42.md" in buttons[0].callback_data


async def test_handle_approve_channel_post(outbox_root: Path):
    """Approving a channel post sends it to the channel and moves to .processed."""
    set_outbox_root(outbox_root)

    post_file = outbox_root / "channel_posts" / "2026-03-31.md"
    post_file.write_text("My first channel post!")

    callback = AsyncMock()
    callback.data = "ws:approve:channel_posts:2026-03-31.md"
    callback.message = AsyncMock()
    callback.message.bot = AsyncMock()

    callback.message.bot.send_message.return_value = AsyncMock(message_id=42)

    settings = type(
        "S", (), {"channel_id": "@zhvusha", "workspace_path": str(outbox_root.parent)}
    )()

    with (
        patch(
            "src.skills.workspace_session.handlers.get_settings",
            return_value=settings,
        ),
        patch(
            "src.skills.workspace_session.handlers.save_published_post",
            new_callable=AsyncMock,
        ) as mock_archive,
    ):
        await handle_approve(callback)

    callback.answer.assert_awaited()
    # File moved to .processed
    assert not post_file.exists()
    assert (outbox_root / ".processed" / "2026-03-31.md").exists()
    # Message sent to channel
    callback.message.bot.send_message.assert_awaited_once()
    call_kwargs = callback.message.bot.send_message.call_args.kwargs
    assert call_kwargs["chat_id"] == "@zhvusha"
    assert "My first channel post!" in call_kwargs["text"]
    # Post archived
    mock_archive.assert_awaited_once()


async def test_handle_approve_channel_post_splits_long_post(outbox_root: Path):
    """Long channel posts are published in chunks, then archived once."""
    set_outbox_root(outbox_root)

    post_file = outbox_root / "channel_posts" / "2026-05-06.md"
    long_text = ("абзац\n\n" * 900).strip()
    post_file.write_text(long_text)

    callback = AsyncMock()
    callback.data = "ws:approve:channel_posts:2026-05-06.md"
    callback.message = AsyncMock()
    callback.message.bot = AsyncMock()
    callback.message.bot.send_message.side_effect = [
        MagicMock(message_id=message_id) for message_id in range(101, 111)
    ]

    settings = type(
        "S", (), {"channel_id": "@zhvusha", "workspace_path": str(outbox_root.parent)}
    )()

    with (
        patch(
            "src.skills.workspace_session.handlers.get_settings",
            return_value=settings,
        ),
        patch(
            "src.skills.workspace_session.handlers.save_published_post",
            new_callable=AsyncMock,
        ) as mock_archive,
    ):
        await handle_approve(callback)

    assert callback.message.bot.send_message.await_count > 1
    for call in callback.message.bot.send_message.await_args_list:
        assert call.kwargs["chat_id"] == "@zhvusha"
        assert len(call.kwargs["text"]) <= 4096
    mock_archive.assert_awaited_once_with(
        workspace_root=outbox_root.parent,
        text=long_text,
        message_id=101,
        visual=None,
        media=None,
    )
    assert not post_file.exists()


async def test_handle_approve_channel_post_blocks_required_unready_visual(
    outbox_root: Path,
) -> None:
    set_outbox_root(outbox_root)

    post_file = outbox_root / "channel_posts" / "2026-05-14.md"
    post_file.write_text(
        "---\n"
        "visual:\n"
        "  needed: true\n"
        "  type: screenshot\n"
        "  source_url: https://openai.com/index/work-with-codex-from-anywhere/\n"
        "---\n\n"
        "Post body.",
        encoding="utf-8",
    )

    callback = AsyncMock()
    callback.data = "ws:approve:channel_posts:2026-05-14.md"
    callback.message = AsyncMock()
    callback.message.bot = AsyncMock()

    settings = type(
        "S", (), {"channel_id": "@zhvusha", "workspace_path": str(outbox_root.parent)}
    )()

    with patch(
        "src.skills.workspace_session.handlers.get_settings",
        return_value=settings,
    ):
        await handle_approve(callback)

    callback.answer.assert_awaited_once()
    assert callback.answer.await_args.kwargs["show_alert"] is True
    assert post_file.exists()
    callback.message.bot.send_message.assert_not_awaited()
    callback.message.bot.send_photo.assert_not_awaited()


async def test_handle_approve_channel_post_publishes_ready_visual(
    outbox_root: Path,
) -> None:
    set_outbox_root(outbox_root)

    asset = outbox_root.parent / "agent_runtime" / "browser_artifacts" / "source.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"\x89PNG\r\n\x1a\n")
    post_file = outbox_root / "channel_posts" / "2026-05-14.md"
    post_file.write_text(
        "---\n"
        "visual:\n"
        "  needed: true\n"
        "  type: screenshot\n"
        "  status: ready\n"
        "  asset_path: agent_runtime/browser_artifacts/source.png\n"
        "  source_url: https://openai.com/index/work-with-codex-from-anywhere/\n"
        "  caption: Codex mobile\n"
        "---\n\n"
        "Post body.",
        encoding="utf-8",
    )

    callback = AsyncMock()
    callback.data = "ws:approve:channel_posts:2026-05-14.md"
    callback.message = AsyncMock()
    callback.message.bot = AsyncMock()
    callback.message.bot.send_photo.return_value = MagicMock(message_id=41)
    callback.message.bot.send_message.return_value = MagicMock(message_id=42)

    settings = type(
        "S", (), {"channel_id": "@zhvusha", "workspace_path": str(outbox_root.parent)}
    )()

    with (
        patch(
            "src.skills.workspace_session.handlers.get_settings",
            return_value=settings,
        ),
        patch(
            "src.skills.workspace_session.handlers.save_published_post",
            new_callable=AsyncMock,
        ) as mock_archive,
    ):
        await handle_approve(callback)

    callback.message.bot.send_photo.assert_awaited_once()
    assert callback.message.bot.send_photo.await_args.kwargs["caption"] == (
        "Codex mobile"
    )
    callback.message.bot.send_message.assert_awaited_once()
    assert callback.message.bot.send_message.await_args.kwargs["text"] == "Post body."
    mock_archive.assert_awaited_once()
    archive_kwargs = mock_archive.await_args.kwargs
    assert archive_kwargs["text"] == "Post body."
    assert archive_kwargs["visual"]["intent"] == "source_screenshot"
    assert archive_kwargs["media"]["message_id"] == 41
    assert archive_kwargs["media"]["text_message_id"] == 42
    assert not post_file.exists()


async def test_handle_approve_kwork_draft(outbox_root: Path):
    """Approving kwork draft shows copyable text."""
    set_outbox_root(outbox_root)

    draft_file = outbox_root / "kwork_drafts" / "project_42.md"
    draft_file.write_text("Draft response text")

    callback = AsyncMock()
    callback.data = "ws:approve:kwork_drafts:project_42.md"
    callback.message = AsyncMock()
    callback.message.bot = AsyncMock()

    settings = type("S", (), {"channel_id": "@zhvusha"})()

    with patch(
        "src.skills.workspace_session.handlers.get_settings",
        return_value=settings,
    ):
        await handle_approve(callback)

    callback.answer.assert_awaited()
    # Kwork drafts → edit message with copyable text
    callback.message.edit_text.assert_awaited_once()
    call_args = callback.message.edit_text.call_args
    edit_text = call_args.kwargs.get(
        "text", call_args.args[0] if call_args.args else ""
    )
    assert "Draft response text" in edit_text
    # File moved
    assert not draft_file.exists()


async def test_handle_skip(outbox_root: Path):
    """Skipping an item moves it to .processed without action."""
    set_outbox_root(outbox_root)

    post_file = outbox_root / "channel_posts" / "2026-03-31.md"
    post_file.write_text("Skipped post")

    callback = AsyncMock()
    callback.data = "ws:skip:channel_posts:2026-03-31.md"
    callback.message = AsyncMock()

    await handle_skip(callback)

    callback.answer.assert_awaited()
    assert not post_file.exists()
    assert (outbox_root / ".processed" / "2026-03-31.md").exists()
    # Message deleted
    callback.message.delete.assert_awaited()


async def test_handle_approve_missing_file(outbox_root: Path):
    """Approving a non-existent file shows error."""
    set_outbox_root(outbox_root)

    callback = AsyncMock()
    callback.data = "ws:approve:channel_posts:nonexistent.md"
    callback.message = AsyncMock()

    await handle_approve(callback)

    callback.answer.assert_awaited()
    # Check it was called with show_alert
    call_kwargs = callback.answer.call_args.kwargs
    assert call_kwargs.get("show_alert") is True

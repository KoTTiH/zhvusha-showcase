"""Media publishing contract for saved channel drafts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from src.skills.base import AgentContext
from src.skills.channel_writer.media import (
    TELEGRAM_PHOTO_CAPTION_MAX_LENGTH,
    validate_approved_media,
)
from src.skills.channel_writer.skill import ChannelWriterSkill
from src.skills.post_drafts.models import PostDraft
from src.skills.post_drafts.store import (
    load_post_draft,
    save_draft_raw,
    write_post_draft,
)


def _ctx(bot: object) -> AgentContext:
    return AgentContext(user_id=1, chat_id=1, mode="personal", message_id=1, bot=bot)


async def test_visual_required_draft_without_approved_asset_is_not_published(
    tmp_path: Path,
) -> None:
    path = write_post_draft(
        tmp_path,
        PostDraft(
            slug="needs-visual",
            title="Needs visual",
            source_cluster="self-coding",
            text="body",
            created_at=datetime(2026, 5, 8, 12, 0, tzinfo=UTC),
            visual={
                "intent": "generated",
                "required": True,
                "status": "planned",
                "prompt": "Карта мысли",
            },
        ),
    )
    raw, body = load_post_draft(path)
    save_draft_raw(path, raw, body)
    bot = AsyncMock()
    skill = ChannelWriterSkill(channel_id="@test", workspace_root=tmp_path)

    result = await skill.execute("/post_draft publish needs-visual", _ctx(bot))

    assert result.success is False
    assert "approved" in result.response
    bot.send_message.assert_not_awaited()


async def test_validate_approved_media_rejects_path_escape(tmp_path: Path) -> None:
    visual = {
        "intent": "generated",
        "required": True,
        "status": "approved",
        "asset_path": "../secret.png",
    }

    result = validate_approved_media(visual, workspace_root=tmp_path)

    assert result.allowed is False
    assert "workspace" in result.reason


async def test_source_media_requires_public_source_url(tmp_path: Path) -> None:
    asset = tmp_path / "agent_runtime" / "browser_artifacts" / "source.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"\x89PNG\r\n\x1a\n")

    missing = validate_approved_media(
        {
            "intent": "source_screenshot",
            "required": True,
            "status": "approved",
            "asset_path": "agent_runtime/browser_artifacts/source.png",
        },
        workspace_root=tmp_path,
    )
    private = validate_approved_media(
        {
            "intent": "source_screenshot",
            "required": True,
            "status": "approved",
            "asset_path": "agent_runtime/browser_artifacts/source.png",
            "source_url": "http://127.0.0.1/dashboard",
        },
        workspace_root=tmp_path,
    )
    public = validate_approved_media(
        {
            "intent": "source_screenshot",
            "required": True,
            "status": "approved",
            "asset_path": "agent_runtime/browser_artifacts/source.png",
            "source_url": "https://example.com/report",
        },
        workspace_root=tmp_path,
    )

    assert missing.allowed is False
    assert "source_url" in missing.reason
    assert private.allowed is False
    assert "public" in private.reason
    assert public.allowed is True
    assert public.source_url == "https://example.com/report"


async def test_approved_media_is_sent_before_separate_text_post_and_archived(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "agent_runtime" / "channel_visual_artifacts" / "card.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"\x89PNG\r\n\x1a\n")
    path = write_post_draft(
        tmp_path,
        PostDraft(
            slug="with-visual",
            title="With visual",
            source_cluster="self-coding",
            text="body",
            created_at=datetime(2026, 5, 8, 12, 0, tzinfo=UTC),
            visual={
                "intent": "generated",
                "required": True,
                "status": "approved",
                "asset_path": "agent_runtime/channel_visual_artifacts/card.png",
                "caption": "Карта мысли",
            },
        ),
    )
    bot = AsyncMock()
    bot.send_message.return_value = MagicMock(message_id=601)
    bot.send_photo.return_value = MagicMock(message_id=602)
    skill = ChannelWriterSkill(channel_id="@test", workspace_root=tmp_path)

    result = await skill.execute("/post_draft publish with-visual", _ctx(bot))

    assert result.success is True
    bot.send_photo.assert_awaited_once()
    bot.send_message.assert_awaited_once_with(
        chat_id="@test",
        text="body",
        parse_mode=None,
    )
    assert bot.send_photo.await_args.kwargs["caption"] == "Карта мысли"
    raw, _body = load_post_draft(path)
    assert raw["message_id"] == 601
    assert raw["media"]["message_id"] == 602
    assert raw["media"]["text_message_id"] == 601
    assert raw["media"]["text_parts"] == 1
    assert raw["media"]["attached_to_text"] is False


async def test_approved_media_with_long_text_splits_media_from_text(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "agent_runtime" / "channel_visual_artifacts" / "card.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"\x89PNG\r\n\x1a\n")
    path = write_post_draft(
        tmp_path,
        PostDraft(
            slug="long-visual",
            title="Long visual",
            source_cluster="self-coding",
            text="x" * (TELEGRAM_PHOTO_CAPTION_MAX_LENGTH + 1),
            created_at=datetime(2026, 5, 8, 12, 0, tzinfo=UTC),
            visual={
                "intent": "generated",
                "required": True,
                "status": "approved",
                "asset_path": "agent_runtime/channel_visual_artifacts/card.png",
                "caption": "Карта мысли",
            },
        ),
    )
    bot = AsyncMock()
    bot.send_message.return_value = MagicMock(message_id=701)
    bot.send_photo.return_value = MagicMock(message_id=702)
    skill = ChannelWriterSkill(channel_id="@test", workspace_root=tmp_path)

    result = await skill.execute("/post_draft publish long-visual", _ctx(bot))

    assert result.success is True
    bot.send_photo.assert_awaited_once()
    assert bot.send_photo.await_args.kwargs["caption"] == "Карта мысли"
    bot.send_message.assert_awaited_once()
    assert bot.send_message.await_args.kwargs["text"] == (
        "x" * (TELEGRAM_PHOTO_CAPTION_MAX_LENGTH + 1)
    )
    raw, _body = load_post_draft(path)
    assert raw["message_id"] == 701
    assert raw["media"]["message_id"] == 702
    assert raw["media"]["text_message_id"] == 701
    assert raw["media"]["attached_to_text"] is False

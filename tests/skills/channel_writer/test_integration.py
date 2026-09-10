"""Integration tests for the migrated channel_writer v4 skill.

Ported from the old ``tests/test_channel_writer.py`` under the v4
``AgentContext`` API. Exercises the full ``can_handle`` / ``execute`` flow
against an ``AsyncMock``-backed bot, with workspace archiving mocked out
for the archive-specific test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from src.skills.base import AgentContext
from src.skills.channel_writer.skill import ChannelWriterSkill
from src.skills.post_drafts.models import PostDraft
from src.skills.post_drafts.store import load_post_draft, write_post_draft

if TYPE_CHECKING:
    from pathlib import Path


def _make_context(bot: object = None) -> AgentContext:
    return AgentContext(
        user_id=1,
        chat_id=42,
        mode="personal",
        message_id=1,
        bot=bot,
    )


async def test_can_handle_post_command(tmp_path: Path) -> None:
    skill = ChannelWriterSkill(channel_id="@test", workspace_root=tmp_path)
    ctx = _make_context()
    assert await skill.can_handle("/post hello", ctx) == 0.9


async def test_can_handle_natural_post_publish(tmp_path: Path) -> None:
    skill = ChannelWriterSkill(channel_id="@test", workspace_root=tmp_path)
    ctx = _make_context()
    assert await skill.can_handle("опубликуй пост: hello", ctx) >= 0.9
    assert await skill.can_handle("обсудим пост для канала", ctx) == 0.0


async def test_can_handle_ignores_other(tmp_path: Path) -> None:
    skill = ChannelWriterSkill(channel_id="@test", workspace_root=tmp_path)
    ctx = _make_context()
    assert await skill.can_handle("hello", ctx) == 0.0


async def test_execute_sends_message(tmp_path: Path) -> None:
    bot = AsyncMock()
    bot.send_message.return_value = AsyncMock(message_id=42)
    skill = ChannelWriterSkill(channel_id="@test", workspace_root=tmp_path)
    ctx = _make_context(bot=bot)
    with patch(
        "src.skills.channel_writer.skill.save_published_post", new_callable=AsyncMock
    ):
        result = await skill.execute("/post hello world", ctx)

    assert result.success is True
    bot.send_message.assert_awaited_once_with(
        chat_id="@test",
        text="hello world",
        parse_mode=None,
    )


async def test_execute_sends_natural_post_message(tmp_path: Path) -> None:
    bot = AsyncMock()
    bot.send_message.return_value = AsyncMock(message_id=42)
    skill = ChannelWriterSkill(channel_id="@test", workspace_root=tmp_path)
    ctx = _make_context(bot=bot)
    with patch(
        "src.skills.channel_writer.skill.save_published_post", new_callable=AsyncMock
    ):
        result = await skill.execute("опубликуй пост: hello world", ctx)

    assert result.success is True
    bot.send_message.assert_awaited_once_with(
        chat_id="@test",
        text="hello world",
        parse_mode=None,
    )


async def test_execute_sends_natural_post_with_extra_spacing(tmp_path: Path) -> None:
    bot = AsyncMock()
    bot.send_message.return_value = AsyncMock(message_id=42)
    skill = ChannelWriterSkill(channel_id="@test", workspace_root=tmp_path)
    ctx = _make_context(bot=bot)
    with patch(
        "src.skills.channel_writer.skill.save_published_post", new_callable=AsyncMock
    ):
        result = await skill.execute("опубликуй   пост: hello world", ctx)

    assert result.success is True
    bot.send_message.assert_awaited_once_with(
        chat_id="@test",
        text="hello world",
        parse_mode=None,
    )


async def test_natural_post_without_text_requests_missing_input(
    tmp_path: Path,
) -> None:
    from src.skills.invocation import (
        InMemorySkillApprovalStore,
        SkillInvocationService,
    )

    async def _approval_classifier(text: str) -> str:
        del text
        return "yes"

    skill = ChannelWriterSkill(channel_id="@test", workspace_root=tmp_path)
    service = SkillInvocationService(
        approval_store=InMemorySkillApprovalStore(),
        approval_classifier=_approval_classifier,
        is_skill_allowed=lambda _name, _mode: True,
    )

    outcome = await service.dispatch("опубликуй пост", _make_context(), [skill])

    assert outcome.handled is True
    assert outcome.result is not None
    assert outcome.result.metadata["requires_user_input"] is True
    assert outcome.result.metadata["pending_decision"]["kind"] == (
        "missing_required_input"
    )
    assert outcome.result.metadata["missing_fields"] == ["post_text"]


async def test_execute_fails_without_bot(tmp_path: Path) -> None:
    skill = ChannelWriterSkill(channel_id="@test", workspace_root=tmp_path)
    ctx = _make_context()
    result = await skill.execute("/post hello", ctx)
    assert result.success is False


async def test_execute_fails_with_empty_text(tmp_path: Path) -> None:
    bot = AsyncMock()
    skill = ChannelWriterSkill(channel_id="@test", workspace_root=tmp_path)
    ctx = _make_context(bot=bot)
    result = await skill.execute("/post ", ctx)
    assert result.success is False
    bot.send_message.assert_not_awaited()


async def test_execute_rejects_non_personal_mode(tmp_path: Path) -> None:
    bot = AsyncMock()
    skill = ChannelWriterSkill(channel_id="@test", workspace_root=tmp_path)
    ctx = AgentContext(
        user_id=1,
        chat_id=42,
        mode="assistant",
        message_id=1,
        bot=bot,
    )
    result = await skill.execute("/post hello", ctx)
    assert result.success is False
    bot.send_message.assert_not_awaited()


async def test_execute_archives_published_post(tmp_path: Path) -> None:
    bot = AsyncMock()
    bot.send_message.return_value = AsyncMock(message_id=99)
    skill = ChannelWriterSkill(channel_id="@test", workspace_root=tmp_path)
    ctx = _make_context(bot=bot)

    with patch(
        "src.skills.channel_writer.skill.save_published_post", new_callable=AsyncMock
    ) as mock_archive:
        result = await skill.execute("/post archived text", ctx)

    assert result.success is True
    mock_archive.assert_awaited_once_with(
        workspace_root=tmp_path,
        text="archived text",
        message_id=99,
    )


async def test_execute_splits_long_post_and_archives_original(tmp_path: Path) -> None:
    bot = AsyncMock()
    bot.send_message.side_effect = [
        MagicMock(message_id=message_id) for message_id in range(201, 211)
    ]
    skill = ChannelWriterSkill(channel_id="@test", workspace_root=tmp_path)
    ctx = _make_context(bot=bot)
    long_text = ("живой абзац\n\n" * 800).strip()

    with patch(
        "src.skills.channel_writer.skill.save_published_post", new_callable=AsyncMock
    ) as mock_archive:
        result = await skill.execute(f"/post {long_text}", ctx)

    assert result.success is True
    assert bot.send_message.await_count > 1
    for call in bot.send_message.await_args_list:
        assert call.kwargs["chat_id"] == "@test"
        assert len(call.kwargs["text"]) <= 4096
    mock_archive.assert_awaited_once_with(
        workspace_root=tmp_path,
        text=long_text,
        message_id=201,
    )


async def test_prepare_returns_inline_plan(tmp_path: Path) -> None:
    skill = ChannelWriterSkill(channel_id="@test", workspace_root=tmp_path)
    ctx = _make_context()
    plan = await skill.prepare("/post hi", ctx)
    assert plan.skill_name == "channel_writer"
    assert plan.skill_type == "inline"
    assert plan.llm_calls_planned == 1


async def test_dry_run_reports_success(tmp_path: Path) -> None:
    skill = ChannelWriterSkill(channel_id="@test", workspace_root=tmp_path)
    ctx = _make_context()
    plan = await skill.prepare("/post hi", ctx)
    sim = await skill.dry_run(plan)
    assert sim.would_succeed is True
    assert sim.dependencies_available is True


async def test_execute_publishes_saved_draft_and_marks_it_published(
    tmp_path: Path,
) -> None:
    from datetime import UTC, datetime

    path = write_post_draft(
        tmp_path,
        PostDraft(
            slug="ai-clients",
            title="AI clients",
            source_cluster="ai-clients",
            text="draft body",
            created_at=datetime(2026, 5, 7, 12, 0, tzinfo=UTC),
            pillar_alignment={"money": 0.9},
        ),
    )
    bot = AsyncMock()
    bot.send_message.return_value = MagicMock(message_id=501)
    skill = ChannelWriterSkill(channel_id="@test", workspace_root=tmp_path)
    ctx = _make_context(bot=bot)

    with patch(
        "src.skills.channel_writer.skill.save_published_post", new_callable=AsyncMock
    ) as mock_archive:
        result = await skill.execute("/post_draft publish ai-clients", ctx)

    assert result.success
    bot.send_message.assert_awaited_once_with(
        chat_id="@test",
        text="draft body\n",
        parse_mode=None,
    )
    raw, _body = load_post_draft(path)
    assert raw["status"] == "published"
    assert raw["message_id"] == 501
    mock_archive.assert_awaited_once()


async def test_execute_publishes_saved_draft_from_natural_request(
    tmp_path: Path,
) -> None:
    from datetime import UTC, datetime

    path = write_post_draft(
        tmp_path,
        PostDraft(
            slug="ai-clients",
            title="AI clients",
            source_cluster="ai-clients",
            text="draft body",
            created_at=datetime(2026, 5, 7, 12, 0, tzinfo=UTC),
            pillar_alignment={"money": 0.9},
        ),
    )
    bot = AsyncMock()
    bot.send_message.return_value = MagicMock(message_id=501)
    skill = ChannelWriterSkill(channel_id="@test", workspace_root=tmp_path)
    ctx = _make_context(bot=bot)

    with patch(
        "src.skills.channel_writer.skill.save_published_post", new_callable=AsyncMock
    ):
        result = await skill.execute("опубликуй черновик ai-clients", ctx)

    assert result.success
    raw, _body = load_post_draft(path)
    assert raw["status"] == "published"

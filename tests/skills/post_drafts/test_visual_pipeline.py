"""Channel draft visual planning, safety and publish gates."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from src.skills.base import AgentContext
from src.skills.channel_writer.skill import ChannelWriterSkill
from src.skills.post_drafts.models import PostDraft, PostTopic, build_post_draft
from src.skills.post_drafts.skill import PostDraftsSkill
from src.skills.post_drafts.store import (
    load_post_draft,
    save_draft_raw,
    write_post_draft,
)
from src.skills.post_drafts.style_check import check_post_style, clean_draft_text
from src.skills.post_drafts.visual_assets import approve_visual_asset
from src.skills.post_drafts.visual_plan import plan_visual_for_draft


def _ctx(bot: object) -> AgentContext:
    return AgentContext(
        user_id=1,
        chat_id=1,
        mode="personal",
        message_id=1,
        bot=bot,
    )


def _message(message_id: int) -> MagicMock:
    return MagicMock(message_id=message_id)


async def test_channel_visual_pipeline_classifies_styles_safety_and_publishes_ready_media(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 5, 8, 12, 0, tzinfo=UTC)

    internal = build_post_draft(
        PostTopic(
            cluster_key="self-coding-architecture",
            title="Жвуша: self-coding architecture",
            summary="Как Agent Runtime помогает Жвуше объяснять свою работу.",
            final_priority=90,
            pillar_alignment={"money": 0.9},
        ),
        now=now,
    )
    assert internal.visual is not None
    assert internal.visual["intent"] == "generated"
    assert internal.visual["required"] is True
    assert "screenshot" not in internal.visual["prompt"].lower()

    external = build_post_draft(
        PostTopic(
            cluster_key="external-research",
            title="Public model report",
            summary=(
                "Разбор публичного отчёта. "
                "Источник: https://example.com/research/report"
            ),
            final_priority=80,
            pillar_alignment={"money": 0.9},
        ),
        now=now,
    )
    assert external.visual is not None
    assert external.visual["intent"] == "source_screenshot"
    assert external.visual["source_url"] == "https://example.com/research/report"

    intimate = build_post_draft(
        PostTopic(
            cluster_key="private-diary",
            title="Личная заметка без визуала",
            summary="Интимная мысль, где картинка только ослабит голос.",
            final_priority=40,
            pillar_alignment={"money": 0.7},
        ),
        now=now,
    )
    assert intimate.visual is not None
    assert intimate.visual["intent"] == "none"
    assert intimate.visual["required"] is False

    long_wall = "Служебный план\n\n" + ("слишком длинная строка " * 90)
    cleaned, service_notes = clean_draft_text(long_wall)
    style = check_post_style(cleaned, extra_notes=service_notes)
    assert "service_heading" in style["warnings"]
    assert "wall_of_text" in style["warnings"]
    assert "Служебный план" not in cleaned

    denied = plan_visual_for_draft(
        title="Internal dashboard debug",
        source_cluster="self-coding",
        text="```python\nprint('/home/nikita/.env TOKEN stack trace')\n```",
    )
    assert denied["intent"] == "denied"
    assert denied["required"] is False
    assert denied["status"] == "denied"
    assert denied["denial_reason"]

    draft_path = write_post_draft(tmp_path, internal)
    raw, body = load_post_draft(draft_path)
    assert body.endswith("\n")
    assert raw["visual"]["intent"] == "generated"
    assert raw["style"]["status"] == "ok"

    raw["visual"]["status"] = "planned"
    raw["visual"].pop("asset_path", None)
    save_draft_raw(draft_path, raw, body)

    bot = AsyncMock()
    bot.send_message.return_value = _message(501)
    bot.send_photo.return_value = _message(502)
    skill = ChannelWriterSkill(channel_id="@test", workspace_root=tmp_path)

    blocked = await skill.execute(
        "/post_draft publish self-coding-architecture", _ctx(bot)
    )
    assert blocked.success is False
    assert "визуал" in blocked.response.lower()
    bot.send_message.assert_not_awaited()
    bot.send_photo.assert_not_awaited()

    asset_path = tmp_path / "agent_runtime" / "channel_visual_artifacts" / "card.png"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    approved = approve_visual_asset(
        raw["visual"],
        workspace_root=tmp_path,
        asset_path="agent_runtime/channel_visual_artifacts/card.png",
        caption="Карта архитектуры Жвуши",
    )
    raw["visual"] = approved
    save_draft_raw(draft_path, raw, body)

    published = await skill.execute(
        "/post_draft publish self-coding-architecture",
        _ctx(bot),
    )

    assert published.success is True
    bot.send_photo.assert_awaited_once()
    bot.send_message.assert_awaited_once()
    assert bot.send_photo.await_args.kwargs["caption"] == "Карта архитектуры Жвуши"
    assert bot.send_message.await_args.kwargs["text"] == body.strip()
    saved_raw, _saved_body = load_post_draft(draft_path)
    assert saved_raw["status"] == "published"
    assert saved_raw["message_id"] == 501
    assert saved_raw["media"]["message_id"] == 502
    assert saved_raw["media"]["text_message_id"] == 501
    assert saved_raw["media"]["attached_to_text"] is False
    archive_text = next((tmp_path / "channel" / "posts").glob("*.md")).read_text(
        encoding="utf-8"
    )
    assert "message_id: 501" in archive_text
    assert "visual:" in archive_text
    assert "media:" in archive_text

    legacy = PostDraft(
        slug="legacy",
        title="Legacy text",
        source_cluster="manual",
        text="old body",
        created_at=now,
        visual=None,
        style=None,
    )
    legacy_path = write_post_draft(tmp_path, legacy)
    legacy_raw, _legacy_body = load_post_draft(legacy_path)
    assert "visual" not in legacy_raw
    assert "style" not in legacy_raw


async def test_post_drafts_generate_can_attach_prepared_visual_metadata(
    tmp_path: Path,
) -> None:
    class _Provider:
        async def list_post_topics(
            self,
            *,
            limit: int = 10,
            min_money_alignment: float = 0.5,
        ) -> list[PostTopic]:
            del limit, min_money_alignment
            return [
                PostTopic(
                    cluster_key="self-coding",
                    title="Жвуша self-coding",
                    summary="Как устроен Agent Runtime.",
                    final_priority=90,
                    pillar_alignment={"money": 0.9},
                )
            ]

    async def prepare(draft: PostDraft, context: AgentContext) -> dict[str, Any]:
        del context
        assert draft.visual is not None
        return {
            **draft.visual,
            "status": "ready",
            "asset_path": "agent_runtime/channel_visual_artifacts/generated.png",
        }

    skill = PostDraftsSkill(
        admin_user_id=1,
        workspace_root=tmp_path,
        topic_provider=_Provider(),
        visual_preparer=prepare,
    )

    result = await skill.execute("/post_drafts generate 1", _ctx(object()))
    raw, _body = load_post_draft(next((tmp_path / "channel" / "drafts").glob("*.md")))

    assert result.success is True
    assert raw["visual"]["status"] == "ready"
    assert raw["visual"]["asset_path"].endswith("generated.png")

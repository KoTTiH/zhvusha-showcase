"""Channel writer skill — publishes messages to the Telegram channel.

First skill migrated to the v4 ``InlineSkill`` contract (phase 3). Triggered
by a ``/post <text>`` prefix, sends the text to the configured channel, and
archives the published post to the workspace (``channel/posts/``).

Metadata lives in the sibling ``skill.yaml`` manifest. The class attributes
declared below must match that manifest — validation happens at startup via
:func:`src.skills.manifest.validate_manifest_matches_class`.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar, Literal

import structlog

from src.skills.base import (
    AgentContext,
    ExecutionPlan,
    InlineSkill,
    SideEffect,
    SkillResult,
)
from src.skills.channel_writer.archive import save_published_post
from src.skills.channel_writer.media import (
    send_approved_media,
    validate_approved_media,
)
from src.skills.post_drafts.store import (
    find_draft_path,
    load_post_draft,
    mark_draft_publish_result,
    mark_draft_published,
)
from src.utils.telegram import send_long_message

if TYPE_CHECKING:
    from pathlib import Path

    from src.memory import EpisodicMemoryProtocol as EpisodicMemory

logger = structlog.get_logger()

POST_PREFIX = "/post "
POST_DRAFT_PREFIX = "/post_draft "
_POST_NATURAL_PREFIXES: tuple[str, ...] = (
    "опубликуй пост в канал",
    "опубликуй пост",
    "опубликуй в канал",
    "запости",
)
_DRAFT_PUBLISH_PREFIXES: tuple[str, ...] = (
    "опубликуй пост-черновик",
    "опубликуй черновик поста",
    "опубликуй черновик",
)


def _normalize_chat_route_text(text: str) -> str:
    return " ".join(text.strip().lower().replace("ё", "е").split())


def _strip_natural_payload(raw_payload: str) -> str:
    return raw_payload.strip(" \t\n\r:-—")


def _payload_after_prefix(original: str, prefix: str) -> str:
    pattern = r"^\s*" + r"\s+".join(re.escape(part) for part in prefix.split())
    pattern += r"\s*[:\-—]?\s*"
    return _strip_natural_payload(re.sub(pattern, "", original, count=1, flags=re.I))


def _strip_prefixed_payload(
    *,
    original: str,
    normalized: str,
    prefixes: tuple[str, ...],
) -> str | None:
    for prefix in prefixes:
        if (
            normalized == prefix
            or normalized.startswith(prefix + " ")
            or normalized.startswith(prefix + ":")
        ):
            return _payload_after_prefix(original, prefix)
    return None


def _normalize_message_to_command(message: str) -> str | None:
    text = message.strip()
    if text.startswith(POST_PREFIX) or text.startswith(POST_DRAFT_PREFIX):
        return text
    normalized = _normalize_chat_route_text(text)
    post_payload = _strip_prefixed_payload(
        original=text,
        normalized=normalized,
        prefixes=_POST_NATURAL_PREFIXES,
    )
    if post_payload is not None:
        return f"{POST_PREFIX}{post_payload}"
    draft_slug = _strip_prefixed_payload(
        original=text,
        normalized=normalized,
        prefixes=_DRAFT_PUBLISH_PREFIXES,
    )
    if draft_slug is not None:
        return f"{POST_DRAFT_PREFIX}publish {draft_slug.split(maxsplit=1)[0]}"
    return None


class ChannelWriterSkill(InlineSkill):
    """Inline skill that posts a user-supplied text to the Telegram channel."""

    # === Identity (must match skill.yaml) ===
    name: ClassVar[str] = "channel_writer"
    description: ClassVar[str] = "Posts messages to the Telegram channel"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "strategist"

    # === Routing ===
    triggers: ClassVar[list[str]] = [POST_PREFIX, POST_DRAFT_PREFIX]

    # === Cost & approval ===
    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "low"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = (
        "required"
    )

    # === Side effects ===
    side_effects: ClassVar[list[SideEffect]] = [
        SideEffect.POSTS_TO_CHANNEL,
        SideEffect.SENDS_TELEGRAM_MESSAGE,
        SideEffect.WRITES_WORKSPACE,
    ]

    # === Mode tags ===
    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    def __init__(
        self,
        channel_id: str,
        workspace_root: Path,
        episodic: EpisodicMemory | None = None,
    ) -> None:
        self.channel_id = channel_id
        self._workspace_root = workspace_root
        self._episodic = episodic

    async def can_handle(self, message: str, context: AgentContext) -> float:
        del context  # not used for prefix-based routing
        if message.startswith(POST_PREFIX) or message.startswith(POST_DRAFT_PREFIX):
            return 0.9
        if _normalize_message_to_command(message) is not None:
            return 0.93
        return 0.0

    async def prepare(self, message: str, context: AgentContext) -> ExecutionPlan:
        del context
        command = _normalize_message_to_command(message) or message.strip()
        if command.startswith(POST_DRAFT_PREFIX):
            summary = f"Опубликовать сохранённый черновик: {command.removeprefix(POST_DRAFT_PREFIX).strip()}"
        else:
            summary = f"Опубликовать пост в канал: {command.removeprefix(POST_PREFIX).strip()[:120]}"
        missing_fields: list[str] = []
        if command == POST_PREFIX:
            summary = "Нужен текст поста для публикации в канал."
            missing_fields.append("post_text")
        elif command.startswith(POST_DRAFT_PREFIX):
            parts = command.removeprefix(POST_DRAFT_PREFIX).strip().split(maxsplit=1)
            if len(parts) < 2 or parts[0] != "publish":
                summary = "Нужен slug черновика для публикации."
                missing_fields.append("draft_slug")
        return ExecutionPlan(
            skill_name=self.name,
            skill_type="inline",
            human_summary=summary,
            estimated_tokens=0,
            estimated_cost_usd=Decimal("0"),
            estimated_duration_seconds=3.0,
            files_to_modify=[self._workspace_root],
            side_effects_invoked=list(self.side_effects),
            llm_calls_planned=1,
            metadata={
                "internal_action": command,
                **(
                    {
                        "requires_user_input": True,
                        "missing_fields": missing_fields,
                    }
                    if missing_fields
                    else {}
                ),
            },
        )

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        if context.mode != "personal":
            return SkillResult(success=False, response="")

        bot = context.bot
        if bot is None:
            return SkillResult(
                success=False,
                response="Bot instance not available in context.",
            )

        command = _normalize_message_to_command(message) or message
        if command.startswith(POST_DRAFT_PREFIX):
            return await self._publish_draft(command, context)

        text = command.removeprefix(POST_PREFIX).strip()
        if not text:
            return SkillResult(
                success=False,
                response="Пустое сообщение. Используй: /post <текст>",
            )

        messages = await send_long_message(bot, chat_id=self.channel_id, text=text)
        first_message = messages[0]
        logger.info(
            "channel_post_sent",
            channel=self.channel_id,
            length=len(text),
            parts=len(messages),
        )

        await save_published_post(
            workspace_root=self._workspace_root,
            text=text,
            message_id=first_message.message_id,
        )

        if self._episodic is not None:
            await self._episodic.record(
                content=f"Published post to channel: {text[:100]}",
                user_id=context.user_id,
                chat_type="personal",
                role="assistant",
                source="channel",
                importance=0.5,
            )

        return SkillResult(
            success=True,
            response=f"Опубликовано в {self.channel_id}",
        )

    async def _publish_draft(self, message: str, context: AgentContext) -> SkillResult:
        bot = context.bot
        if bot is None:
            return SkillResult(
                success=False,
                response="Bot instance not available in context.",
            )
        parts = message.removeprefix(POST_DRAFT_PREFIX).strip().split(maxsplit=1)
        if len(parts) < 2 or parts[0] != "publish":
            return SkillResult(
                success=False,
                response="Используй: `/post_draft publish <slug>`.",
            )
        slug = parts[1].strip()
        path = find_draft_path(self._workspace_root, slug)
        if path is None:
            return SkillResult(
                success=False,
                response=f"Черновик `{slug}` не найден.",
            )
        raw, text = load_post_draft(path)
        if raw.get("status") == "published":
            return SkillResult(
                success=True,
                response=f"Черновик `{slug}` уже опубликован.",
            )
        media_check = validate_approved_media(
            raw.get("visual") if isinstance(raw.get("visual"), dict) else None,
            workspace_root=self._workspace_root,
        )
        if not media_check.allowed:
            return SkillResult(
                success=False,
                response=(
                    f"Черновик требует готовый approved визуал: {media_check.reason}."
                ),
                metadata={"draft_path": str(path), "reason": media_check.reason},
            )
        if media_check.should_publish:
            post_text = text.rstrip()
            try:
                media = await send_approved_media(
                    bot,
                    chat_id=self.channel_id,
                    validation=media_check,
                )
            except Exception as exc:
                logger.warning(
                    "channel_draft_media_failed_before_publish",
                    channel=self.channel_id,
                    draft=str(path),
                    error=str(exc),
                )
                return SkillResult(
                    success=False,
                    response=(
                        f"Черновик `{slug}` не опубликован: visual media не ушёл: {exc}"
                    ),
                    metadata={"draft_path": str(path), "error": str(exc)[:500]},
                )
            if media is None:
                return SkillResult(
                    success=False,
                    response=f"Черновик `{slug}` не опубликован: media не подготовлен.",
                    metadata={"draft_path": str(path)},
                )
            messages = await send_long_message(
                bot,
                chat_id=self.channel_id,
                text=post_text,
            )
            first_message = messages[0]
            message_id = int(first_message.message_id)
            media["text_message_id"] = message_id
            media["text_parts"] = len(messages)
            mark_draft_publish_result(
                path,
                status="published",
                message_id=message_id,
                media=media,
            )
            await save_published_post(
                workspace_root=self._workspace_root,
                text=post_text,
                message_id=message_id,
                visual=raw.get("visual")
                if isinstance(raw.get("visual"), dict)
                else None,
                media=media,
            )
            logger.info(
                "channel_draft_published",
                channel=self.channel_id,
                draft=str(path),
                message_id=message_id,
                media_message_id=media["message_id"],
            )
            return SkillResult(
                success=True,
                response=f"Черновик `{slug}` опубликован в {self.channel_id}",
                metadata={
                    "draft_path": str(path),
                    "message_id": message_id,
                    "media": media,
                },
            )

        messages = await send_long_message(bot, chat_id=self.channel_id, text=text)
        first_message = messages[0]
        mark_draft_published(path, message_id=first_message.message_id)
        await save_published_post(
            workspace_root=self._workspace_root,
            text=text,
            message_id=first_message.message_id,
            visual=raw.get("visual") if isinstance(raw.get("visual"), dict) else None,
            media=None,
        )
        logger.info(
            "channel_draft_published",
            channel=self.channel_id,
            draft=str(path),
            message_id=first_message.message_id,
            media_message_id=None,
        )
        return SkillResult(
            success=True,
            response=f"Черновик `{slug}` опубликован в {self.channel_id}",
            metadata={
                "draft_path": str(path),
                "message_id": first_message.message_id,
                "media": None,
            },
        )

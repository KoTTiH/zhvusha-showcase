"""Attachment routing for active Agent Runtime jobs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from aiogram import Router
from aiogram.filters import BaseFilter

from src.agent_runtime.models import AgentJobStatus
from src.agent_runtime.routing import latest_active_job_for_chat
from src.bot.utils import send_long_message
from src.dialogue import DialogueStateUpdater, FileDialogueStateStore
from src.skills.chat_self_coding.attachments import (
    format_attachment_context,
    save_message_attachments,
)

if TYPE_CHECKING:
    from pathlib import Path

    from aiogram.types import Message

    from src.agent_runtime.runtime import AgentRuntime

logger = structlog.get_logger()
router = Router(name="agent_runtime_attachments")

_admin_user_id: int | None = None
_workspace_root: Path | None = None
_runtime: AgentRuntime | None = None
_source_compare_background_runner: Any | None = None


def set_agent_runtime_attachment_deps(
    *,
    admin_user_id: int,
    workspace_root: Path,
    runtime: AgentRuntime,
    source_compare_background_runner: Any | None = None,
) -> None:
    """Inject runtime dependencies from bot startup."""
    global _admin_user_id, _workspace_root, _runtime, _source_compare_background_runner
    _admin_user_id = admin_user_id
    _workspace_root = workspace_root
    _runtime = runtime
    _source_compare_background_runner = source_compare_background_runner


def reset_agent_runtime_attachment_deps_for_tests() -> None:
    """Clear module globals used by tests."""
    global _admin_user_id, _workspace_root, _runtime, _source_compare_background_runner
    _admin_user_id = None
    _workspace_root = None
    _runtime = None
    _source_compare_background_runner = None


class AgentRuntimeAttachmentFilter(BaseFilter):
    """Match attachments that should be attached to an active agent job."""

    async def __call__(self, message: Message, mode: str = "personal") -> bool:
        if mode != "personal":
            return False
        if (
            _admin_user_id is None
            or _workspace_root is None
            or _runtime is None
            or not _message_has_supported_attachment(message)
        ):
            return False
        user = getattr(message, "from_user", None)
        user_id = getattr(user, "id", None)
        if user_id != _admin_user_id:
            return False
        chat = getattr(message, "chat", None)
        chat_id = getattr(chat, "id", None)
        if not isinstance(chat_id, int):
            return False
        return await latest_active_job_for_chat(_runtime, chat_id) is not None


@router.message(AgentRuntimeAttachmentFilter())
async def handle_agent_runtime_attachment(
    message: Message,
    mode: str = "personal",
    album: list[Any] | None = None,
) -> None:
    """Persist attachments and attach raw paths to the active agent job."""
    del mode
    if _runtime is None or _workspace_root is None:
        return
    chat = getattr(message, "chat", None)
    chat_id = getattr(chat, "id", None)
    if not isinstance(chat_id, int):
        return
    job = await latest_active_job_for_chat(_runtime, chat_id)
    if job is None:
        return

    messages = album or [message]
    attachments = await save_message_attachments(
        messages,
        workspace_root=_workspace_root,
        base_dir="agent_runtime_uploads",
    )
    if not attachments:
        await message.answer("Не смогла сохранить вложение для agent-задачи.")
        return

    caption = _caption_from_messages(messages)
    context_line = format_attachment_context(
        attachments,
        caption=caption,
        target_label=f"agent job {job.id}",
    )
    for attachment in attachments:
        await _runtime.attach_artifact(job.id, str(attachment.path))
    await _runtime.attach_followup(job.id, context_line)
    DialogueStateUpdater(FileDialogueStateStore(_workspace_root)).record_observation(
        chat_id=chat_id,
        mode="personal",
        kind="agent_runtime_attachment",
        summary=_attachment_observation_summary(
            count=len(attachments),
            caption=caption,
        ),
        source="agent_runtime_attachments",
    )

    if (
        job.kind == "source_compare"
        and job.status is AgentJobStatus.AWAITING_INPUT
        and _source_compare_background_runner is not None
    ):
        await _start_source_compare_job(job.id, message)
        await message.answer(
            "Поняла, получила вложение. Запустила agent-задачу и пришлю итог отдельно."
        )
    else:
        await message.answer(
            f"Сохранила {len(attachments)} вложение(я) и добавила к текущей agent-задаче."
        )

    logger.info(
        "agent_runtime_attachment_saved",
        job_id=job.id,
        chat_id=chat_id,
        count=len(attachments),
    )


async def _start_source_compare_job(job_id: str, message: Message) -> None:
    if _source_compare_background_runner is None:
        return

    async def completion(text: str) -> None:
        bot = getattr(message, "bot", None)
        chat = getattr(message, "chat", None)
        chat_id = getattr(chat, "id", None)
        if bot is not None and isinstance(chat_id, int):
            await send_long_message(bot, chat_id, text)

    await _source_compare_background_runner.start_existing_background(
        job_id=job_id,
        completion_callback=completion,
    )


def _message_has_supported_attachment(message: Message) -> bool:
    return any(
        bool(getattr(message, attr, None))
        for attr in (
            "photo",
            "document",
            "video",
            "animation",
            "audio",
            "voice",
            "video_note",
        )
    )


def _caption_from_messages(messages: list[Any]) -> str:
    for message in messages:
        caption = getattr(message, "caption", None)
        if isinstance(caption, str) and caption.strip():
            return caption.strip()
    return ""


def _attachment_observation_summary(*, count: int, caption: str) -> str:
    summary = f"Пользователь прислал вложение(я): {count}."
    cleaned_caption = caption.strip()
    if cleaned_caption:
        summary += f" Подпись: {cleaned_caption[:240]}"
    return summary

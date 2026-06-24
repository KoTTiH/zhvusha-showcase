"""Telegram attachment handler for the /код self-coding room."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from aiogram import Router
from aiogram.filters import BaseFilter

from src.dialogue import DialogueStateUpdater, FileDialogueStateStore
from src.skills.chat_self_coding.attachments import (
    format_attachment_context,
    save_message_attachments,
)
from src.skills.chat_self_coding.intent_classifier import Stage

if TYPE_CHECKING:
    from pathlib import Path

    from aiogram.types import Message

    from src.skills.chat_self_coding.state import StateStore

logger = structlog.get_logger()
router = Router(name="self_coding_attachments")

_admin_user_id: int | None = None
_state_store: StateStore | None = None
_workspace_root: Path | None = None


def set_self_coding_attachment_deps(
    *,
    admin_user_id: int,
    state_store: StateStore,
    workspace_root: Path,
) -> None:
    """Inject chat-mode dependencies from bot startup."""
    global _admin_user_id, _state_store, _workspace_root
    _admin_user_id = admin_user_id
    _state_store = state_store
    _workspace_root = workspace_root


def reset_self_coding_attachment_deps_for_tests() -> None:
    """Clear module globals used by tests."""
    global _admin_user_id, _state_store, _workspace_root
    _admin_user_id = None
    _state_store = None
    _workspace_root = None


class SelfCodingAttachmentFilter(BaseFilter):
    """Match attachments only inside an open personal /код session."""

    async def __call__(self, message: Message, mode: str = "personal") -> bool:
        if not _message_has_supported_attachment(message):
            return False
        if mode != "personal":
            return False
        if _admin_user_id is None or _state_store is None:
            return False
        user = getattr(message, "from_user", None)
        user_id = getattr(user, "id", None)
        if user_id != _admin_user_id:
            return False
        state = await _state_store.load(int(user_id))
        return state is not None and state.is_open


@router.message(SelfCodingAttachmentFilter())
async def handle_self_coding_attachment(
    message: Message,
    mode: str = "personal",
    album: list[Any] | None = None,
) -> None:
    """Persist attachments and append their raw paths to /код context."""
    del mode
    if _state_store is None or _workspace_root is None:
        return
    user = getattr(message, "from_user", None)
    user_id = getattr(user, "id", None)
    if not isinstance(user_id, int):
        return

    state = await _state_store.load(user_id)
    if state is None or not state.is_open:
        return

    messages = album or [message]
    attachments = await save_message_attachments(
        messages,
        workspace_root=_workspace_root,
    )
    if not attachments:
        await message.answer("Не смогла сохранить вложение для /код.")
        return

    caption = _caption_from_messages(messages)
    context_line = format_attachment_context(attachments, caption=caption)
    await _state_store.save(state.append_message(f"Никита: {context_line}"))
    chat = getattr(message, "chat", None)
    chat_id = getattr(chat, "id", None)
    DialogueStateUpdater(FileDialogueStateStore(_workspace_root)).record_observation(
        chat_id=chat_id if isinstance(chat_id, int) else user_id,
        mode="personal",
        kind="self_coding_attachment",
        summary=_attachment_observation_summary(
            count=len(attachments),
            caption=caption,
        ),
        source="self_coding_attachments",
    )

    await message.answer(
        _attachment_saved_reply(count=len(attachments), stage=state.stage)
    )
    logger.info(
        "self_coding_attachment_saved",
        user_id=user_id,
        count=len(attachments),
        stage=state.stage.value,
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


def _attachment_saved_reply(*, count: int, stage: Stage) -> str:
    noun = "вложение" if count == 1 else "вложения"
    if stage in {Stage.DRAFTING, Stage.RUNNING}:
        return (
            f"Сохранила {count} {noun} в контекст /код. Текущий шаг уже мог "
            "не увидеть его; если нужно учесть прямо в задаче, после завершения "
            "скажи «пересобери план»."
        )
    return (
        f"Сохранила {count} {noun} в контекст /код. Когда скажешь «оформи план» "
        "или «пересобери план», сессия получит пути к оригиналам."
    )


def _attachment_observation_summary(*, count: int, caption: str) -> str:
    summary = f"Пользователь прислал вложение(я): {count}."
    cleaned_caption = caption.strip()
    if cleaned_caption:
        summary += f" Подпись: {cleaned_caption[:240]}"
    return summary

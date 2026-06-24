from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from aiogram import BaseMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from aiogram.types import Message, TelegramObject

_ROLE_USER = "user"
_ROLE_ASSISTANT = "assistant"


class ChatLoggerMiddleware(BaseMiddleware):
    """Log chat messages to JSONL files.

    Personal mode: full text (needed for morning session context).
    Assistant/social: metadata only (privacy protection).

    Registered as inner middleware — only runs when a handler matched.
    """

    def __init__(self, log_dir: Path) -> None:
        super().__init__()
        self._log_dir = log_dir

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        self._log_message(event, data)  # type: ignore[arg-type]
        return await handler(event, data)

    def _log_message(self, event: Message, data: dict[str, Any]) -> None:
        # Photos logged by photo handler with full metadata (description, paths).
        # Skip here to avoid duplicate log entries.
        if event.photo:
            return

        mode = data.get("mode", "personal")
        now = datetime.now(tz=UTC)
        today = now.strftime("%Y-%m-%d")

        user_id = event.from_user.id if event.from_user else 0

        entry: dict[str, Any] = {
            "ts": now.isoformat(),
            "role": _ROLE_USER,
            "source": "telegram",
            "user_id": user_id,
            "username": (event.from_user.username if event.from_user else None),
            "text": event.text,
            "chat_id": event.chat.id,
            "chat_type": str(getattr(event.chat, "type", "")),
            "message_id": getattr(event, "message_id", None),
            "reply_to_message_id": _reply_to_message_id(event),
            "mode": mode,
        }

        _write_entry(self._log_dir, event.chat.id, today, entry)


def _write_entry(
    log_dir: Path, chat_id: int | str, today: str, entry: dict[str, Any]
) -> None:
    """Append a JSONL entry to logs/{chat_id}/chat_{date}.jsonl."""
    chat_dir = log_dir / str(chat_id)
    chat_dir.mkdir(parents=True, exist_ok=True)
    log_file = chat_dir / f"chat_{today}.jsonl"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_bot_response(
    *,
    log_dir: Path,
    text: str,
    chat_id: int | str,
    mode: str,
    source: str = "telegram",
    source_actor: str = "",
) -> None:
    """Log a bot response to today's chat JSONL."""
    now = datetime.now(tz=UTC)
    today = now.strftime("%Y-%m-%d")

    from src.utils.observation_mask import mask_observations

    entry: dict[str, Any] = {
        "ts": now.isoformat(),
        "role": _ROLE_ASSISTANT,
        "source": source,
        "text": mask_observations(text),
        "chat_id": chat_id,
        "mode": mode,
        "author_label": _author_label(_ROLE_ASSISTANT, source_actor),
    }
    if source_actor:
        entry["source_actor"] = source_actor
        if source_actor == "codex":
            entry["codex"] = True

    _write_entry(log_dir, chat_id, today, entry)


def log_interface_message(
    *,
    log_dir: Path,
    text: str,
    chat_id: int | str,
    role: str,
    source: str,
    mode: str,
    source_actor: str = "",
    user_id: int | None = None,
    username: str | None = None,
    message_id: int | str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log a non-Telegram chat interface message to today's chat JSONL."""
    now = datetime.now(tz=UTC)
    today = now.strftime("%Y-%m-%d")
    entry: dict[str, Any] = {
        "ts": now.isoformat(),
        "role": role,
        "source": source,
        "text": text,
        "chat_id": chat_id,
        "mode": mode,
        "author_label": _author_label(role, source_actor),
    }
    if source_actor:
        entry["source_actor"] = source_actor
        if source_actor == "codex":
            entry["codex"] = True
    if user_id is not None:
        entry["user_id"] = user_id
    if username:
        entry["username"] = username
    if message_id is not None:
        entry["message_id"] = message_id
    if extra:
        entry.update(extra)

    _write_entry(log_dir, chat_id, today, entry)


def log_photo_message(
    *,
    log_dir: Path,
    user_id: int,
    username: str,
    caption: str,
    chat_id: int | str,
    mode: str,
    photo_paths: list[str],
    photo_description: str,
    message_id: int | None = None,
    chat_type: str = "",
    reply_to_message_id: int | None = None,
) -> None:
    """Log a user photo message to today's chat JSONL."""
    now = datetime.now(tz=UTC)
    today = now.strftime("%Y-%m-%d")

    entry: dict[str, Any] = {
        "ts": now.isoformat(),
        "role": _ROLE_USER,
        "source": "telegram",
        "user_id": user_id,
        "username": username,
        "text": caption,
        "chat_id": chat_id,
        "chat_type": chat_type if isinstance(chat_type, str) else "",
        "message_id": message_id if isinstance(message_id, int) else None,
        "reply_to_message_id": (
            reply_to_message_id if isinstance(reply_to_message_id, int) else None
        ),
        "mode": mode,
        "photo_paths": photo_paths,
        "photo_description": photo_description,
    }

    _write_entry(log_dir, chat_id, today, entry)


def _reply_to_message_id(event: Message) -> int | None:
    reply_to = getattr(event, "reply_to_message", None)
    message_id = getattr(reply_to, "message_id", None)
    return message_id if isinstance(message_id, int) else None


def _author_label(role: str, source_actor: str) -> str:
    if source_actor == "codex":
        return "Codex"
    if role == _ROLE_ASSISTANT:
        return "Жвуша"
    return "Никита"

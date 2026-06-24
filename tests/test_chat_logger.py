from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

if TYPE_CHECKING:
    from pathlib import Path

from src.bot.middleware.chat_logger import (
    ChatLoggerMiddleware,
    log_bot_response,
    log_interface_message,
    log_photo_message,
)


def _make_event(
    text: str = "hello",
    user_id: int = 12345,
    username: str = "nikita",
    chat_id: int = 12345,
    message_id: int = 77,
) -> MagicMock:
    event = MagicMock()
    event.text = text
    event.photo = None  # text message, not photo
    event.message_id = message_id
    event.reply_to_message = None
    event.from_user = MagicMock()
    event.from_user.id = user_id
    event.from_user.username = username
    event.chat = MagicMock()
    event.chat.id = chat_id
    event.chat.type = "private"
    return event


async def test_personal_mode_logs_full_text(tmp_path: Path):
    middleware = ChatLoggerMiddleware(log_dir=tmp_path)
    handler = AsyncMock(return_value="ok")
    data: dict[str, Any] = {"mode": "personal"}
    event = _make_event(text="привет жвуша", chat_id=12345)

    await middleware(handler, event, data)

    files = list(tmp_path.glob("12345/chat_*.jsonl"))
    assert len(files) == 1
    entry = json.loads(files[0].read_text().strip())
    assert entry["text"] == "привет жвуша"
    assert entry["user_id"] == 12345
    assert entry["mode"] == "personal"
    assert entry["role"] == "user"
    assert entry["source"] == "telegram"
    assert entry["message_id"] == 77
    assert entry["chat_type"] == "private"


async def test_logs_reply_metadata(tmp_path: Path) -> None:
    middleware = ChatLoggerMiddleware(log_dir=tmp_path)
    handler = AsyncMock(return_value="ok")
    event = _make_event(text="ответ", chat_id=12345, message_id=88)
    event.reply_to_message = MagicMock()
    event.reply_to_message.message_id = 77

    await middleware(handler, event, {"mode": "personal"})

    files = list(tmp_path.glob("12345/chat_*.jsonl"))
    entry = json.loads(files[0].read_text().strip())
    assert entry["message_id"] == 88
    assert entry["reply_to_message_id"] == 77


async def test_social_mode_logs_full_text(tmp_path: Path):
    middleware = ChatLoggerMiddleware(log_dir=tmp_path)
    handler = AsyncMock(return_value="ok")
    data: dict[str, Any] = {"mode": "social"}
    event = _make_event(text="group message", chat_id=99999)

    await middleware(handler, event, data)

    files = list(tmp_path.glob("99999/chat_*.jsonl"))
    assert len(files) == 1
    entry = json.loads(files[0].read_text().strip())
    assert entry["text"] == "group message"


async def test_log_file_in_chat_subdir(tmp_path: Path):
    from datetime import UTC, datetime

    middleware = ChatLoggerMiddleware(log_dir=tmp_path)
    handler = AsyncMock()
    data: dict[str, Any] = {"mode": "personal"}
    event = _make_event(chat_id=12345)

    await middleware(handler, event, data)

    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    expected = tmp_path / "12345" / f"chat_{today}.jsonl"
    assert expected.exists()


async def test_handler_still_called(tmp_path: Path):
    middleware = ChatLoggerMiddleware(log_dir=tmp_path)
    handler = AsyncMock(return_value="response")
    data: dict[str, Any] = {"mode": "personal"}
    event = _make_event()

    result = await middleware(handler, event, data)

    handler.assert_awaited_once()
    assert result == "response"


async def test_log_dir_created_if_missing(tmp_path: Path):
    log_dir = tmp_path / "subdir" / "logs"
    middleware = ChatLoggerMiddleware(log_dir=log_dir)
    handler = AsyncMock()
    data: dict[str, Any] = {"mode": "personal"}
    event = _make_event(chat_id=12345)

    await middleware(handler, event, data)

    assert (log_dir / "12345").exists()


async def test_separate_files_per_chat(tmp_path: Path):
    middleware = ChatLoggerMiddleware(log_dir=tmp_path)
    handler = AsyncMock()

    event_a = _make_event(text="hello from A", chat_id=111)
    event_b = _make_event(text="hello from B", chat_id=222)

    await middleware(handler, event_a, {"mode": "personal"})
    await middleware(handler, event_b, {"mode": "assistant"})

    assert len(list(tmp_path.glob("111/chat_*.jsonl"))) == 1
    assert len(list(tmp_path.glob("222/chat_*.jsonl"))) == 1


def test_log_bot_response_personal(tmp_path: Path):
    log_bot_response(
        log_dir=tmp_path,
        text="Привет, Никита!",
        chat_id=12345,
        mode="personal",
    )

    files = list(tmp_path.glob("12345/chat_*.jsonl"))
    assert len(files) == 1
    entry = json.loads(files[0].read_text().strip())
    assert entry["role"] == "assistant"
    assert entry["text"] == "Привет, Никита!"


def test_log_interface_message_marks_codex_author(tmp_path: Path) -> None:
    log_interface_message(
        log_dir=tmp_path,
        text="Проверь статус сборки.",
        chat_id="vscode",
        role="user",
        source="vscode",
        mode="personal",
        source_actor="codex",
    )

    files = list(tmp_path.glob("vscode/chat_*.jsonl"))
    assert len(files) == 1
    entry = json.loads(files[0].read_text().strip())
    assert entry["role"] == "user"
    assert entry["source"] == "vscode"
    assert entry["source_actor"] == "codex"
    assert entry["codex"] is True
    assert entry["author_label"] == "Codex"


def test_log_bot_response_social(tmp_path: Path):
    log_bot_response(
        log_dir=tmp_path,
        text="group response",
        chat_id=99999,
        mode="social",
    )

    files = list(tmp_path.glob("99999/chat_*.jsonl"))
    assert len(files) == 1
    entry = json.loads(files[0].read_text().strip())
    assert entry["role"] == "assistant"
    assert entry["text"] == "group response"


def test_log_photo_message_writes_entry(tmp_path: Path):
    log_photo_message(
        log_dir=tmp_path,
        user_id=12345,
        username="nikita",
        caption="мой кот",
        chat_id=12345,
        mode="personal",
        photo_paths=["media/2026-04-01_42_0.jpg"],
        photo_description="Рыжий кот на диване",
    )

    files = list(tmp_path.glob("12345/chat_*.jsonl"))
    assert len(files) == 1
    entry = json.loads(files[0].read_text().strip())
    assert entry["role"] == "user"
    assert entry["text"] == "мой кот"
    assert entry["photo_paths"] == ["media/2026-04-01_42_0.jpg"]
    assert entry["photo_description"] == "Рыжий кот на диване"
    assert entry["user_id"] == 12345


def test_log_photo_message_empty_caption(tmp_path: Path):
    log_photo_message(
        log_dir=tmp_path,
        user_id=12345,
        username="nikita",
        caption="",
        chat_id=12345,
        mode="personal",
        photo_paths=["media/2026-04-01_42_0.jpg", "media/2026-04-01_42_1.jpg"],
        photo_description="Два скриншота кода",
    )

    files = list(tmp_path.glob("12345/chat_*.jsonl"))
    entry = json.loads(files[0].read_text().strip())
    assert entry["text"] == ""
    assert len(entry["photo_paths"]) == 2

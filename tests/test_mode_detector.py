from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.bot.middleware.mode_detector import ModeDetectorMiddleware

ADMIN_ID = 12345


def _make_event(
    chat_type: str = "private",
    user_id: int | None = ADMIN_ID,
) -> MagicMock:
    event = MagicMock()
    event.chat = MagicMock()
    event.chat.type = chat_type
    if user_id is not None:
        event.from_user = MagicMock()
        event.from_user.id = user_id
    else:
        event.from_user = None
    return event


@pytest.fixture
def middleware() -> ModeDetectorMiddleware:
    return ModeDetectorMiddleware(admin_user_id=ADMIN_ID)


async def test_private_admin_returns_personal(middleware: ModeDetectorMiddleware):
    handler = AsyncMock()
    data: dict[str, Any] = {}
    event = _make_event(chat_type="private", user_id=ADMIN_ID)

    await middleware(handler, event, data)

    assert data["mode"] == "personal"


async def test_private_stranger_returns_assistant(middleware: ModeDetectorMiddleware):
    handler = AsyncMock()
    data: dict[str, Any] = {}
    event = _make_event(chat_type="private", user_id=99999)

    await middleware(handler, event, data)

    assert data["mode"] == "assistant"


async def test_group_returns_social(middleware: ModeDetectorMiddleware):
    handler = AsyncMock()
    data: dict[str, Any] = {}
    event = _make_event(chat_type="group", user_id=ADMIN_ID)

    await middleware(handler, event, data)

    assert data["mode"] == "social"


async def test_supergroup_returns_social(middleware: ModeDetectorMiddleware):
    handler = AsyncMock()
    data: dict[str, Any] = {}
    event = _make_event(chat_type="supergroup", user_id=ADMIN_ID)

    await middleware(handler, event, data)

    assert data["mode"] == "social"


async def test_mode_injected_into_data(middleware: ModeDetectorMiddleware):
    handler = AsyncMock()
    data: dict[str, Any] = {}
    event = _make_event()

    await middleware(handler, event, data)

    assert "mode" in data


async def test_handler_is_called(middleware: ModeDetectorMiddleware):
    handler = AsyncMock(return_value="ok")
    data: dict[str, Any] = {}
    event = _make_event()

    result = await middleware(handler, event, data)

    handler.assert_awaited_once_with(event, data)
    assert result == "ok"


async def test_no_from_user_returns_assistant(middleware: ModeDetectorMiddleware):
    handler = AsyncMock()
    data: dict[str, Any] = {}
    event = _make_event(chat_type="private", user_id=None)

    await middleware(handler, event, data)

    assert data["mode"] == "assistant"

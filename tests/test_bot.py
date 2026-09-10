from __future__ import annotations

from unittest.mock import AsyncMock

from aiogram.types import Chat, Message, User
from src.bot.main import handle_start, handle_text


def _make_message(text: str, user_id: int = 12345) -> AsyncMock:
    msg = AsyncMock(spec=Message)
    msg.text = text
    msg.message_id = 1
    msg.from_user = AsyncMock(spec=User)
    msg.from_user.id = user_id
    msg.chat = AsyncMock(spec=Chat)
    msg.chat.id = user_id
    msg.answer = AsyncMock()
    msg.bot = AsyncMock()
    return msg


async def test_start_personal():
    msg = _make_message("/start", user_id=12345)
    await handle_start(msg, mode="personal")
    msg.answer.assert_awaited_once()
    call_text = msg.answer.call_args[0][0]
    assert "активна" in call_text


async def test_start_assistant():
    msg = _make_message("/start", user_id=99999)
    await handle_start(msg, mode="assistant")
    msg.answer.assert_awaited_once()
    call_text = msg.answer.call_args[0][0]
    assert "помощник" in call_text.lower() or "Жвуша" in call_text


async def test_start_social():
    msg = _make_message("/start")
    await handle_start(msg, mode="social")
    msg.answer.assert_awaited_once()


async def test_handle_text_fallback():
    """Unknown slash command in personal mode shows fallback."""
    msg = _make_message("/unknown_command")
    await handle_text(msg, mode="personal")
    msg.answer.assert_awaited_once()
    call_text = msg.answer.call_args[0][0]
    assert "команд" in call_text.lower()

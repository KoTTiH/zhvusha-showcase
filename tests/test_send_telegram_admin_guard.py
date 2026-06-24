"""SendTelegramTool must refuse any chat_id that is not admin_chat_id.

Daemon is untrusted surface: LLM-planned params may reach execute() via the
approval pipeline. The tool is the last line of defense against accidentally
(or maliciously) addressing arbitrary Telegram users.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from src.daemon.tools.send_telegram import SendTelegramTool


async def test_sends_when_chat_id_matches_admin() -> None:
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=101))
    tool = SendTelegramTool(bot, admin_chat_id=123)

    result = await tool.execute({"text": "hi", "chat_id": 123})

    assert result.success is True
    bot.send_message.assert_awaited_once_with(chat_id=123, text="hi")


async def test_sends_when_chat_id_omitted_defaults_to_admin() -> None:
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=101))
    tool = SendTelegramTool(bot, admin_chat_id=123)

    result = await tool.execute({"text": "hi"})

    assert result.success is True
    bot.send_message.assert_awaited_once_with(chat_id=123, text="hi")


async def test_blocks_when_chat_id_is_different_int() -> None:
    bot = MagicMock()
    bot.send_message = AsyncMock()
    tool = SendTelegramTool(bot, admin_chat_id=123)

    result = await tool.execute({"text": "hi", "chat_id": 999})

    assert result.success is False
    assert "block" in result.message.lower()
    bot.send_message.assert_not_awaited()


async def test_blocks_when_chat_id_is_string_even_if_numerically_admin() -> None:
    bot = MagicMock()
    bot.send_message = AsyncMock()
    tool = SendTelegramTool(bot, admin_chat_id=123)

    result = await tool.execute({"text": "hi", "chat_id": "123"})

    assert result.success is False
    bot.send_message.assert_not_awaited()


async def test_blocks_when_chat_id_is_none_explicit() -> None:
    bot = MagicMock()
    bot.send_message = AsyncMock()
    tool = SendTelegramTool(bot, admin_chat_id=123)

    result = await tool.execute({"text": "hi", "chat_id": None})

    assert result.success is False
    bot.send_message.assert_not_awaited()


async def test_blocks_with_empty_text_independently_of_chat_id() -> None:
    bot = MagicMock()
    bot.send_message = AsyncMock()
    tool = SendTelegramTool(bot, admin_chat_id=123)

    result = await tool.execute({"text": "", "chat_id": 123})

    assert result.success is False
    assert "empty" in result.message.lower()
    bot.send_message.assert_not_awaited()

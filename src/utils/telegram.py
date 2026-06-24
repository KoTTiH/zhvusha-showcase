"""Telegram-safe text sending helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.utils.text import _TELEGRAM_MAX_LENGTH, _split_text

if TYPE_CHECKING:
    from aiogram import Bot


async def send_long_message(
    bot: Bot,
    chat_id: int | str,
    text: str,
    *,
    parse_mode: str | None = None,
    max_length: int = _TELEGRAM_MAX_LENGTH,
) -> list[Any]:
    """Split and send messages exceeding Telegram's 4096-char limit."""
    if len(text) <= max_length:
        msg = await bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
        return [msg]

    messages: list[Any] = []
    for part in _split_text(text, max_length):
        msg = await bot.send_message(chat_id=chat_id, text=part, parse_mode=parse_mode)
        messages.append(msg)
    return messages

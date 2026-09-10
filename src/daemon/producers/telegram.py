"""Telegram signal producer — bridges aiogram messages to Redis Streams."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.daemon.signals import Signal

if TYPE_CHECKING:
    from src.daemon.stream import SignalStream


async def push_user_message(
    stream: SignalStream,
    *,
    user_id: int,
    chat_id: int,
    text: str,
    message_id: int,
) -> None:
    """Push a user message as a critical signal."""
    signal = Signal(
        source="telegram_chat",
        priority="critical",
        signal_type="user_message",
        payload={
            "user_id": user_id,
            "chat_id": chat_id,
            "text": text,
            "message_id": message_id,
        },
        requires_response=True,
    )
    await stream.push(signal)

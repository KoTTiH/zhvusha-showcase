"""Send message to Telegram via aiogram Bot."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from src.daemon.tools.base import DaemonTool, ToolResult

logger = structlog.get_logger()

if TYPE_CHECKING:
    from aiogram import Bot


class SendTelegramTool(DaemonTool):
    """Send a message to a Telegram chat."""

    name = "send_telegram"
    description = "Отправить сообщение в Telegram"
    requires_approval = True

    def __init__(self, bot: Bot, admin_chat_id: int) -> None:
        self._bot = bot
        self._admin_chat_id = admin_chat_id

    _MAX_LENGTH = 4096

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Send message. Params: text (str), chat_id (int, optional).

        chat_id is hard-guarded to admin_chat_id — daemon cannot address
        arbitrary users. A non-admin chat_id (including stringified or
        explicit None) returns failure without hitting the Telegram API.
        """
        text = params.get("text", "")
        if not text:
            return ToolResult(success=False, message="Empty text")

        chat_id = params.get("chat_id", self._admin_chat_id)
        if not isinstance(chat_id, int) or isinstance(chat_id, bool):
            logger.warning(
                "telegram_tool_non_admin_blocked",
                reason="non_int_chat_id",
                chat_id_repr=repr(chat_id),
            )
            return ToolResult(
                success=False,
                message=f"send_telegram blocked: chat_id must be int (got {type(chat_id).__name__})",
            )
        if chat_id != self._admin_chat_id:
            logger.warning(
                "telegram_tool_non_admin_blocked",
                reason="chat_id_mismatch",
                chat_id=chat_id,
                admin_chat_id=self._admin_chat_id,
            )
            return ToolResult(
                success=False,
                message=f"send_telegram blocked: chat_id {chat_id} != admin",
            )

        if len(text) > self._MAX_LENGTH:
            logger.warning(
                "telegram_message_truncated",
                original_len=len(text),
                max_len=self._MAX_LENGTH,
            )
            text = text[: self._MAX_LENGTH]

        try:
            msg = await self._bot.send_message(chat_id=chat_id, text=text)
            return ToolResult(
                success=True,
                message=f"Sent to {chat_id}",
                data={"message_id": msg.message_id},
            )
        except Exception as e:
            return ToolResult(success=False, message=f"Send failed: {e}")

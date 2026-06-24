from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from aiogram import BaseMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from aiogram.types import Message, TelegramObject


class ModeDetectorMiddleware(BaseMiddleware):
    """Detect operating mode from chat type and user identity.

    Sets data["mode"] to one of: "personal", "assistant", "social".
    Must be registered as an outer middleware on dp.message.
    """

    def __init__(self, admin_user_id: int) -> None:
        super().__init__()
        self._admin_user_id = admin_user_id

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        data["mode"] = self._detect_mode(event)  # type: ignore[arg-type]
        return await handler(event, data)

    def _detect_mode(
        self, event: Message
    ) -> Literal["personal", "assistant", "social"]:
        chat_type = event.chat.type
        if chat_type in ("group", "supergroup"):
            return "social"
        user_id = event.from_user.id if event.from_user else 0
        if user_id == self._admin_user_id:
            return "personal"
        return "assistant"

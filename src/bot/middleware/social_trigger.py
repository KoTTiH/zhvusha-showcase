from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiogram import BaseMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from aiogram.types import Message, TelegramObject


class SocialTriggerMiddleware(BaseMiddleware):
    """Gate group messages — only pass through when Zhvusha is triggered.

    Trigger conditions (any one is enough):
    1. Bot @username mentioned in entities
    2. Trigger word ("жвуш") found in text (case-insensitive)
    3. Message is a reply to one of the bot's messages

    Non-social modes always pass through.
    Must be registered AFTER ModeDetectorMiddleware.
    """

    _TRIGGER_STEMS: frozenset[str] = frozenset(
        {"жвуш", "жвуша", "жвушу", "жвушей", "жвуше", "жвушк"}
    )

    def __init__(self, bot_username: str) -> None:
        super().__init__()
        self._bot_username = bot_username.lower().lstrip("@")

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        mode = data.get("mode", "personal")
        if mode != "social":
            return await handler(event, data)

        if self._is_triggered(event):  # type: ignore[arg-type]
            return await handler(event, data)

        return None  # silent drop

    def _is_triggered(self, event: Message) -> bool:
        return (
            self._has_mention(event)
            or self._has_trigger_word(event)
            or self._is_reply_to_bot(event)
        )

    def _has_mention(self, event: Message) -> bool:
        if not (event.entities and event.text):
            return False
        for entity in event.entities:
            if entity.type == "mention":
                mention = event.text[entity.offset : entity.offset + entity.length]
                if mention.lower().lstrip("@") == self._bot_username:
                    return True
        return False

    def _has_trigger_word(self, event: Message) -> bool:
        if not event.text:
            return False
        lower_text = event.text.lower()
        return any(stem in lower_text for stem in self._TRIGGER_STEMS)

    def _is_reply_to_bot(self, event: Message) -> bool:
        if not (event.reply_to_message and event.reply_to_message.from_user):
            return False
        reply_user = event.reply_to_message.from_user
        if not reply_user.is_bot or not reply_user.username:
            return False
        return reply_user.username.lower() == self._bot_username

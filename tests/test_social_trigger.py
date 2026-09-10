from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.bot.middleware.social_trigger import SocialTriggerMiddleware

BOT_USERNAME = "zhvusha_bot"


def _make_event(
    text: str | None = "some message",
    *,
    entities: list[dict[str, Any]] | None = None,
    reply_to_bot: bool = False,
) -> MagicMock:
    event = MagicMock()
    event.text = text

    if entities:
        mock_entities = []
        for e in entities:
            ent = MagicMock()
            ent.type = e["type"]
            ent.offset = e["offset"]
            ent.length = e["length"]
            mock_entities.append(ent)
        event.entities = mock_entities
    else:
        event.entities = None

    if reply_to_bot:
        event.reply_to_message = MagicMock()
        event.reply_to_message.from_user = MagicMock()
        event.reply_to_message.from_user.is_bot = True
        event.reply_to_message.from_user.username = BOT_USERNAME
    else:
        event.reply_to_message = None

    return event


@pytest.fixture
def middleware() -> SocialTriggerMiddleware:
    return SocialTriggerMiddleware(bot_username=BOT_USERNAME)


async def test_personal_mode_always_passes(middleware: SocialTriggerMiddleware):
    handler = AsyncMock(return_value="ok")
    data: dict[str, Any] = {"mode": "personal"}
    event = _make_event("hello")

    result = await middleware(handler, event, data)

    handler.assert_awaited_once()
    assert result == "ok"


async def test_assistant_mode_always_passes(middleware: SocialTriggerMiddleware):
    handler = AsyncMock(return_value="ok")
    data: dict[str, Any] = {"mode": "assistant"}
    event = _make_event("hello")

    result = await middleware(handler, event, data)

    handler.assert_awaited_once()
    assert result == "ok"


async def test_social_with_mention_passes(middleware: SocialTriggerMiddleware):
    handler = AsyncMock(return_value="ok")
    data: dict[str, Any] = {"mode": "social"}
    event = _make_event(
        f"hey @{BOT_USERNAME} what's up",
        entities=[{"type": "mention", "offset": 4, "length": len(BOT_USERNAME) + 1}],
    )

    result = await middleware(handler, event, data)

    handler.assert_awaited_once()
    assert result == "ok"


async def test_social_with_trigger_word_passes(middleware: SocialTriggerMiddleware):
    handler = AsyncMock(return_value="ok")
    data: dict[str, Any] = {"mode": "social"}
    event = _make_event("привет жвуша как дела")

    await middleware(handler, event, data)

    handler.assert_awaited_once()


async def test_social_with_trigger_word_case_insensitive(
    middleware: SocialTriggerMiddleware,
):
    handler = AsyncMock(return_value="ok")
    data: dict[str, Any] = {"mode": "social"}
    event = _make_event("А Жвуша тут?")

    await middleware(handler, event, data)

    handler.assert_awaited_once()


async def test_social_with_reply_to_bot_passes(middleware: SocialTriggerMiddleware):
    handler = AsyncMock(return_value="ok")
    data: dict[str, Any] = {"mode": "social"}
    event = _make_event("I agree", reply_to_bot=True)

    await middleware(handler, event, data)

    handler.assert_awaited_once()


async def test_social_without_trigger_drops(middleware: SocialTriggerMiddleware):
    handler = AsyncMock(return_value="ok")
    data: dict[str, Any] = {"mode": "social"}
    event = _make_event("random chat message")

    result = await middleware(handler, event, data)

    handler.assert_not_awaited()
    assert result is None


async def test_social_no_text_drops(middleware: SocialTriggerMiddleware):
    handler = AsyncMock(return_value="ok")
    data: dict[str, Any] = {"mode": "social"}
    event = _make_event(None)

    result = await middleware(handler, event, data)

    handler.assert_not_awaited()
    assert result is None


async def test_mode_missing_defaults_personal(middleware: SocialTriggerMiddleware):
    handler = AsyncMock(return_value="ok")
    data: dict[str, Any] = {}
    event = _make_event("hello")

    result = await middleware(handler, event, data)

    handler.assert_awaited_once()
    assert result == "ok"

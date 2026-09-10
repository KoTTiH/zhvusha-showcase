from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from src.bot.middleware.album_collector import AlbumCollectorMiddleware


def _make_photo_message(
    message_id: int,
    media_group_id: str | None = None,
    caption: str | None = None,
) -> MagicMock:
    msg = MagicMock()
    msg.message_id = message_id
    msg.photo = [MagicMock()]  # at least one PhotoSize
    msg.media_group_id = media_group_id
    msg.caption = caption
    return msg


def _make_text_message(message_id: int) -> MagicMock:
    msg = MagicMock()
    msg.message_id = message_id
    msg.photo = None
    msg.media_group_id = None
    return msg


async def test_single_photo_passes_through():
    """Photo without media_group_id passes immediately with album=[msg]."""
    middleware = AlbumCollectorMiddleware()
    handler = AsyncMock()
    msg = _make_photo_message(1)

    await middleware(handler, msg, {"mode": "personal"})

    handler.assert_awaited_once()
    call_data = handler.call_args[0][1]
    assert call_data["album"] == [msg]


async def test_text_message_passes_through():
    """Non-photo messages are not intercepted."""
    middleware = AlbumCollectorMiddleware()
    handler = AsyncMock()
    msg = _make_text_message(1)

    await middleware(handler, msg, {"mode": "personal"})

    handler.assert_awaited_once()


async def test_album_bundles_same_media_group():
    """Multiple photos with same media_group_id are bundled together."""
    middleware = AlbumCollectorMiddleware()

    captured: dict[str, Any] = {}

    async def capture_handler(event: Any, data: dict[str, Any]) -> None:
        captured["album"] = data["album"]
        captured["event"] = event

    msg1 = _make_photo_message(10, media_group_id="album-1", caption="My album")
    msg2 = _make_photo_message(11, media_group_id="album-1")
    msg3 = _make_photo_message(12, media_group_id="album-1")

    data = {"mode": "personal"}

    # Send all three — handler should NOT be called immediately
    await middleware(capture_handler, msg1, dict(data))
    await middleware(capture_handler, msg2, dict(data))
    await middleware(capture_handler, msg3, dict(data))

    # Wait for the flush timer
    await asyncio.sleep(0.7)

    assert "album" in captured
    assert len(captured["album"]) == 3
    # Sorted by message_id
    assert captured["album"][0].message_id == 10
    assert captured["album"][2].message_id == 12
    # Event is the first message
    assert captured["event"].message_id == 10


async def test_album_does_not_call_handler_per_message():
    """Individual album photos should not trigger the handler."""
    middleware = AlbumCollectorMiddleware()
    call_count = 0

    async def counting_handler(event: Any, data: dict[str, Any]) -> None:
        nonlocal call_count
        call_count += 1

    msg1 = _make_photo_message(1, media_group_id="grp")
    msg2 = _make_photo_message(2, media_group_id="grp")

    await middleware(counting_handler, msg1, {})
    await middleware(counting_handler, msg2, {})

    # Before flush — no calls yet
    assert call_count == 0

    await asyncio.sleep(0.7)

    # After flush — exactly one call
    assert call_count == 1

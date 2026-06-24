"""Album collector middleware — buffers Telegram photo albums.

Telegram sends album photos as separate messages with the same
``media_group_id``.  This middleware collects them into a single
list and passes ``data["album"]`` to the handler after a short
delay so all album members arrive.

Single photos (no ``media_group_id``) pass through immediately
with ``data["album"] = [message]``.

Register as an outer middleware AFTER ModeDetector and SocialTrigger.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog
from aiogram import BaseMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from aiogram.types import Message, TelegramObject

logger = structlog.get_logger()

# How long to wait for additional album members (seconds).
_ALBUM_WAIT: float = 0.5


class AlbumCollectorMiddleware(BaseMiddleware):
    """Collect album photos into a single handler call."""

    def __init__(self) -> None:
        super().__init__()
        self._albums: dict[str, list[Message]] = {}
        self._album_locks: dict[str, asyncio.Lock] = {}
        self._album_data: dict[str, dict[str, Any]] = {}
        self._album_handlers: dict[
            str, Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]]
        ] = {}
        self._album_tasks: dict[str, asyncio.Task[Any]] = {}

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        message: Message = event  # type: ignore[assignment]

        # Only intercept photo messages
        if not message.photo:
            return await handler(event, data)

        media_group_id = message.media_group_id

        # Single photo (no album) — pass through immediately
        if media_group_id is None:
            data["album"] = [message]
            return await handler(event, data)

        # Album photo — buffer it
        if media_group_id not in self._albums:
            self._albums[media_group_id] = []
            self._album_locks[media_group_id] = asyncio.Lock()
            self._album_data[media_group_id] = dict(data)
            self._album_handlers[media_group_id] = handler

        self._albums[media_group_id].append(message)

        # Start timer on first message of the album
        if media_group_id not in self._album_tasks:
            self._album_tasks[media_group_id] = asyncio.create_task(
                self._flush_album(media_group_id)
            )

        return None  # suppress individual handler calls

    async def _flush_album(self, media_group_id: str) -> None:
        """Wait for album members, then dispatch to handler."""
        await asyncio.sleep(_ALBUM_WAIT)

        messages = self._albums.pop(media_group_id, [])
        data = self._album_data.pop(media_group_id, {})
        handler = self._album_handlers.pop(media_group_id, None)
        self._album_locks.pop(media_group_id, None)
        self._album_tasks.pop(media_group_id, None)

        if not messages or handler is None:
            return

        # Sort by message_id to preserve order
        messages.sort(key=lambda m: m.message_id)

        # Pass all collected messages as album
        data["album"] = messages

        # Dispatch with the first message as the event
        try:
            await handler(messages[0], data)
        except Exception:
            logger.exception("album_flush_handler_error", media_group_id=media_group_id)

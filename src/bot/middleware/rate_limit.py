"""Per-user daily message rate limit for non-admin chats.

Protects the LLM budget from a single chatty user (or troll) burning
hundreds of dollars in a day. Admin (``admin_user_id``) is never capped.
The counter is stored in Redis with a 36-hour TTL keyed by ``user_id``
and the calendar date — so each day starts fresh and a key set late in
the day still expires cleanly.

If Redis is unavailable (no client passed, or the call raises), the
middleware fails *open*: the message is allowed through. Bot uptime is
more important than perfect cap enforcement; the alternative is locking
the bot out of every conversation when Redis hiccups.

Registered as an outer middleware on ``dp.message`` so the cap kicks in
before any handler-level work (LLM calls, tool dispatch).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog
from aiogram import BaseMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from aiogram.types import Message, TelegramObject
    from redis.asyncio import Redis

logger = structlog.get_logger()

_TTL_SECONDS = 36 * 3600  # 36h — survives a late-night spike across midnight


class RateLimitMiddleware(BaseMiddleware):
    """Cap daily messages per non-admin user via Redis counters."""

    def __init__(
        self,
        admin_user_id: int,
        daily_limit: int,
        redis: Redis | None,
    ) -> None:
        super().__init__()
        self._admin_user_id = admin_user_id
        self._daily_limit = daily_limit
        self._redis = redis

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if not self._enabled():
            return await handler(event, data)

        message: Message = event  # type: ignore[assignment]
        if message.from_user is None:
            return await handler(event, data)
        user_id = message.from_user.id
        if user_id == self._admin_user_id:
            return await handler(event, data)

        try:
            count = await self._increment(user_id)
        except Exception:
            logger.warning("rate_limit_redis_error", exc_info=True, user_id=user_id)
            return await handler(event, data)

        if count > self._daily_limit:
            # Reply once on the first over-limit message, then go silent so
            # we don't echo the refusal on every spam attempt.
            if count == self._daily_limit + 1:
                await message.answer(
                    "Достиг дневного лимита сообщений. Давай продолжим завтра 🐊"
                )
            logger.info(
                "rate_limit_blocked",
                user_id=user_id,
                count=count,
                limit=self._daily_limit,
            )
            return None

        return await handler(event, data)

    def _enabled(self) -> bool:
        return self._daily_limit > 0 and self._redis is not None

    async def _increment(self, user_id: int) -> int:
        """Atomically bump the day's counter and ensure TTL is set.

        SET-then-INCR is racy across calls; INCR on a missing key creates
        it at 1, then EXPIRE attaches the TTL. EXPIRE on an existing key
        is a no-op for the value, so it's safe to call every time.
        """
        assert self._redis is not None  # narrowed by _enabled()
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        key = f"bot:msg_count:{user_id}:{today}"
        count_raw = await self._redis.incr(key)
        await self._redis.expire(key, _TTL_SECONDS)
        return int(count_raw)

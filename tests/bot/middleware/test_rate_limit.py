"""Tests for ``RateLimitMiddleware`` — per-user daily message cap.

Admin must never be capped (he pays for the bot). Non-admin must be
capped at ``daily_limit`` and replied to once on the first over-limit
message; subsequent over-limit messages drop silently. Redis errors
must fail open — the bot keeps working even if Redis is flaky.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.bot.middleware.rate_limit import RateLimitMiddleware


def _make_message(user_id: int) -> AsyncMock:
    msg = AsyncMock()
    msg.from_user = MagicMock()
    msg.from_user.id = user_id
    msg.answer = AsyncMock()
    return msg


def _fake_redis(starting_count: int = 0) -> MagicMock:
    """Stub the Redis methods the middleware uses (``incr`` + ``expire``).

    Each ``incr`` call bumps the counter and returns the post-increment
    value, so the test can simulate "this is the Nth message of the day"
    by pre-setting ``starting_count``.
    """
    redis = MagicMock()
    counter = {"value": starting_count}

    async def _incr(_key: str) -> int:
        counter["value"] += 1
        return counter["value"]

    redis.incr = AsyncMock(side_effect=_incr)
    redis.expire = AsyncMock()
    return redis


@pytest.fixture
def downstream() -> AsyncMock:
    """Mock the next handler in the middleware chain."""
    return AsyncMock(return_value="handled")


class TestAdminBypass:
    async def test_admin_is_never_counted(self, downstream: AsyncMock) -> None:
        redis = _fake_redis(starting_count=999_999)
        mw = RateLimitMiddleware(admin_user_id=12345, daily_limit=30, redis=redis)
        msg = _make_message(user_id=12345)

        result = await mw(downstream, msg, {})

        assert result == "handled"
        downstream.assert_awaited_once()
        # Admin bypass must skip Redis entirely — no I/O, no warming the
        # counter for the user.
        redis.incr.assert_not_called()


class TestNonAdminCap:
    async def test_under_limit_passes_through(self, downstream: AsyncMock) -> None:
        redis = _fake_redis(starting_count=10)  # next incr → 11
        mw = RateLimitMiddleware(admin_user_id=12345, daily_limit=30, redis=redis)
        msg = _make_message(user_id=99)

        result = await mw(downstream, msg, {})

        assert result == "handled"
        downstream.assert_awaited_once()
        msg.answer.assert_not_called()
        # TTL is renewed on every increment so a key set near midnight
        # doesn't expire prematurely.
        redis.expire.assert_awaited_once()

    async def test_first_over_limit_replies_once_and_blocks(
        self, downstream: AsyncMock
    ) -> None:
        redis = _fake_redis(starting_count=30)  # next incr → 31, limit+1
        mw = RateLimitMiddleware(admin_user_id=12345, daily_limit=30, redis=redis)
        msg = _make_message(user_id=99)

        result = await mw(downstream, msg, {})

        assert result is None
        downstream.assert_not_called()
        msg.answer.assert_awaited_once()
        # Refusal copy mentions the daily limit so the user knows why.
        text = msg.answer.call_args.args[0]
        assert "лимит" in text.lower() or "limit" in text.lower()

    async def test_subsequent_over_limit_drops_silently(
        self, downstream: AsyncMock
    ) -> None:
        """Avoid spamming the refusal copy on every over-cap message —
        only the first crossing speaks."""
        redis = _fake_redis(starting_count=50)  # next incr → 51, well past
        mw = RateLimitMiddleware(admin_user_id=12345, daily_limit=30, redis=redis)
        msg = _make_message(user_id=99)

        result = await mw(downstream, msg, {})

        assert result is None
        downstream.assert_not_called()
        msg.answer.assert_not_called()


class TestDisabledStates:
    async def test_zero_limit_disables_cap(self, downstream: AsyncMock) -> None:
        """``daily_limit=0`` is the legacy "no cap" knob — redirect through."""
        redis = _fake_redis()
        mw = RateLimitMiddleware(admin_user_id=12345, daily_limit=0, redis=redis)
        msg = _make_message(user_id=99)

        result = await mw(downstream, msg, {})

        assert result == "handled"
        redis.incr.assert_not_called()

    async def test_no_redis_fails_open(self, downstream: AsyncMock) -> None:
        """If Redis isn't configured, bot must still work."""
        mw = RateLimitMiddleware(admin_user_id=12345, daily_limit=30, redis=None)
        msg = _make_message(user_id=99)

        result = await mw(downstream, msg, {})

        assert result == "handled"

    async def test_redis_error_fails_open(self, downstream: AsyncMock) -> None:
        """A flaky Redis must not lock the bot out of conversations.
        We log the error and pass the message through — better an extra
        LLM call than a dead bot."""
        redis = MagicMock()
        redis.incr = AsyncMock(side_effect=ConnectionError("redis down"))
        mw = RateLimitMiddleware(admin_user_id=12345, daily_limit=30, redis=redis)
        msg = _make_message(user_id=99)

        result = await mw(downstream, msg, {})

        assert result == "handled"
        downstream.assert_awaited_once()


class TestNoUser:
    async def test_message_without_user_passes_through(
        self, downstream: AsyncMock
    ) -> None:
        """Channel-style messages may have ``from_user=None``. Don't crash;
        let the next handler decide what to do."""
        redis = _fake_redis()
        mw = RateLimitMiddleware(admin_user_id=12345, daily_limit=30, redis=redis)
        msg = AsyncMock()
        msg.from_user = None

        result = await mw(downstream, msg, {})

        assert result == "handled"
        redis.incr.assert_not_called()


class TestKeyShape:
    async def test_redis_key_includes_user_id_and_date(
        self, downstream: AsyncMock
    ) -> None:
        """Counter must reset across calendar days. Key carries the date so
        a key from yesterday isn't reused today."""
        from datetime import UTC, datetime

        redis = _fake_redis()
        mw = RateLimitMiddleware(admin_user_id=12345, daily_limit=30, redis=redis)
        msg = _make_message(user_id=99)

        await mw(downstream, msg, {})

        key: Any = redis.incr.call_args.args[0]
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        assert "99" in key
        assert today in key

"""Contract tests for ``CapsEnforcer`` (Phase 13).

Sliding-window rate limit on top of a redis.asyncio-shaped client. Tests
substitute a tiny in-memory ``FakeRedis`` for the sorted-set ops we use
(``zadd`` / ``zcount`` / ``zremrangebyscore`` / ``expire``) — no fakeredis
dependency, no live Redis. Fail-open behaviour is covered with both a
``None`` client and a stub that raises on every call.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.contract


class FakeRedis:
    """Minimal in-memory stand-in for ``redis.asyncio.Redis`` sorted sets."""

    def __init__(self) -> None:
        self._zsets: dict[str, list[tuple[float, str]]] = {}
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def zadd(self, key: str, mapping: dict[str, float]) -> int:
        self.calls.append(("zadd", (key, dict(mapping))))
        zset = self._zsets.setdefault(key, [])
        added = 0
        for member, score in mapping.items():
            if not any(m == member for _, m in zset):
                zset.append((score, member))
                added += 1
        return added

    async def zremrangebyscore(
        self, key: str, min_score: float, max_score: float
    ) -> int:
        self.calls.append(("zremrangebyscore", (key, min_score, max_score)))
        zset = self._zsets.get(key, [])
        before = len(zset)
        zset[:] = [(s, m) for s, m in zset if not (min_score <= s <= max_score)]
        return before - len(zset)

    async def zcount(self, key: str, min_score: float, max_score: float) -> int:
        self.calls.append(("zcount", (key, min_score, max_score)))
        zset = self._zsets.get(key, [])
        return sum(1 for s, _ in zset if min_score <= s <= max_score)

    async def expire(self, key: str, seconds: int) -> int:
        self.calls.append(("expire", (key, seconds)))
        return 1


class FailingRedis:
    """Redis-like stub that raises on every call (covers fail-open path)."""

    async def zadd(self, *_: Any, **__: Any) -> int:
        raise ConnectionError("redis down")

    async def zremrangebyscore(self, *_: Any, **__: Any) -> int:
        raise ConnectionError("redis down")

    async def zcount(self, *_: Any, **__: Any) -> int:
        raise ConnectionError("redis down")

    async def expire(self, *_: Any, **__: Any) -> int:
        raise ConnectionError("redis down")


def _enforcer(
    *,
    redis: Any,
    max_per_hour: int = 3,
    max_per_day: int = 6,
    now: float = 1000.0,
) -> Any:
    from src.skills.implement_spec.caps_enforcer import CapsEnforcer

    return CapsEnforcer(
        redis=redis,
        max_per_hour=max_per_hour,
        max_per_day=max_per_day,
        skill_name="implement_spec",
        clock=lambda: now,
    )


class TestCheckUnderLimit:
    async def test_no_invocations_allows(self) -> None:
        result = await _enforcer(redis=FakeRedis()).check()
        assert result.allowed is True
        assert result.reason is None

    async def test_zero_caps_disable_blocking_and_tracking(self) -> None:
        redis = FakeRedis()
        enforcer = _enforcer(redis=redis, max_per_hour=0, max_per_day=0)

        result = await enforcer.check()
        await enforcer.record()

        assert result.allowed is True
        assert result.reason is None
        assert redis.calls == []

    async def test_under_hourly_limit_allows(self) -> None:
        redis = FakeRedis()
        enforcer = _enforcer(redis=redis, max_per_hour=3, now=10000.0)
        await enforcer.record()
        await enforcer.record()
        result = await enforcer.check()
        assert result.allowed is True
        assert result.reason is None


class TestCheckBlocked:
    async def test_hourly_limit_blocks_with_reason(self) -> None:
        redis = FakeRedis()
        enforcer = _enforcer(redis=redis, max_per_hour=2, max_per_day=10, now=10000.0)
        await enforcer.record()
        await enforcer.record()
        result = await enforcer.check()
        assert result.allowed is False
        assert result.reason is not None
        assert "час" in result.reason.lower()

    async def test_daily_limit_blocks_with_reason(self) -> None:
        redis = FakeRedis()
        enforcer = _enforcer(redis=redis, max_per_hour=100, max_per_day=2, now=10000.0)
        await enforcer.record()
        await enforcer.record()
        result = await enforcer.check()
        assert result.allowed is False
        assert result.reason is not None
        assert "день" in result.reason.lower() or "сут" in result.reason.lower()


class TestSlidingWindow:
    async def test_old_hourly_invocations_dont_count(self) -> None:
        redis = FakeRedis()
        early = _enforcer(redis=redis, max_per_hour=1, max_per_day=10, now=0.0)
        await early.record()
        # >1h later — earlier invocation falls outside the hour window.
        late = _enforcer(redis=redis, max_per_hour=1, max_per_day=10, now=4000.0)
        result = await late.check()
        assert result.allowed is True

    async def test_records_within_hour_still_count(self) -> None:
        redis = FakeRedis()
        early = _enforcer(redis=redis, max_per_hour=1, max_per_day=10, now=0.0)
        await early.record()
        # 30 min later — still inside window, hits the cap.
        late = _enforcer(redis=redis, max_per_hour=1, max_per_day=10, now=1800.0)
        result = await late.check()
        assert result.allowed is False


class TestRecord:
    async def test_record_writes_to_redis(self) -> None:
        redis = FakeRedis()
        await _enforcer(redis=redis).record()
        zadds = [c for c in redis.calls if c[0] == "zadd"]
        assert len(zadds) == 1

    async def test_record_sets_expire_on_set(self) -> None:
        redis = FakeRedis()
        await _enforcer(redis=redis).record()
        expires = [c for c in redis.calls if c[0] == "expire"]
        assert expires, "expected EXPIRE on the invocations key"

    async def test_check_does_not_record(self) -> None:
        redis = FakeRedis()
        await _enforcer(redis=redis).check()
        zadds = [c for c in redis.calls if c[0] == "zadd"]
        assert zadds == []

    async def test_record_uses_unique_members(self) -> None:
        # Two records at the *same* clock value must both land in the
        # sorted set — otherwise the count is silently wrong.
        redis = FakeRedis()
        enforcer = _enforcer(redis=redis, now=10000.0)
        await enforcer.record()
        await enforcer.record()
        members = redis._zsets[next(iter(redis._zsets))]
        assert len(members) == 2


class TestFailOpen:
    async def test_no_redis_check_allows(self) -> None:
        result = await _enforcer(redis=None).check()
        assert result.allowed is True

    async def test_no_redis_record_noop(self) -> None:
        # Must not raise.
        await _enforcer(redis=None).record()

    async def test_redis_error_check_allows(self) -> None:
        result = await _enforcer(redis=FailingRedis()).check()
        assert result.allowed is True

    async def test_redis_error_record_noop(self) -> None:
        # Must not raise even if Redis is broken.
        await _enforcer(redis=FailingRedis()).record()

"""Sliding-window rate limit for ImplementSpecSkill (Phase 13).

Implements optional per-skill ``delegation_caps`` for Codex backend
invocations per hour and per day. Backed by a Redis sorted set so two
bot processes share the same counter.

``max_per_hour <= 0`` and ``max_per_day <= 0`` disable the corresponding
caps. This keeps the enforcer available for deployments that want a guardrail
without blocking the default local self-coding loop.

Fail-open by design: if Redis is unavailable (``None`` client or any
exception), ``check`` returns ``allowed=True`` and ``record`` becomes a
no-op. The reasoning matches the KB #90 throttling guidance — losing
budget telemetry is bad, but blocking Жвуша whenever Redis hiccups is
worse. Every fail-open path emits a ``structlog.warning`` so the gap
shows up in monitoring.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable

logger = structlog.get_logger()

_HOUR_SECONDS = 3600
_DAY_SECONDS = 86400


@dataclass(frozen=True)
class CapsCheckResult:
    """Outcome of :meth:`CapsEnforcer.check`.

    ``allowed=True`` always sets ``reason=None``. ``allowed=False`` carries
    a Russian-language ``reason`` ready to surface to Никита in chat.
    """

    allowed: bool
    reason: str | None = None


class _RedisProto(Protocol):
    """Minimal subset of ``redis.asyncio.Redis`` we depend on.

    The real client matches; tests substitute an in-memory fake. The
    contract is intentionally narrow — adding new operations means
    expanding this Protocol AND the test fakes in lockstep.
    """

    async def zadd(self, key: str, mapping: dict[str, float]) -> int: ...
    async def zremrangebyscore(
        self, key: str, min_score: float, max_score: float
    ) -> int: ...
    async def zcount(self, key: str, min_score: float, max_score: float) -> int: ...
    async def expire(self, key: str, seconds: int) -> int: ...


class CapsEnforcer:
    """Enforce ``max_per_hour`` / ``max_per_day`` invocation caps.

    Storage layout — one sorted set per skill, score is the unix
    timestamp of the invocation, member is ``"<ts>-<uuid>"`` so two
    invocations at the same clock value still produce distinct entries.
    Counts are evaluated on demand via ``zcount`` over the relevant
    window; cleanup runs on every ``record`` so the set is bounded by
    the day window.
    """

    def __init__(
        self,
        *,
        redis: _RedisProto | None,
        max_per_hour: int,
        max_per_day: int,
        skill_name: str = "implement_spec",
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._redis = redis
        self._max_per_hour = max_per_hour
        self._max_per_day = max_per_day
        self._key = f"caps:{skill_name}:invocations"
        self._clock = clock or time.time

    async def check(self) -> CapsCheckResult:
        """Return whether a new invocation is allowed *right now*.

        Does not record the invocation — call :meth:`record` separately
        once the SDK call is actually about to start. Splitting check
        from record lets the skill surface a friendly error message
        before paying for any side effect.
        """
        if self._redis is None:
            return CapsCheckResult(allowed=True)
        try:
            now = self._clock()
            if self._max_per_hour > 0:
                hourly = await self._redis.zcount(self._key, now - _HOUR_SECONDS, now)
                if hourly >= self._max_per_hour:
                    return CapsCheckResult(
                        allowed=False,
                        reason=(
                            f"Лимит {self._max_per_hour} вызовов/час превышен "
                            f"({hourly} в последний час). Попробуй позже."
                        ),
                    )
            if self._max_per_day > 0:
                daily = await self._redis.zcount(self._key, now - _DAY_SECONDS, now)
                if daily >= self._max_per_day:
                    return CapsCheckResult(
                        allowed=False,
                        reason=(
                            f"Лимит {self._max_per_day} вызовов/день превышен "
                            f"({daily} за последние сутки)."
                        ),
                    )
            return CapsCheckResult(allowed=True)
        except Exception:
            logger.warning(
                "caps_check_failed_fail_open",
                key=self._key,
                exc_info=True,
            )
            return CapsCheckResult(allowed=True)

    async def record(self) -> None:
        """Mark a fresh invocation against the cap."""
        if self._redis is None:
            return
        if self._max_per_hour <= 0 and self._max_per_day <= 0:
            return
        try:
            now = self._clock()
            member = f"{now:.9f}-{uuid.uuid4().hex[:8]}"
            await self._redis.zadd(self._key, {member: now})
            await self._redis.zremrangebyscore(self._key, 0, now - _DAY_SECONDS)
            await self._redis.expire(self._key, _DAY_SECONDS * 2)
        except Exception:
            logger.warning(
                "caps_record_failed",
                key=self._key,
                exc_info=True,
            )

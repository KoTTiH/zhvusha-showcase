"""APScheduler-based cron producer for scheduled signals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.daemon.signals import Signal

if TYPE_CHECKING:
    from src.daemon.stream import SignalStream


class CronProducer:
    """Pushes scheduled signals to Redis Streams via APScheduler."""

    def __init__(self, stream: SignalStream) -> None:
        self._stream = stream
        self._scheduler: Any = None

    async def start(self) -> None:
        """Start the scheduler with default jobs."""
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler

            scheduler = AsyncIOScheduler()
            scheduler.add_job(
                self._morning_briefing,
                "cron",
                hour=8,
                minute=0,
                id="morning_briefing",
            )
            scheduler.start()
            self._scheduler = scheduler
        except ImportError:
            pass  # apscheduler not installed

    async def stop(self) -> None:
        """Stop the scheduler."""
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)

    async def _morning_briefing(self) -> None:
        """Push a morning briefing signal."""
        signal = Signal(
            source="cron",
            priority="normal",
            signal_type="morning_briefing",
            payload={"task": "morning_briefing"},
        )
        await self._stream.push(signal)

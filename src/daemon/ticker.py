"""Adaptive ticker for the daemon main loop.

Sleeps longer when idle (up to MAX_INTERVAL), wakes instantly
on critical signals via asyncio.Event.
"""

from __future__ import annotations

import asyncio


class AdaptiveTicker:
    """Event-driven ticker that adapts to activity."""

    MIN_INTERVAL: float = 10.0
    MAX_INTERVAL: float = 300.0

    def __init__(self, initial_interval: float = 30.0) -> None:
        self.interval: float = initial_interval
        self._wake_event = asyncio.Event()

    async def wait_for_next_tick(self) -> bool:
        """Wait for next tick.

        Returns True if woken by event, False if timeout.
        """
        try:
            await asyncio.wait_for(self._wake_event.wait(), timeout=self.interval)
            self._wake_event.clear()
            self.interval = max(self.MIN_INTERVAL, self.interval / 2)
            return True
        except TimeoutError:
            self.interval = min(self.MAX_INTERVAL, self.interval * 2)
            return False

    def wake(self) -> None:
        """Wake the ticker immediately (called by producers)."""
        self._wake_event.set()

    def reset(self) -> None:
        """Reset interval to initial value."""
        self.interval = 30.0
        self._wake_event.clear()

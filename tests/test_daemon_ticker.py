"""Tests for AdaptiveTicker."""

from __future__ import annotations

import asyncio

from src.daemon.ticker import AdaptiveTicker


class TestAdaptiveTicker:
    async def test_timeout_increases_interval(self) -> None:
        ticker = AdaptiveTicker(initial_interval=0.05)
        original = ticker.interval

        woken = await ticker.wait_for_next_tick()

        assert woken is False
        assert ticker.interval > original

    async def test_wake_decreases_interval(self) -> None:
        ticker = AdaptiveTicker(initial_interval=100.0)

        async def wake_soon() -> None:
            await asyncio.sleep(0.01)
            ticker.wake()

        task = asyncio.create_task(wake_soon())
        woken = await ticker.wait_for_next_tick()
        await task

        assert woken is True
        assert ticker.interval < 100.0

    async def test_min_interval(self) -> None:
        ticker = AdaptiveTicker(initial_interval=0.05)
        ticker.interval = AdaptiveTicker.MIN_INTERVAL

        async def wake_soon() -> None:
            await asyncio.sleep(0.01)
            ticker.wake()

        task = asyncio.create_task(wake_soon())
        await ticker.wait_for_next_tick()
        await task

        assert ticker.interval >= AdaptiveTicker.MIN_INTERVAL

    async def test_max_interval(self) -> None:
        ticker = AdaptiveTicker(initial_interval=0.01)
        ticker.MAX_INTERVAL = 0.05  # type: ignore[misc]
        ticker.interval = ticker.MAX_INTERVAL

        woken = await ticker.wait_for_next_tick()

        assert woken is False
        assert ticker.interval <= ticker.MAX_INTERVAL

    def test_reset(self) -> None:
        ticker = AdaptiveTicker(initial_interval=100.0)
        ticker.interval = 5.0
        ticker.reset()
        assert ticker.interval == 30.0

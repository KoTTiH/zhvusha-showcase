"""Dashboard must flush pending updates on its own, not rely on the next
incoming call. Before this fix, ``schedule_update`` would set
``_pending = True`` when called within the rate-limit window; the flag
then sat unflushed until the next caller happened to arrive. Between
those, the pinned dashboard message froze with stale numbers.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

from src.monitoring.dashboard import UsageDashboard
from src.monitoring.usage_tracker import UsageTracker

if TYPE_CHECKING:
    from pathlib import Path


def _make_dashboard(
    tmp_path: Path, *, update_interval: int = 1
) -> tuple[UsageDashboard, UsageTracker, AsyncMock]:
    tracker = UsageTracker(tmp_path / "monitoring")
    bot = AsyncMock()
    sent_msg = AsyncMock()
    sent_msg.message_id = 7
    bot.send_message = AsyncMock(return_value=sent_msg)
    bot.edit_message_text = AsyncMock()
    bot.pin_chat_message = AsyncMock()

    dashboard = UsageDashboard(
        bot=bot,
        admin_chat_id=1,
        tracker=tracker,
        state_dir=tmp_path / "monitoring",
        update_interval=update_interval,
    )
    return dashboard, tracker, bot


async def test_background_flush_runs_pending_updates(tmp_path: Path) -> None:
    """Pending flag set inside the rate-limit window must be flushed by
    the dashboard's own background task, not by a subsequent caller."""
    dashboard, _tracker, bot = _make_dashboard(tmp_path, update_interval=1)
    await dashboard.initialize()
    bot.edit_message_text.reset_mock()

    # First update consumes the interval, second marks pending and returns.
    dashboard.schedule_update()
    dashboard.schedule_update()
    assert dashboard._pending is True

    # Give the background loop time to fire a flush (interval = 1s + slack).
    await asyncio.sleep(1.3)

    bot.edit_message_text.assert_awaited()
    assert dashboard._pending is False

    await dashboard.stop()


async def test_initialize_starts_background_task(tmp_path: Path) -> None:
    dashboard, _, _ = _make_dashboard(tmp_path)
    await dashboard.initialize()
    assert dashboard._flush_task is not None
    assert not dashboard._flush_task.done()
    await dashboard.stop()


async def test_stop_cancels_background_task(tmp_path: Path) -> None:
    dashboard, _, _ = _make_dashboard(tmp_path)
    await dashboard.initialize()
    task = dashboard._flush_task
    assert task is not None

    await dashboard.stop()

    assert task.done()
    assert dashboard._flush_task is None


async def test_background_flush_is_idempotent_when_nothing_pending(
    tmp_path: Path,
) -> None:
    """No pending flag → background loop must not spam edit calls."""
    dashboard, _, bot = _make_dashboard(tmp_path, update_interval=1)
    await dashboard.initialize()
    initial_edit_calls = bot.edit_message_text.await_count

    # Let the loop tick a couple of times without any schedule_update calls
    await asyncio.sleep(1.2)

    # Only the initial edit (from initialize's existing-message flow may be 0
    # or 1) plus no new edits from background — exact count must not grow.
    final_edit_calls = bot.edit_message_text.await_count
    assert final_edit_calls == initial_edit_calls

    await dashboard.stop()

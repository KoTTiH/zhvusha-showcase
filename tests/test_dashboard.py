"""Tests for UsageDashboard."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from src.monitoring.codex_limits import CodexLimitSnapshot
from src.monitoring.dashboard import UsageDashboard
from src.monitoring.usage_tracker import UsageTracker

if TYPE_CHECKING:
    from pathlib import Path


def _make_dashboard(
    tmp_path: Path,
    *,
    update_interval: int = 30,
    codex_limits_provider: Any = None,
) -> tuple[UsageDashboard, UsageTracker, AsyncMock]:
    tracker = UsageTracker(tmp_path / "monitoring")
    bot = AsyncMock()
    sent_msg = AsyncMock()
    sent_msg.message_id = 42
    bot.send_message = AsyncMock(return_value=sent_msg)
    bot.edit_message_text = AsyncMock()
    bot.pin_chat_message = AsyncMock()

    dashboard = UsageDashboard(
        bot=bot,
        admin_chat_id=12345,
        tracker=tracker,
        state_dir=tmp_path / "monitoring",
        update_interval=update_interval,
        codex_limits_provider=codex_limits_provider,
    )
    return dashboard, tracker, bot


def test_format_dashboard_output(tmp_path: Path) -> None:
    dashboard, tracker, _ = _make_dashboard(tmp_path)
    tracker.record_api_call("haiku", input_tokens=100, output_tokens=50)
    tracker.record_api_call("sonnet", input_tokens=500, output_tokens=200)
    tracker.record_gemini_call()
    tracker.record_cli_session()

    text = dashboard._format_dashboard()

    assert "API:" in text
    assert "Haiku: 1" in text
    assert "Sonnet: 1" in text
    assert "CLI:" in text
    assert "Gemini:" in text
    assert "$" in text


def test_format_dashboard_codex_limits_line(tmp_path: Path) -> None:
    dashboard, tracker, _ = _make_dashboard(
        tmp_path,
        codex_limits_provider=lambda: CodexLimitSnapshot(
            primary_used_percent=6.0,
            primary_window_minutes=300,
            primary_resets_at=1778233294,
            secondary_used_percent=36.0,
            secondary_window_minutes=10080,
            secondary_resets_at=1778553681,
            plan_type="prolite",
        ),
    )
    tracker.record_cli_call(caller="chat_agentic")

    text = dashboard._format_dashboard()
    lines = text.splitlines()

    assert "Codex CLI:" in text
    assert lines[0] == "🟣 7д 36% · 5ч 6%"
    assert "Сегодня" in lines[1]
    assert "5ч 6%" in text
    assert "7д 36%" in text
    assert "prolite" not in text
    assert lines[0].index("7д 36%") < lines[0].index("5ч 6%")


def test_format_dashboard_cache_line(tmp_path: Path) -> None:
    """Cache hit rate is shown when cache metrics are present."""
    dashboard, tracker, _ = _make_dashboard(tmp_path)
    tracker.record_api_call(
        "sonnet",
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=900,
        cache_write_tokens=100,
    )

    text = dashboard._format_dashboard()

    assert "Cache:" in text
    assert "90%" in text


def test_format_dashboard_no_cache_line_without_metrics(tmp_path: Path) -> None:
    """Cache line is hidden when no cache activity."""
    dashboard, tracker, _ = _make_dashboard(tmp_path)
    tracker.record_api_call("sonnet", input_tokens=500, output_tokens=200)

    text = dashboard._format_dashboard()

    assert "Cache:" not in text


async def test_initialize_creates_new_message(tmp_path: Path) -> None:
    dashboard, _, bot = _make_dashboard(tmp_path)

    await dashboard.initialize()

    bot.send_message.assert_awaited_once()
    bot.pin_chat_message.assert_awaited_once()
    assert dashboard._message_id == 42

    # State saved to file
    state_file = tmp_path / "monitoring" / "dashboard_state.json"
    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert data["message_id"] == 42


async def test_initialize_reuses_existing(tmp_path: Path) -> None:
    dashboard, _, bot = _make_dashboard(tmp_path)

    # Pre-save state
    state_dir = tmp_path / "monitoring"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "dashboard_state.json").write_text(json.dumps({"message_id": 99}))

    await dashboard.initialize()

    # Should edit existing, not send new
    bot.edit_message_text.assert_awaited_once()
    bot.send_message.assert_not_awaited()
    assert dashboard._message_id == 99


async def test_initialize_reuses_existing_when_text_is_unchanged(
    tmp_path: Path,
) -> None:
    dashboard, _, bot = _make_dashboard(tmp_path)

    state_dir = tmp_path / "monitoring"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "dashboard_state.json").write_text(json.dumps({"message_id": 99}))
    bot.edit_message_text = AsyncMock(
        side_effect=Exception("Bad Request: message is not modified")
    )

    await dashboard.initialize()

    bot.edit_message_text.assert_awaited_once()
    bot.send_message.assert_not_awaited()
    assert dashboard._message_id == 99


async def test_initialize_creates_new_if_deleted(tmp_path: Path) -> None:
    dashboard, _, bot = _make_dashboard(tmp_path)

    # Pre-save state with old message
    state_dir = tmp_path / "monitoring"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "dashboard_state.json").write_text(json.dumps({"message_id": 99}))

    # Edit fails (message was deleted)
    bot.edit_message_text = AsyncMock(side_effect=Exception("Message not found"))

    await dashboard.initialize()

    # Should fall back to creating new message
    bot.send_message.assert_awaited_once()
    assert dashboard._message_id == 42


def test_rate_limits_updates(tmp_path: Path) -> None:
    dashboard, _, _bot = _make_dashboard(tmp_path, update_interval=30)
    dashboard._message_id = 42

    # First update goes through
    dashboard.schedule_update()
    assert dashboard._last_update > 0

    # Second update within interval is buffered
    dashboard.schedule_update()
    assert dashboard._pending is True


async def test_flush_pending(tmp_path: Path) -> None:
    dashboard, _, bot = _make_dashboard(tmp_path, update_interval=30)
    dashboard._message_id = 42
    dashboard._pending = True

    await dashboard.flush_pending()

    bot.edit_message_text.assert_awaited_once()
    assert dashboard._pending is False


async def test_flush_loop_refreshes_codex_limits_without_usage_event(
    tmp_path: Path,
) -> None:
    calls = 0

    def limits_provider() -> CodexLimitSnapshot:
        nonlocal calls
        calls += 1
        return CodexLimitSnapshot(
            primary_used_percent=float(calls),
            primary_window_minutes=300,
            secondary_used_percent=40.0,
            secondary_window_minutes=10080,
        )

    dashboard, _, bot = _make_dashboard(
        tmp_path,
        update_interval=1,
        codex_limits_provider=limits_provider,
    )
    dashboard._message_id = 42

    task = asyncio.create_task(dashboard._flush_loop())
    try:
        await asyncio.sleep(1.2)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    bot.edit_message_text.assert_awaited()

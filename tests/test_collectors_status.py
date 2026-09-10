"""Tests for collector status file and Phase 3 integration."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from src.collectors.base import CollectorStatus
from src.skills.workspace_session.collector import (
    _write_status_file,
    collect_phase3_sources,
)


@pytest.fixture
def phase3_settings(tmp_path: Path) -> SimpleNamespace:
    workspace = tmp_path / "workspace"
    (workspace / "inbox").mkdir(parents=True)
    (workspace / "knowledge" / "youtube").mkdir(parents=True)
    (workspace / "knowledge" / "channels").mkdir(parents=True)
    return SimpleNamespace(
        workspace_path=str(workspace),
        project_path="",
        git_max_commits=100,
        firefox_profile_path=str(tmp_path / "places.sqlite"),
        chrome_history_path="",
        youtube_takeout_path="",
        youtube_api_key="",
        youtube_scan_enabled=False,
        youtube_transcribe_top_n=3,
        telegram_api_id=0,
        telegram_api_hash="",
        telethon_session_path="",
        monitored_channel_ids="",
        channel_read_delay_seconds=0.0,
        admin_user_id=12345,
    )


def test_status_file_written_after_collection(tmp_path: Path):
    """Status file is created with all collector statuses."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()

    statuses = [
        CollectorStatus(
            name="Browser", success=True, entries_count=42, message="42 entries"
        ),
        CollectorStatus(
            name="YouTube", success=True, entries_count=5, message="5 videos"
        ),
    ]

    today = date(2026, 4, 2)
    _write_status_file(inbox, statuses, today)

    path = inbox / "collectors_status_2026-04-02.md"
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Browser" in content
    assert "✅" in content
    assert "42 entries" in content


def test_failed_collector_shows_error(tmp_path: Path):
    """Failed collector shows ❌ with error reason."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()

    statuses = [
        CollectorStatus(name="Browser", success=False, error="SQLite backup failed"),
    ]

    _write_status_file(inbox, statuses, date(2026, 4, 2))

    content = (inbox / "collectors_status_2026-04-02.md").read_text(encoding="utf-8")
    assert "❌" in content
    assert "SQLite backup failed" in content


def test_partial_failure_shows_warning(tmp_path: Path):
    """Partially failed collector shows ⚠️."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()

    statuses = [
        CollectorStatus(
            name="Channels",
            success=True,
            entries_count=8,
            message="8/10 channels read",
            error="2 skipped: FloodWait",
        ),
    ]

    _write_status_file(inbox, statuses, date(2026, 4, 2))

    content = (inbox / "collectors_status_2026-04-02.md").read_text(encoding="utf-8")
    assert "⚠️" in content
    assert "FloodWait" in content


async def test_collect_phase3_skips_unconfigured(phase3_settings: SimpleNamespace):
    """Collectors without config are skipped."""
    # Only firefox_profile_path is set (but file doesn't exist)
    phase3_settings.firefox_profile_path = ""
    inbox = Path(phase3_settings.workspace_path) / "inbox"

    statuses = await collect_phase3_sources(inbox, phase3_settings)
    assert len(statuses) == 0  # All unconfigured, all skipped


async def test_collect_phase3_browser_failure_isolated(
    phase3_settings: SimpleNamespace,
):
    """Browser collector failure doesn't crash the pipeline."""
    inbox = Path(phase3_settings.workspace_path) / "inbox"

    with patch(
        "src.skills.workspace_session.collector._run_browser_collector",
        AsyncMock(
            return_value=(
                CollectorStatus(name="Browser", success=False, error="test error"),
                [],
            )
        ),
    ):
        statuses = await collect_phase3_sources(inbox, phase3_settings)

    assert len(statuses) == 1
    assert not statuses[0].success


async def test_collect_phase3_passes_recovery_window_to_youtube(
    phase3_settings: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """YouTube Takeout fallback receives the same window as other collectors."""
    from src.skills.workspace_session import collector as workspace_collector

    phase3_settings.firefox_profile_path = ""
    phase3_settings.youtube_takeout_path = "watch-history.json"
    fixed_now = datetime(2026, 5, 24, 16, 0, tzinfo=UTC)
    seen_since: datetime | None = None

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: object | None = None) -> datetime:
            return fixed_now if tz is not None else fixed_now.replace(tzinfo=None)

    async def fake_run_youtube_collector(
        settings: SimpleNamespace,
        episodic: object | None,
        browser_entries: list[dict[str, object]] | None = None,
        since: datetime | None = None,
    ) -> CollectorStatus:
        nonlocal seen_since
        del settings, episodic, browser_entries
        seen_since = since
        return CollectorStatus(name="YouTube", success=True, message="ok")

    monkeypatch.setattr(workspace_collector, "datetime", FixedDateTime)
    monkeypatch.setattr(
        workspace_collector,
        "_run_youtube_collector",
        fake_run_youtube_collector,
    )

    await workspace_collector.collect_phase3_sources(
        Path(phase3_settings.workspace_path) / "inbox",
        phase3_settings,
        lookback_hours=70,
    )

    assert seen_since == fixed_now - timedelta(hours=70)

"""Tests for collector base types and infrastructure."""

from __future__ import annotations

from datetime import UTC, datetime

from src.collectors.base import CollectorStatus, InboxEntry


def test_collector_status_success_format():
    status = CollectorStatus(
        name="Browser",
        success=True,
        entries_count=42,
        message="42 entries (Firefox 30, Chrome 12)",
    )
    line = status.format_line()
    assert line == "- Browser: ✅ 42 entries (Firefox 30, Chrome 12)"


def test_collector_status_warning_format():
    status = CollectorStatus(
        name="YouTube",
        success=True,
        entries_count=5,
        message="5 videos analyzed",
        error="API quota at 80%, used fallback",
    )
    line = status.format_line()
    assert "⚠️" in line
    assert "API quota" in line


def test_collector_status_failure_format():
    status = CollectorStatus(
        name="Channels",
        success=False,
        error="Telethon connection failed",
    )
    line = status.format_line()
    assert "❌" in line
    assert "Telethon connection failed" in line


def test_inbox_entry_defaults():
    entry = InboxEntry(
        content="test",
        source="browser",
        timestamp=datetime.now(tz=UTC),
    )
    assert entry.importance == 0.5
    assert entry.metadata == {}


def test_inbox_entry_with_metadata():
    entry = InboxEntry(
        content="test",
        source="youtube",
        timestamp=datetime.now(tz=UTC),
        importance=0.8,
        metadata={"video_id": "abc123"},
    )
    assert entry.importance == 0.8
    assert entry.metadata["video_id"] == "abc123"

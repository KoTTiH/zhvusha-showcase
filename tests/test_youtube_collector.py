"""Tests for YouTube collector."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from src.collectors.youtube import (
    YouTubeAnalysis,
    YouTubeCollector,
    _extract_video_id,
    _get_transcript,
    _search_youtube,
)


@pytest.fixture
def yt_settings(tmp_path: Path) -> SimpleNamespace:
    workspace = tmp_path / "workspace"
    (workspace / "inbox").mkdir(parents=True)
    (workspace / "knowledge" / "youtube").mkdir(parents=True)
    return SimpleNamespace(
        workspace_path=str(workspace),
        youtube_takeout_path=str(tmp_path / "watch-history.json"),
        youtube_api_key="",
        youtube_scan_enabled=False,
        youtube_transcribe_top_n=3,
        admin_user_id=12345,
    )


def _make_takeout_json(path: Path, entries: list[dict[str, str]]) -> None:
    """Create a Google Takeout watch-history.json file."""
    path.write_text(json.dumps(entries), encoding="utf-8")


_SAMPLE_TAKEOUT = [
    {
        "title": "Watched Building AI Agents",
        "titleUrl": "https://www.youtube.com/watch?v=abc123",
        "time": "2026-04-01T14:30:00.000Z",
        "subtitles": [{"name": "TechChannel"}],
    },
    {
        "title": "Watched VLESS Tutorial",
        "titleUrl": "https://www.youtube.com/watch?v=def456",
        "time": "2026-04-01T18:00:00.000Z",
        "subtitles": [{"name": "VPNGuru"}],
    },
]


# --- Helper functions ---


def test_extract_video_id_standard_url() -> None:
    assert _extract_video_id("https://www.youtube.com/watch?v=abc123") == "abc123"


def test_extract_video_id_short_url() -> None:
    assert _extract_video_id("https://youtu.be/abc123") == "abc123"


def test_extract_video_id_no_id() -> None:
    assert _extract_video_id("https://youtube.com/") == ""


def test_search_youtube_import_error() -> None:
    """Returns empty list when library not installed."""
    with patch.dict("sys.modules", {"youtubesearchpython": None}):
        result = _search_youtube("test")
        assert result == []


def test_get_transcript_import_error() -> None:
    """Returns None when library not installed."""
    with patch.dict("sys.modules", {"youtube_transcript_api": None}):
        result = _get_transcript("abc123")
        assert result is None


# --- from_browser_history ---


def test_from_browser_history_filters_youtube() -> None:
    entries = [
        {
            "url": "https://www.youtube.com/watch?v=vid1",
            "title": "Video 1",
            "domain": "youtube.com",
            "visit_time": None,
        },
        {
            "url": "https://github.com/test",
            "title": "GitHub",
            "domain": "github.com",
            "visit_time": None,
        },
        {
            "url": "https://youtu.be/vid2",
            "title": "Video 2",
            "domain": "youtu.be",
            "visit_time": None,
        },
    ]
    result = YouTubeCollector.from_browser_history(entries)
    assert len(result) == 2
    ids = {e.video_id for e in result}
    assert ids == {"vid1", "vid2"}


def test_from_browser_history_deduplicates() -> None:
    entries = [
        {
            "url": "https://www.youtube.com/watch?v=vid1",
            "title": "V1",
            "domain": "youtube.com",
            "visit_time": None,
        },
        {
            "url": "https://www.youtube.com/watch?v=vid1",
            "title": "V1 again",
            "domain": "youtube.com",
            "visit_time": None,
        },
    ]
    result = YouTubeCollector.from_browser_history(entries)
    assert len(result) == 1


def test_from_browser_history_empty() -> None:
    assert YouTubeCollector.from_browser_history([]) == []


async def test_parse_takeout_watch_history(yt_settings: SimpleNamespace):
    """Parses Google Takeout watch-history.json correctly."""
    _make_takeout_json(Path(yt_settings.youtube_takeout_path), _SAMPLE_TAKEOUT)

    collector = YouTubeCollector(yt_settings)
    since = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    entries = await collector.parse_watch_history(since=since)

    assert len(entries) == 2
    assert entries[0].title == "Building AI Agents"
    assert entries[0].channel == "TechChannel"
    assert entries[0].video_id == "abc123"


async def test_parse_takeout_filters_by_date(yt_settings: SimpleNamespace):
    """Only entries after `since` are returned."""
    _make_takeout_json(Path(yt_settings.youtube_takeout_path), _SAMPLE_TAKEOUT)

    collector = YouTubeCollector(yt_settings)
    since = datetime(2026, 4, 1, 16, 0, tzinfo=UTC)
    entries = await collector.parse_watch_history(since=since)

    assert len(entries) == 1
    assert entries[0].title == "VLESS Tutorial"


async def test_scan_feed_returns_results(yt_settings: SimpleNamespace):
    """Feed scanning returns results via youtube-search-python mock."""
    collector = YouTubeCollector(yt_settings)

    mock_result = {
        "result": [
            {
                "id": "xyz789",
                "title": "aiogram 3 new features",
                "channel": {"name": "PythonDev"},
                "link": "https://youtube.com/watch?v=xyz789",
                "duration": "30:00",
            },
        ]
    }

    with patch(
        "src.collectors.youtube._search_youtube",
        return_value=mock_result["result"],
    ):
        entries = await collector.scan_feed(interests=["aiogram"])

    assert len(entries) == 1
    assert entries[0].video_id == "xyz789"
    assert entries[0].channel == "PythonDev"


async def test_transcribe_and_analyze(yt_settings: SimpleNamespace):
    """Transcription + analysis returns YouTubeAnalysis."""
    collector = YouTubeCollector(yt_settings)

    mock_transcript = [
        {"text": "В этом видео мы разберём aiogram 3.", "start": 0.0},
        {"text": "Новые фичи включают middleware chaining.", "start": 5.0},
    ]

    mock_llm = AsyncMock(return_value="Ключевые идеи:\n- middleware chaining\n- FSM")

    with (
        patch(
            "src.collectors.youtube._get_transcript",
            return_value=mock_transcript,
        ),
        patch.object(collector, "_call_llm", mock_llm),
    ):
        analysis = await collector.transcribe_and_analyze(
            "https://youtube.com/watch?v=test1"
        )

    assert analysis is not None
    assert analysis.video_id == "test1"
    assert analysis.transcript_length > 0


async def test_saves_analysis_to_knowledge(yt_settings: SimpleNamespace):
    """Analysis is saved to knowledge/youtube/ directory."""
    collector = YouTubeCollector(yt_settings)

    analysis = YouTubeAnalysis(
        video_id="test1",
        title="Test Video",
        key_ideas=["idea 1", "idea 2"],
        useful_for_nikita="Very useful",
        tools_mentioned=["tool1"],
        transcript_length=500,
    )

    collector._save_analysis(analysis)

    knowledge_file = (
        Path(yt_settings.workspace_path) / "knowledge" / "youtube" / "test1.md"
    )
    assert knowledge_file.exists()
    content = knowledge_file.read_text(encoding="utf-8")
    assert "Test Video" in content
    assert "idea 1" in content


async def test_handles_missing_subtitles(yt_settings: SimpleNamespace):
    """Returns None when transcript is not available."""
    collector = YouTubeCollector(yt_settings)

    with patch(
        "src.collectors.youtube._get_transcript",
        return_value=None,
    ):
        analysis = await collector.transcribe_and_analyze(
            "https://youtube.com/watch?v=nosubs"
        )

    assert analysis is None


async def test_writes_inbox_summary(
    yt_settings: SimpleNamespace, mock_episodic: AsyncMock
):
    """collect_and_save writes inbox summary and records episodes."""
    _make_takeout_json(Path(yt_settings.youtube_takeout_path), _SAMPLE_TAKEOUT)

    collector = YouTubeCollector(yt_settings)

    # Use explicit since to include test data (default 24h window misses old entries)
    watched = await collector.parse_watch_history(
        since=datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    )
    with (
        patch.object(
            collector,
            "parse_watch_history",
            AsyncMock(return_value=watched),
        ),
        patch.object(collector, "scan_feed", AsyncMock(return_value=[])),
    ):
        summary = await collector.collect_and_save(episodic=mock_episodic)

    assert "Building AI Agents" in summary or "AI Agents" in summary
    inbox_files = list(
        (Path(yt_settings.workspace_path) / "inbox").glob("youtube_*.md")
    )
    assert len(inbox_files) == 1
    assert mock_episodic.record.await_count >= 1


async def test_collect_and_save_passes_since_to_takeout_fallback(
    yt_settings: SimpleNamespace,
) -> None:
    """The Takeout fallback must honor the recovery window from /morning."""
    collector = YouTubeCollector(yt_settings)
    since = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
    parse_watch_history = AsyncMock(return_value=[])

    with (
        patch.object(collector, "parse_watch_history", parse_watch_history),
        patch.object(collector, "scan_feed", AsyncMock(return_value=[])),
    ):
        await collector.collect_and_save(browser_entries=None, since=since)

    parse_watch_history.assert_awaited_once_with(since=since)

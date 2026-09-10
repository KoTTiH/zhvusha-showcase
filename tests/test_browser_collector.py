"""Tests for browser history collector."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from src.collectors.browser import (
    BrowserCollectionResult,
    BrowserEntry,
    BrowserHistoryCollector,
)


def _make_firefox_db(db_path: Path, entries: list[tuple[str, str, int, int]]) -> None:
    """Create a minimal Firefox places.sqlite with history entries.

    entries: list of (url, title, visit_count, visit_date_us)
    Firefox stores timestamps as microseconds since epoch.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE moz_places "
        "(id INTEGER PRIMARY KEY, url TEXT, title TEXT, visit_count INTEGER, "
        "last_visit_date INTEGER)"
    )
    conn.execute(
        "CREATE TABLE moz_historyvisits "
        "(id INTEGER PRIMARY KEY, place_id INTEGER, visit_date INTEGER, "
        "visit_type INTEGER)"
    )
    for i, (url, title, visit_count, visit_date_us) in enumerate(entries, 1):
        conn.execute(
            "INSERT INTO moz_places VALUES (?, ?, ?, ?, ?)",
            (i, url, title, visit_count, visit_date_us),
        )
        conn.execute(
            "INSERT INTO moz_historyvisits VALUES (?, ?, ?, ?)",
            (i, i, visit_date_us, 1),
        )
    conn.commit()
    conn.close()


def _make_chrome_db(db_path: Path, entries: list[tuple[str, str, int, int]]) -> None:
    """Create a minimal Chrome History sqlite with history entries.

    entries: list of (url, title, visit_count, last_visit_time_chrome)
    Chrome stores timestamps as microseconds since 1601-01-01.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE urls "
        "(id INTEGER PRIMARY KEY, url TEXT, title TEXT, visit_count INTEGER, "
        "last_visit_time INTEGER)"
    )
    conn.execute(
        "CREATE TABLE visits "
        "(id INTEGER PRIMARY KEY, url INTEGER, visit_time INTEGER, transition INTEGER)"
    )
    for i, (url, title, visit_count, chrome_ts) in enumerate(entries, 1):
        conn.execute(
            "INSERT INTO urls VALUES (?, ?, ?, ?, ?)",
            (i, url, title, visit_count, chrome_ts),
        )
        conn.execute(
            "INSERT INTO visits VALUES (?, ?, ?, ?)",
            (i, i, chrome_ts, 0),
        )
    conn.commit()
    conn.close()


# Unix timestamp for 2026-04-01 12:00:00 UTC
_TS_2026 = 1775044800
# As Firefox microseconds
_FIREFOX_TS = _TS_2026 * 1_000_000
# As Chrome microseconds (since 1601-01-01)
_CHROME_TS = (_TS_2026 + 11644473600) * 1_000_000


@pytest.fixture
def collector_settings(tmp_path: Path) -> SimpleNamespace:
    firefox_db = tmp_path / "places.sqlite"
    chrome_db = tmp_path / "History"
    return SimpleNamespace(
        firefox_profile_path=str(firefox_db),
        chrome_history_path=str(chrome_db),
        workspace_path=str(tmp_path / "workspace"),
        admin_user_id=12345,
    )


async def test_read_firefox_history(
    tmp_path: Path, collector_settings: SimpleNamespace
):
    """Firefox places.sqlite is read correctly."""
    db_path = Path(collector_settings.firefox_profile_path)
    _make_firefox_db(
        db_path,
        [
            ("https://github.com/test", "GitHub Test", 3, _FIREFOX_TS),
            ("https://kwork.ru/projects", "Kwork Projects", 5, _FIREFOX_TS),
        ],
    )

    collector = BrowserHistoryCollector(collector_settings)
    since = datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
    entries = collector._read_firefox(since, limit=100)

    assert len(entries) == 2
    assert entries[0].browser == "firefox"
    assert entries[0].domain == "github.com"


async def test_read_chrome_history(tmp_path: Path, collector_settings: SimpleNamespace):
    """Chrome History sqlite is read with correct timestamp conversion."""
    db_path = Path(collector_settings.chrome_history_path)
    _make_chrome_db(
        db_path,
        [
            ("https://youtube.com/watch?v=abc", "AI Video", 1, _CHROME_TS),
        ],
    )

    collector = BrowserHistoryCollector(collector_settings)
    since = datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
    entries = collector._read_chrome(since, limit=100)

    assert len(entries) == 1
    assert entries[0].browser == "chrome"
    assert entries[0].domain == "youtube.com"
    # Verify timestamp conversion: year should be 2026
    assert entries[0].visit_time.year == 2026


async def test_chrome_timestamp_conversion_validation(
    tmp_path: Path, collector_settings: SimpleNamespace
):
    """Chrome entries with invalid timestamps (year outside 2020-2030) are skipped."""
    db_path = Path(collector_settings.chrome_history_path)
    # Invalid timestamp: way too small (year ~1601)
    _make_chrome_db(
        db_path,
        [
            ("https://example.com", "Bad Entry", 1, 1000),
        ],
    )

    collector = BrowserHistoryCollector(collector_settings)
    since = datetime(2020, 1, 1, 0, 0, tzinfo=UTC)
    entries = collector._read_chrome(since, limit=100)

    assert len(entries) == 0


async def test_deduplicates_same_url(
    tmp_path: Path, collector_settings: SimpleNamespace
):
    """Same URL in both browsers is deduplicated."""
    firefox_db = Path(collector_settings.firefox_profile_path)
    chrome_db = Path(collector_settings.chrome_history_path)

    _make_firefox_db(
        firefox_db,
        [
            ("https://github.com/test", "GitHub", 2, _FIREFOX_TS),
        ],
    )
    _make_chrome_db(
        chrome_db,
        [
            ("https://github.com/test", "GitHub", 3, _CHROME_TS),
        ],
    )

    collector = BrowserHistoryCollector(collector_settings)
    since = datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
    result = await collector.collect(since=since)

    # Should be deduplicated to 1 entry with combined visit count
    assert len(result.entries) == 1
    assert result.entries[0].visit_count == 5  # 2 + 3


async def test_extracts_top_domains(
    tmp_path: Path, collector_settings: SimpleNamespace
):
    """Top domains are extracted from results."""
    firefox_db = Path(collector_settings.firefox_profile_path)
    _make_firefox_db(
        firefox_db,
        [
            ("https://github.com/repo1", "Repo 1", 1, _FIREFOX_TS),
            ("https://github.com/repo2", "Repo 2", 1, _FIREFOX_TS),
            ("https://kwork.ru/p1", "Kwork 1", 1, _FIREFOX_TS),
        ],
    )

    collector = BrowserHistoryCollector(collector_settings)
    since = datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
    result = await collector.collect(since=since)

    assert result.top_domains[0] == ("github.com", 2)


async def test_extracts_topics_from_domains(
    tmp_path: Path, collector_settings: SimpleNamespace
):
    """Known domains map to topics."""
    firefox_db = Path(collector_settings.firefox_profile_path)
    _make_firefox_db(
        firefox_db,
        [
            ("https://github.com/test", "Some Repo", 1, _FIREFOX_TS),
            ("https://stackoverflow.com/q/123", "Debug Q", 1, _FIREFOX_TS),
            ("https://kwork.ru/projects", "Kwork", 1, _FIREFOX_TS),
        ],
    )

    collector = BrowserHistoryCollector(collector_settings)
    since = datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
    result = await collector.collect(since=since)

    assert "programming" in result.topics
    assert "debugging" in result.topics
    assert "freelancing" in result.topics


async def test_writes_inbox_summary(
    tmp_path: Path, collector_settings: SimpleNamespace
):
    """collect_and_save writes a summary file to inbox/."""
    # Use a recent timestamp so entries fall within the default 24h window
    recent_ts = int(datetime.now(tz=UTC).timestamp()) * 1_000_000
    firefox_db = Path(collector_settings.firefox_profile_path)
    _make_firefox_db(
        firefox_db,
        [
            ("https://github.com/test", "GitHub Test", 3, recent_ts),
        ],
    )

    workspace = Path(collector_settings.workspace_path)
    (workspace / "inbox").mkdir(parents=True)

    collector = BrowserHistoryCollector(collector_settings)
    summary = await collector.collect_and_save(episodic=None)

    assert "github.com" in summary.lower() or "GitHub" in summary
    # Check inbox file was created
    inbox_files = list((workspace / "inbox").glob("browser_*.md"))
    assert len(inbox_files) == 1


async def test_records_episodes_for_notable_entries(
    tmp_path: Path, collector_settings: SimpleNamespace, mock_episodic: AsyncMock
):
    """Only notable entries (3+ visits) are recorded as episodes."""
    recent_ts = int(datetime.now(tz=UTC).timestamp()) * 1_000_000
    firefox_db = Path(collector_settings.firefox_profile_path)
    _make_firefox_db(
        firefox_db,
        [
            ("https://github.com/hot", "Hot Repo", 5, recent_ts),  # notable: 5 visits
            ("https://example.com/once", "One Visit", 1, recent_ts),  # not notable
        ],
    )

    workspace = Path(collector_settings.workspace_path)
    (workspace / "inbox").mkdir(parents=True)

    collector = BrowserHistoryCollector(collector_settings)
    await collector.collect_and_save(episodic=mock_episodic)

    # Only the 5-visit entry should be recorded
    assert mock_episodic.record.await_count >= 1
    call_kwargs = mock_episodic.record.call_args_list[0].kwargs
    assert call_kwargs["source"] == "browser"
    assert (
        "Hot Repo" in call_kwargs["content"] or "github.com" in call_kwargs["content"]
    )


# --- Browser locked (open) scenarios ---


async def test_firefox_locked_returns_empty(
    tmp_path: Path, collector_settings: SimpleNamespace
):
    """When Firefox is open and locks the DB, _read_firefox raises and collect() handles it."""
    # Create a valid DB so the path-exists check passes
    db_path = Path(collector_settings.firefox_profile_path)
    _make_firefox_db(db_path, [("https://example.com", "Test", 1, _FIREFOX_TS)])

    collector = BrowserHistoryCollector(collector_settings)
    # Simulate browser holding exclusive lock → _safe_copy_sqlite raises OSError
    with patch.object(
        BrowserHistoryCollector,
        "_safe_copy_sqlite",
        side_effect=OSError("SQLite backup timed out — close the browser"),
    ):
        since = datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
        result = await collector.collect(since=since)

    # Firefox failed, Chrome path is empty → zero entries, no crash
    assert result.entries == []
    assert result.firefox_count == 0


async def test_chrome_locked_returns_empty(
    tmp_path: Path, collector_settings: SimpleNamespace
):
    """When Chrome is open and locks the DB, collect() handles it gracefully."""
    db_path = Path(collector_settings.chrome_history_path)
    _make_chrome_db(db_path, [("https://example.com", "Test", 1, _CHROME_TS)])
    # Clear Firefox path so only Chrome is attempted
    collector_settings.firefox_profile_path = ""

    collector = BrowserHistoryCollector(collector_settings)
    with patch.object(
        BrowserHistoryCollector,
        "_safe_copy_sqlite",
        side_effect=OSError("SQLite backup timed out — close the browser"),
    ):
        since = datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
        result = await collector.collect(since=since)

    assert result.entries == []
    assert result.chrome_count == 0


async def test_both_locked_collect_and_save_returns_message(
    tmp_path: Path, collector_settings: SimpleNamespace
):
    """When both browsers are locked, collect_and_save returns a friendly message."""
    db_path_ff = Path(collector_settings.firefox_profile_path)
    db_path_cr = Path(collector_settings.chrome_history_path)
    _make_firefox_db(db_path_ff, [("https://a.com", "A", 1, _FIREFOX_TS)])
    _make_chrome_db(db_path_cr, [("https://b.com", "B", 1, _CHROME_TS)])

    workspace = Path(collector_settings.workspace_path)
    (workspace / "inbox").mkdir(parents=True)

    collector = BrowserHistoryCollector(collector_settings)
    with patch.object(
        BrowserHistoryCollector,
        "_safe_copy_sqlite",
        side_effect=OSError("SQLite backup timed out — close the browser"),
    ):
        summary = await collector.collect_and_save(episodic=None)

    assert summary == "No browser history entries found."


def _make_entry(
    url: str,
    title: str,
    domain: str,
    hour: int,
    visit_count: int = 1,
) -> BrowserEntry:
    """Helper to build a BrowserEntry at a given hour on 2026-04-01."""
    return BrowserEntry(
        url=url,
        title=title,
        visit_time=datetime(2026, 4, 1, hour, 0, tzinfo=UTC),
        visit_count=visit_count,
        browser="firefox",
        domain=domain,
    )


# --- Pattern extraction tests ---


def test_extract_patterns_domain_concentration() -> None:
    """3+ visits to the same domain within a 2-hour sliding window → concentration pattern."""
    entries = [
        _make_entry("https://kwork.ru/p1", "Kwork 1", "kwork.ru", hour=9),
        _make_entry("https://kwork.ru/p2", "Kwork 2", "kwork.ru", hour=9),
        _make_entry("https://kwork.ru/p3", "Kwork 3", "kwork.ru", hour=10),
    ]
    patterns = BrowserHistoryCollector._extract_patterns(entries)

    # Should detect concentration (all 3 within 2 hours: 09:00-10:00)
    concentration = [p for p in patterns if "kwork.ru" in p]
    assert len(concentration) == 1
    assert "3" in concentration[0]
    assert "09:00" in concentration[0]


def test_extract_patterns_no_concentration_below_threshold() -> None:
    """Fewer than 3 visits to a domain in any window → no pattern."""
    entries = [
        _make_entry("https://kwork.ru/p1", "Kwork 1", "kwork.ru", hour=9),
        _make_entry("https://kwork.ru/p2", "Kwork 2", "kwork.ru", hour=10),
        _make_entry("https://github.com/r1", "GitHub 1", "github.com", hour=9),
    ]
    patterns = BrowserHistoryCollector._extract_patterns(entries)

    assert patterns == []


def test_extract_patterns_topic_deep_dive() -> None:
    """3+ entries sharing a topic (across different domains) → deep-dive pattern."""
    entries = [
        _make_entry("https://github.com/r1", "Repo 1", "github.com", hour=10),
        _make_entry("https://gitlab.com/r2", "Repo 2", "gitlab.com", hour=11),
        _make_entry("https://myapi.dev/docs", "API Docs", "myapi.dev", hour=13),
    ]
    patterns = BrowserHistoryCollector._extract_patterns(entries)

    # github and gitlab share "programming"; myapi.dev matches "api" → "programming"
    deep_dive = [p for p in patterns if "programming" in p.lower()]
    assert len(deep_dive) == 1
    assert "3" in deep_dive[0]


def test_extract_patterns_concentration_suppresses_topic_for_same_domain() -> None:
    """When a domain concentration is detected, a single-domain topic deep-dive
    for the same domain is not double-reported."""
    entries = [
        _make_entry("https://kwork.ru/p1", "Kwork 1", "kwork.ru", hour=9),
        _make_entry("https://kwork.ru/p2", "Kwork 2", "kwork.ru", hour=9),
        _make_entry("https://kwork.ru/p3", "Kwork 3", "kwork.ru", hour=10),
    ]
    patterns = BrowserHistoryCollector._extract_patterns(entries)

    # Should report domain concentration but NOT a separate freelancing deep-dive
    # (because kwork.ru is already the covered domain)
    concentration_patterns = [p for p in patterns if "kwork.ru" in p]
    deep_dive_patterns = [p for p in patterns if "freelancing" in p.lower()]
    assert len(concentration_patterns) == 1
    assert len(deep_dive_patterns) == 0


def test_format_summary_includes_patterns_section() -> None:
    """_format_summary places a ## Patterns section before ## Topics when patterns exist."""
    entries = [
        _make_entry("https://kwork.ru/p1", "Kwork 1", "kwork.ru", hour=9),
        _make_entry("https://kwork.ru/p2", "Kwork 2", "kwork.ru", hour=9),
        _make_entry("https://kwork.ru/p3", "Kwork 3", "kwork.ru", hour=10),
    ]
    result = BrowserCollectionResult(
        entries=entries,
        firefox_count=3,
        chrome_count=0,
        top_domains=[("kwork.ru", 3)],
        topics=["freelancing"],
    )
    summary = BrowserHistoryCollector._format_summary(result, "2026-04-01")

    assert "## Patterns" in summary
    assert "kwork.ru" in summary
    # Patterns section must appear before Topics section
    assert summary.index("## Patterns") < summary.index("## Topics")


def test_format_summary_no_patterns_section_when_empty() -> None:
    """_format_summary omits ## Patterns when there are no patterns."""
    entries = [
        _make_entry("https://kwork.ru/p1", "Kwork 1", "kwork.ru", hour=9),
    ]
    result = BrowserCollectionResult(
        entries=entries,
        firefox_count=1,
        chrome_count=0,
        top_domains=[("kwork.ru", 1)],
        topics=["freelancing"],
    )
    summary = BrowserHistoryCollector._format_summary(result, "2026-04-01")

    assert "## Patterns" not in summary


async def test_locked_browser_cleans_temp_file(
    tmp_path: Path, collector_settings: SimpleNamespace
):
    """Temp file is cleaned up even when _safe_copy_sqlite raises."""
    db_path = Path(collector_settings.firefox_profile_path)
    _make_firefox_db(db_path, [("https://example.com", "Test", 1, _FIREFOX_TS)])

    collector = BrowserHistoryCollector(collector_settings)
    created_tmp: list[str] = []

    def fake_copy(_self: object, _src: str, dst: str, timeout: int = 5) -> None:
        # Create the file (simulating partial write before timeout)
        Path(dst).write_bytes(b"partial")
        created_tmp.append(dst)
        raise OSError("SQLite backup timed out — close the browser")

    with patch.object(BrowserHistoryCollector, "_safe_copy_sqlite", fake_copy):
        since = datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
        result = await collector.collect(since=since)

    assert result.entries == []
    # The temp file should have been cleaned up by the finally block
    assert len(created_tmp) == 1
    assert not Path(created_tmp[0]).exists()

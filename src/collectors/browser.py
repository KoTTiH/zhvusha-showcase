"""Browser history collector for Firefox and Chrome."""

from __future__ import annotations

import sqlite3
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import structlog

if TYPE_CHECKING:
    from types import SimpleNamespace

    from src.memory import EpisodicMemoryProtocol as EpisodicMemory

logger = structlog.get_logger()

# Chrome epoch offset: microseconds between 1601-01-01 and 1970-01-01
_CHROME_EPOCH_OFFSET_S = 11644473600

# Domain → topic mapping for known sites
_DOMAIN_TOPICS: dict[str, str] = {
    "github.com": "programming",
    "gitlab.com": "programming",
    "stackoverflow.com": "debugging",
    "kwork.ru": "freelancing",
    "fl.ru": "freelancing",
    "freelance.habr.com": "freelancing",
    "youtube.com": "video content",
    "habr.com": "tech articles",
    "dev.to": "tech articles",
    "medium.com": "tech articles",
    "docs.python.org": "python",
    "pypi.org": "python",
    "npmjs.com": "javascript",
    "reddit.com": "community",
    "t.me": "telegram",
    "web.telegram.org": "telegram",
    "claude.ai": "ai tools",
    "platform.claude.com": "ai tools",
    "chatgpt.com": "ai tools",
    "gemini.google.com": "ai tools",
    "chat.deepseek.com": "ai tools",
    "yandex.ru": "search",
    "google.com": "search",
    "duckduckgo.com": "search",
}

# Keyword → topic fallback for domains not in the map
_DOMAIN_KEYWORD_TOPICS: dict[str, str] = {
    "docs": "documentation",
    "wiki": "documentation",
    "api": "programming",
    "git": "programming",
    "code": "programming",
    "dev": "programming",
    "ai": "ai tools",
    "llm": "ai tools",
    "chat": "communication",
    "mail": "communication",
    "freelance": "freelancing",
    "shop": "shopping",
    "store": "shopping",
    "news": "news",
    "blog": "tech articles",
}

# Minimum visits to consider an entry "notable" for episode recording
_NOTABLE_VISIT_THRESHOLD = 3


@dataclass
class BrowserEntry:
    url: str
    title: str
    visit_time: datetime
    visit_count: int
    browser: str  # "firefox" | "chrome"
    domain: str


@dataclass
class BrowserCollectionResult:
    entries: list[BrowserEntry]
    firefox_count: int
    chrome_count: int
    top_domains: list[tuple[str, int]]
    topics: list[str]


class BrowserHistoryCollector:
    """Reads browser history from Firefox and Chrome SQLite files.

    IMPORTANT: browsers lock their SQLite files while running.
    Uses sqlite3 backup API for WAL-safe copying.
    """

    def __init__(self, config: SimpleNamespace) -> None:
        self._firefox_path = str(Path(config.firefox_profile_path).expanduser())
        self._chrome_path = str(Path(config.chrome_history_path).expanduser())
        self._workspace = Path(
            getattr(config, "workspace_path", "~/zhvusha-workspace")
        ).expanduser()
        self._admin_user_id = getattr(config, "admin_user_id", 0)

    async def collect(
        self,
        since: datetime | None = None,
        limit: int = 200,
    ) -> BrowserCollectionResult:
        """Collect browser history from both browsers."""
        if since is None:
            since = datetime.now(tz=UTC) - timedelta(hours=24)

        firefox_entries: list[BrowserEntry] = []
        chrome_entries: list[BrowserEntry] = []

        if self._firefox_path:
            try:
                firefox_entries = self._read_firefox(since, limit)
            except OSError as exc:
                # SQLite lock / backup timeout = browser running — expected, quiet warning
                logger.warning("browser_firefox_locked", error=str(exc))
            except Exception:
                logger.warning("browser_firefox_read_failed", exc_info=True)

        if self._chrome_path:
            try:
                chrome_entries = self._read_chrome(since, limit)
            except OSError as exc:
                logger.warning("browser_chrome_locked", error=str(exc))
            except Exception:
                logger.warning("browser_chrome_read_failed", exc_info=True)

        all_entries = self._deduplicate(firefox_entries + chrome_entries)
        top_domains = self._extract_top_domains(all_entries)
        topics = self._extract_topics(all_entries)

        return BrowserCollectionResult(
            entries=all_entries,
            firefox_count=len(firefox_entries),
            chrome_count=len(chrome_entries),
            top_domains=top_domains,
            topics=topics,
        )

    def _read_firefox(self, since: datetime, limit: int) -> list[BrowserEntry]:
        """Read Firefox history from places.sqlite."""
        src_path = Path(self._firefox_path)
        if not src_path.exists():
            logger.warning("browser_firefox_not_found", path=str(src_path))
            return []

        since_us = int(since.timestamp() * 1_000_000)

        with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
            tmp_path = tmp.name

        entries: list[BrowserEntry] = []
        try:
            self._safe_copy_sqlite(str(src_path), tmp_path)
            conn = sqlite3.connect(tmp_path)
            rows = conn.execute(
                "SELECT p.url, p.title, p.visit_count, h.visit_date "
                "FROM moz_places p "
                "JOIN moz_historyvisits h ON h.place_id = p.id "
                "WHERE h.visit_date > ? "
                "ORDER BY h.visit_date DESC LIMIT ?",
                (since_us, limit),
            ).fetchall()
            conn.close()

            for url, title, visit_count, visit_date_us in rows:
                visit_time = datetime.fromtimestamp(visit_date_us / 1_000_000, tz=UTC)
                domain = urlparse(url).netloc.removeprefix("www.")
                entries.append(
                    BrowserEntry(
                        url=url,
                        title=title or "",
                        visit_time=visit_time,
                        visit_count=visit_count or 1,
                        browser="firefox",
                        domain=domain,
                    )
                )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        logger.info("browser_firefox_read", count=len(entries))
        return entries

    def _read_chrome(self, since: datetime, limit: int) -> list[BrowserEntry]:
        """Read Chrome history from History sqlite.

        Chrome stores timestamps as microseconds since 1601-01-01.
        """
        src_path = Path(self._chrome_path)
        if not src_path.exists():
            logger.warning("browser_chrome_not_found", path=str(src_path))
            return []

        since_chrome = (int(since.timestamp()) + _CHROME_EPOCH_OFFSET_S) * 1_000_000

        with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
            tmp_path = tmp.name

        entries: list[BrowserEntry] = []
        try:
            self._safe_copy_sqlite(str(src_path), tmp_path)
            conn = sqlite3.connect(tmp_path)
            rows = conn.execute(
                "SELECT u.url, u.title, u.visit_count, v.visit_time "
                "FROM urls u "
                "JOIN visits v ON v.url = u.id "
                "WHERE v.visit_time > ? "
                "ORDER BY v.visit_time DESC LIMIT ?",
                (since_chrome, limit),
            ).fetchall()
            conn.close()

            for url, title, visit_count, chrome_ts in rows:
                unix_ts = (chrome_ts / 1_000_000) - _CHROME_EPOCH_OFFSET_S
                visit_time = datetime.fromtimestamp(unix_ts, tz=UTC)

                # Validate year is reasonable (2020-2030)
                if not (2020 <= visit_time.year <= 2030):
                    logger.warning(
                        "browser_chrome_invalid_timestamp",
                        raw_ts=chrome_ts,
                        year=visit_time.year,
                    )
                    continue

                domain = urlparse(url).netloc.removeprefix("www.")
                entries.append(
                    BrowserEntry(
                        url=url,
                        title=title or "",
                        visit_time=visit_time,
                        visit_count=visit_count or 1,
                        browser="chrome",
                        domain=domain,
                    )
                )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        logger.info("browser_chrome_read", count=len(entries))
        return entries

    @staticmethod
    def _safe_copy_sqlite(src: str, dst: str, timeout: int = 5) -> None:
        """Copy SQLite file using backup API (WAL-safe).

        Uses a subprocess to avoid hanging when browser
        holds an exclusive lock (common with WAL mode).
        """
        import subprocess
        import sys

        script = (
            f"import sqlite3; "
            f"s=sqlite3.connect({src!r},timeout={timeout}); "
            f"d=sqlite3.connect({dst!r}); "
            f"s.backup(d); d.close(); s.close()"
        )
        try:
            subprocess.run(
                [sys.executable, "-c", script],
                timeout=timeout,
                check=True,
                capture_output=True,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "browser_sqlite_backup_timeout",
                src=src,
                hint="browser is running and holds exclusive lock",
            )
            raise OSError(  # noqa: B904
                f"SQLite backup timed out for {src} — close the browser"
            )
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "browser_sqlite_backup_failed",
                src=src,
                stderr=exc.stderr.decode()[:200],
            )
            raise

    @staticmethod
    def _deduplicate(entries: list[BrowserEntry]) -> list[BrowserEntry]:
        """Deduplicate entries by URL, merging visit counts."""
        seen: dict[str, BrowserEntry] = {}
        for entry in entries:
            if entry.url in seen:
                existing = seen[entry.url]
                existing.visit_count += entry.visit_count
                if entry.visit_time > existing.visit_time:
                    existing.visit_time = entry.visit_time
            else:
                seen[entry.url] = BrowserEntry(
                    url=entry.url,
                    title=entry.title,
                    visit_time=entry.visit_time,
                    visit_count=entry.visit_count,
                    browser=entry.browser,
                    domain=entry.domain,
                )
        return list(seen.values())

    @staticmethod
    def _extract_top_domains(
        entries: list[BrowserEntry],
    ) -> list[tuple[str, int]]:
        """Extract top domains sorted by visit count."""
        counter: Counter[str] = Counter()
        for e in entries:
            counter[e.domain] += 1
        return counter.most_common(10)

    @staticmethod
    def _extract_topics(entries: list[BrowserEntry]) -> list[str]:
        """Extract topics from domains using known mapping + keyword fallback."""
        topics: set[str] = set()
        for e in entries:
            if e.domain in _DOMAIN_TOPICS:
                topics.add(_DOMAIN_TOPICS[e.domain])
                continue
            for keyword, topic in _DOMAIN_KEYWORD_TOPICS.items():
                if keyword in e.domain:
                    topics.add(topic)
                    break
        return sorted(topics)

    async def collect_and_save(
        self,
        episodic: EpisodicMemory | None = None,
        result: BrowserCollectionResult | None = None,
    ) -> str:
        """Full pipeline: collect -> save to inbox -> record episodes.

        If `result` is provided, reuses it instead of re-running collect()
        (avoids double SQLite backup attempt in the orchestrator path).
        """
        if result is None:
            result = await self.collect()

        if not result.entries:
            return "No browser history entries found."

        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        summary = self._format_summary(result, today)

        # Write to inbox
        inbox_dir = self._workspace / "inbox"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        inbox_path = inbox_dir / f"browser_{today}.md"
        inbox_path.write_text(summary, encoding="utf-8")
        logger.info("browser_inbox_written", path=str(inbox_path))

        # Record notable entries as episodes
        if episodic is not None:
            for entry in result.entries:
                if entry.visit_count >= _NOTABLE_VISIT_THRESHOLD:
                    importance = min(0.3 + (entry.visit_count - 3) * 0.1, 0.8)
                    await episodic.record(
                        content=f"Browsed: {entry.title} ({entry.domain})",
                        user_id=self._admin_user_id,
                        chat_type="personal",
                        role="user",
                        source="browser",
                        importance=importance,
                        person_name="Никита",
                        significance="inner_circle",
                        domain="chat",
                    )

        return summary

    @staticmethod
    def _find_domain_concentration(
        domain: str,
        domain_entries: list[BrowserEntry],
    ) -> tuple[int, str] | None:
        """Find the first 2-hour window with 3+ visits for a single domain.

        Uses a left-anchored sliding window over time-sorted entries.
        Returns (visit_count, pattern_string) or None if no window qualifies.
        """
        sorted_entries = sorted(domain_entries, key=lambda e: e.visit_time)
        for window_start in range(len(sorted_entries)):
            anchor = sorted_entries[window_start].visit_time
            window_end = window_start
            while (
                window_end < len(sorted_entries)
                and (sorted_entries[window_end].visit_time - anchor).total_seconds()
                <= 7200
            ):  # 2 hours
                window_end += 1
            window_count = window_end - window_start
            if window_count >= 3:
                hour_start = anchor.hour
                hour_end = sorted_entries[window_end - 1].visit_time.hour + 1
                pattern = (
                    f"Active {domain} monitoring"
                    f" (visited {window_count} times"
                    f" between {hour_start:02d}:00-{hour_end:02d}:00)"
                )
                return window_count, pattern
        return None

    @staticmethod
    def _group_by_topic(
        entries: list[BrowserEntry],
    ) -> defaultdict[str, list[BrowserEntry]]:
        """Map entries to topics using domain map then keyword fallback."""
        topic_entries: defaultdict[str, list[BrowserEntry]] = defaultdict(list)
        for entry in entries:
            topic: str | None = _DOMAIN_TOPICS.get(entry.domain)
            if topic is None:
                for keyword, kw_topic in _DOMAIN_KEYWORD_TOPICS.items():
                    if keyword in entry.domain:
                        topic = kw_topic
                        break
            if topic is not None:
                topic_entries[topic].append(entry)
        return topic_entries

    @staticmethod
    def _extract_patterns(entries: list[BrowserEntry]) -> list[str]:
        """Detect temporal and topical concentration patterns in browser history.

        Checks two pattern types:
        - Domain concentration: 3+ visits to the same domain within any 2-hour window.
        - Topic deep-dive: 3+ entries that share the same topic via domain/keyword maps.

        Returns a list of human-readable pattern strings.
        """
        patterns: list[str] = []

        # Concentration: group by domain, find qualifying 2-hour windows
        by_domain: defaultdict[str, list[BrowserEntry]] = defaultdict(list)
        for entry in entries:
            by_domain[entry.domain].append(entry)

        concentration_domains: set[str] = set()
        raw_concentration: list[tuple[int, str]] = []

        for domain, domain_entries in by_domain.items():
            result = BrowserHistoryCollector._find_domain_concentration(
                domain, domain_entries
            )
            if result is not None:
                count, pattern_str = result
                concentration_domains.add(domain)
                raw_concentration.append((count, pattern_str))

        for _, pattern_str in sorted(raw_concentration, key=lambda t: -t[0]):
            patterns.append(pattern_str)

        # Topic deep-dives not already covered by single-domain concentrations above
        topic_entries = BrowserHistoryCollector._group_by_topic(entries)
        for topic, t_entries in sorted(
            topic_entries.items(), key=lambda kv: -len(kv[1])
        ):
            if len(t_entries) >= 3:
                domains_in_topic = {e.domain for e in t_entries}
                if not domains_in_topic.issubset(concentration_domains):
                    patterns.append(
                        f'Deep-dive into "{topic}"'
                        f" ({len(t_entries)} pages across"
                        f" {len(domains_in_topic)} domain(s))"
                    )

        return patterns

    @staticmethod
    def _format_summary(result: BrowserCollectionResult, date: str) -> str:
        """Format browser history as markdown summary."""
        lines = [
            f"# Browser History — {date}",
            "",
        ]

        patterns = BrowserHistoryCollector._extract_patterns(result.entries)
        if patterns:
            lines.append("## Patterns")
            for pattern in patterns:
                lines.append(f"- {pattern}")
            lines.append("")

        if result.topics:
            lines.append("## Topics")
            for topic in result.topics:
                lines.append(f"- {topic}")
            lines.append("")

        if result.top_domains:
            lines.append("## Top domains")
            for domain, count in result.top_domains:
                lines.append(f"- {domain} ({count} visits)")
            lines.append("")

        notable = [
            e for e in result.entries if e.visit_count >= _NOTABLE_VISIT_THRESHOLD
        ]
        if notable:
            lines.append("## Notable pages")
            for entry in notable[:10]:
                lines.append(
                    f'- "{entry.title}" ({entry.domain}, {entry.visit_count} visits)'
                )
            lines.append("")

        lines.append(
            f"Total: {len(result.entries)} entries "
            f"(Firefox {result.firefox_count}, Chrome {result.chrome_count})"
        )
        return "\n".join(lines)

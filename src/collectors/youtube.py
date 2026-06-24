"""YouTube collector: watch history, feed scanning, transcription."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

import structlog

if TYPE_CHECKING:
    from types import SimpleNamespace

    from src.memory import EpisodicMemoryProtocol as EpisodicMemory

logger = structlog.get_logger()


@dataclass
class YouTubeEntry:
    video_id: str
    title: str
    channel: str
    url: str
    watched_at: datetime | None = None
    duration: str | None = None


@dataclass
class YouTubeAnalysis:
    video_id: str
    title: str
    key_ideas: list[str]
    useful_for_nikita: str
    tools_mentioned: list[str]
    transcript_length: int


def _extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/")
    qs = parse_qs(parsed.query)
    return qs.get("v", [""])[0]


def _search_youtube(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Search YouTube using youtube-search-python. Returns raw results."""
    try:
        from youtubesearchpython import VideosSearch

        search = VideosSearch(query, limit=max_results)
        result = search.result()
        return result.get("result", [])  # type: ignore[no-any-return]
    except ImportError:
        logger.warning("youtube_search_python_not_installed")
        return []
    except Exception:
        logger.warning("youtube_search_failed", query=query, exc_info=True)
        return []


def _get_transcript(
    video_id: str, languages: list[str] | None = None
) -> list[dict[str, Any]] | None:
    """Get transcript via youtube-transcript-api. Returns None if unavailable."""
    if languages is None:
        languages = ["ru", "en"]
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        fetcher = YouTubeTranscriptApi()
        transcript = fetcher.fetch(video_id, languages=languages)
        return [
            {"text": snippet.text, "start": snippet.start} for snippet in transcript
        ]
    except ImportError:
        logger.warning("youtube_transcript_api_not_installed")
        return None
    except Exception:
        logger.info("youtube_transcript_unavailable", video_id=video_id)
        return None


class YouTubeCollector:
    """Three-level YouTube intelligence.

    Level 1: Watch history from browser history (primary) or Google Takeout (fallback)
    Level 2: Feed scanning (youtube-search-python)
    Level 3: Transcription + analysis of interesting videos
    """

    def __init__(self, config: SimpleNamespace) -> None:
        self._workspace = Path(
            getattr(config, "workspace_path", "~/zhvusha-workspace")
        ).expanduser()
        self._takeout_path = getattr(config, "youtube_takeout_path", "")
        self._api_key = getattr(config, "youtube_api_key", "")
        self._scan_enabled = getattr(config, "youtube_scan_enabled", False)
        self._transcribe_top_n = getattr(config, "youtube_transcribe_top_n", 3)
        self._admin_user_id = getattr(config, "admin_user_id", 0)
        self._knowledge_dir = self._workspace / "knowledge" / "youtube"

    # --- Level 1: Watch History ---

    @staticmethod
    def from_browser_history(
        browser_entries: list[dict[str, Any]],
    ) -> list[YouTubeEntry]:
        """Extract YouTube entries from browser history.

        Args:
            browser_entries: list of dicts with keys: url, title, visit_time, domain.
        """
        entries: list[YouTubeEntry] = []
        seen_ids: set[str] = set()

        for item in browser_entries:
            domain = item.get("domain", "")
            if "youtube.com" not in domain and "youtu.be" not in domain:
                continue

            url = item.get("url", "")
            video_id = _extract_video_id(url)
            if not video_id or video_id in seen_ids:
                continue
            seen_ids.add(video_id)

            entries.append(
                YouTubeEntry(
                    video_id=video_id,
                    title=item.get("title", ""),
                    channel="",
                    url=url,
                    watched_at=item.get("visit_time"),
                )
            )

        return entries

    async def parse_watch_history(
        self,
        since: datetime | None = None,
    ) -> list[YouTubeEntry]:
        """Parse YouTube watch history from Google Takeout JSON."""
        if not self._takeout_path:
            return []

        takeout_path = Path(self._takeout_path)
        if not takeout_path.exists():
            logger.warning("youtube_takeout_not_found", path=str(takeout_path))
            return []

        if since is None:
            since = datetime.now(tz=UTC) - timedelta(hours=24)

        raw = json.loads(takeout_path.read_text(encoding="utf-8"))
        entries: list[YouTubeEntry] = []

        for item in raw:
            title = item.get("title", "")
            if title.startswith("Watched "):
                title = title[len("Watched ") :]

            url = item.get("titleUrl", "")
            if not url:
                continue

            time_str = item.get("time", "")
            if not time_str:
                continue

            watched_at = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            if watched_at < since:
                continue

            channel = ""
            subs = item.get("subtitles", [])
            if subs and isinstance(subs, list):
                channel = subs[0].get("name", "")

            video_id = _extract_video_id(url)

            entries.append(
                YouTubeEntry(
                    video_id=video_id,
                    title=title,
                    channel=channel,
                    url=url,
                    watched_at=watched_at,
                )
            )

        logger.info("youtube_takeout_parsed", count=len(entries))
        return entries

    # --- Level 2: Feed Scanning ---

    async def scan_feed(
        self,
        interests: list[str],
        max_results: int = 20,
    ) -> list[YouTubeEntry]:
        """Scan YouTube for videos matching current interests."""
        if not self._scan_enabled and not interests:
            return []

        all_entries: list[YouTubeEntry] = []
        seen_ids: set[str] = set()

        for interest in interests[:5]:  # Max 5 search queries
            results = _search_youtube(
                interest, max_results=max_results // len(interests)
            )
            for item in results:
                vid = item.get("id", "")
                if vid in seen_ids:
                    continue
                seen_ids.add(vid)

                channel_info = item.get("channel", {})
                channel_name = (
                    channel_info.get("name", "")
                    if isinstance(channel_info, dict)
                    else str(channel_info)
                )

                all_entries.append(
                    YouTubeEntry(
                        video_id=vid,
                        title=item.get("title", ""),
                        channel=channel_name,
                        url=item.get("link", f"https://youtube.com/watch?v={vid}"),
                        duration=item.get("duration"),
                    )
                )

        logger.info("youtube_feed_scanned", count=len(all_entries))
        return all_entries

    # --- Level 3: Transcription + Analysis ---

    async def transcribe_and_analyze(
        self,
        video_url: str,
    ) -> YouTubeAnalysis | None:
        """Full analysis of a single video via transcript."""
        video_id = _extract_video_id(video_url)
        if not video_id:
            return None

        transcript = _get_transcript(video_id)
        if transcript is None:
            logger.info(
                "youtube_no_transcript",
                video_id=video_id,
                note="видео без RU/EN субтитров, пропустила",
            )
            return None

        full_text = " ".join(seg["text"] for seg in transcript)
        truncated = full_text[:12000]

        llm_response = await self._call_llm(
            f"Проанализируй транскрипт видео хладнокровно и скептически. "
            f"Игнорируй эмоциональную подачу — извлекай только факты.\n"
            f"Выдели:\n"
            f"- Ключевые идеи (3-5 пунктов)\n"
            f"- Что полезно для Никиты (разработчик, фрилансер)\n"
            f"- Конкретные инструменты/ресурсы упомянутые в видео\n"
            f"- Если источник использует манипулятивные приёмы — отметить\n"
            f"Транскрипт: {truncated}"
        )

        analysis = YouTubeAnalysis(
            video_id=video_id,
            title=video_id,  # Will be updated if available
            key_ideas=[llm_response],
            useful_for_nikita="",
            tools_mentioned=[],
            transcript_length=len(full_text),
        )

        self._save_analysis(analysis)
        return analysis

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM (strategist tier — morning session). Override in tests."""
        try:
            from src.llm.protocols import LLMRequest
            from src.llm.router import get_router

            router = get_router()
            response = await router.generate(
                LLMRequest(
                    prompt=prompt,
                    tier="strategist",
                    caller="youtube_analysis",
                )
            )
            return response.text
        except Exception:
            logger.warning("youtube_llm_call_failed", exc_info=True)
            return ""

    def _save_analysis(self, analysis: YouTubeAnalysis) -> None:
        """Save analysis to knowledge/youtube/{video_id}.md."""
        self._knowledge_dir.mkdir(parents=True, exist_ok=True)
        file_path = self._knowledge_dir / f"{analysis.video_id}.md"

        content = f"# {analysis.title}\n\n"
        content += "## Key Ideas\n"
        for idea in analysis.key_ideas:
            content += f"- {idea}\n"
        content += f"\n## Useful for Nikita\n{analysis.useful_for_nikita}\n"
        if analysis.tools_mentioned:
            content += "\n## Tools Mentioned\n"
            for tool in analysis.tools_mentioned:
                content += f"- {tool}\n"
        content += f"\n---\nTranscript length: {analysis.transcript_length} chars\n"

        file_path.write_text(content, encoding="utf-8")
        logger.info("youtube_analysis_saved", path=str(file_path))

    # --- Full Pipeline ---

    async def collect_and_save(
        self,
        episodic: EpisodicMemory | None = None,
        browser_entries: list[dict[str, Any]] | None = None,
        since: datetime | None = None,
    ) -> str:
        """Full pipeline for morning session.

        Args:
            browser_entries: YouTube URLs from BrowserHistoryCollector (primary).
                             Falls back to Takeout if not provided.
            since: lower bound for the Takeout fallback.
        """
        if browser_entries:
            watched = self.from_browser_history(browser_entries)
        else:
            watched = await self.parse_watch_history(since=since)
        feed: list[YouTubeEntry] = []
        if self._scan_enabled:
            feed = await self.scan_feed(interests=[])

        if not watched and not feed:
            return "No YouTube data available."

        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        summary = self._format_summary(watched, feed, today)

        # Write to inbox
        inbox_dir = self._workspace / "inbox"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        inbox_path = inbox_dir / f"youtube_{today}.md"
        inbox_path.write_text(summary, encoding="utf-8")
        logger.info("youtube_inbox_written", path=str(inbox_path))

        # Record watched videos as episodes
        if episodic is not None:
            for entry in watched:
                await episodic.record(
                    content=f"Watched: {entry.title} ({entry.channel})",
                    user_id=self._admin_user_id,
                    chat_type="personal",
                    role="user",
                    source="youtube",
                    importance=0.4,
                    person_name="Никита",
                    significance="inner_circle",
                    domain="chat",
                )

        return summary

    @staticmethod
    def _format_summary(
        watched: list[YouTubeEntry],
        feed: list[YouTubeEntry],
        date: str,
    ) -> str:
        """Format YouTube data as markdown summary."""
        lines = [f"# YouTube — {date}", ""]

        if watched:
            lines.append("## Watched")
            for entry in watched:
                channel_part = f" ({entry.channel})" if entry.channel else ""
                lines.append(f'- "{entry.title}"{channel_part}')
            lines.append("")

        if feed:
            lines.append("## Feed Discoveries")
            for entry in feed:
                channel_part = f" ({entry.channel})" if entry.channel else ""
                dur_part = f", {entry.duration}" if entry.duration else ""
                lines.append(f'- "{entry.title}"{channel_part}{dur_part}')
            lines.append("")

        return "\n".join(lines)

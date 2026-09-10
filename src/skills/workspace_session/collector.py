from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog
import yaml

from src.collectors.base import CollectorStatus

if TYPE_CHECKING:
    from pathlib import Path

    from src.memory import EpisodicMemoryProtocol as EpisodicMemory

logger = structlog.get_logger()


async def collect_inbox(
    inbox_dir: Path,
    *,
    redis: Any | None = None,
    today: date | None = None,
    lookback_hours: int = 24,
    episodic: EpisodicMemory | None = None,
    include_self_coding_archive: bool = False,
    settings: Any | None = None,
) -> tuple[Path, list[CollectorStatus]]:
    """Collect today's data and write to inbox/YYYY-MM-DD.md.

    Returns the path to the created file.
    Skips if today's file already exists.

    Sections (deterministic, no random top-3 filter):
        1. Chat Log — full diaries across the lookback window
        2. Photo Descriptions — Gemini vision output from chat logs
        3. Published Posts — channel posts written in the window
        4. Self-Coding Archive — optional at collector level; /morning opts in
        5. People — new contacts + promotions
    """
    del redis
    today = today or datetime.now(tz=UTC).date()
    filename = f"{today.isoformat()}.md"
    file_path = inbox_dir / filename

    if file_path.exists():
        logger.info("inbox_already_collected", date=str(today))
        return file_path, []

    # Lookback range includes today plus the preceding whole date buckets.
    # JSONL/media/archive files are date-partitioned, not timestamp-indexed;
    # including today is the conservative choice for ad-hoc `/morning 336`
    # runs so fresh self-coding chat/commits are not silently missed.
    lookback_days = max(1, (lookback_hours + 23) // 24)
    date_range = [today - timedelta(days=i) for i in range(0, lookback_days + 1)]

    sections: list[str] = [
        f"# Inbox — {today.isoformat()}",
        "",
    ]

    logs_dir = inbox_dir.parent / "logs"

    # Chat log section — full range across lookback window
    sections.append("## Chat Log")
    sections.append("")
    chat_log = _read_chat_log_range(logs_dir, date_range)
    sections.append(chat_log or "No chat activity recorded.")
    sections.append("")

    # Photo descriptions — Gemini vision output from chat_log JSONL
    photo_block = _read_photo_descriptions(logs_dir, date_range)
    if photo_block:
        sections.append("## Photo Descriptions")
        sections.append("")
        sections.append(photo_block)
        sections.append("")

    # Published posts — channel posts written in the window
    posts_block = _read_published_posts(inbox_dir.parent, date_range)
    if posts_block:
        sections.append("## Published Posts")
        sections.append("")
        sections.append(posts_block)
        sections.append("")

    if include_self_coding_archive:
        # Self-coding archive — durable cycle history from ImplementSpec/CycleAnalyzer.
        archive_block = _read_self_coding_archive(inbox_dir.parent, date_range)
        if archive_block:
            sections.append("## Self-Coding Archive")
            sections.append("")
            sections.append(archive_block)
            sections.append("")

    # People section
    sections.append("## People")
    sections.append("")
    people_dir = inbox_dir.parent / "memory" / "people"
    people_summary = _read_people_updates(people_dir, inbox_dir, today)
    if people_summary:
        sections.append(people_summary)
    else:
        sections.append("No new people activity.")
    sections.append("")

    file_path.write_text("\n".join(sections), encoding="utf-8")
    logger.info("inbox_collected", date=str(today), path=str(file_path))

    # Phase 3 collectors — isolated so failures never break morning session
    statuses: list[CollectorStatus] = []
    try:
        if settings is None:
            from src.core.config import get_settings

            settings = get_settings()
        statuses = await collect_phase3_sources(
            inbox_dir,
            settings,
            episodic=episodic,
            lookback_hours=lookback_hours,
            today=today,
        )
    except Exception:
        logger.warning("phase3_collectors_failed", exc_info=True)

    return file_path, statuses


def _read_chat_log_range(logs_dir: Path, date_range: list[date]) -> str:
    """Read chat logs for every date in `date_range` across all chat dirs.

    Replaces the legacy "yesterday only" behaviour — now respects the
    caller's lookback window so `/morning 72` gets 3 days of context.
    """
    if not logs_dir.is_dir():
        return ""

    all_parts: list[str] = []
    for chat_dir in sorted(logs_dir.iterdir()):
        chat_parts: list[str] = []
        for day in sorted(date_range):
            log_file = chat_dir / f"chat_{day.isoformat()}.jsonl"
            if not log_file.is_file():
                continue
            parts = _parse_log_file(log_file)
            if parts:
                chat_parts.append(f"#### {day.isoformat()}")
                chat_parts.extend(parts)
        if chat_parts:
            all_parts.append(f"### Chat {chat_dir.name}")
            all_parts.extend(chat_parts)

    return "\n".join(all_parts)


def _parse_log_file(log_file: Path) -> list[str]:
    """Parse a single JSONL log file into formatted lines."""
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    parts: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = entry.get("ts", "")[:16]
        role = entry.get("role", "user")
        label = "🤖" if role == "assistant" else "👤"
        text = entry.get("text", "")
        if text:
            parts.append(f"- {label} [{ts}] {text}")
    return parts


def _read_photo_descriptions(logs_dir: Path, date_range: list[date]) -> str:
    """Extract Gemini vision descriptions from chat log entries.

    Photos are logged with `photo_description` field by `log_photo_message`
    in chat_logger middleware. This function surfaces those descriptions
    to the morning session so the Codex session can see what the user showed Zhvusha
    (otherwise they stay buried in JSONL).
    """
    if not logs_dir.is_dir():
        return ""

    parts: list[str] = []
    for chat_dir in sorted(logs_dir.iterdir()):
        for day in sorted(date_range):
            log_file = chat_dir / f"chat_{day.isoformat()}.jsonl"
            if not log_file.is_file():
                continue
            for line in log_file.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                description = entry.get("photo_description")
                if not description:
                    continue
                ts = entry.get("ts", "")[:16]
                caption = entry.get("text") or "(без подписи)"
                paths = entry.get("photo_paths") or []
                parts.append(
                    f"- [{ts}] chat={chat_dir.name} "
                    f'caption="{caption[:100]}" '
                    f"photos={len(paths)}\n  {description}"
                )
    return "\n".join(parts)


def _read_published_posts(ws_root: Path, date_range: list[date]) -> str:
    """Read channel posts published during the lookback window.

    Posts are archived by `save_published_post` in channel_writer/archive.py
    as `channel/posts/{date}_{n}.md` with frontmatter (date, message_id).
    """
    posts_dir = ws_root / "channel" / "posts"
    if not posts_dir.is_dir():
        return ""

    parts: list[str] = []
    date_strs = {d.isoformat() for d in date_range}
    for post_file in sorted(posts_dir.glob("*.md")):
        # Filename format: {YYYY-MM-DD}_{n}.md
        name_parts = post_file.stem.split("_", 1)
        if not name_parts or name_parts[0] not in date_strs:
            continue
        try:
            content = post_file.read_text(encoding="utf-8")
        except OSError:
            continue
        # Strip frontmatter
        body = content
        if content.startswith("---"):
            sep_end = content.find("---", 3)
            if sep_end != -1:
                body = content[sep_end + 3 :].lstrip()
        snippet = body[:300].strip()
        parts.append(f"- {post_file.stem}: {snippet}")
    return "\n".join(parts)


def _read_self_coding_archive(ws_root: Path, date_range: list[date]) -> str:
    """Read self-coding archive nodes created during the lookback window."""
    archive_dir = ws_root / "self_coding_archive"
    if not archive_dir.is_dir():
        return ""

    wanted_dates = {day.isoformat() for day in date_range}
    parts: list[str] = []
    for node_dir in sorted(
        (path for path in archive_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ):
        metadata = _read_archive_metadata(node_dir)
        created_at = str(metadata.get("created_at") or "")
        if wanted_dates and created_at[:10] not in wanted_dates:
            continue
        slug = str(metadata.get("slug") or node_dir.name)
        status = str(metadata.get("status") or "unknown")
        spec_slug = str(metadata.get("spec_slug") or "unknown")
        commit_sha = str(metadata.get("commit_sha") or "no-commit")
        actor = _archive_actor(metadata)
        backend = _archive_backend(metadata)
        insight = _read_archive_insight_snippet(node_dir)
        chat_context = _read_archive_chat_context_snippet(node_dir)
        parts.append(
            f"- [{created_at[:16] or 'unknown'}] {slug} "
            f"status={status} spec={spec_slug} commit={commit_sha[:12]} "
            f"actor={actor} backend={backend}"
        )
        if insight:
            parts.append(f"  insight: {insight}")
        if chat_context:
            parts.append(f"  chat_context: {chat_context}")
    return "\n".join(parts)


def _read_archive_metadata(node_dir: Path) -> dict[str, Any]:
    path = node_dir / "metadata.yaml"
    if not path.is_file():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _archive_actor(metadata: dict[str, Any]) -> str:
    nested = metadata.get("metadata")
    if isinstance(nested, dict):
        actor = nested.get("self_coding_actor")
        if actor:
            return str(actor)

    tags = metadata.get("tags")
    if isinstance(tags, list) and "self-coding" in {str(tag) for tag in tags}:
        return "zhvusha"

    # Every node in this archive is created by the self-coding cycle.
    return "zhvusha"


def _archive_backend(metadata: dict[str, Any]) -> str:
    nested = metadata.get("metadata")
    if isinstance(nested, dict):
        backend = nested.get("agent_backend")
        if backend:
            return str(backend)

    model_config = metadata.get("model_config")
    if isinstance(model_config, dict):
        backend = model_config.get("backend") or model_config.get("executor")
        if backend:
            return str(backend)

    return "unknown"


def _read_archive_insight_snippet(node_dir: Path) -> str:
    path = node_dir / "insight.md"
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("##"):
            return stripped[:300]
    return ""


def _read_archive_chat_context_snippet(node_dir: Path) -> str:
    path = node_dir / "chat_context.md"
    if not path.is_file():
        return ""
    try:
        lines = [
            line.lstrip("- ").strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("- ")
        ]
    except OSError:
        return ""
    return " | ".join(lines)[:300]


def _read_people_updates(people_dir: Path, inbox_dir: Path, today: date) -> str:
    """Summarize new people and promotions for the morning session."""
    parts: list[str] = []

    # Read promotions flag file
    promotions_file = inbox_dir / "promotions.md"
    if promotions_file.exists():
        content = promotions_file.read_text(encoding="utf-8").strip()
        if content:
            parts.append("### Promotions")
            parts.append(content)
        promotions_file.unlink()

    # Find profiles created yesterday
    new_people = _find_new_contacts(people_dir, today)
    if new_people:
        parts.append("### New contacts yesterday")
        parts.extend(new_people)

    return "\n".join(parts)


def _find_new_contacts(people_dir: Path, today: date) -> list[str]:
    """Find profiles created yesterday."""
    if not people_dir.is_dir():
        return []
    yesterday_str = (today - timedelta(days=1)).isoformat()
    result: list[str] = []
    for user_dir in sorted(people_dir.iterdir()):
        profile = user_dir / "profile.md"
        if not profile.is_file():
            continue
        try:
            text = profile.read_text(encoding="utf-8")
        except OSError:
            continue
        if f"first_seen: {yesterday_str}" not in text:
            continue
        for line in text.split("\n"):
            if line.startswith("username:"):
                name = line.split(": ", 1)[1].strip()
                result.append(f"- {name or user_dir.name} (id: {user_dir.name})")
                break
    return result


async def collect_phase3_sources(
    inbox_dir: Path,
    settings: Any,
    *,
    episodic: EpisodicMemory | None = None,
    today: date | None = None,
    lookback_hours: int = 24,
) -> list[CollectorStatus]:
    """Run all Phase 3 data collectors, each isolated by try/except.

    Browser runs first — its YouTube entries feed into YouTubeCollector.
    """
    today = today or datetime.now(tz=UTC).date()
    since = datetime.now(tz=UTC) - timedelta(hours=lookback_hours)
    statuses: list[CollectorStatus] = []
    browser_entries: list[dict[str, object]] = []

    # Browser history (runs first — YouTube depends on it)
    if getattr(settings, "firefox_profile_path", "") or getattr(
        settings, "chrome_history_path", ""
    ):
        status, browser_entries = await _run_browser_collector(
            settings, episodic, since=since
        )
        statuses.append(status)

    # YouTube — uses browser entries, falls back to Takeout
    youtube_entries = [
        e
        for e in browser_entries
        if "youtube.com" in str(e.get("domain", ""))
        or "youtu.be" in str(e.get("domain", ""))
    ]
    takeout_path: str = getattr(settings, "youtube_takeout_path", "")
    if youtube_entries or takeout_path:
        statuses.append(
            await _run_youtube_collector(
                settings, episodic, youtube_entries, since=since
            )
        )

    # Telegram channels
    if getattr(settings, "telegram_api_id", 0) and getattr(
        settings, "monitored_channel_ids", ""
    ):
        statuses.append(await _run_telegram_collector(settings, episodic, since=since))

    # Git changes — project_path has a default, so always attempt
    if getattr(settings, "project_path", ""):
        # /morning's lookback is the authoritative recovery window. Stored SHA
        # state may already reflect a bad/rolled-back run, so use time here.
        statuses.append(
            await _run_git_collector(
                settings,
                episodic,
                since=since,
                force_since=True,
            )
        )

    # Write status file
    if statuses:
        _write_status_file(inbox_dir, statuses, today)

    return statuses


async def _run_browser_collector(
    settings: Any,
    episodic: EpisodicMemory | None,
    since: datetime | None = None,
) -> tuple[CollectorStatus, list[dict[str, object]]]:
    """Run browser collector. Returns status + raw entries for YouTube."""
    try:
        from src.collectors.browser import BrowserHistoryCollector

        collector = BrowserHistoryCollector(settings)
        result = await collector.collect(since=since)
        raw_entries: list[dict[str, object]] = [
            {
                "url": e.url,
                "title": e.title,
                "visit_time": e.visit_time,
                "domain": e.domain,
            }
            for e in result.entries
        ]
        # Reuse result from above — avoids a second SQLite backup attempt
        summary = await collector.collect_and_save(episodic=episodic, result=result)
        if not result.entries:
            return (
                CollectorStatus(
                    name="Browser", success=False, error="Нет данных браузера"
                ),
                [],
            )
        lines = summary.split("\n")
        total_line = [line for line in lines if line.startswith("Total:")]
        msg = total_line[0] if total_line else f"{len(result.entries)} записей"
        return CollectorStatus(name="Browser", success=True, message=msg), raw_entries
    except Exception as exc:
        logger.warning("collector_browser_failed", exc_info=True)
        return CollectorStatus(name="Browser", success=False, error=str(exc)), []


async def _run_youtube_collector(
    settings: Any,
    episodic: EpisodicMemory | None,
    browser_entries: list[dict[str, object]] | None = None,
    since: datetime | None = None,
) -> CollectorStatus:
    try:
        from src.collectors.youtube import YouTubeCollector

        collector = YouTubeCollector(settings)
        summary = await collector.collect_and_save(
            episodic=episodic,
            browser_entries=browser_entries,
            since=since,
        )
        return CollectorStatus(name="YouTube", success=True, message=summary[:100])
    except Exception as exc:
        logger.warning("collector_youtube_failed", exc_info=True)
        return CollectorStatus(name="YouTube", success=False, error=str(exc))


async def _run_telegram_collector(
    settings: Any,
    episodic: EpisodicMemory | None,
    since: datetime | None = None,
) -> CollectorStatus:
    try:
        from src.collectors.telegram_channels import TelegramChannelCollector

        collector = TelegramChannelCollector(settings)
        try:
            await collector.connect()
            summary = await collector.collect_and_save(episodic=episodic, since=since)
            return CollectorStatus(name="Channels", success=True, message=summary[:100])
        finally:
            await collector.disconnect()
    except Exception as exc:
        logger.warning("collector_telegram_failed", exc_info=True)
        return CollectorStatus(name="Channels", success=False, error=str(exc))


async def _run_git_collector(
    settings: Any,
    episodic: EpisodicMemory | None,
    since: datetime | None = None,
    *,
    force_since: bool = False,
) -> CollectorStatus:
    try:
        from src.collectors.git import GitChangesCollector

        collector = GitChangesCollector(settings)
        summary = await collector.collect_and_save(
            episodic=episodic,
            since=since,
            force_since=force_since,
        )
        if summary.startswith("Git: ошибка"):
            return CollectorStatus(
                name="Git",
                success=False,
                error=summary.removeprefix("Git: ошибка — "),
            )
        return CollectorStatus(name="Git", success=True, message=summary[:100])
    except Exception as exc:
        logger.warning("collector_git_failed", exc_info=True)
        return CollectorStatus(name="Git", success=False, error=str(exc))


def _write_status_file(
    inbox_dir: Path,
    statuses: list[CollectorStatus],
    today: date,
) -> None:
    """Write collectors_status_{date}.md to inbox."""
    lines = [f"# Collectors Status — {today.isoformat()}", ""]
    for status in statuses:
        lines.append(status.format_line())
    lines.append("")

    path = inbox_dir / f"collectors_status_{today.isoformat()}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("collectors_status_written", path=str(path))

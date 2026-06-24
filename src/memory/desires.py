"""Morning analytics processor for Zhvusha's desire system.

Runs during /morning to:
1. Crystallize dreams into wishlist candidates (>7 days + 3 episodic matches)
2. Escalate stale dreams (>15 days)
3. Recommend condensation (5+ dreams)
4. Enforce wishlist limits (10 want / 5 wip / 10 done)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import structlog

from src.memory.protocols import DesireProcessorProtocol

if TYPE_CHECKING:
    from pathlib import Path

    from src.memory.protocols import EpisodicMemoryProtocol

logger = structlog.get_logger()

_DREAM_RE = re.compile(r"^- \[(\d{4}-\d{2}-\d{2})\]\s+(.+)$", re.MULTILINE)
_SECTION_RE = re.compile(r"^## (.+)$", re.MULTILINE)

_CRYSTALLIZE_DAYS = 7
_CRYSTALLIZE_MIN_EPISODES = 3
_STALE_DAYS = 15
_CONDENSE_THRESHOLD = 5
_LIMIT_WANT = 10
_LIMIT_WIP = 5
_LIMIT_DONE = 10


@dataclass
class DreamEntry:
    """A single dream parsed from dreams.md."""

    date: date
    text: str


def _parse_dreams(text: str) -> list[DreamEntry]:
    """Parse dreams.md format: ``- [YYYY-MM-DD] dream text``."""
    entries: list[DreamEntry] = []
    for match in _DREAM_RE.finditer(text):
        try:
            d = date.fromisoformat(match.group(1))
        except ValueError:
            continue
        entries.append(DreamEntry(date=d, text=match.group(2).strip()))
    return entries


def _parse_wishlist_sections(text: str) -> dict[str, list[str]]:
    """Parse wishlist.md into sections by ``## `` headers."""
    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    for line in text.splitlines():
        header_match = _SECTION_RE.match(line)
        if header_match:
            current_section = header_match.group(1).strip()
            sections[current_section] = []
            continue
        if current_section is not None and line.strip().startswith("- "):
            sections[current_section].append(line.strip())

    return sections


class DesireProcessor(DesireProcessorProtocol):
    """Orchestrates morning desire analytics.

    Implements :class:`DesireProcessorProtocol`. Uses an injected
    :class:`EpisodicMemoryProtocol` for dream-crystallisation scoring
    (optional — pipeline degrades gracefully when not provided).
    """

    def __init__(
        self,
        workspace_root: Path,
        episodic: EpisodicMemoryProtocol | None = None,
    ) -> None:
        self._root = workspace_root
        self._episodic = episodic
        self._dreams_path = workspace_root / "personality" / "dreams.md"
        self._wishlist_path = workspace_root / "personality" / "wishlist.md"

    async def run_all(self) -> str:
        """Run all 4 steps. Returns summary for inbox."""
        parts: list[str] = []

        crystal = await self._check_crystallization()
        if crystal:
            parts.append(crystal)

        stale = self._escalate_stale()
        if stale:
            parts.append(stale)

        condense = self._condense_dreams()
        if condense:
            parts.append(condense)

        archive = self._enforce_wishlist_limits()
        if archive:
            parts.append(archive)

        return "\n\n".join(parts)

    async def _check_crystallization(self) -> str:
        """Dreams >7 days with 3+ episodic matches become wishlist candidates."""
        if self._episodic is None:
            logger.warning("desire_crystallization_skipped", reason="no_episodic")
            return ""

        if not self._dreams_path.exists():
            return ""

        dreams = _parse_dreams(self._dreams_path.read_text(encoding="utf-8"))
        today = datetime.now(tz=UTC).date()
        candidates: list[str] = []

        for dream in dreams:
            age = (today - dream.date).days
            if age < _CRYSTALLIZE_DAYS:
                continue

            episodes = await self._episodic.retrieve(query=dream.text, limit=5)
            if len(episodes) < _CRYSTALLIZE_MIN_EPISODES:
                continue

            # Write candidate to outbox
            h = hashlib.md5(dream.text.encode()).hexdigest()[:8]  # noqa: S324
            candidate_dir = self._root / "outbox" / "dream_candidates"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidate_file = candidate_dir / f"{today.isoformat()}_{h}.md"
            candidate_file.write_text(
                f"# Кристаллизация мечты\n\n"
                f"**Мечта:** {dream.text}\n"
                f"**Дата:** {dream.date.isoformat()}\n"
                f"**Возраст:** {age} дней\n"
                f"**Совпадений в памяти:** {len(episodes)}\n",
                encoding="utf-8",
            )
            candidates.append(dream.text)
            logger.info("dream_crystallized", dream=dream.text, episodes=len(episodes))

        if not candidates:
            return ""
        return (
            "💎 Кристаллизация мечт:\n"
            + "\n".join(f"- {c}" for c in candidates)
            + "\nКандидаты записаны в outbox/dream_candidates/"
        )

    def _escalate_stale(self) -> str:
        """Dreams >15 days — flag for Nikita's attention."""
        if not self._dreams_path.exists():
            return ""

        dreams = _parse_dreams(self._dreams_path.read_text(encoding="utf-8"))
        today = datetime.now(tz=UTC).date()
        stale: list[tuple[str, int]] = []

        for dream in dreams:
            age = (today - dream.date).days
            if age >= _STALE_DAYS:
                stale.append((dream.text, age))

        if not stale:
            return ""

        lines = ["⏰ Застоявшиеся мечты:"]
        for text, age in stale:
            lines.append(f"- {text} (дней: {age})")
        lines.append("Что делаем? Думаем / ждём / удаляем?")
        return "\n".join(lines)

    def _condense_dreams(self) -> str:
        """If 5+ dreams, recommend condensation."""
        if not self._dreams_path.exists():
            return ""

        dreams = _parse_dreams(self._dreams_path.read_text(encoding="utf-8"))
        if len(dreams) < _CONDENSE_THRESHOLD:
            return ""

        return (
            f"📦 У меня {len(dreams)} мечт — можно подумать, "
            "какие связаны и объединить."
        )

    def _enforce_wishlist_limits(self) -> str:
        """Move overflow items from wishlist to archive."""
        if not self._wishlist_path.exists():
            return ""

        text = self._wishlist_path.read_text(encoding="utf-8")
        sections = _parse_wishlist_sections(text)

        archived: list[str] = []
        limits = {"Хочу": _LIMIT_WANT, "В работе": _LIMIT_WIP, "Готово": _LIMIT_DONE}

        for section_name, limit in limits.items():
            items = sections.get(section_name, [])
            if len(items) <= limit:
                continue

            # Archive OLDEST (top of list), keep NEWEST (bottom)
            overflow = items[: len(items) - limit]
            sections[section_name] = items[len(items) - limit :]
            archived.extend(overflow)

        if not archived:
            return ""

        # Write archive
        archive_path = self._root / "personality" / "history" / "wishlist_archive.md"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        existing = (
            archive_path.read_text(encoding="utf-8") if archive_path.exists() else ""
        )
        archive_path.write_text(
            existing + "\n".join(archived) + "\n",
            encoding="utf-8",
        )

        # Rewrite wishlist with trimmed sections
        self._rewrite_wishlist(sections)

        logger.info("wishlist_overflow_archived", count=len(archived))
        return f"📋 Wishlist: {len(archived)} элемент(ов) перенесено в архив."

    def _rewrite_wishlist(self, sections: dict[str, list[str]]) -> None:
        """Rewrite wishlist.md with updated sections.

        Preserves standard section order, then appends any unknown sections
        so custom additions (e.g. "## Отложено") are not lost.
        """
        standard_order = ["Хочу", "В работе", "Готово"]
        lines = ["# Wishlist", ""]
        for section_name in standard_order:
            lines.append(f"## {section_name}")
            lines.append("")
            for item in sections.get(section_name, []):
                lines.append(item)
            lines.append("")
        # Preserve non-standard sections
        for section_name, items in sections.items():
            if section_name not in standard_order:
                lines.append(f"## {section_name}")
                lines.append("")
                for item in items:
                    lines.append(item)
                lines.append("")
        self._wishlist_path.write_text("\n".join(lines), encoding="utf-8")

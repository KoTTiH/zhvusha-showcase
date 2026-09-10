"""Base types and protocol for all data collectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime  # noqa: TC003
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.memory import EpisodicMemoryProtocol as EpisodicMemory


@runtime_checkable
class BaseCollector(Protocol):
    """Protocol that all collectors must satisfy."""

    async def collect_and_save(
        self,
        episodic: EpisodicMemory | None = None,
    ) -> str:
        """Run full pipeline: collect → save to inbox → record episodes.

        Returns summary text.
        """
        ...


@dataclass
class CollectionResult:
    """Generic result from any collector."""

    source: str
    entries_count: int
    summary: str
    inbox_path: str = ""
    errors: list[str] = field(default_factory=list)


@dataclass
class CollectorStatus:
    """Status of a single collector run."""

    name: str
    success: bool
    entries_count: int = 0
    message: str = ""
    error: str = ""

    def format_line(self) -> str:
        """Format as a single status line for the status file."""
        if self.success and not self.error:
            return f"- {self.name}: ✅ {self.message}"
        if self.success and self.error:
            return f"- {self.name}: ⚠️ {self.message} ({self.error})"
        return f"- {self.name}: ❌ {self.error}"

    def format_ru(self) -> str:
        """Format as a user-facing Russian status line."""
        names = {
            "Browser": "Браузер",
            "YouTube": "YouTube",
            "Channels": "Каналы",
        }
        name = names.get(self.name, self.name)
        if self.success and not self.error:
            return f"✅ {name}: {self.message}"
        if self.success and self.error:
            return f"⚠️ {name}: {self.message} ({self.error})"
        return f"❌ {name}: {self.error}"


@dataclass
class InboxEntry:
    """A single entry from any data source before processing."""

    content: str
    source: str
    timestamp: datetime
    importance: float = 0.5
    metadata: dict[str, str] = field(default_factory=dict)

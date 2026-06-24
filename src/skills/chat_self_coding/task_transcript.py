"""Durable transcript artifacts for one /код task."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

import structlog

if TYPE_CHECKING:
    from pathlib import Path

    from src.skills.chat_self_coding.events import BlockEvent, BlockPublisher

logger = structlog.get_logger()


class TaskTranscriptStore(Protocol):
    """Append-only transcript store keyed by ``code_task_id``."""

    async def append(
        self,
        *,
        task_id: str,
        user_id: int,
        kind: str,
        text: str,
        slug: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class TaskTranscriptEntry:
    task_id: str
    user_id: int
    kind: str
    text: str
    slug: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())

    def to_json_line(self) -> str:
        return json.dumps(
            {
                "task_id": self.task_id,
                "user_id": self.user_id,
                "kind": self.kind,
                "text": self.text,
                "slug": self.slug,
                "payload": self.payload,
                "created_at": self.created_at,
            },
            ensure_ascii=False,
        )


class FileTaskTranscriptStore:
    """Filesystem JSONL store for task transcripts.

    The file is append-only and intentionally simple: it survives bot
    restarts, is easy to inspect manually, and can be rendered into a compact
    final report without depending on Redis Pub/Sub replay.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def path_for(self, task_id: str) -> Path:
        safe = _safe_task_id(task_id)
        return self._root / f"{safe}.jsonl"

    async def append(
        self,
        *,
        task_id: str,
        user_id: int,
        kind: str,
        text: str,
        slug: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        entry = TaskTranscriptEntry(
            task_id=task_id,
            user_id=user_id,
            kind=kind,
            text=text,
            slug=slug,
            payload=dict(payload or {}),
        )
        path = self.path_for(task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(entry.to_json_line() + "\n")

    def render_markdown(self, task_id: str) -> str:
        path = self.path_for(task_id)
        lines = [
            "# /код task transcript",
            "",
            f"- task_id: {task_id}",
            "",
            "## Events",
        ]
        if not path.exists():
            lines.append("- Нет записей.")
            return "\n".join(lines) + "\n"

        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(
                    "chat_self_coding_transcript_bad_line",
                    task_id=task_id,
                    raw=raw[:200],
                )
                continue
            kind = str(entry.get("kind", "event"))
            text = str(entry.get("text", "")).strip()
            slug = str(entry.get("slug", "")).strip()
            slug_suffix = f" [{slug}]" if slug else ""
            lines.append(f"- `{kind}`{slug_suffix}: {text}")
        return "\n".join(lines) + "\n"


class TranscriptBlockPublisher:
    """Record block events into transcript before forwarding them."""

    def __init__(
        self,
        *,
        delegate: BlockPublisher,
        transcript_store: TaskTranscriptStore,
    ) -> None:
        self._delegate = delegate
        self._transcript_store = transcript_store

    async def publish(self, event: BlockEvent) -> None:
        if event.task_id:
            await self._append_event(event)
        await self._delegate.publish(event)

    async def _append_event(self, event: BlockEvent) -> None:
        try:
            await self._transcript_store.append(
                task_id=event.task_id,
                user_id=event.user_id,
                kind="block_event",
                text=_block_event_text(event),
                slug=event.slug,
                payload={
                    "event_type": event.event_type.value,
                    **event.payload,
                },
            )
        except Exception:
            logger.warning(
                "chat_self_coding_transcript_block_append_failed",
                task_id=event.task_id,
                event_type=event.event_type.value,
                exc_info=True,
            )


def _block_event_text(event: BlockEvent) -> str:
    detail = (
        event.payload.get("detail")
        or event.payload.get("summary")
        or event.payload.get("description")
        or event.payload.get("reason")
        or ""
    )
    return f"{event.event_type.value}: {str(detail).strip()}"


def _safe_task_id(task_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", task_id.strip()).strip(".-_")
    if not safe:
        raise ValueError("task_id is required")
    return safe

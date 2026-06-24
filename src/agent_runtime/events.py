"""Agent Runtime event streams."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from src.agent_runtime.models import AgentEvent

if TYPE_CHECKING:
    from pathlib import Path


class AgentEventStream(Protocol):
    """Append/read interface for runtime events."""

    async def emit(self, event: AgentEvent) -> None: ...
    def events_for(self, job_id: str) -> list[AgentEvent]: ...


class InMemoryAgentEventStream:
    """Test-friendly event stream implementation."""

    def __init__(self) -> None:
        self._events: list[AgentEvent] = []

    async def emit(self, event: AgentEvent) -> None:
        self._events.append(event)

    def events_for(self, job_id: str) -> list[AgentEvent]:
        return [event for event in self._events if event.job_id == job_id]


class FileAgentEventStream:
    """Durable JSONL event stream for one runtime directory."""

    def __init__(self, root: Path) -> None:
        self._root = root

    async def emit(self, event: AgentEvent) -> None:
        path = self._path_for(event.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(event.model_dump_json() + "\n")

    def events_for(self, job_id: str) -> list[AgentEvent]:
        path = self._path_for(job_id)
        if not path.exists():
            return []
        return [
            AgentEvent.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _path_for(self, job_id: str) -> Path:
        return self._root / "events" / f"{job_id}.jsonl"

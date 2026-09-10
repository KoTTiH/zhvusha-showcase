"""File-backed append-only LifeRuntime store."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from pydantic import ValidationError

from src.life_runtime.models import LifeEvent, LifeTick, SelfState, SelfStateMode

if TYPE_CHECKING:
    from pathlib import Path

_T = TypeVar("_T", LifeEvent, LifeTick)


class FileLifeRuntimeStore:
    """Persist LifeRuntime audit records separately from core memory."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._events_path = root / "events.jsonl"
        self._ticks_path = root / "ticks.jsonl"
        self._state_path = root / "self_state.json"

    def append_event(self, event: LifeEvent) -> bool:
        """Append an event unless its dedupe key was already observed."""

        if event.dedupe_key and event.dedupe_key in self._seen_event_dedupe_keys():
            return False
        self._append_jsonl(self._events_path, event)
        return True

    def list_events(self) -> tuple[LifeEvent, ...]:
        """Return all stored events in append order."""

        return self._read_jsonl(self._events_path, LifeEvent)

    def append_tick(self, tick: LifeTick) -> None:
        """Append one completed LifeTick."""

        self._append_jsonl(self._ticks_path, tick)

    def list_ticks(self) -> tuple[LifeTick, ...]:
        """Return all stored ticks in append order."""

        return self._read_jsonl(self._ticks_path, LifeTick)

    def load_state(self) -> SelfState:
        """Load the latest durable SelfState or recover fail-closed."""

        if not self._state_path.exists():
            return SelfState()
        try:
            return SelfState.model_validate_json(
                self._state_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError):
            return SelfState(
                mode=SelfStateMode.RECOVERING,
                attention_summary="state recovery after corrupted LifeRuntime state",
            )

    def save_state(self, state: SelfState) -> None:
        """Atomically write the current durable SelfState snapshot."""

        self._root.mkdir(parents=True, exist_ok=True)
        tmp_path = self._state_path.with_suffix(".json.tmp")
        tmp_path.write_text(state.model_dump_json(), encoding="utf-8")
        tmp_path.replace(self._state_path)

    def _seen_event_dedupe_keys(self) -> set[str]:
        return {event.dedupe_key for event in self.list_events() if event.dedupe_key}

    def _append_jsonl(self, path: Path, item: LifeEvent | LifeTick) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(item.model_dump_json())
            handle.write("\n")

    def _read_jsonl(self, path: Path, model: type[_T]) -> tuple[_T, ...]:
        if not path.exists():
            return ()
        items: list[_T] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            items.append(model.model_validate_json(line))
        return tuple(items)

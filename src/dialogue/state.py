"""Persistent working memory for one chat dialogue."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_MAX_CORRECTIONS = 8


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _timestamp_for_filename() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _clean_tuple(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    cleaned: list[str] = []
    for item in values:
        text = _clean_text(item)
        if text and text not in cleaned:
            cleaned.append(text)
    return tuple(cleaned)


class DialogueState(BaseModel):
    """Compact semantic state that survives across turns.

    Raw JSONL chat logs remain the source of truth. This model stores only the
    current working meaning needed by routing, chat responses and tools.
    """

    model_config = ConfigDict(extra="ignore")

    chat_id: str
    mode: str = "personal"
    active_topic: str = ""
    pending_action: str = ""
    selected_skill: str = ""
    last_intent: str = ""
    last_user_message: str = ""
    last_assistant_response: str = ""
    recipient_hint: str = ""
    executable_chat_id: str = ""
    draft_message: str = ""
    missing_fields: tuple[str, ...] = Field(default_factory=tuple)
    corrections: tuple[str, ...] = Field(default_factory=tuple)
    last_tool: str = ""
    last_result: str = ""
    confidence: float = 0.0
    turn_count: int = 0
    last_update_source: str = ""
    updated_at: str = Field(default_factory=_now_iso)

    @field_validator(
        "chat_id",
        "mode",
        "active_topic",
        "pending_action",
        "selected_skill",
        "last_intent",
        "last_user_message",
        "last_assistant_response",
        "recipient_hint",
        "executable_chat_id",
        "draft_message",
        "last_tool",
        "last_result",
        "last_update_source",
        mode="before",
    )
    @classmethod
    def _strip_string_fields(cls, value: object) -> str:
        return _clean_text(value)

    @field_validator("missing_fields", "corrections", mode="before")
    @classmethod
    def _normalize_tuple_fields(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, tuple | list):
            return _clean_tuple(value)
        text = _clean_text(value)
        return (text,) if text else ()

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: object) -> float:
        if not isinstance(value, str | bytes | bytearray | int | float):
            return 0.0
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(confidence, 1.0))

    @field_validator("executable_chat_id", mode="after")
    @classmethod
    def _sanitize_executable_chat_id(cls, value: str) -> str:
        return value if _looks_like_executable_chat_id(value) else ""

    @model_validator(mode="after")
    def _missing_chat_id_clears_executable_target(self) -> DialogueState:
        if "chat_id" in self.missing_fields:
            self.executable_chat_id = ""
        return self

    def has_signal(self) -> bool:
        """Return True when state has useful meaning beyond bookkeeping."""
        return any(
            (
                self.active_topic,
                self.pending_action,
                self.selected_skill,
                self.last_intent,
                self.recipient_hint,
                self.executable_chat_id,
                self.draft_message,
                self.missing_fields,
                self.corrections,
                self.last_tool,
                self.last_result,
            )
        )


class DialogueStatePatch(BaseModel):
    """Incremental update for ``DialogueState``."""

    model_config = ConfigDict(extra="ignore")

    mode: str | None = None
    active_topic: str | None = None
    pending_action: str | None = None
    selected_skill: str | None = None
    last_intent: str | None = None
    last_user_message: str | None = None
    last_assistant_response: str | None = None
    recipient_hint: str | None = None
    executable_chat_id: str | None = None
    draft_message: str | None = None
    missing_fields: tuple[str, ...] | None = None
    last_tool: str | None = None
    last_result: str | None = None
    confidence: float | None = None
    source: str = ""
    append_correction: str = ""
    increment_turn: bool = False
    clear_pending_action: bool = False
    clear_executable_chat_id: bool = False
    clear_missing_fields: bool = False

    @field_validator(
        "mode",
        "active_topic",
        "pending_action",
        "selected_skill",
        "last_intent",
        "last_user_message",
        "last_assistant_response",
        "recipient_hint",
        "executable_chat_id",
        "draft_message",
        "last_tool",
        "last_result",
        "source",
        "append_correction",
        mode="before",
    )
    @classmethod
    def _strip_optional_string_fields(cls, value: object) -> str | None:
        if value is None:
            return None
        return _clean_text(value)

    @field_validator("missing_fields", mode="before")
    @classmethod
    def _normalize_missing_fields(cls, value: object) -> tuple[str, ...] | None:
        if value is None:
            return None
        if isinstance(value, tuple | list):
            return _clean_tuple(value)
        text = _clean_text(value)
        return (text,) if text else ()

    @field_validator("executable_chat_id", mode="after")
    @classmethod
    def _sanitize_executable_chat_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value if _looks_like_executable_chat_id(value) else ""

    def apply_to(self, state: DialogueState) -> DialogueState:
        updates = self._field_updates()
        self._apply_control_updates(state, updates)
        return state.model_copy(update=updates)

    def _field_updates(self) -> dict[str, Any]:
        updates: dict[str, Any] = {"updated_at": _now_iso()}
        for field_name in (
            "mode",
            "active_topic",
            "pending_action",
            "selected_skill",
            "last_intent",
            "last_user_message",
            "last_assistant_response",
            "recipient_hint",
            "executable_chat_id",
            "draft_message",
            "last_tool",
            "last_result",
        ):
            value = getattr(self, field_name)
            if value is not None:
                updates[field_name] = value

        if self.missing_fields is not None:
            updates["missing_fields"] = self.missing_fields
        if self.confidence is not None:
            updates["confidence"] = max(0.0, min(float(self.confidence), 1.0))
        if self.source:
            updates["last_update_source"] = self.source
        return updates

    def _apply_control_updates(
        self,
        state: DialogueState,
        updates: dict[str, Any],
    ) -> None:
        if self.increment_turn:
            updates["turn_count"] = state.turn_count + 1
        if self.clear_pending_action:
            updates["pending_action"] = ""
        if self.clear_executable_chat_id:
            updates["executable_chat_id"] = ""
        if self.clear_missing_fields:
            updates["missing_fields"] = ()
        if self.append_correction:
            corrections = (*state.corrections, self.append_correction)
            updates["corrections"] = corrections[-_MAX_CORRECTIONS:]

        missing_fields = tuple(updates.get("missing_fields", state.missing_fields))
        if (
            updates.get("executable_chat_id")
            and self.missing_fields is None
            and "chat_id" in missing_fields
        ):
            missing_fields = tuple(
                field for field in missing_fields if field != "chat_id"
            )
            updates["missing_fields"] = missing_fields
        if "chat_id" in missing_fields:
            updates["executable_chat_id"] = ""


class FileDialogueStateStore:
    """JSON-file store colocated with append-only chat logs."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        max_age_seconds: int | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._workspace_root = workspace_root
        self._max_age_seconds = max_age_seconds
        self._clock = clock or (lambda: datetime.now(tz=UTC))

    def load(self, chat_id: int | str | None) -> DialogueState:
        normalized_chat_id = _normalize_chat_id(chat_id)
        path = self._state_path(normalized_chat_id)
        if not path.is_file():
            return DialogueState(chat_id=normalized_chat_id)
        try:
            payload = path.read_text(encoding="utf-8")
        except OSError:
            return DialogueState(chat_id=normalized_chat_id)
        try:
            state = DialogueState.model_validate_json(payload)
        except ValueError:
            self._quarantine_corrupt_state(path)
            return DialogueState(chat_id=normalized_chat_id)
        if self._is_expired(state):
            self._quarantine_expired_state(path)
            return DialogueState(chat_id=normalized_chat_id)
        if state.chat_id != normalized_chat_id:
            return state.model_copy(update={"chat_id": normalized_chat_id})
        return state

    def save(self, state: DialogueState) -> None:
        path = self._state_path(state.chat_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        tmp_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def apply_patch(
        self,
        chat_id: int | str | None,
        patch: DialogueStatePatch,
    ) -> DialogueState:
        state = patch.apply_to(self.load(chat_id))
        self.save(state)
        return state

    def _state_path(self, chat_id: str) -> Path:
        return self._workspace_root / "logs" / chat_id / "dialogue_state.json"

    def _quarantine_corrupt_state(self, path: Path) -> None:
        target = path.with_name(
            f"dialogue_state.corrupt-{_timestamp_for_filename()}.json"
        )
        try:
            path.replace(target)
        except OSError:
            return

    def _quarantine_expired_state(self, path: Path) -> None:
        target = path.with_name(
            f"dialogue_state.expired-{_timestamp_for_filename()}.json"
        )
        try:
            path.replace(target)
        except OSError:
            return

    def _is_expired(self, state: DialogueState) -> bool:
        if self._max_age_seconds is None:
            return False
        updated_at = _parse_datetime(state.updated_at)
        if updated_at is None:
            return True
        return (self._clock().astimezone(UTC) - updated_at).total_seconds() > max(
            self._max_age_seconds,
            0,
        )


def render_dialogue_context(
    state: DialogueState,
    *,
    decision_recent_messages: str = "",
) -> str:
    """Render compact prompt context without exposing raw JSON."""
    if not state.has_signal() and not decision_recent_messages.strip():
        return ""

    lines: list[str] = []
    _append_line(lines, "active_topic", state.active_topic)
    _append_line(lines, "pending_action", state.pending_action)
    _append_line(lines, "selected_skill", state.selected_skill)
    _append_line(lines, "last_intent", state.last_intent)
    _append_line(lines, "recipient_hint", state.recipient_hint)
    if state.executable_chat_id:
        _append_line(lines, "executable_chat_id", state.executable_chat_id)
    elif state.recipient_hint or "chat_id" in state.missing_fields:
        lines.append("executable_chat_id: missing")
    _append_line(lines, "draft_message", state.draft_message)
    if state.missing_fields:
        lines.append(f"missing_fields: {', '.join(state.missing_fields)}")
    if state.corrections:
        lines.append("corrections:")
        lines.extend(f"- {correction}" for correction in state.corrections)
    _append_line(lines, "last_tool", state.last_tool)
    _append_line(lines, "last_result", state.last_result)

    parts = [
        "<DIALOGUE_STATE>",
        "\n".join(lines) if lines else "state: empty",
        "</DIALOGUE_STATE>",
    ]
    recent = decision_recent_messages.strip()
    if recent:
        parts.extend(
            [
                "",
                "<RECENT_DECISION_CONTEXT>",
                recent,
                "</RECENT_DECISION_CONTEXT>",
            ]
        )
    return "\n".join(parts)


def render_dialogue_status(state: DialogueState) -> str:
    """Render safe operator-facing dialogue memory status.

    This intentionally reports presence and structural fields, not raw message
    text or draft content.
    """
    if not state.has_signal():
        return "Диалоговая память пуста."

    lines = ["Диалоговая память:"]
    _append_line(lines, "mode", state.mode)
    lines.append(f"turn_count: {state.turn_count}")
    _append_line(lines, "pending_action", state.pending_action)
    _append_line(lines, "selected_skill", state.selected_skill)
    _append_line(lines, "active_topic", state.active_topic)
    _append_line(lines, "last_intent", state.last_intent)
    _append_line(lines, "recipient_hint", state.recipient_hint)
    if state.executable_chat_id:
        _append_line(lines, "executable_chat_id", state.executable_chat_id)
    elif state.recipient_hint or "chat_id" in state.missing_fields:
        lines.append("executable_chat_id: missing")
    if state.draft_message:
        lines.append("draft_message: present")
    if state.missing_fields:
        lines.append(f"missing_fields: {', '.join(state.missing_fields)}")
    if state.corrections:
        lines.append(f"corrections: {len(state.corrections)}")
    _append_line(lines, "last_tool", state.last_tool)
    _append_line(lines, "last_result", state.last_result)
    _append_line(lines, "updated_at", state.updated_at)
    return "\n".join(lines)


def dialogue_state_from_metadata(value: object) -> DialogueState | None:
    """Parse ``AgentContext.metadata['dialogue_state']`` safely."""
    if isinstance(value, DialogueState):
        return value
    if isinstance(value, dict):
        try:
            return DialogueState.model_validate(value)
        except ValueError:
            return None
    return None


def dialogue_state_patch_from_metadata(value: object) -> DialogueStatePatch | None:
    """Parse a state patch carried in skill metadata."""
    if isinstance(value, DialogueStatePatch):
        return value
    if isinstance(value, dict):
        try:
            return DialogueStatePatch.model_validate(value)
        except ValueError:
            return None
    return None


def _append_line(lines: list[str], key: str, value: str) -> None:
    if value:
        lines.append(f"{key}: {value}")


def _normalize_chat_id(chat_id: int | str | None) -> str:
    return str(chat_id or "unknown").strip() or "unknown"


def _parse_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _looks_like_executable_chat_id(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if text.startswith("@"):
        username = text[1:]
        return bool(username) and all(
            char.isascii() and (char.isalnum() or char == "_") for char in username
        )
    number = text[1:] if text.startswith("-") else text
    return bool(number) and number.isdigit()

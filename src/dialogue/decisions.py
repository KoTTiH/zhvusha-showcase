"""Shared pending-decision contract for Zhvusha's cognitive loop."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from pathlib import Path

DecisionOutcome = Literal[
    "approve",
    "reject",
    "revise",
    "ask_more",
    "defer",
    "new_topic",
]
DecisionSignal = Literal["yes", "no", "later", "ambiguous"]

_DEFAULT_OUTCOMES: tuple[DecisionOutcome, ...] = (
    "approve",
    "reject",
    "revise",
    "ask_more",
    "defer",
    "new_topic",
)
_COGNITIVE_QUALIFIER_MARKERS = (
    " но ",
    ", но",
    "но ",
    "только ",
    " если ",
    ", если",
    "сначала",
    "покажи",
    "что именно",
    "а что",
    "почему",
    "?",
    "не ",
    "не\t",
    "не,",
    "не.",
    " а ",
    ", а ",
)


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


class PendingDecision(BaseModel):
    """A user-facing decision waiting for Zhvusha's next cognitive turn.

    It intentionally stores structured proposal data instead of final chat text.
    The owner can be a skill, router, daemon, worker or tool layer, but the
    decision itself belongs to Zhvusha's central loop.
    """

    model_config = ConfigDict(extra="ignore")

    decision_id: str
    kind: str
    owner: str
    action: str
    summary: str
    proposal: dict[str, Any] = Field(default_factory=dict)
    required_consent: bool = True
    allowed_outcomes: tuple[DecisionOutcome, ...] = _DEFAULT_OUTCOMES
    missing_fields: tuple[str, ...] = Field(default_factory=tuple)
    constraints: tuple[str, ...] = Field(default_factory=tuple)
    context_snapshot: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)

    @field_validator("decision_id", "kind", "owner", "action", "summary", mode="before")
    @classmethod
    def _strip_required_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("missing_fields", "constraints", mode="before")
    @classmethod
    def _normalize_tuple(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, tuple | list):
            return tuple(_clean_unique(value))
        text = str(value or "").strip()
        return (text,) if text else ()

    @field_validator("allowed_outcomes", mode="before")
    @classmethod
    def _normalize_outcomes(cls, value: object) -> tuple[DecisionOutcome, ...]:
        if value is None:
            return _DEFAULT_OUTCOMES
        if not isinstance(value, tuple | list):
            return _DEFAULT_OUTCOMES
        allowed: list[DecisionOutcome] = []
        for item in value:
            text = str(item or "").strip()
            if text in _DEFAULT_OUTCOMES and text not in allowed:
                allowed.append(text)
        return tuple(allowed) or _DEFAULT_OUTCOMES


class DecisionResolution(BaseModel):
    """Zhvusha's structured resolution for one pending decision."""

    model_config = ConfigDict(extra="ignore")

    decision_id: str
    outcome: DecisionOutcome
    reason: str = ""
    revised_proposal: dict[str, Any] | None = None
    missing_fields: tuple[str, ...] = Field(default_factory=tuple)
    user_message: str = ""
    confidence: float = 0.0
    source: str = ""

    @field_validator("decision_id", "reason", "user_message", "source", mode="before")
    @classmethod
    def _strip_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("missing_fields", mode="before")
    @classmethod
    def _normalize_missing_fields(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, tuple | list):
            return tuple(_clean_unique(value))
        text = str(value or "").strip()
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


def should_defer_to_cognitive_loop(text: str) -> bool:
    """Return True when a reply contains a condition, correction or question."""
    normalized = f" {text.strip().lower()} "
    if not normalized.strip():
        return True
    return any(marker in normalized for marker in _COGNITIVE_QUALIFIER_MARKERS)


def resolution_from_approval_signal(
    signal: DecisionSignal,
    pending: PendingDecision,
    *,
    user_message: str = "",
) -> DecisionResolution:
    """Map a legacy approval signal into the shared decision vocabulary."""
    if signal == "yes":
        outcome: DecisionOutcome = "approve"
    elif signal == "no":
        outcome = "reject"
    elif signal == "later":
        outcome = "defer"
    else:
        outcome = "ask_more"
    return DecisionResolution(
        decision_id=pending.decision_id,
        outcome=outcome,
        reason=f"legacy approval signal: {signal}",
        user_message=user_message,
        confidence=1.0 if signal != "ambiguous" else 0.0,
        source="approval_signal",
    )


class FilePendingDecisionStore:
    """Durable per-chat store for structured pending decisions."""

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root

    def save(self, chat_id: int | str | None, decision: PendingDecision) -> None:
        path = self._pending_path(chat_id, decision.decision_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        tmp_path.write_text(decision.model_dump_json(indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def get(
        self,
        chat_id: int | str | None,
        decision_id: str,
    ) -> PendingDecision | None:
        path = self._pending_path(chat_id, decision_id)
        if not path.is_file():
            return None
        try:
            return PendingDecision.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def list_pending(self, chat_id: int | str | None) -> tuple[PendingDecision, ...]:
        root = self._pending_root(chat_id)
        if not root.is_dir():
            return ()
        pending: list[PendingDecision] = []
        for path in sorted(root.glob("*.json")):
            try:
                pending.append(
                    PendingDecision.model_validate_json(
                        path.read_text(encoding="utf-8")
                    )
                )
            except (OSError, ValueError):
                continue
        return tuple(sorted(pending, key=lambda item: item.created_at))

    def resolve(
        self,
        chat_id: int | str | None,
        resolution: DecisionResolution,
    ) -> PendingDecision | None:
        decision = self.get(chat_id, resolution.decision_id)
        if decision is None:
            return None
        source_path = self._pending_path(chat_id, resolution.decision_id)
        resolved_path = (
            self._pending_root(chat_id)
            / "resolved"
            / f"{_safe_filename(resolution.decision_id)}.json"
        )
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "decision": decision.model_dump(mode="json"),
            "resolution": resolution.model_dump(mode="json"),
            "resolved_at": _now_iso(),
        }
        tmp_path = resolved_path.with_name(f"{resolved_path.name}.tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(resolved_path)
        try:
            source_path.unlink()
        except OSError:
            return decision
        return decision

    def _pending_root(self, chat_id: int | str | None) -> Path:
        return (
            self._workspace_root
            / "logs"
            / _normalize_chat_id(chat_id)
            / "pending_decisions"
        )

    def _pending_path(self, chat_id: int | str | None, decision_id: str) -> Path:
        return self._pending_root(chat_id) / f"{_safe_filename(decision_id)}.json"


def _clean_unique(values: tuple[object, ...] | list[object]) -> list[str]:
    cleaned: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _normalize_chat_id(chat_id: int | str | None) -> str:
    return str(chat_id or "unknown").strip() or "unknown"


def _safe_filename(value: str) -> str:
    cleaned = str(value or "").strip()
    safe = "".join(
        char if char.isalnum() or char in {"-", "_"} else "-" for char in cleaned
    )
    return safe.strip("-_") or "unknown"

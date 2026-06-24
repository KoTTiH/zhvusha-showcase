"""LifeRuntime attention scoring for one bounded tick."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.life_runtime.models import (
    AttentionItem,
    AttentionStatus,
    DriveVector,
    LifeEvent,
    LifeEventKind,
    LifePriority,
    SelfState,
    default_attention_decay,
)

if TYPE_CHECKING:
    from datetime import datetime


def select_attention_item(
    *,
    event: LifeEvent,
    state: SelfState,
    drives: DriveVector,
    now: datetime,
) -> AttentionItem:
    """Select a deterministic attention item for the current trigger."""

    risk = _risk_for_event(event)
    urgency = _urgency_for_priority(event.priority)
    open_loop_pressure = min(len(state.open_loops) / 5.0, 1.0)
    salience = _clamp(
        0.2
        + urgency * 0.25
        + drives.silence_pressure * 0.2
        + drives.relational_continuity * 0.15
        + open_loop_pressure * 0.2
        - risk * 0.15
    )
    summary = _summary_for_event(event=event, state=state)
    return AttentionItem(
        id=f"attention:{event.id}",
        event_id=event.id,
        summary=summary,
        status=AttentionStatus.SELECTED,
        salience=salience,
        urgency=urgency,
        novelty=0.55 if event.kind is LifeEventKind.SILENCE_TICK else 0.7,
        relation_to_nikita=drives.care_for_nikita,
        risk=risk,
        estimated_cost=0.15,
        decay_after=default_attention_decay(now),
        evidence=(event.id, *state.open_loops[:3]),
        created_at=now,
    )


def _urgency_for_priority(priority: LifePriority) -> float:
    if priority is LifePriority.CRITICAL:
        return 1.0
    if priority is LifePriority.BACKGROUND:
        return 0.25
    return 0.55


def _risk_for_event(event: LifeEvent) -> float:
    if event.kind in {LifeEventKind.FAILED_JOB, LifeEventKind.BUDGET_GUARD}:
        return 0.8
    raw = event.payload.get("risk", 0.2)
    return _clamp(_as_float(raw, default=0.2))


def _summary_for_event(*, event: LifeEvent, state: SelfState) -> str:
    if event.kind is LifeEventKind.SILENCE_TICK:
        focus = state.current_focus or (state.open_loops[0] if state.open_loops else "")
        if focus:
            return f"silence tick while focus is `{focus}`"
        return "silence tick with no current focus"
    if event.payload:
        reason = str(event.payload.get("reason", "")).strip()
        if reason:
            return f"{event.kind.value}: {reason}"
    return event.kind.value


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float) -> float:
    return max(0.0, min(value, 1.0))

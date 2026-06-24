"""Bounded LifeRuntime tick runner."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from src.life_runtime.agent_runtime_bridge import build_life_reflection_action_request
from src.life_runtime.attention import select_attention_item
from src.life_runtime.drives import build_drive_vector
from src.life_runtime.models import (
    AttentionItem,
    InnerDecision,
    InnerDecisionType,
    LifeEvent,
    LifeEventKind,
    LifeTick,
    SelfState,
    SelfStateMode,
)
from src.life_runtime.safety import LifeRuntimeSafetyGuard

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.life_runtime.store import FileLifeRuntimeStore


class LifeTickRunner:
    """Run exactly one read-only LifeRuntime tick and stop."""

    def __init__(
        self,
        *,
        store: FileLifeRuntimeStore,
        safety_guard: LifeRuntimeSafetyGuard | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._safety_guard = safety_guard or LifeRuntimeSafetyGuard()
        self._clock = clock or (lambda: datetime.now(UTC))

    def run_once(self, event: LifeEvent) -> LifeTick:
        """Process one event, append audit/state records and return one tick."""

        started_at = self._clock()
        self._store.append_event(event)
        loaded_state = self._store.load_state()
        loaded_state_hash = _hash_state(loaded_state)
        drives = build_drive_vector(loaded_state)
        attention = select_attention_item(
            event=event,
            state=loaded_state,
            drives=drives,
            now=started_at,
        )
        tick_id = f"life-tick-{uuid4().hex}"
        decision = _build_decision(
            tick_id=tick_id,
            event=event,
            state=loaded_state,
            attention=attention,
        )
        safety_verdict = self._safety_guard.evaluate(decision)
        finished_at = self._clock()
        state_delta: dict[str, str] = {}
        if safety_verdict.allowed:
            updated_state = _updated_state(
                state=loaded_state,
                tick_id=tick_id,
                attention=attention,
                decision=decision,
                now=finished_at,
            )
            self._store.save_state(updated_state)
            state_delta = {
                "mode": updated_state.mode.value,
                "current_focus": updated_state.current_focus,
                "last_tick_id": updated_state.last_tick_id,
            }
        tick = LifeTick(
            id=tick_id,
            trigger_event_id=event.id,
            started_at=started_at,
            finished_at=finished_at,
            loaded_state_hash=loaded_state_hash,
            selected_attention_id=attention.id,
            drive_vector=drives,
            decision=decision,
            safety_verdict=safety_verdict,
            state_delta=state_delta,
            result_summary=_result_summary(safety_verdict.allowed),
        )
        self._store.append_tick(tick)
        return tick


def _build_decision(
    *,
    tick_id: str,
    event: LifeEvent,
    state: SelfState,
    attention: AttentionItem,
) -> InnerDecision:
    if attention.risk >= 0.75:
        return InnerDecision(
            decision_type=InnerDecisionType.DEFER,
            reason="attention item is too risky for read-only MVP",
        )
    if event.kind in {
        LifeEventKind.SILENCE_TICK,
        LifeEventKind.DESIRE_STALE,
        LifeEventKind.HOMEOSTASIS_DRIFT,
    }:
        return InnerDecision(
            decision_type=InnerDecisionType.REFLECT,
            reason=(
                "bounded read-only reflection for inner continuity"
                if state.open_loops or state.active_desires
                else "bounded read-only reflection after silence"
            ),
            action_request=build_life_reflection_action_request(
                tick_id=tick_id,
                reason=attention.summary,
            ),
        )
    return InnerDecision(
        decision_type=InnerDecisionType.THINK,
        reason="event can be handled as internal thought without tool access",
    )


def _updated_state(
    *,
    state: SelfState,
    tick_id: str,
    attention: AttentionItem,
    decision: InnerDecision,
    now: datetime,
) -> SelfState:
    return state.model_copy(
        update={
            "mode": SelfStateMode.REFLECTING
            if decision.decision_type == "reflect"
            else SelfStateMode.ATTENDING,
            "current_focus": attention.summary,
            "attention_summary": (
                f"{attention.summary}; decision={decision.decision_type.value}"
            ),
            "recent_decision_ids": (*state.recent_decision_ids, decision.id)[-20:],
            "last_tick_id": tick_id,
            "last_tick_at": now,
            "updated_at": now,
        }
    )


def _hash_state(state: SelfState) -> str:
    return hashlib.sha256(state.model_dump_json().encode("utf-8")).hexdigest()


def _result_summary(allowed: bool) -> str:
    if allowed:
        return "one bounded read-only LifeRuntime decision emitted and stopped"
    return "LifeRuntime decision blocked by safety guard and stopped"

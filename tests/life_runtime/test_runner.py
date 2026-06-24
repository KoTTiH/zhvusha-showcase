"""Read-only LifeRuntime tick contract."""

from __future__ import annotations

from datetime import UTC, datetime


def test_life_tick_readonly_mvp_emits_one_safe_inner_decision_and_stops(
    tmp_path,
) -> None:
    from src.life_runtime import (
        FileLifeRuntimeStore,
        LifeEvent,
        LifeTickRunner,
    )

    now = datetime(2026, 5, 14, 10, 30, tzinfo=UTC)
    store = FileLifeRuntimeStore(tmp_path)
    store.save_state(
        store.load_state().model_copy(
            update={
                "open_loops": ("finish LifeRuntime read-only MVP",),
                "active_desires": ("keep inner continuity between messages",),
            }
        )
    )
    runner = LifeTickRunner(store=store, clock=lambda: now)

    tick = runner.run_once(
        LifeEvent(
            id="silence:2026-05-14T10:30:00Z",
            kind="silence_tick",
            source="daemon",
            priority="normal",
            payload={"reason": "idle timeout"},
            observed_at=now,
        )
    )

    assert tick.trigger_event_id == "silence:2026-05-14T10:30:00Z"
    assert tick.decision.decision_type == "reflect"
    assert tick.decision.requires_approval is False
    assert tick.decision.action_request is not None
    assert tick.decision.action_request.profile_id == "life_reflection.readonly"
    assert "send_message" in tick.decision.action_request.denied_capabilities
    assert "write_files" in tick.decision.action_request.denied_capabilities
    assert tick.safety_verdict.allowed is True
    assert "read-only" in tick.result_summary
    assert len(store.list_ticks()) == 1

    state = store.load_state()
    assert state.last_tick_at == now
    assert state.current_focus
    assert state.recent_decision_ids == (tick.decision.id,)

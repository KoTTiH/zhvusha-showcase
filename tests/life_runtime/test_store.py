"""LifeRuntime append-only store contract."""

from __future__ import annotations

from datetime import UTC, datetime


def test_life_runtime_store_recovers_state_and_dedupes_events(tmp_path) -> None:
    from src.life_runtime import FileLifeRuntimeStore, LifeEvent, SelfState

    store = FileLifeRuntimeStore(tmp_path)
    state = SelfState(
        current_focus="LifeRuntime MVP",
        open_loops=("write tests",),
        updated_at=datetime(2026, 5, 14, 10, tzinfo=UTC),
    )
    event = LifeEvent(
        id="event-1",
        kind="workspace_signal",
        source="workspace",
        priority="normal",
        dedupe_key="workspace:event-1",
        observed_at=datetime(2026, 5, 14, 10, tzinfo=UTC),
    )

    assert store.append_event(event) is True
    assert store.append_event(event) is False
    store.save_state(state)

    restarted = FileLifeRuntimeStore(tmp_path)

    assert restarted.load_state().current_focus == "LifeRuntime MVP"
    assert [item.id for item in restarted.list_events()] == ["event-1"]

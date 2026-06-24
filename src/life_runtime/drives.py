"""Deterministic LifeRuntime drive scoring."""

from __future__ import annotations

from src.life_runtime.models import DriveVector, SelfState


def build_drive_vector(state: SelfState) -> DriveVector:
    """Build the MVP drive vector from durable state snapshots."""

    open_loop_pressure = min(len(state.open_loops) / 5.0, 1.0)
    question_pressure = min(len(state.unresolved_questions) / 3.0, 1.0)
    desire_pressure = min(len(state.active_desires) / 4.0, 1.0)
    budget = state.budget_state.strip().lower()
    energy_budget = 0.3 if budget in {"low", "exhausted", "blocked"} else 0.7
    caution = 0.85 if budget in {"low", "exhausted", "blocked"} else 0.65
    return DriveVector(
        curiosity=_clamp(0.45 + question_pressure * 0.25),
        care_for_nikita=_clamp(0.7 + desire_pressure * 0.15),
        honesty_pressure=0.75,
        caution=caution,
        complexity_growth=_clamp(0.45 + desire_pressure * 0.2),
        relational_continuity=_clamp(0.65 + open_loop_pressure * 0.2),
        energy_budget=energy_budget,
        learning_pressure=_clamp(0.45 + open_loop_pressure * 0.2),
        action_pressure=_clamp(0.15 + desire_pressure * 0.2),
        silence_pressure=_clamp(0.35 + open_loop_pressure * 0.35),
    )


def _clamp(value: float) -> float:
    return max(0.0, min(value, 1.0))

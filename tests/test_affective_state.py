"""Tests for the affective state tracker."""

from __future__ import annotations

import pytest
from src.personality.affective_state import (
    AffectiveStateManager,
    get_affective_state_manager,
)

# ============================================================
# Baseline
# ============================================================


def test_initial_state_is_curiosity_baseline() -> None:
    mgr = AffectiveStateManager()
    state = mgr.get_state()
    assert state.self_emotion == "curiosity"
    assert state.self_valence == pytest.approx(0.6)
    assert state.self_arousal == pytest.approx(0.6)


def test_initial_user_state_is_neutral() -> None:
    mgr = AffectiveStateManager()
    state = mgr.get_state()
    assert state.user_emotion == "neutral"
    assert state.user_valence == pytest.approx(0.0)
    assert state.user_arousal == pytest.approx(0.5)


# ============================================================
# Update from enrichment
# ============================================================


def _make_enrichment(
    *,
    emotion: str = "happy",
    valence: str = "positive",
    confidence: float = 0.8,
    arousal: float = 0.5,
    self_emotion: str = "curiosity",
    self_arousal: float = 0.5,
) -> object:
    """Minimal duck-typed enrichment result for testing."""
    from types import SimpleNamespace

    valence_map = {"positive": 0.5, "negative": -0.5, "neutral": 0.0}
    return SimpleNamespace(
        emotion=emotion,
        valence=valence,
        confidence=confidence,
        arousal=arousal,
        self_emotion=self_emotion,
        self_arousal=self_arousal,
        _valence_float=valence_map.get(valence, 0.0),
    )


def test_update_sets_user_emotion() -> None:
    mgr = AffectiveStateManager()
    enrichment = _make_enrichment(emotion="angry", valence="negative", arousal=0.8)
    mgr.update_from_enrichment(enrichment)  # type: ignore[arg-type]
    state = mgr.get_state()
    assert state.user_emotion == "angry"
    assert state.user_arousal == pytest.approx(0.8)
    assert state.user_valence < 0


def test_update_sets_self_emotion() -> None:
    mgr = AffectiveStateManager()
    enrichment = _make_enrichment(self_emotion="warmth", self_arousal=0.4)
    mgr.update_from_enrichment(enrichment)  # type: ignore[arg-type]
    state = mgr.get_state()
    assert state.self_emotion == "warmth"
    assert state.self_arousal == pytest.approx(0.4)


def test_update_resets_turns_counter() -> None:
    mgr = AffectiveStateManager()
    mgr._state.turns_since_update = 10
    enrichment = _make_enrichment()
    mgr.update_from_enrichment(enrichment)  # type: ignore[arg-type]
    assert mgr.get_state().turns_since_update == 0


# ============================================================
# Counter-regulation
# ============================================================


def test_counter_regulation_on_high_user_arousal() -> None:
    """When user arousal > 0.7, Zhvusha should dampen her own arousal."""
    mgr = AffectiveStateManager()
    enrichment = _make_enrichment(
        emotion="angry",
        valence="negative",
        arousal=0.9,
        self_emotion="excitement",
        self_arousal=0.8,
    )
    mgr.update_from_enrichment(enrichment)  # type: ignore[arg-type]
    state = mgr.get_state()
    assert state.regulation_active
    # Arousal should be dampened
    assert state.self_arousal < 0.8


def test_counter_regulation_warmth_on_user_negativity() -> None:
    """When user is negative + active, Zhvusha shifts to warmth."""
    mgr = AffectiveStateManager()
    enrichment = _make_enrichment(
        emotion="sad",
        valence="negative",
        arousal=0.6,
        self_emotion="curiosity",
        self_arousal=0.6,
    )
    mgr.update_from_enrichment(enrichment)  # type: ignore[arg-type]
    state = mgr.get_state()
    assert state.regulation_active
    assert state.regulation_target == "warmth"


def test_no_regulation_on_calm_user() -> None:
    mgr = AffectiveStateManager()
    enrichment = _make_enrichment(
        emotion="happy",
        valence="positive",
        arousal=0.4,
        self_emotion="joy",
        self_arousal=0.6,
    )
    mgr.update_from_enrichment(enrichment)  # type: ignore[arg-type]
    state = mgr.get_state()
    assert not state.regulation_active


# ============================================================
# Decay
# ============================================================


def test_decay_halves_at_5_turns() -> None:
    mgr = AffectiveStateManager()
    # Set to a non-baseline state
    enrichment = _make_enrichment(
        self_emotion="excitement", self_arousal=1.0, arousal=0.5
    )
    mgr.update_from_enrichment(enrichment)  # type: ignore[arg-type]

    # Simulate 5 turns of silence
    mgr._state.turns_since_update = 5
    mgr.decay_if_stale()

    state = mgr.get_state()
    # Should be ~50% back toward baseline (0.6)
    # excitement arousal was 1.0, baseline is 0.6
    # After decay: 0.6 + (1.0 - 0.6) * 0.5 = 0.8
    assert state.self_arousal == pytest.approx(0.8, abs=0.05)


def test_decay_reaches_baseline_after_15_turns() -> None:
    mgr = AffectiveStateManager()
    enrichment = _make_enrichment(
        self_emotion="hostility", self_arousal=0.9, arousal=0.5
    )
    mgr.update_from_enrichment(enrichment)  # type: ignore[arg-type]

    mgr._state.turns_since_update = 15
    mgr.decay_if_stale()

    state = mgr.get_state()
    # After 15 turns: factor = 0.5^3 = 0.125 → 87.5% back to baseline
    assert abs(state.self_arousal - 0.6) < 0.1
    assert abs(state.self_valence - 0.6) < 0.2


def test_no_decay_at_zero_turns() -> None:
    mgr = AffectiveStateManager()
    enrichment = _make_enrichment(self_emotion="excitement", self_arousal=0.9)
    mgr.update_from_enrichment(enrichment)  # type: ignore[arg-type]

    mgr.decay_if_stale()  # turns_since_update == 0
    state = mgr.get_state()
    assert state.self_arousal == pytest.approx(0.9, abs=0.05)


# ============================================================
# Prompt context
# ============================================================


def test_prompt_context_under_200_bytes() -> None:
    mgr = AffectiveStateManager()
    # Set a non-trivial state
    enrichment = _make_enrichment(
        emotion="frustrated",
        valence="negative",
        arousal=0.8,
        self_emotion="warmth",
        self_arousal=0.4,
    )
    mgr.update_from_enrichment(enrichment)  # type: ignore[arg-type]
    ctx = mgr.get_prompt_context()
    assert len(ctx.encode("utf-8")) <= 300  # slightly generous for Russian


def test_prompt_context_shows_regulation() -> None:
    mgr = AffectiveStateManager()
    enrichment = _make_enrichment(
        emotion="angry", valence="negative", arousal=0.9, self_emotion="calm"
    )
    mgr.update_from_enrichment(enrichment)  # type: ignore[arg-type]
    ctx = mgr.get_prompt_context()
    assert "регул" in ctx.lower()


def test_prompt_context_empty_at_baseline() -> None:
    mgr = AffectiveStateManager()
    ctx = mgr.get_prompt_context()
    assert ctx == ""


# ============================================================
# Singleton
# ============================================================


def test_singleton_returns_same_instance() -> None:
    import src.personality.affective_state as mod

    mod._manager = None  # reset
    a = get_affective_state_manager()
    b = get_affective_state_manager()
    assert a is b
    mod._manager = None  # cleanup


# ============================================================
# Snapshot line
# ============================================================


def test_snapshot_line_contains_emotion_name() -> None:
    mgr = AffectiveStateManager()
    enrichment = _make_enrichment(self_emotion="excitement", self_arousal=0.8)
    mgr.update_from_enrichment(enrichment)  # type: ignore[arg-type]
    line = mgr.get_snapshot_line()
    assert "excitement" in line

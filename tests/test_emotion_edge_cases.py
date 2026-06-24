"""Edge case tests for the emotion system.

Covers bugs found during architectural review:
1. decay_if_stale() integration with get_prompt_context()
2. _analyze_emotional_patterns valence_sum bug
3. get_prompt_context() idempotency
4. Counter-regulation with unknown emotion
5. Boundary float values
6. Singleton thread safety
7. Atlas-Enricher Literal sync
8. Multiple rapid updates
9. Decay after regulation
10. Emotional log dedup
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from src.memory.sonnet_enricher import EnrichmentResult
from src.personality.affective_state import AffectiveStateManager
from src.personality.emotion_atlas import (
    _COMPLEMENT_MAP,
    EMOTION_ATLAS,
    get_decay_target,
)

# ============================================================
# 1. Decay integrates with get_prompt_context()
# ============================================================


def test_decay_applied_through_prompt_context() -> None:
    """get_prompt_context() should apply decay automatically."""
    mgr = AffectiveStateManager()
    result = EnrichmentResult(
        importance=0.5,
        valence="negative",
        intent="emotional",
        emotion="angry",
        confidence=0.9,
        arousal=0.3,
        is_feedback=False,
        feedback_strength=0.0,
        reasoning="test",
        self_emotion="frustration",
        self_arousal=0.9,
    )
    mgr.update_from_enrichment(result)

    # Simulate many turns by calling get_prompt_context repeatedly
    for _ in range(25):
        mgr.get_prompt_context()

    # After 25 calls, decay should have snapped to baseline
    assert mgr._is_at_baseline
    assert mgr.get_prompt_context() == ""


def test_decay_gradual_through_prompt_context() -> None:
    """Arousal should decrease gradually across prompt context calls."""
    mgr = AffectiveStateManager()
    result = EnrichmentResult(
        importance=0.5,
        valence="neutral",
        intent="statement",
        emotion="neutral",
        confidence=0.5,
        arousal=0.5,
        is_feedback=False,
        feedback_strength=0.0,
        reasoning="test",
        self_emotion="excitement",
        self_arousal=1.0,
    )
    mgr.update_from_enrichment(result)

    arousals = []
    for _ in range(10):
        mgr.get_prompt_context()
        arousals.append(mgr.get_state().self_arousal)

    # Each subsequent value should be closer to baseline (0.6)
    for i in range(1, len(arousals)):
        assert abs(arousals[i] - 0.6) <= abs(arousals[i - 1] - 0.6) + 0.01


# ============================================================
# 2. _analyze_emotional_patterns correctness
# ============================================================


def test_analyze_patterns_uses_both_arousals() -> None:
    """Summary should include both self_arousal and user arousal."""
    from src.memory.consolidation import ConsolidationEngine

    meta = json.dumps(
        {
            "enrichment": {
                "self_emotion": "curiosity",
                "self_arousal": 0.8,
                "arousal": 0.3,
            }
        }
    )
    episodes = [
        SimpleNamespace(
            id=1,
            role="assistant",
            content="test",
            user_id=12345,
            valence="neutral",
            importance=0.5,
            timestamp=datetime.now(tz=UTC),
            metadata_json=meta,
        )
    ]

    summary = ConsolidationEngine._analyze_emotional_patterns(episodes)
    assert "self_arousal=0.80" in summary
    assert "user_arousal=0.30" in summary


def test_analyze_patterns_handles_missing_arousal_fields() -> None:
    """Episodes with partial enrichment data should use defaults."""
    from src.memory.consolidation import ConsolidationEngine

    meta = json.dumps({"enrichment": {"self_emotion": "joy"}})
    episodes = [
        SimpleNamespace(
            id=1,
            role="assistant",
            content="test",
            user_id=12345,
            valence="positive",
            importance=0.5,
            timestamp=datetime.now(tz=UTC),
            metadata_json=meta,
        )
    ]

    summary = ConsolidationEngine._analyze_emotional_patterns(episodes)
    assert "joy" in summary
    # Default arousal is 0.5
    assert "0.50" in summary


# ============================================================
# 3. get_prompt_context() idempotency
# ============================================================


def test_double_prompt_context_call_doesnt_crash() -> None:
    """Calling get_prompt_context() twice in a row should not crash."""
    mgr = AffectiveStateManager()
    result = EnrichmentResult(
        importance=0.5,
        valence="positive",
        intent="statement",
        emotion="happy",
        confidence=0.8,
        arousal=0.5,
        is_feedback=False,
        feedback_strength=0.0,
        reasoning="test",
        self_emotion="joy",
        self_arousal=0.7,
    )
    mgr.update_from_enrichment(result)

    ctx1 = mgr.get_prompt_context()
    ctx2 = mgr.get_prompt_context()

    # Both should be valid strings
    assert isinstance(ctx1, str)
    assert isinstance(ctx2, str)
    # Second call has one more turn of decay
    assert mgr.get_state().turns_since_update == 2


# ============================================================
# 4. Counter-regulation with unknown self_emotion
# ============================================================


def test_regulation_with_emotion_not_in_atlas() -> None:
    """If LLM returns an emotion not in atlas, regulation should fall back to 'calm'."""
    mgr = AffectiveStateManager()
    # Manually set a state with unknown emotion
    mgr._state.self_emotion = "unknown_emotion_xyz"
    mgr._state.user_arousal = 0.9
    mgr._state.user_valence = -0.5
    mgr._is_at_baseline = False

    mgr._apply_counter_regulation()

    assert mgr._state.regulation_active
    # Should fall back to "calm" since unknown emotion has no complement
    assert mgr._state.regulation_target == "calm"


# ============================================================
# 5. Boundary float values
# ============================================================


@pytest.mark.parametrize(
    "arousal,self_arousal",
    [
        (0.0, 0.0),  # minimum
        (1.0, 1.0),  # maximum
        (0.5, 0.5),  # exact middle
        (0.001, 0.999),  # near boundaries
    ],
)
def test_boundary_arousal_values(arousal: float, self_arousal: float) -> None:
    """Extreme arousal values should not crash or produce NaN."""
    result = EnrichmentResult(
        importance=0.5,
        valence="neutral",
        intent="statement",
        emotion="neutral",
        confidence=0.5,
        arousal=arousal,
        is_feedback=False,
        feedback_strength=0.0,
        reasoning="test",
        self_emotion="curiosity",
        self_arousal=self_arousal,
    )

    mgr = AffectiveStateManager()
    mgr.update_from_enrichment(result)

    state = mgr.get_state()
    assert 0.0 <= state.self_arousal <= 1.0
    assert 0.0 <= state.user_arousal <= 1.0
    # No NaN
    assert state.self_arousal == state.self_arousal  # NaN != NaN
    assert state.self_valence == state.self_valence


def test_zero_arousal_regulation() -> None:
    """User arousal=0 should not trigger regulation."""
    mgr = AffectiveStateManager()
    result = EnrichmentResult(
        importance=0.5,
        valence="negative",
        intent="emotional",
        emotion="sad",
        confidence=0.8,
        arousal=0.0,
        is_feedback=False,
        feedback_strength=0.0,
        reasoning="test",
        self_emotion="warmth",
        self_arousal=0.4,
    )
    mgr.update_from_enrichment(result)
    assert not mgr.get_state().regulation_active


# ============================================================
# 6. Atlas integrity checks
# ============================================================


def test_complement_map_covers_all_atlas_entries() -> None:
    """Every emotion in EMOTION_ATLAS must have a complement mapping."""
    for name in EMOTION_ATLAS:
        assert name in _COMPLEMENT_MAP, f"{name} missing from complement map"


def test_complement_map_targets_are_valid_atlas_entries() -> None:
    """Every complement target must itself be in the atlas."""
    for source, target in _COMPLEMENT_MAP.items():
        assert target in EMOTION_ATLAS, (
            f"Complement {source}→{target}: target not in atlas"
        )


def test_all_decay_targets_are_curiosity() -> None:
    """All emotions should decay to curiosity (Zhvusha's baseline)."""
    for name in EMOTION_ATLAS:
        assert get_decay_target(name) == "curiosity"


def test_enricher_self_emotion_literal_matches_atlas() -> None:
    """self_emotion Literal in EnrichmentResult must match atlas keys exactly."""
    import typing

    er_fields = EnrichmentResult.model_fields
    literal_type = er_fields["self_emotion"].annotation
    literal_values = set(typing.get_args(literal_type))
    atlas_keys = set(EMOTION_ATLAS.keys())

    assert literal_values == atlas_keys, (
        f"Mismatch: "
        f"in Literal but not atlas: {literal_values - atlas_keys}, "
        f"in atlas but not Literal: {atlas_keys - literal_values}"
    )


# ============================================================
# 7. Multiple rapid updates
# ============================================================


def test_rapid_updates_dont_accumulate_regulation() -> None:
    """Multiple updates in a row should use latest state, not accumulate."""
    mgr = AffectiveStateManager()

    # First: angry user → regulation
    r1 = EnrichmentResult(
        importance=0.7,
        valence="negative",
        intent="emotional",
        emotion="angry",
        confidence=0.9,
        arousal=0.9,
        is_feedback=False,
        feedback_strength=0.0,
        reasoning="test",
        self_emotion="warmth",
        self_arousal=0.8,
    )
    mgr.update_from_enrichment(r1)
    assert mgr.get_state().regulation_active

    # Second: calm user → no regulation
    r2 = EnrichmentResult(
        importance=0.5,
        valence="positive",
        intent="statement",
        emotion="happy",
        confidence=0.8,
        arousal=0.3,
        is_feedback=False,
        feedback_strength=0.0,
        reasoning="test",
        self_emotion="joy",
        self_arousal=0.6,
    )
    mgr.update_from_enrichment(r2)
    assert not mgr.get_state().regulation_active
    assert mgr.get_state().self_emotion == "joy"


# ============================================================
# 8. Decay resets regulation
# ============================================================


def test_decay_to_baseline_clears_regulation() -> None:
    """When decay snaps to baseline, regulation should be cleared."""
    mgr = AffectiveStateManager()
    result = EnrichmentResult(
        importance=0.7,
        valence="negative",
        intent="emotional",
        emotion="angry",
        confidence=0.9,
        arousal=0.9,
        is_feedback=False,
        feedback_strength=0.0,
        reasoning="test",
        self_emotion="warmth",
        self_arousal=0.4,
    )
    mgr.update_from_enrichment(result)
    assert mgr.get_state().regulation_active

    # Force decay to baseline
    mgr._state.turns_since_update = 30
    mgr.decay_if_stale()

    assert mgr._is_at_baseline
    assert not mgr.get_state().regulation_active
    assert mgr.get_state().regulation_target == ""


# ============================================================
# 9. Emotional log snapshot
# ============================================================


def test_emotional_log_no_duplicate_dates(tmp_path: object) -> None:
    """Multiple snapshots on the same day should both appear."""
    from src.memory.consolidation import ConsolidationEngine

    engine = ConsolidationEngine.__new__(ConsolidationEngine)
    engine.personality_dir = tmp_path  # type: ignore[assignment]

    engine._write_emotional_snapshot("curiosity×5, arousal=0.6")
    engine._write_emotional_snapshot("frustration×3, arousal=0.8")

    log_path = tmp_path / "emotional_log.md"  # type: ignore[operator]
    content = log_path.read_text()
    # Both lines should be present (same date is fine — different summaries)
    assert content.count("- [") == 2


def test_emotional_log_respects_30_entry_cap(tmp_path: object) -> None:
    """Log should keep only the last 30 entries."""
    from src.memory.consolidation import ConsolidationEngine

    engine = ConsolidationEngine.__new__(ConsolidationEngine)
    engine.personality_dir = tmp_path  # type: ignore[assignment]

    for i in range(35):
        engine._write_emotional_snapshot(f"entry_{i}")

    log_path = tmp_path / "emotional_log.md"  # type: ignore[operator]
    content = log_path.read_text()
    entry_count = content.count("- [")
    assert entry_count == 30
    # Should keep the latest entries
    assert "entry_34" in content
    assert "entry_0" not in content


# ============================================================
# 10. Circumplex coordinates sanity
# ============================================================


def test_negative_emotions_have_negative_valence() -> None:
    """Emotions in anger/sadness/fear/confusion clusters should have negative valence."""
    negative_clusters = {"anger", "sadness", "fear", "confusion"}
    for name, concept in EMOTION_ATLAS.items():
        if concept.cluster in negative_clusters:
            assert concept.valence < 0, (
                f"{name} (cluster={concept.cluster}) has positive valence={concept.valence}"
            )


def test_positive_emotions_have_positive_valence() -> None:
    """Emotions in joy/calm/curiosity/tender/pride should have positive valence."""
    positive_clusters = {"joy", "calm", "curiosity", "tender", "pride"}
    for name, concept in EMOTION_ATLAS.items():
        if concept.cluster in positive_clusters:
            assert concept.valence > 0, (
                f"{name} (cluster={concept.cluster}) has negative valence={concept.valence}"
            )


def test_high_arousal_emotions() -> None:
    """Anger and fear clusters should have high arousal (>=0.6)."""
    high_arousal_clusters = {"anger", "fear"}
    for name, concept in EMOTION_ATLAS.items():
        if concept.cluster in high_arousal_clusters:
            assert concept.arousal >= 0.6, (
                f"{name} (cluster={concept.cluster}) has low arousal={concept.arousal}"
            )


def test_calm_cluster_has_low_arousal() -> None:
    """Calm cluster emotions should have low arousal (<=0.3)."""
    for name, concept in EMOTION_ATLAS.items():
        if concept.cluster == "calm":
            assert concept.arousal <= 0.3, (
                f"{name} has high arousal={concept.arousal} in calm cluster"
            )

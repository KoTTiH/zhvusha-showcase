"""Contract tests for the Personality capability module (v4).

Verify that:
1. Concrete implementations satisfy their declared protocols
   (`isinstance(impl, Protocol)` because protocols are
   `@runtime_checkable`).
2. Public API is reachable through the `src.personality` package
   façade only (no internal-module imports needed by clients).
3. The frozen domain types behave as immutable (writes raise).
4. The mode-guard policy (`should_update_personality`) returns
   the documented values for all three modes.

These tests use the public protocol surface only — they do NOT
import from internal modules (`affective_state`, `evolution`,
`homeostasis`, `guard`, `constants`, `emotion_atlas`).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from src.personality import (
    EMOTION_ATLAS,
    PERSONALITY_COMPACT,
    AffectiveSnapshot,
    AffectiveStateManager,
    AffectiveStateProtocol,
    EmotionConcept,
    HomeostasisCheck,
    HomeostasisCorrection,
    HomeostasisProtocol,
    PersonalityEvolution,
    PersonalityEvolutionProtocol,
    get_affective_state_manager,
    get_complement,
    should_update_personality,
)


@pytest.mark.contract
class TestProtocolConformance:
    """Concrete implementations satisfy their declared protocols."""

    def test_homeostasis_check_implements_protocol(self) -> None:
        instance = HomeostasisCheck()
        assert isinstance(instance, HomeostasisProtocol)

    def test_personality_evolution_implements_protocol(self, tmp_path: Path) -> None:
        instance = PersonalityEvolution(tmp_path)
        assert isinstance(instance, PersonalityEvolutionProtocol)

    def test_affective_state_manager_implements_protocol(self) -> None:
        instance = AffectiveStateManager()
        assert isinstance(instance, AffectiveStateProtocol)


@pytest.mark.contract
class TestFrozenDomainTypes:
    """HomeostasisCorrection and AffectiveSnapshot are frozen."""

    def test_homeostasis_correction_is_frozen(self) -> None:
        correction = HomeostasisCorrection(
            gene="initiative",
            direction="too_low",
            evidence="0 предложений",
            suggestion="попробовать предложить что-то завтра",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            correction.gene = "honesty"  # type: ignore[misc]

    def test_affective_snapshot_is_frozen(self) -> None:
        snapshot = get_affective_state_manager().get_state()
        assert isinstance(snapshot, AffectiveSnapshot)
        with pytest.raises(dataclasses.FrozenInstanceError):
            snapshot.self_arousal = 0.99  # type: ignore[misc]


@pytest.mark.contract
class TestModeGuard:
    """should_update_personality enforces the mode policy."""

    def test_personal_mode_allows_personality_update(self) -> None:
        assert should_update_personality("personal") is True

    def test_assistant_mode_blocks_personality_update(self) -> None:
        assert should_update_personality("assistant") is False

    def test_social_mode_blocks_personality_update(self) -> None:
        assert should_update_personality("social") is False


@pytest.mark.contract
class TestEmotionAtlasReexport:
    """Emotion atlas accessible through the package façade."""

    def test_atlas_contains_baseline_emotion(self) -> None:
        assert "curiosity" in EMOTION_ATLAS
        concept = EMOTION_ATLAS["curiosity"]
        assert isinstance(concept, EmotionConcept)
        assert concept.valence == pytest.approx(0.6)

    def test_complement_helper_resolves(self) -> None:
        assert get_complement("frustration") == "calm"


@pytest.mark.contract
class TestConstantsReexport:
    """PERSONALITY_COMPACT constant accessible through the façade."""

    def test_personality_compact_is_non_empty_string(self) -> None:
        assert isinstance(PERSONALITY_COMPACT, str)
        assert "Жвуша" in PERSONALITY_COMPACT


@pytest.mark.contract
class TestAffectiveSnapshotInvariants:
    """AffectiveSnapshot mirrors the manager's internal state."""

    def test_snapshot_starts_at_baseline(self) -> None:
        manager = AffectiveStateManager()
        snapshot = manager.get_state()
        assert snapshot.self_emotion == "curiosity"
        assert snapshot.regulation_active is False
        assert snapshot.turns_since_update == 0

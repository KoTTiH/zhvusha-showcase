"""Tests for the emotion atlas — static registry of 30 emotion concepts."""

from __future__ import annotations

import pytest
from src.personality.emotion_atlas import (
    EMOTION_ATLAS,
    get_cluster_members,
    get_complement,
    get_decay_target,
)

# ============================================================
# Atlas structure
# ============================================================


def test_atlas_has_30_emotions() -> None:
    assert len(EMOTION_ATLAS) == 30


def test_all_emotions_have_valid_valence_range() -> None:
    for name, concept in EMOTION_ATLAS.items():
        assert -1.0 <= concept.valence <= 1.0, f"{name}: valence={concept.valence}"


def test_all_emotions_have_valid_arousal_range() -> None:
    for name, concept in EMOTION_ATLAS.items():
        assert 0.0 <= concept.arousal <= 1.0, f"{name}: arousal={concept.arousal}"


def test_all_emotions_have_russian_name() -> None:
    for name, concept in EMOTION_ATLAS.items():
        assert concept.name_ru, f"{name}: missing name_ru"


def test_atlas_keys_match_name_field() -> None:
    for key, concept in EMOTION_ATLAS.items():
        assert key == concept.name, f"key={key} != concept.name={concept.name}"


# ============================================================
# Clusters
# ============================================================

_EXPECTED_CLUSTERS = {
    "joy",
    "sadness",
    "anger",
    "fear",
    "calm",
    "curiosity",
    "tender",
    "brooding",
    "pride",
    "confusion",
}


def test_clusters_cover_all_emotions() -> None:
    """Every emotion must belong to a known cluster."""
    actual_clusters = {c.cluster for c in EMOTION_ATLAS.values()}
    assert actual_clusters == _EXPECTED_CLUSTERS


def test_cluster_members_returns_siblings() -> None:
    members = get_cluster_members("curiosity")
    assert "curiosity" in members
    assert "wonder" in members
    assert "fascination" in members
    # Should NOT include emotions from other clusters
    assert "joy" not in members


def test_cluster_members_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_cluster_members("nonexistent_emotion")


# ============================================================
# Complement mapping
# ============================================================


def test_complement_mapping_returns_valid_emotion() -> None:
    comp = get_complement("anxiety")
    assert comp in EMOTION_ATLAS


def test_complement_shifts_arousal_down_or_valence_up() -> None:
    """Complement should generally have lower arousal OR higher valence."""
    for name, concept in EMOTION_ATLAS.items():
        comp_name = get_complement(name)
        comp = EMOTION_ATLAS[comp_name]
        # At least one of: lower arousal, higher valence, or same (for calm emotions)
        assert (
            comp.arousal <= concept.arousal + 0.1
            or comp.valence >= concept.valence - 0.1
        ), f"{name}→{comp_name}: complement not calming"


def test_complement_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_complement("nonexistent_emotion")


# ============================================================
# Decay target
# ============================================================


def test_decay_target_defaults_to_curiosity() -> None:
    """Most emotions should decay to curiosity (Zhvusha's baseline)."""
    curiosity_count = sum(
        1 for name in EMOTION_ATLAS if get_decay_target(name) == "curiosity"
    )
    # At least 80% should decay to curiosity
    assert curiosity_count >= 24, f"Only {curiosity_count}/30 decay to curiosity"


def test_decay_target_returns_valid_emotion() -> None:
    for name in EMOTION_ATLAS:
        target = get_decay_target(name)
        assert target in EMOTION_ATLAS, f"{name} decays to unknown {target}"


def test_curiosity_decays_to_itself() -> None:
    assert get_decay_target("curiosity") == "curiosity"


# ============================================================
# EmotionConcept immutability
# ============================================================


def test_emotion_concept_is_frozen() -> None:
    concept = EMOTION_ATLAS["curiosity"]
    with pytest.raises(AttributeError):
        concept.valence = 0.0  # type: ignore[misc]

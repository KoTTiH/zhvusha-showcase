"""Tests for emotional_stability homeostasis gene."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

from src.personality.homeostasis import HomeostasisCheck


def _make_episode(
    role: str = "assistant",
    content: str = "test",
    user_id: int = 12345,
    valence: str = "neutral",
    metadata_json: str | None = None,
) -> object:
    return SimpleNamespace(
        id=1,
        role=role,
        content=content,
        user_id=user_id,
        valence=valence,
        importance=0.5,
        timestamp=datetime.now(tz=UTC),
        metadata_json=metadata_json,
    )


def _meta_with_arousal(self_arousal: float) -> str:
    return json.dumps({"enrichment": {"self_arousal": self_arousal}})


async def test_detects_negative_streak(tmp_path: object) -> None:
    """5+ consecutive negative-valence assistant responses → correction."""
    check = HomeostasisCheck()
    episodes = [_make_episode(role="assistant", valence="negative") for _ in range(6)]
    corrections = await check.check(tmp_path / "genes.md", episodes, 12345)  # type: ignore[operator]
    assert any(
        c.gene == "emotional_stability" and c.direction == "too_low"
        for c in corrections
    )


async def test_detects_arousal_spikes(tmp_path: object) -> None:
    """3+ high-arousal episodes → correction."""
    check = HomeostasisCheck()
    episodes = [
        _make_episode(
            role="assistant",
            metadata_json=_meta_with_arousal(0.9),
        )
        for _ in range(4)
    ]
    corrections = await check.check(tmp_path / "genes.md", episodes, 12345)  # type: ignore[operator]
    assert any(
        c.gene == "emotional_stability" and c.direction == "too_high"
        for c in corrections
    )


async def test_no_correction_on_healthy_state(tmp_path: object) -> None:
    """Balanced episodes → no emotional_stability corrections."""
    check = HomeostasisCheck()
    episodes = [
        _make_episode(role="assistant", valence="positive"),
        _make_episode(role="assistant", valence="neutral"),
        _make_episode(role="assistant", valence="positive"),
    ]
    corrections = await check.check(tmp_path / "genes.md", episodes, 12345)  # type: ignore[operator]
    emotional_corrections = [c for c in corrections if c.gene == "emotional_stability"]
    assert len(emotional_corrections) == 0


async def test_reads_arousal_from_metadata_json(tmp_path: object) -> None:
    """Arousal value is correctly read from metadata_json enrichment."""
    check = HomeostasisCheck()
    # 2 low-arousal + 2 high-arousal = exactly 2, not enough for correction
    episodes = [
        _make_episode(
            role="assistant",
            metadata_json=_meta_with_arousal(0.3),
        ),
        _make_episode(
            role="assistant",
            metadata_json=_meta_with_arousal(0.3),
        ),
        _make_episode(
            role="assistant",
            metadata_json=_meta_with_arousal(0.85),
        ),
        _make_episode(
            role="assistant",
            metadata_json=_meta_with_arousal(0.85),
        ),
    ]
    corrections = await check.check(tmp_path / "genes.md", episodes, 12345)  # type: ignore[operator]
    arousal_corrections = [
        c
        for c in corrections
        if c.gene == "emotional_stability" and c.direction == "too_high"
    ]
    assert len(arousal_corrections) == 0  # only 2 spikes, threshold is >3


async def test_graceful_without_enrichment_metadata(tmp_path: object) -> None:
    """Episodes without metadata_json don't crash the check."""
    check = HomeostasisCheck()
    episodes = [_make_episode(role="assistant", metadata_json=None) for _ in range(5)]
    corrections = await check.check(tmp_path / "genes.md", episodes, 12345)  # type: ignore[operator]
    # Should not crash, and no arousal correction
    arousal_corrections = [
        c
        for c in corrections
        if c.gene == "emotional_stability" and c.direction == "too_high"
    ]
    assert len(arousal_corrections) == 0

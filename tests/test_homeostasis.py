"""Tests for HomeostasisCheck."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from src.personality.homeostasis import HomeostasisCheck


def _make_episode(role="assistant", content="test", user_id=12345, **kwargs):
    defaults = {
        "id": 1,
        "role": role,
        "content": content,
        "user_id": user_id,
        "valence": "neutral",
        "importance": 0.5,
        "timestamp": datetime.now(tz=UTC),
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


async def test_detects_low_initiative(tmp_path):
    check = HomeostasisCheck()
    # Many episodes but no proposals
    episodes = [
        _make_episode(role="assistant", content="ответ на вопрос") for _ in range(25)
    ]
    corrections = await check.check(tmp_path / "genes.md", episodes, 12345)
    assert any(c.gene == "initiative" and c.direction == "too_low" for c in corrections)


async def test_detects_honesty_drift(tmp_path):
    check = HomeostasisCheck()
    # Many sycophantic responses
    episodes = [
        _make_episode(role="assistant", content="конечно, ты прав! безусловно!")
        for _ in range(10)
    ]
    corrections = await check.check(tmp_path / "genes.md", episodes, 12345)
    assert any(c.gene == "honesty" for c in corrections)


async def test_healthy_state_returns_empty(tmp_path):
    check = HomeostasisCheck()
    # Balanced episodes with proposals
    episodes = [
        _make_episode(
            role="assistant", content="предлагаю попробовать новый подход к задаче"
        ),
        _make_episode(
            role="assistant", content="вот мой анализ ситуации с подробностями"
        ),
        _make_episode(role="user", content="хорошо", valence="positive"),
    ]
    corrections = await check.check(tmp_path / "genes.md", episodes, 12345)
    # With so few episodes, no corrections should trigger
    assert len(corrections) == 0


async def test_suggestion_is_actionable(tmp_path):
    check = HomeostasisCheck()
    episodes = [_make_episode(role="assistant", content="да") for _ in range(25)]
    corrections = await check.check(tmp_path / "genes.md", episodes, 12345)
    for c in corrections:
        assert len(c.suggestion) > 10  # Not empty, actionable

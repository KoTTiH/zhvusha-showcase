"""Tests for MetacognitionTracker."""

from __future__ import annotations

from src.core.decision import MetacognitionTracker


async def test_default_threshold():
    tracker = MetacognitionTracker()
    threshold = await tracker.get_system1_threshold("chat")
    assert threshold == 0.7


async def test_threshold_lowers_after_good_outcomes():
    tracker = MetacognitionTracker()
    for _ in range(3):
        await tracker.record_outcome("chat", "system1", was_correct=True)
    threshold = await tracker.get_system1_threshold("chat")
    assert threshold < 0.7  # Should be 0.6


async def test_threshold_raises_after_bad_outcomes():
    tracker = MetacognitionTracker()
    for _ in range(2):
        await tracker.record_outcome("chat", "system1", was_correct=False)
    threshold = await tracker.get_system1_threshold("chat")
    assert threshold > 0.7  # Should be 0.8


async def test_strategy_suggestion_after_failures():
    tracker = MetacognitionTracker()
    for _ in range(3):
        await tracker.record_outcome("kwork", "system1", was_correct=False)
    suggestion = await tracker.should_suggest_strategy_change("kwork")
    assert suggestion is not None
    assert "kwork" in suggestion


async def test_no_suggestion_when_healthy():
    tracker = MetacognitionTracker()
    suggestion = await tracker.should_suggest_strategy_change("chat")
    assert suggestion is None

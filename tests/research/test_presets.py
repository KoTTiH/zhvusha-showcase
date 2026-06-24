"""Tests for research presets."""

from src.research.presets import PRESETS


def test_bug_investigation_preset_exists_with_correct_config() -> None:
    preset = PRESETS["bug_investigation"]
    assert preset.use_kb is True
    assert preset.use_code_search is True
    assert preset.use_web is False
    assert preset.budget_seconds == 60
    assert preset.max_sources == 5

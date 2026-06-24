"""Top-level contract test for the spec tier classifier public API."""

from __future__ import annotations

import pytest
from src.skills.ideation_to_spec import tier_classifier

pytestmark = pytest.mark.contract


def test_classify_spec_tier_exists_and_old_name_removed() -> None:
    classify_spec_tier = getattr(tier_classifier, "classify_spec_tier", None)

    assert classify_spec_tier is not None
    assert (
        classify_spec_tier(
            whitelist_paths=["src/skills/weather/skill.py"],
            goal="add weather",
            modifies_capabilities=[],
        )
        == 1
    )
    assert getattr(tier_classifier, "classify_tier", None) is None

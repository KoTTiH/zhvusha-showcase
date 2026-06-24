"""Contract tests for the migrated chat_response v4 skill."""

from __future__ import annotations

import pytest
from src.skills.base import BaseSkill, InlineSkill, SideEffect
from src.skills.chat_response.skill import ChatResponseSkill
from src.skills.manifest import (
    load_manifest_for_skill_class,
    validate_manifest_matches_class,
)

pytestmark = pytest.mark.contract


class TestContract:
    def test_inherits_inline_skill(self) -> None:
        assert issubclass(ChatResponseSkill, InlineSkill)
        assert issubclass(ChatResponseSkill, BaseSkill)

    def test_skill_type_is_inline(self) -> None:
        assert ChatResponseSkill.skill_type == "inline"

    def test_manifest_loads(self) -> None:
        manifest = load_manifest_for_skill_class(ChatResponseSkill)
        assert manifest.name == "chat_response"
        assert manifest.type == "inline"
        assert manifest.llm_tier == "analyst"
        assert manifest.version == "1.0.0"

    def test_manifest_matches_class(self) -> None:
        manifest = load_manifest_for_skill_class(ChatResponseSkill)
        validate_manifest_matches_class(manifest, ChatResponseSkill)

    def test_triggers_empty_catchall(self) -> None:
        """chat_response is score-based, no deterministic trigger list."""
        assert ChatResponseSkill.triggers == []

    def test_side_effects_cover_conversation(self) -> None:
        assert SideEffect.CALLS_LLM in ChatResponseSkill.side_effects
        assert SideEffect.READS_FROM_KB in ChatResponseSkill.side_effects
        assert SideEffect.SENDS_TELEGRAM_MESSAGE in ChatResponseSkill.side_effects
        assert SideEffect.MODIFIES_MEMORY in ChatResponseSkill.side_effects

    def test_mode_tags_all_modes(self) -> None:
        assert set(ChatResponseSkill.mode_tags) == {"personal", "assistant", "social"}

    def test_modifies_no_core_capabilities(self) -> None:
        assert ChatResponseSkill.modifies == []

    def test_approval_policy_auto(self) -> None:
        assert ChatResponseSkill.approval_policy == "auto"

    def test_cost_estimate_medium(self) -> None:
        assert ChatResponseSkill.cost_estimate == "medium"

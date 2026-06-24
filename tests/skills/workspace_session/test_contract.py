"""Contract tests for the migrated workspace_session v4 skill."""

from __future__ import annotations

import pytest
from src.skills.base import BaseSkill, InlineSkill, SideEffect
from src.skills.manifest import (
    load_manifest_for_skill_class,
    validate_manifest_matches_class,
)
from src.skills.workspace_session.skill import WorkspaceSessionSkill

pytestmark = pytest.mark.contract


class TestContract:
    def test_inherits_inline_skill(self) -> None:
        assert issubclass(WorkspaceSessionSkill, InlineSkill)
        assert issubclass(WorkspaceSessionSkill, BaseSkill)

    def test_skill_type_is_inline(self) -> None:
        assert WorkspaceSessionSkill.skill_type == "inline"

    def test_manifest_loads(self) -> None:
        manifest = load_manifest_for_skill_class(WorkspaceSessionSkill)
        assert manifest.name == "workspace_session"
        assert manifest.type == "inline"
        assert manifest.llm_tier == "strategist"
        assert manifest.version == "1.0.0"

    def test_manifest_matches_class(self) -> None:
        manifest = load_manifest_for_skill_class(WorkspaceSessionSkill)
        validate_manifest_matches_class(manifest, WorkspaceSessionSkill)

    def test_triggers_contain_morning(self) -> None:
        assert "/morning" in WorkspaceSessionSkill.triggers

    def test_session_prompt_preserves_no_downgrade_self_growth_principle(self) -> None:
        from src.skills.workspace_session.skill import SESSION_PROMPT

        assert "Принцип развития Жвуши" in SESSION_PROMPT
        assert "обогащение" in SESSION_PROMPT
        assert "без явного Никитиного разрешения" in SESSION_PROMPT

    def test_cost_estimate_is_high(self) -> None:
        assert WorkspaceSessionSkill.cost_estimate == "high"

    def test_side_effects_cover_session(self) -> None:
        assert (
            SideEffect.CALLS_LLM_TIER_STRATEGIST in WorkspaceSessionSkill.side_effects
        )
        assert SideEffect.SPAWNS_SUBPROCESS in WorkspaceSessionSkill.side_effects
        assert SideEffect.WRITES_WORKSPACE in WorkspaceSessionSkill.side_effects
        assert SideEffect.READS_WORKSPACE in WorkspaceSessionSkill.side_effects
        assert SideEffect.SENDS_TELEGRAM_MESSAGE in WorkspaceSessionSkill.side_effects

    def test_modifies_no_core_capabilities(self) -> None:
        assert WorkspaceSessionSkill.modifies == []

    def test_mode_tags_personal_only(self) -> None:
        assert WorkspaceSessionSkill.mode_tags == ["personal"]

    def test_approval_policy_auto(self) -> None:
        assert WorkspaceSessionSkill.approval_policy == "auto"

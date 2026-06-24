"""Contract tests for the migrated delegate v4 skill."""

from __future__ import annotations

import pytest
from src.skills.base import BaseSkill, DelegatedSkill, SideEffect
from src.skills.delegate.skill import DelegateSkill
from src.skills.manifest import (
    load_manifest_for_skill_class,
    validate_manifest_matches_class,
)

pytestmark = pytest.mark.contract


class TestContract:
    def test_inherits_delegated_skill(self) -> None:
        assert issubclass(DelegateSkill, DelegatedSkill)
        assert issubclass(DelegateSkill, BaseSkill)

    def test_skill_type_is_delegated(self) -> None:
        assert DelegateSkill.skill_type == "delegated"

    def test_manifest_loads(self) -> None:
        manifest = load_manifest_for_skill_class(DelegateSkill)
        assert manifest.name == "delegate"
        assert manifest.type == "delegated"
        assert manifest.llm_tier == "strategist"
        assert manifest.version == "1.0.0"

    def test_manifest_matches_class(self) -> None:
        manifest = load_manifest_for_skill_class(DelegateSkill)
        validate_manifest_matches_class(manifest, DelegateSkill)

    def test_executor_is_codex_cli(self) -> None:
        assert DelegateSkill.executor == "codex_cli"

    def test_approval_policy_is_required(self) -> None:
        assert DelegateSkill.approval_policy == "required"

    def test_side_effects_cover_delegation(self) -> None:
        assert SideEffect.DELEGATES_TO_CODE_AGENT in DelegateSkill.side_effects
        assert SideEffect.CALLS_LLM_TIER_STRATEGIST in DelegateSkill.side_effects
        assert SideEffect.NETWORK_IO_EXTERNAL in DelegateSkill.side_effects
        assert SideEffect.SPAWNS_SUBPROCESS in DelegateSkill.side_effects

    def test_triggers_contain_delegate_prefix(self) -> None:
        assert "/delegate" in DelegateSkill.triggers

    def test_modifies_no_core_capabilities(self) -> None:
        assert DelegateSkill.modifies == []

    def test_mode_tags_personal_only(self) -> None:
        assert DelegateSkill.mode_tags == ["personal"]

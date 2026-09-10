"""Contract tests for SpecCommandSkill (Phase 11).

Verifies the v4 ``InlineSkill`` contract and its ``skill.yaml`` manifest:
inheritance, skill_type, manifest loads + matches class, side_effects cover
declared behaviour, approval_policy is auto, mode_tags are personal-only,
and the recursive self-improvement gate (``modifies``) is empty (no Tier 3
upgrade).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract


class TestSpecCommandContract:
    def test_inherits_inline_skill(self) -> None:
        from src.skills.base import BaseSkill, InlineSkill
        from src.skills.spec_command.skill import SpecCommandSkill

        assert issubclass(SpecCommandSkill, InlineSkill)
        assert issubclass(SpecCommandSkill, BaseSkill)

    def test_skill_type_is_inline(self) -> None:
        from src.skills.spec_command.skill import SpecCommandSkill

        assert SpecCommandSkill.skill_type == "inline"

    def test_manifest_loads(self) -> None:
        from src.skills.manifest import load_manifest_for_skill_class
        from src.skills.spec_command.skill import SpecCommandSkill

        manifest = load_manifest_for_skill_class(SpecCommandSkill)
        assert manifest.name == "spec_command"
        assert manifest.type == "inline"
        assert manifest.llm_tier == "worker"

    def test_manifest_matches_class(self) -> None:
        from src.skills.manifest import (
            load_manifest_for_skill_class,
            validate_manifest_matches_class,
        )
        from src.skills.spec_command.skill import SpecCommandSkill

        manifest = load_manifest_for_skill_class(SpecCommandSkill)
        validate_manifest_matches_class(manifest, SpecCommandSkill)

    def test_triggers_include_spec_prefix(self) -> None:
        from src.skills.spec_command.skill import SpecCommandSkill

        assert any(t.startswith("/spec") for t in SpecCommandSkill.triggers)

    def test_approval_policy_is_auto(self) -> None:
        from src.skills.spec_command.skill import SpecCommandSkill

        assert SpecCommandSkill.approval_policy == "auto"

    def test_mode_tags_personal_only(self) -> None:
        from src.skills.spec_command.skill import SpecCommandSkill

        assert SpecCommandSkill.mode_tags == ["personal"]

    def test_modifies_no_core_capabilities(self) -> None:
        from src.skills.spec_command.skill import SpecCommandSkill

        assert SpecCommandSkill.modifies == []

    def test_side_effects_cover_filesystem_and_telegram(self) -> None:
        from src.skills.base import SideEffect
        from src.skills.spec_command.skill import SpecCommandSkill

        side_effects = SpecCommandSkill.side_effects
        assert SideEffect.READS_FILESYSTEM in side_effects
        assert SideEffect.WRITES_FILESYSTEM in side_effects
        assert SideEffect.SENDS_TELEGRAM_MESSAGE in side_effects

    def test_cost_estimate_is_low(self) -> None:
        from src.skills.spec_command.skill import SpecCommandSkill

        assert SpecCommandSkill.cost_estimate == "low"

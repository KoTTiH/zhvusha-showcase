"""Tier classifier — table-driven contract tests.

Mirrors ``scripts/check_tier3_protection.sh`` and the Tier 3 list in
``src/skills/spec_command/parser.py``. Phase 12.

The classifier deliberately does NOT call an LLM. It is a deterministic
algorithm so the same spec always classifies the same way regardless of
which model is used to draft it.
"""

from __future__ import annotations

import pytest
from src.skills.base import CoreCapability

pytestmark = pytest.mark.contract


class TestTierClassifierPathBased:
    """Path-based rules — Tier 3 paths win regardless of keywords."""

    @pytest.mark.parametrize(
        "path",
        [
            "src/skills/base.py",
            "src/skills/__init__.py",
            "src/skills/registry.py",
            "src/personality/decision.py",
            ".importlinter",
            "AGENTS.md",
            "CLAUDE.md",
            "scripts/check_tier3_protection.sh",
            "src/safety/guard.py",
            "src/safety/__init__.py",
            "src/llm/protocols.py",
            "src/memory/protocols.py",
            "src/skills/protocols.py",  # any /protocols.py is Tier 3
        ],
    )
    def test_tier3_paths_classify_as_tier3(self, path: str) -> None:
        from src.skills.ideation_to_spec.tier_classifier import classify_spec_tier

        assert (
            classify_spec_tier(
                whitelist_paths=[path], goal="trivial change", modifies_capabilities=[]
            )
            == 3
        )

    @pytest.mark.parametrize(
        "path",
        [
            "src/skills/chat_response/skill.py",  # existing skill module
            "src/memory/consolidation.py",  # existing capability internal
            "src/llm/router.py",  # router itself
            "src/personality/evolution.py",
        ],
    )
    def test_existing_capability_paths_classify_as_tier2(self, path: str) -> None:
        from src.skills.ideation_to_spec.tier_classifier import classify_spec_tier

        assert (
            classify_spec_tier(
                whitelist_paths=[path], goal="fix something", modifies_capabilities=[]
            )
            == 2
        )

    @pytest.mark.parametrize(
        "path",
        [
            "src/skills/weather/skill.py",  # entirely new skill subpackage
            "src/skills/weather/__init__.py",
            "src/skills/weather/skill.yaml",
            "tests/skills/weather/test_contract.py",
            "src/research/extra_preset.py",  # new file in leaf module
        ],
    )
    def test_new_files_classify_as_tier1(self, path: str) -> None:
        from src.skills.ideation_to_spec.tier_classifier import classify_spec_tier

        assert (
            classify_spec_tier(
                whitelist_paths=[path],
                goal="add a new feature",
                modifies_capabilities=[],
            )
            == 1
        )


class TestTierClassifierCapabilityModifies:
    """Any modifies_capabilities entry forces Tier 3."""

    def test_modifies_skills_dispatcher_forces_tier3(self) -> None:
        from src.skills.ideation_to_spec.tier_classifier import classify_spec_tier

        assert (
            classify_spec_tier(
                whitelist_paths=["src/skills/weather/skill.py"],
                goal="add weather",
                modifies_capabilities=[CoreCapability.SKILLS_DISPATCHER],
            )
            == 3
        )

    def test_modifies_safety_module_forces_tier3(self) -> None:
        from src.skills.ideation_to_spec.tier_classifier import classify_spec_tier

        assert (
            classify_spec_tier(
                whitelist_paths=["tests/skills/weather/test_contract.py"],
                goal="cosmetic test change",
                modifies_capabilities=[CoreCapability.SAFETY_MODULE],
            )
            == 3
        )


class TestTierClassifierKeywordEscalation:
    """Refactor keywords on existing capability → Tier 2 even if path looks Tier 1."""

    def test_refactor_keyword_with_existing_skill_path_is_tier2(self) -> None:
        from src.skills.ideation_to_spec.tier_classifier import classify_spec_tier

        # ``src/skills/chat_response`` already exists → Tier 2.
        assert (
            classify_spec_tier(
                whitelist_paths=["src/skills/chat_response/skill.py"],
                goal="refactor chat response handling",
                modifies_capabilities=[],
            )
            == 2
        )

    def test_refactor_keyword_only_in_new_skill_remains_tier1(self) -> None:
        from src.skills.ideation_to_spec.tier_classifier import classify_spec_tier

        # A purely new skill subdirectory: 1 even with refactor keyword.
        assert (
            classify_spec_tier(
                whitelist_paths=[
                    "src/skills/newfeature/__init__.py",
                    "src/skills/newfeature/skill.py",
                ],
                goal="rewrite the way we say hello",
                modifies_capabilities=[],
            )
            == 1
        )


class TestTierClassifierPersonalityAnchorContract:
    """Shared identity-contract changes are protected even inside chat_response."""

    def test_shared_personality_anchor_contract_is_tier3(self) -> None:
        from src.skills.ideation_to_spec.tier_classifier import classify_spec_tier

        assert (
            classify_spec_tier(
                whitelist_paths=["src/skills/chat_response/prompts.py"],
                goal=(
                    "жёстко закрепить личность Жвуши в общем PERSONALITY_ANCHOR "
                    "как общий identity contract"
                ),
                modifies_capabilities=[],
            )
            == 3
        )

    def test_local_chat_prompt_fix_without_shared_contract_stays_tier2(self) -> None:
        from src.skills.ideation_to_spec.tier_classifier import classify_spec_tier

        assert (
            classify_spec_tier(
                whitelist_paths=["src/skills/chat_response/prompts.py"],
                goal="поправить локальную формулировку ответа в chat response",
                modifies_capabilities=[],
            )
            == 2
        )


class TestTierClassifierMixedPaths:
    """Most-restrictive wins: any single Tier 3 path → Tier 3."""

    def test_one_tier3_path_among_many_forces_tier3(self) -> None:
        from src.skills.ideation_to_spec.tier_classifier import classify_spec_tier

        assert (
            classify_spec_tier(
                whitelist_paths=[
                    "src/skills/weather/skill.py",
                    "src/skills/base.py",  # Tier 3
                ],
                goal="add weather",
                modifies_capabilities=[],
            )
            == 3
        )

    def test_one_tier2_path_among_new_paths_forces_tier2(self) -> None:
        from src.skills.ideation_to_spec.tier_classifier import classify_spec_tier

        assert (
            classify_spec_tier(
                whitelist_paths=[
                    "src/skills/weather/skill.py",  # new
                    "src/memory/consolidation.py",  # existing -> Tier 2
                ],
                goal="extend weather + memory tweak",
                modifies_capabilities=[],
            )
            == 2
        )


class TestTierClassifierEmptyWhitelist:
    """Empty whitelist is invalid for callers but classifier returns Tier 1 default."""

    def test_empty_whitelist_returns_tier1(self) -> None:
        from src.skills.ideation_to_spec.tier_classifier import classify_spec_tier

        # SpecModel rejects empty whitelist; classifier should not crash.
        assert (
            classify_spec_tier(
                whitelist_paths=[], goal="anything", modifies_capabilities=[]
            )
            == 1
        )


class TestClassifySpecTierRename:
    """Verify the rename classify_tier → classify_spec_tier is complete."""

    def test_classify_spec_tier_exists_and_old_name_gone(self) -> None:
        import importlib

        mod = importlib.import_module("src.skills.ideation_to_spec.tier_classifier")
        # New name works
        assert hasattr(mod, "classify_spec_tier")
        result = mod.classify_spec_tier(
            whitelist_paths=["src/skills/weather/skill.py"],
            goal="add weather",
            modifies_capabilities=[],
        )
        assert result == 1

        # Old name is gone
        assert not hasattr(mod, "classify_tier")

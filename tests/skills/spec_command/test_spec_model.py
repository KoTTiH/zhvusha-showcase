"""Contract tests for SpecModel — Pydantic schema for tasks/*.yaml.

See tasks/README.md and v4plans plan for the full lifecycle. The model is the
single source of truth for spec validation: ideation_to_spec writes through it,
spec_command reads through it, implement_spec validates through it before
execution, and check_whitelist.sh parses YAML via python-yaml against the same
field names.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError


def _minimal_spec_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return a minimal-valid SpecModel kwargs dict, with overrides applied."""
    base: dict[str, Any] = {
        "slug": "weather-skill",
        "title": "Add /weather skill returning temperature",
        "created_at": datetime(2026, 4, 26, 12, 0, tzinfo=UTC),
        "created_by": "nikita",
        "tier": 1,
        "goal": "Дать Жвуше команду /weather, которая возвращает температуру.",
        "failing_test": {
            "file": "tests/skills/weather/test_contract.py",
            "name": "test_returns_temp",
            "spec": "Mock API → assert response contains '12.5'.",
        },
        "whitelist_paths": [
            "src/skills/weather/__init__.py",
            "src/skills/weather/skill.py",
        ],
        "blast_radius": ["new skill, no existing skill touched"],
        "rollback_path": ["git revert", "rm -rf src/skills/weather/"],
    }
    base.update(overrides)
    return base


@pytest.mark.contract
class TestSpecModelHappyPath:
    """The minimal valid spec accepted; defaults populated."""

    def test_minimal_valid_spec_loads(self) -> None:
        from src.skills.spec_command.parser import SpecModel, SpecStatus

        spec = SpecModel(**_minimal_spec_kwargs())
        assert spec.slug == "weather-skill"
        assert spec.tier == 1
        assert spec.status == SpecStatus.PENDING_APPROVAL
        assert spec.iterations == 0
        assert spec.failed_attempts == []
        assert spec.research_findings == []
        assert spec.source_provenance == []
        assert spec.rationale == ""
        assert spec.chat_context == []
        assert spec.previous_attempts == []
        assert spec.read_only_paths == []
        assert spec.preserve_behavior == []
        assert spec.allowed_simplifications == []
        assert spec.actual_tokens == 0
        assert spec.actual_cost_usd == 0.0

    def test_status_round_trip_via_yaml(self) -> None:
        """YAML serialization round-trip: SpecModel → dict → SpecModel."""
        from src.skills.spec_command.parser import SpecModel

        original = SpecModel(**_minimal_spec_kwargs())
        as_dict = original.model_dump(mode="json")
        restored = SpecModel.model_validate(as_dict)
        assert restored.slug == original.slug
        assert restored.failing_test.file == original.failing_test.file
        assert restored.tier == original.tier

    def test_chat_context_strips_and_round_trips(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        original = SpecModel(
            **_minimal_spec_kwargs(
                chat_context=[
                    "  Никита: сначала обсудим приветствие  ",
                    "Жвуша: убираем театральность, живость оставляем.",
                ]
            )
        )
        assert original.chat_context == [
            "Никита: сначала обсудим приветствие",
            "Жвуша: убираем театральность, живость оставляем.",
        ]
        restored = SpecModel.model_validate(original.model_dump(mode="json"))
        assert restored.chat_context == original.chat_context

    def test_blank_chat_context_rejected(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError, match="chat_context"):
            SpecModel(**_minimal_spec_kwargs(chat_context=["   "]))

    def test_previous_attempts_strip_and_round_trip(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        original = SpecModel(
            **_minimal_spec_kwargs(
                previous_attempts=[
                    {
                        "archive_slug": "  greeting-calibration-abc123  ",
                        "status": "failed",
                        "tier": 2,
                        "commit_sha": " abc123 ",
                        "insight": "  Too much theatrical greeting tone. ",
                        "tests_summary": " pytest failed ",
                    }
                ]
            )
        )

        assert (
            original.previous_attempts[0].archive_slug == "greeting-calibration-abc123"
        )
        assert (
            original.previous_attempts[0].insight
            == "Too much theatrical greeting tone."
        )

        restored = SpecModel.model_validate(original.model_dump(mode="json"))
        assert restored.previous_attempts == original.previous_attempts

    def test_blank_previous_attempt_insight_rejected(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError, match="previous_attempt"):
            SpecModel(
                **_minimal_spec_kwargs(
                    previous_attempts=[
                        {
                            "archive_slug": "greeting-calibration-abc123",
                            "status": "failed",
                            "tier": 2,
                            "insight": "   ",
                        }
                    ]
                )
            )


@pytest.mark.contract
class TestSpecModelSlug:
    """Slug must be lowercase, dashes, ≤60 chars."""

    def test_uppercase_slug_rejected(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError):
            SpecModel(**_minimal_spec_kwargs(slug="Weather-Skill"))

    def test_underscore_slug_rejected(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError):
            SpecModel(**_minimal_spec_kwargs(slug="weather_skill"))

    def test_long_slug_rejected(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError):
            SpecModel(**_minimal_spec_kwargs(slug="x" * 61))

    def test_dashed_lowercase_slug_accepted(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        spec = SpecModel(**_minimal_spec_kwargs(slug="weather-skill-v2"))
        assert spec.slug == "weather-skill-v2"


@pytest.mark.contract
class TestSpecModelTier3Consistency:
    """Tier 3 path in whitelist requires tier=3 designation; otherwise fail."""

    def test_tier3_path_in_whitelist_with_tier1_raises(self) -> None:
        """The headline failing_test for Phase 10."""
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError, match=r"[Tt]ier.?3"):
            SpecModel(
                **_minimal_spec_kwargs(
                    tier=1,
                    whitelist_paths=["src/skills/base.py"],
                )
            )

    def test_protocols_py_in_whitelist_with_tier2_raises(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError, match=r"[Tt]ier.?3"):
            SpecModel(
                **_minimal_spec_kwargs(
                    tier=2,
                    whitelist_paths=["src/llm/protocols.py"],
                )
            )

    def test_safety_dir_in_whitelist_with_tier1_raises(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError, match=r"[Tt]ier.?3"):
            SpecModel(
                **_minimal_spec_kwargs(
                    tier=1,
                    whitelist_paths=["src/safety/guard.py"],
                )
            )

    def test_importlinter_in_whitelist_with_tier1_raises(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError, match=r"[Tt]ier.?3"):
            SpecModel(
                **_minimal_spec_kwargs(
                    tier=1,
                    whitelist_paths=[".importlinter"],
                )
            )

    def test_tier3_path_with_tier3_accepted(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        spec = SpecModel(
            **_minimal_spec_kwargs(
                tier=3,
                whitelist_paths=["src/skills/base.py"],
            )
        )
        assert spec.tier == 3

    def test_rationale_self_declared_tier3_with_tier2_raises(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError, match=r"rationale.*Tier 3"):
            SpecModel(
                **_minimal_spec_kwargs(
                    tier=2,
                    whitelist_paths=["src/skills/chat_response/prompts.py"],
                    rationale=(
                        "Я фиксирую это как Tier 3, потому что это общий "
                        "contract личности Жвуши."
                    ),
                )
            )


@pytest.mark.contract
class TestSpecModelRequiredFields:
    """Empty whitelist, blast_radius, rollback_path are rejected."""

    def test_empty_whitelist_rejected(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError):
            SpecModel(**_minimal_spec_kwargs(whitelist_paths=[]))

    def test_empty_blast_radius_rejected(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError):
            SpecModel(**_minimal_spec_kwargs(blast_radius=[]))

    def test_empty_rollback_rejected(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError):
            SpecModel(**_minimal_spec_kwargs(rollback_path=[]))

    def test_short_goal_rejected(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError):
            SpecModel(**_minimal_spec_kwargs(goal="too short"))


@pytest.mark.contract
class TestSpecModelStatus:
    """Status enum accepts only known values."""

    def test_unknown_status_rejected(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError):
            SpecModel(**_minimal_spec_kwargs(status="paused"))

    def test_known_statuses_accepted(self) -> None:
        from src.skills.spec_command.parser import SpecModel, SpecStatus

        for status in SpecStatus:
            spec = SpecModel(**_minimal_spec_kwargs(status=status))
            assert spec.status == status


@pytest.mark.contract
class TestSpecModelTierLiteral:
    """Tier accepts only 1, 2, 3."""

    @pytest.mark.parametrize("bad_tier", [0, 4, -1, 1.5])
    def test_invalid_tier_rejected(self, bad_tier: float) -> None:
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError):
            SpecModel(**_minimal_spec_kwargs(tier=bad_tier))


@pytest.mark.contract
class TestSpecModelKind:
    """``spec.kind`` classifies spec intent so the Editor knows whether
    existing tests can be touched.

    FEATURE / FIX / DOCS treat existing tests as immutable. REFACTOR
    explicitly permits structural updates (renamed imports, renamed
    symbol references) — but never assertion-logic edits.

    Default is FEATURE — the safest mode and the one all pre-kind specs
    were authored under, so existing yaml files keep parsing.
    """

    def test_default_kind_is_feature(self) -> None:
        from src.skills.spec_command.parser import SpecKind, SpecModel

        spec = SpecModel(**_minimal_spec_kwargs())
        assert spec.kind == SpecKind.FEATURE

    def test_known_kinds_accepted(self) -> None:
        from src.skills.spec_command.parser import SpecKind, SpecModel

        for kind in SpecKind:
            spec = SpecModel(**_minimal_spec_kwargs(kind=kind))
            assert spec.kind == kind

    def test_unknown_kind_rejected(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError):
            SpecModel(**_minimal_spec_kwargs(kind="rewrite"))

    def test_kind_round_trips_through_model_dump(self) -> None:
        from src.skills.spec_command.parser import SpecKind, SpecModel

        original = SpecModel(**_minimal_spec_kwargs(kind=SpecKind.REFACTOR))
        as_dict = original.model_dump(mode="json")
        restored = SpecModel.model_validate(as_dict)
        assert restored.kind == SpecKind.REFACTOR

    def test_kind_serializes_as_lowercase_string(self) -> None:
        from src.skills.spec_command.parser import SpecKind, SpecModel

        spec = SpecModel(**_minimal_spec_kwargs(kind=SpecKind.REFACTOR))
        as_dict = spec.model_dump(mode="json")
        assert as_dict["kind"] == "refactor"


@pytest.mark.contract
class TestSpecModelExistingTestsToUpdate:
    """``spec.existing_tests_to_update`` is the legitimate channel by
    which a spec declares "this existing test must be mutated, with
    these limits" — Editor (downstream) reads the list and gates
    test edits against it.

    Phase 16 — Architect detects hidden test contracts (fixed-set /
    fixed-length asserts in read_only test files) and either declares
    the unavoidable mutation here, or escalates to a separate
    refactor-kind pre-spec. Without this field, every collection-extension
    feature whose existing test count-asserts the collection silently
    stalls the Editor cycle (kind=feature/fix forbid existing test edits).

    Each entry must declare four things:
    * ``path`` — the test file (under ``tests/``, sanity-guarded).
    * ``test_name`` — the specific test function/method to be touched.
    * ``reason`` — *why* this spec forces the edit (Architect's
      justification, used in code review and in the failure log if the
      Editor cycle aborts).
    * ``allowed_changes`` — the minimal edit envelope (e.g. "add the
      new preset name to the asserted set", "replace == with >= for
      subset check"). Editor uses this as the contract for what it may
      do; anything outside this envelope is still forbidden.

    Default is the empty list — no spec needs the field unless it
    extends a collection or otherwise hits a hidden contract. Existing
    yaml files without the field load unchanged.
    """

    def test_default_is_empty_list(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        spec = SpecModel(**_minimal_spec_kwargs())
        assert spec.existing_tests_to_update == []

    def test_well_formed_entry_accepted(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        spec = SpecModel(
            **_minimal_spec_kwargs(
                existing_tests_to_update=[
                    {
                        "path": "tests/research/test_research_service.py",
                        "test_name": "TestResearchPresets.test_four_presets_defined",
                        "reason": (
                            "Adding 'bug_investigation' to PRESETS breaks the "
                            "fixed-set equality assertion."
                        ),
                        "allowed_changes": (
                            "Add 'bug_investigation' to the asserted set; "
                            "do not change other assertions in this test."
                        ),
                    }
                ]
            )
        )
        assert len(spec.existing_tests_to_update) == 1
        entry = spec.existing_tests_to_update[0]
        assert entry.path == "tests/research/test_research_service.py"
        assert entry.test_name == "TestResearchPresets.test_four_presets_defined"
        assert "bug_investigation" in entry.reason
        assert "asserted set" in entry.allowed_changes

    @pytest.mark.parametrize(
        "missing_key",
        ["path", "test_name", "reason", "allowed_changes"],
    )
    def test_each_field_is_required(self, missing_key: str) -> None:
        from src.skills.spec_command.parser import SpecModel

        entry = {
            "path": "tests/research/test_research_service.py",
            "test_name": "TestResearchPresets.test_four_presets_defined",
            "reason": "Hidden contract on PRESETS membership.",
            "allowed_changes": "Add new entry to the asserted set.",
        }
        del entry[missing_key]
        with pytest.raises(ValidationError):
            SpecModel(**_minimal_spec_kwargs(existing_tests_to_update=[entry]))

    @pytest.mark.parametrize(
        "blank_field",
        ["path", "test_name", "reason", "allowed_changes"],
    )
    def test_blank_string_rejected(self, blank_field: str) -> None:
        from src.skills.spec_command.parser import SpecModel

        entry = {
            "path": "tests/research/test_research_service.py",
            "test_name": "TestResearchPresets.test_four_presets_defined",
            "reason": "Hidden contract on PRESETS membership.",
            "allowed_changes": "Add new entry to the asserted set.",
        }
        entry[blank_field] = "   "
        with pytest.raises(ValidationError):
            SpecModel(**_minimal_spec_kwargs(existing_tests_to_update=[entry]))

    def test_path_must_be_under_tests_directory(self) -> None:
        """Sanity guard — the field is for *tests*, not for arbitrary files.

        If Architect tries to use this channel to mutate a non-test file,
        that's a misuse — the Editor would still refuse on whitelist grounds,
        but rejecting at schema time gives a clearer error.
        """
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError, match=r"tests/"):
            SpecModel(
                **_minimal_spec_kwargs(
                    existing_tests_to_update=[
                        {
                            "path": "src/research/presets.py",
                            "test_name": "n/a",
                            "reason": "Trying to abuse the channel.",
                            "allowed_changes": "rewrite freely",
                        }
                    ]
                )
            )

    def test_round_trip_through_yaml_dict(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        original = SpecModel(
            **_minimal_spec_kwargs(
                existing_tests_to_update=[
                    {
                        "path": "tests/research/test_research_service.py",
                        "test_name": "TestResearchPresets.test_four_presets_defined",
                        "reason": "Adding new preset breaks fixed-set equality.",
                        "allowed_changes": "Add the new preset to the set.",
                    }
                ]
            )
        )
        as_dict = original.model_dump(mode="json")
        restored = SpecModel.model_validate(as_dict)
        assert len(restored.existing_tests_to_update) == 1
        assert (
            restored.existing_tests_to_update[0].path
            == "tests/research/test_research_service.py"
        )

    def test_legacy_yaml_without_field_still_loads(self) -> None:
        """Backward compat — every spec authored before Phase 16 must still
        validate. Default empty list keeps old yaml parsing unchanged."""
        from src.skills.spec_command.parser import SpecModel

        legacy_kwargs = _minimal_spec_kwargs()
        # No existing_tests_to_update key at all — simulating a pre-Phase-16
        # spec.yaml loaded via SpecModel.model_validate.
        assert "existing_tests_to_update" not in legacy_kwargs
        spec = SpecModel.model_validate(legacy_kwargs)
        assert spec.existing_tests_to_update == []


@pytest.mark.contract
class TestSpecModelRationaleAndSourceProvenance:
    """Жвушины generated specs must explain why they exist and what stays.

    Никитины manually-authored historical specs may stay minimal, but anything
    created by Жвуша becomes part of the self-improvement archive and therefore
    needs source evidence + rationale + a no-downgrade contract.
    """

    def test_nikita_spec_keeps_backward_compatible_defaults(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        spec = SpecModel(**_minimal_spec_kwargs(created_by="nikita"))
        assert spec.rationale == ""
        assert spec.source_provenance == []
        assert spec.preserve_behavior == []
        assert spec.allowed_simplifications == []

    def test_legacy_spec_without_created_by_defaults_to_nikita(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        legacy_kwargs = _minimal_spec_kwargs()
        legacy_kwargs.pop("created_by")

        spec = SpecModel.model_validate(legacy_kwargs)

        assert spec.created_by == "nikita"
        assert spec.rationale == ""
        assert spec.source_provenance == []
        assert spec.preserve_behavior == []
        assert spec.allowed_simplifications == []

    def test_zhvusha_spec_requires_rationale(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError, match="rationale"):
            SpecModel(
                **_minimal_spec_kwargs(
                    created_by="zhvusha",
                    source_provenance=[
                        {
                            "url": "src/skills/code_agent/registry.py",
                            "source_type": "local_repo",
                            "trust_tier": "direct",
                            "claim": "Self-coding backend is codex_cli.",
                        }
                    ],
                )
            )

    def test_zhvusha_spec_requires_source_provenance(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError, match="source_provenance"):
            SpecModel(
                **_minimal_spec_kwargs(
                    created_by="zhvusha",
                    rationale="Codex-only migration must be enforced in specs.",
                )
            )

    def test_zhvusha_spec_requires_preserve_behavior(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError, match="preserve_behavior"):
            SpecModel(
                **_minimal_spec_kwargs(
                    created_by="zhvusha",
                    rationale="Codex-only migration must be enforced in specs.",
                    source_provenance=[
                        {
                            "url": "src/skills/code_agent/registry.py",
                            "source_type": "local_repo",
                            "trust_tier": "direct",
                            "claim": "Self-coding backend is codex_cli.",
                        }
                    ],
                )
            )

    def test_zhvusha_spec_with_evidence_and_preservation_is_accepted(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        spec = SpecModel(
            **_minimal_spec_kwargs(
                created_by="zhvusha",
                rationale=(
                    "Codex-only migration changes the self-coding contract, so "
                    "the spec must carry evidence for review."
                ),
                source_provenance=[
                    {
                        "url": "src/skills/code_agent/registry.py",
                        "source_type": "local_repo",
                        "trust_tier": "direct",
                        "claim": "Registry accepts only codex_cli backend order.",
                    }
                ],
                preserve_behavior=[
                    "Existing Codex-only backend gates, tests, fallbacks and "
                    "chat-mode behaviour remain intact.",
                ],
            )
        )
        assert spec.rationale.startswith("Codex-only")
        assert spec.source_provenance[0].source_type == "local_repo"
        assert spec.preserve_behavior[0].startswith("Existing Codex-only")
        assert spec.allowed_simplifications == []

    def test_allowed_simplifications_strip_and_round_trip(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        spec = SpecModel(
            **_minimal_spec_kwargs(
                allowed_simplifications=["  Remove obsolete duplicate message.  "],
            )
        )
        assert spec.allowed_simplifications == ["Remove obsolete duplicate message."]

    def test_blank_preserve_behavior_rejected(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError, match="non-empty"):
            SpecModel(**_minimal_spec_kwargs(preserve_behavior=["   "]))

    def test_source_provenance_rejects_blank_claim(self) -> None:
        from src.skills.spec_command.parser import SpecModel

        with pytest.raises(ValidationError):
            SpecModel(
                **_minimal_spec_kwargs(
                    created_by="zhvusha",
                    rationale="Has source but blank claim should be rejected.",
                    source_provenance=[
                        {
                            "url": "src/skills/code_agent/registry.py",
                            "source_type": "local_repo",
                            "trust_tier": "direct",
                            "claim": " ",
                        }
                    ],
                )
            )

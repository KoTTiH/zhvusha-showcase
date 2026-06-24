"""Tests for ``ideation_to_spec.spec_writer`` — parses SDK output → spec file.

The writer is deliberately small and side-effecty only on disk:

* :func:`extract_yaml_block` pulls the ``\\`\\`\\`yaml...\\`\\`\\``` block out of free
  text returned by the Architect SDK call.
* :func:`build_spec_from_draft` validates the draft against ``SpecModel``,
  re-runs the deterministic ``classify_spec_tier`` and keeps the most
  restrictive tier between the classifier and the Architect draft.
* :func:`write_spec_to_disk` writes the validated spec under
  ``tasks/<YYYY-MM-DD>-<slug>.yaml``, preserving key order.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.contract


def _draft_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "slug": "weather-skill",
        "title": "Add /weather skill",
        "created_at": "2026-04-26T12:00:00+00:00",
        "created_by": "zhvusha",
        "tier": 1,
        "goal": "Дать Жвуше команду /weather, возвращающую температуру по городу.",
        "rationale": (
            "Никита попросил новый weather skill; похожих skill'ов нет, "
            "поэтому это изолированное Tier 1 расширение."
        ),
        "source_provenance": [
            {
                "url": "src/skills/weather/",
                "source_type": "local_repo",
                "trust_tier": "direct",
                "claim": "Weather skill directory does not exist yet.",
            }
        ],
        "preserve_behavior": [
            "Existing skills, fallbacks, tests and chat behaviour stay intact.",
        ],
        "allowed_simplifications": [],
        "failing_test": {
            "file": "tests/skills/weather/test_contract.py",
            "name": "test_returns_temp",
            "spec": "Mock API → response contains '12.5'.",
        },
        "whitelist_paths": [
            "src/skills/weather/__init__.py",
            "src/skills/weather/skill.py",
        ],
        "blast_radius": ["new skill"],
        "rollback_path": ["git revert"],
    }
    base.update(overrides)
    return base


class TestExtractYamlBlock:
    def test_extracts_fenced_yaml_block(self) -> None:
        from src.skills.ideation_to_spec.spec_writer import extract_yaml_block

        text = (
            "Here is your spec.\n\n"
            "```yaml\n"
            "slug: weather-skill\n"
            "title: Add weather\n"
            "```\n"
            "Hope it helps."
        )
        block = extract_yaml_block(text)
        assert "slug: weather-skill" in block
        assert "Hope it helps" not in block

    def test_extracts_first_block_when_multiple(self) -> None:
        from src.skills.ideation_to_spec.spec_writer import extract_yaml_block

        text = "```yaml\nfirst: 1\n```\n```yaml\nsecond: 2\n```"
        block = extract_yaml_block(text)
        assert "first: 1" in block
        assert "second: 2" not in block

    def test_extracts_unfenced_yaml_when_no_block_present(self) -> None:
        """If SDK returns plain YAML without code fences, accept it."""
        from src.skills.ideation_to_spec.spec_writer import extract_yaml_block

        text = "slug: weather-skill\ntitle: x\n"
        block = extract_yaml_block(text)
        assert "slug: weather-skill" in block

    def test_raises_on_empty_input(self) -> None:
        from src.skills.ideation_to_spec.spec_writer import extract_yaml_block

        with pytest.raises(ValueError, match="empty"):
            extract_yaml_block("")


class TestExtractClarificationRequest:
    def test_extracts_single_clarification_line(self) -> None:
        from src.skills.ideation_to_spec.spec_writer import (
            extract_clarification_request,
        )

        question = extract_clarification_request(
            "CLARIFICATION_NEEDED: Сохранять старый fallback?"
        )
        assert question == "Сохранять старый fallback?"

    def test_returns_none_for_yaml(self) -> None:
        from src.skills.ideation_to_spec.spec_writer import (
            extract_clarification_request,
        )

        assert extract_clarification_request("```yaml\nslug: x\n```") is None


class TestBuildSpecFromDraft:
    def test_validates_and_returns_spec_model(self) -> None:
        from src.skills.ideation_to_spec.spec_writer import build_spec_from_draft
        from src.skills.spec_command.parser import SpecModel

        spec = build_spec_from_draft(_draft_dict())
        assert isinstance(spec, SpecModel)
        assert spec.slug == "weather-skill"
        assert spec.tier == 1
        assert spec.rationale.startswith("Никита попросил")
        assert spec.source_provenance[0].trust_tier == "direct"
        assert spec.preserve_behavior[0].startswith("Existing skills")

    def test_classifier_overrides_lower_tier_from_draft(self) -> None:
        """LLM may guess tier=1 for a path that is actually Tier 2."""
        from src.skills.ideation_to_spec.spec_writer import build_spec_from_draft

        draft = _draft_dict(
            tier=1,  # LLM's guess
            whitelist_paths=["src/memory/consolidation.py"],  # actually Tier 2
            goal="refactor consolidation engine for clarity",
        )
        spec = build_spec_from_draft(draft)
        assert spec.tier == 2

    def test_classifier_escalates_to_tier3_when_path_is_tier3(self) -> None:
        from src.skills.ideation_to_spec.spec_writer import build_spec_from_draft

        draft = _draft_dict(
            tier=1,
            whitelist_paths=["src/skills/base.py"],  # Tier 3
            goal="Add helper that exposes new contract on BaseSkill class",
        )
        spec = build_spec_from_draft(draft)
        assert spec.tier == 3

    def test_architect_declared_tier3_is_not_downgraded(self) -> None:
        from src.skills.ideation_to_spec.spec_writer import build_spec_from_draft

        draft = _draft_dict(
            tier=3,
            whitelist_paths=["src/skills/chat_response/prompts.py"],
            goal="поправить локальную формулировку ответа в chat response",
            rationale=(
                "Я фиксирую это как Tier 3, потому что это общий contract "
                "личности Жвуши, а не локальная prompt правка."
            ),
        )
        spec = build_spec_from_draft(draft)
        assert spec.tier == 3

    def test_personality_anchor_contract_draft_becomes_tier3(self) -> None:
        from src.skills.ideation_to_spec.spec_writer import build_spec_from_draft

        draft = _draft_dict(
            tier=2,
            whitelist_paths=["src/skills/chat_response/prompts.py"],
            goal=(
                "жёстко закрепить личность Жвуши в общем prompt anchor, "
                "чтобы PERSONALITY_ANCHOR стал shared identity contract"
            ),
            rationale=(
                "Никита просит общий contract личности, поэтому менять нужно "
                "PERSONALITY_ANCHOR, а не локальный сценарий."
            ),
        )
        spec = build_spec_from_draft(draft)
        assert spec.tier == 3

    def test_invalid_draft_raises_validation_error(self) -> None:
        from pydantic import ValidationError
        from src.skills.ideation_to_spec.spec_writer import build_spec_from_draft

        # Empty whitelist is rejected.
        with pytest.raises(ValidationError):
            build_spec_from_draft(_draft_dict(whitelist_paths=[]))

    def test_normalizes_object_items_in_free_text_risk_lists(self) -> None:
        """Codex Architect may emit a YAML object where the schema wants a
        free-text bullet. For narrative risk fields this is recoverable:
        flatten the object into a readable string before Pydantic validation."""
        from src.skills.ideation_to_spec.spec_writer import build_spec_from_draft

        spec = build_spec_from_draft(
            _draft_dict(
                blast_radius=[
                    "new skill",
                    {
                        "path": "personality/core.md",
                        "risk": "could over-correct Жвуша's voice",
                    },
                ],
                rollback_path=[
                    {
                        "command": "git revert",
                        "reason": "undo prompt calibration",
                    }
                ],
                preserve_behavior=[
                    {
                        "path": "personality/core.md",
                        "keep": "Жвушина voice calibration",
                    }
                ],
            )
        )

        assert spec.blast_radius[1] == (
            "path: personality/core.md; risk: could over-correct Жвуша's voice"
        )
        assert spec.rollback_path == [
            "command: git revert; reason: undo prompt calibration"
        ]
        assert spec.preserve_behavior == [
            "path: personality/core.md; keep: Жвушина voice calibration"
        ]


class TestWriteSpecToDisk:
    def test_writes_spec_under_dated_filename(self, tmp_path: Path) -> None:
        from src.skills.ideation_to_spec.spec_writer import (
            build_spec_from_draft,
            write_spec_to_disk,
        )

        spec = build_spec_from_draft(_draft_dict())
        path = write_spec_to_disk(
            tasks_dir=tmp_path,
            spec=spec,
            now=datetime(2026, 4, 27, 9, 0, tzinfo=UTC),
        )
        assert path.parent == tmp_path
        assert path.name == "2026-04-27-weather-skill.yaml"
        loaded = yaml.safe_load(path.read_text())
        assert loaded["slug"] == "weather-skill"
        assert loaded["status"] == "pending_approval"

    def test_does_not_overwrite_existing(self, tmp_path: Path) -> None:
        from src.skills.ideation_to_spec.spec_writer import (
            build_spec_from_draft,
            write_spec_to_disk,
        )

        spec = build_spec_from_draft(_draft_dict())
        now = datetime(2026, 4, 27, 9, 0, tzinfo=UTC)
        write_spec_to_disk(tasks_dir=tmp_path, spec=spec, now=now)

        # Second write with the same slug + date → conflict → suffix added.
        path2 = write_spec_to_disk(tasks_dir=tmp_path, spec=spec, now=now)
        assert path2.name != "2026-04-27-weather-skill.yaml"
        assert "weather-skill" in path2.name
        # Two distinct files now exist.
        files = sorted(p.name for p in tmp_path.glob("*.yaml"))
        assert len(files) == 2

"""Contract + chain tests for IdeationToSpecSkill (Phase 12).

Verifies v4 ``DelegatedSkill`` contract and runs an end-to-end mocked flow:
free-text request → SDK runner returns canned YAML → skill validates,
classifies tier, writes ``tasks/<YYYY-MM-DD>-<slug>.yaml``.

Tests do not call any real LLM (per AGENTS.md rule 0). The
``sdk_runner`` callable is the only seam we need to mock; research
findings are passed via injected fake ``ResearchService``.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import yaml

pytestmark = pytest.mark.contract


_DRAFT_YAML_TEMPLATE = """```yaml
slug: weather-skill
title: Add /weather skill
created_at: '2026-04-26T12:00:00+00:00'
created_by: zhvusha
tier: 1
goal: |
  Дать Жвуше команду /weather <city>, возвращающую температуру.
  Personal mode для admin_user_id.
rationale: |
  Никита попросил новый weather skill; похожих skill'ов нет, поэтому это
  изолированное Tier 1 расширение с понятным тестом.
source_provenance:
  - url: src/skills/weather/
    source_type: local_repo
    trust_tier: direct
    claim: Weather skill directory does not exist yet.
preserve_behavior:
  - Existing skills, fallbacks, tests and chat behaviour stay intact.
allowed_simplifications: []
failing_test:
  file: tests/skills/weather/test_contract.py
  name: test_returns_temp
  spec: Mock API → response contains '12.5'.
whitelist_paths:
  - src/skills/weather/__init__.py
  - src/skills/weather/skill.py
  - src/skills/weather/skill.yaml
  - tests/skills/weather/test_contract.py
blast_radius:
  - new skill, no existing skill touched
rollback_path:
  - git revert
  - rm -rf src/skills/weather/
research_findings: []
```
"""


def _ctx(*, user_id: int = 12345, mode: str = "personal", bot: Any = None) -> Any:
    from src.skills.base import AgentContext

    return AgentContext(
        user_id=user_id,
        chat_id=user_id,
        mode=mode,  # type: ignore[arg-type]
        bot=bot,
    )


class _FakeResearch:
    """Minimal stand-in for ResearchService used in chain tests."""

    def __init__(self, citations: list[Any] | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self._citations = list(citations or [])

    async def research(self, *, query: str, preset: str, budget_seconds: float) -> Any:
        del budget_seconds
        from src.research.protocols import ResearchResult

        self.calls.append((query, preset))
        return ResearchResult(
            citations=self._citations,
            elapsed_seconds=0.0,
            truncated=False,
            findings_summary="",
            sources_used=["kb"],
        )


class _FakeArchiveContext:
    def __init__(self, attempts: list[Any]) -> None:
        self.attempts = attempts
        self.calls: list[tuple[str, int]] = []

    async def previous_attempts(self, query: str, *, top_k: int = 3) -> list[Any]:
        self.calls.append((query, top_k))
        return self.attempts


class TestContract:
    def test_inherits_delegated_skill(self) -> None:
        from src.skills.base import BaseSkill, DelegatedSkill
        from src.skills.ideation_to_spec.skill import IdeationToSpecSkill

        assert issubclass(IdeationToSpecSkill, DelegatedSkill)
        assert issubclass(IdeationToSpecSkill, BaseSkill)

    def test_skill_type_is_delegated(self) -> None:
        from src.skills.ideation_to_spec.skill import IdeationToSpecSkill

        assert IdeationToSpecSkill.skill_type == "delegated"

    def test_manifest_loads_and_matches_class(self) -> None:
        from src.skills.ideation_to_spec.skill import IdeationToSpecSkill
        from src.skills.manifest import (
            load_manifest_for_skill_class,
            validate_manifest_matches_class,
        )

        manifest = load_manifest_for_skill_class(IdeationToSpecSkill)
        validate_manifest_matches_class(manifest, IdeationToSpecSkill)
        assert manifest.name == "ideation_to_spec"
        assert manifest.executor == "codex_cli"

    def test_approval_policy_is_required(self) -> None:
        from src.skills.ideation_to_spec.skill import IdeationToSpecSkill

        assert IdeationToSpecSkill.approval_policy == "required"

    def test_modifies_no_core_capabilities(self) -> None:
        from src.skills.ideation_to_spec.skill import IdeationToSpecSkill

        assert IdeationToSpecSkill.modifies == []

    def test_executor_is_codex_cli(self) -> None:
        from src.skills.ideation_to_spec.skill import IdeationToSpecSkill

        assert IdeationToSpecSkill.executor == "codex_cli"


def _make_skill(
    tasks_dir: Path,
    *,
    sdk_runner: Any | None = None,
    research: Any | None = None,
    block_publisher: Any | None = None,
    self_critique_runner: Any | None = None,
    archive_context_provider: Any | None = None,
) -> Any:
    from src.skills.ideation_to_spec.skill import IdeationToSpecSkill

    return IdeationToSpecSkill(
        tasks_dir=tasks_dir,
        admin_user_id=12345,
        research_service=research or _FakeResearch(),
        sdk_runner=sdk_runner or AsyncMock(return_value=_DRAFT_YAML_TEMPLATE),
        clock=lambda: datetime(2026, 4, 27, 9, 0, tzinfo=UTC),
        block_publisher=block_publisher,
        self_critique_runner=self_critique_runner,
        archive_context_provider=archive_context_provider,
    )


@pytest.mark.chain
class TestChainHappyPath:
    async def test_writes_validated_spec_from_mock_sdk(self, tmp_path: Path) -> None:
        skill = _make_skill(tmp_path)
        result = await skill.execute("/spec_create добавь skill для /weather", _ctx())

        assert result.success
        files = sorted(tmp_path.glob("*.yaml"))
        assert len(files) == 1
        loaded = yaml.safe_load(files[0].read_text())
        assert loaded["slug"] == "weather-skill"
        assert loaded["status"] == "pending_approval"
        assert loaded["rationale"].startswith("Никита попросил")
        assert loaded["source_provenance"][0]["trust_tier"] == "direct"
        assert files[0].name == "2026-04-27-weather-skill.yaml"
        # Skill response should mention the slug.
        assert "weather-skill" in result.response

    async def test_natural_spec_create_routes_and_strips_request(
        self, tmp_path: Path
    ) -> None:
        research = _FakeResearch()
        skill = _make_skill(tmp_path, research=research)

        assert await skill.can_handle("создай spec для /weather", _ctx()) >= 0.9
        assert await skill.can_handle("обсудим spec для /weather", _ctx()) == 0.0

        result = await skill.execute("создай spec для /weather", _ctx())

        assert result.success
        assert research.calls[0][0] == "для /weather"
        assert "weather-skill" in result.response

    async def test_natural_spec_create_tolerates_extra_spacing(
        self, tmp_path: Path
    ) -> None:
        research = _FakeResearch()
        skill = _make_skill(tmp_path, research=research)

        result = await skill.execute("создай   spec: для /weather", _ctx())

        assert result.success
        assert research.calls[0][0] == "для /weather"

    async def test_skill_result_metadata_carries_slug_and_tier(
        self, tmp_path: Path
    ) -> None:
        """Phase 40 contract: chat-mode skill needs the new slug + tier
        on the result so it can save them to session state. Returning
        them in ``metadata`` keeps the legacy text response unchanged
        for slash-command callers."""
        skill = _make_skill(tmp_path)
        result = await skill.execute("/spec_create добавь skill для /weather", _ctx())
        assert result.success
        assert result.metadata.get("slug") == "weather-skill"
        assert result.metadata.get("tier") == 1

    async def test_research_service_invoked(self, tmp_path: Path) -> None:
        research = _FakeResearch()
        skill = _make_skill(tmp_path, research=research)
        await skill.execute("/spec_create добавь skill для /weather", _ctx())
        assert len(research.calls) == 1
        query, preset = research.calls[0]
        assert "weather" in query.lower()
        assert preset in {
            "foundational",
            "current_practices",
            "api_integration",
            "hot_topic",
        }

    async def test_runtime_research_citations_are_persisted_when_draft_omits_them(
        self, tmp_path: Path
    ) -> None:
        """Runtime read-only evidence must survive the LLM draft boundary.

        The Architect prompt receives these findings, but the deterministic
        writer still persists them so a spec cannot silently drop real Telegram
        evidence or an explicit unavailable-source unknown.
        """
        from src.research.protocols import Citation

        research = _FakeResearch(
            citations=[
                Citation(
                    source="telegram_mcp",
                    ref="telegram://personal/dialog/42",
                    excerpt="confirmed: Тоша писал Никите про @Anroxa2748",
                ),
                Citation(
                    source="unknown",
                    ref="unavailable:web_research",
                    excerpt="UNKNOWN: agent_profile.web_research.readonly disabled",
                ),
            ]
        )
        sdk_runner = AsyncMock(return_value=_DRAFT_YAML_TEMPLATE)
        skill = _make_skill(tmp_path, research=research, sdk_runner=sdk_runner)

        result = await skill.execute(
            "/spec_create найди Telegram context для задачи", _ctx()
        )

        assert result.success
        sdk_prompt = sdk_runner.await_args.kwargs["user_prompt"]
        assert "[telegram_mcp] telegram://personal/dialog/42" in sdk_prompt
        assert "[unknown] unavailable:web_research" in sdk_prompt

        loaded = yaml.safe_load(next(tmp_path.glob("*.yaml")).read_text())
        findings = loaded["research_findings"]
        assert {finding["source"]: finding["excerpt"] for finding in findings} == {
            "telegram://personal/dialog/42": (
                "confirmed: Тоша писал Никите про @Anroxa2748"
            ),
            "unavailable:web_research": (
                "UNKNOWN: agent_profile.web_research.readonly disabled"
            ),
        }
        assert findings[0]["relevance"] == ("Research evidence from telegram_mcp.")
        assert findings[1]["relevance"] == (
            "Explicit unavailable research source; do not claim this context "
            "as evidence."
        )

    async def test_sdk_runner_called_once(self, tmp_path: Path) -> None:
        sdk_runner = AsyncMock(return_value=_DRAFT_YAML_TEMPLATE)
        skill = _make_skill(tmp_path, sdk_runner=sdk_runner)
        await skill.execute("/spec_create добавь skill для /weather", _ctx())
        sdk_runner.assert_awaited_once()

    async def test_chat_self_coding_discussion_context_is_persisted(
        self, tmp_path: Path
    ) -> None:
        skill = _make_skill(tmp_path)
        result = await skill.execute(
            "/spec_create "
            "Контекст предварительного обсуждения в режиме /код:\n"
            "Никита: хочу исправить приветствия\n"
            "Жвуша: убираем театральность, живость оставляем.\n\n"
            "Текущая команда Никиты:\n"
            "оформи план\n\n"
            "Составь spec на основе всего обсуждения.",
            _ctx(),
        )

        assert result.success
        loaded = yaml.safe_load(next(tmp_path.glob("*.yaml")).read_text())
        assert loaded["chat_context"] == [
            "Никита: хочу исправить приветствия",
            "Жвуша: убираем театральность, живость оставляем.",
        ]

    async def test_archive_previous_attempts_are_prompted_and_persisted(
        self, tmp_path: Path
    ) -> None:
        from src.skills.spec_command.parser import PreviousAttempt

        sdk_runner = AsyncMock(return_value=_DRAFT_YAML_TEMPLATE)
        archive_context = _FakeArchiveContext(
            [
                PreviousAttempt(
                    archive_slug="greeting-calibration-failed",
                    status="failed",
                    tier=2,
                    commit_sha=None,
                    insight="Greeting fix failed because personality context was not preserved.",
                    tests_summary="Spec validation failed.",
                )
            ]
        )
        skill = _make_skill(
            tmp_path,
            sdk_runner=sdk_runner,
            archive_context_provider=archive_context,
        )

        result = await skill.execute(
            "/spec_create исправь калибровку приветствий Жвуши", _ctx()
        )

        assert result.success
        sdk_prompt = sdk_runner.await_args.kwargs["user_prompt"]
        assert "greeting-calibration-failed" in sdk_prompt
        assert "personality context" in sdk_prompt
        loaded = yaml.safe_load(next(tmp_path.glob("*.yaml")).read_text())
        assert loaded["previous_attempts"][0]["archive_slug"] == (
            "greeting-calibration-failed"
        )
        assert archive_context.calls == [("исправь калибровку приветствий Жвуши", 3)]


@pytest.mark.chain
class TestChainNonAdmin:
    async def test_non_admin_rejected_in_can_handle(self, tmp_path: Path) -> None:
        skill = _make_skill(tmp_path)
        confidence = await skill.can_handle("/spec_create x", _ctx(user_id=99))
        assert confidence == 0.0


@pytest.mark.chain
class TestChainBadDraft:
    async def test_clarification_needed_returns_dialogue_metadata(
        self, tmp_path: Path
    ) -> None:
        sdk_runner = AsyncMock(
            return_value=(
                "CLARIFICATION_NEEDED: Ты разрешаешь упростить старый "
                "fallback или нужно сохранить оба пути?"
            )
        )
        skill = _make_skill(tmp_path, sdk_runner=sdk_runner)

        result = await skill.execute("/spec_create измени fallback", _ctx())

        assert result.success is False
        assert result.metadata["needs_clarification"] is True
        assert "разрешаешь" in result.response
        assert list(tmp_path.glob("*.yaml")) == []

    async def test_invalid_yaml_returns_failure(self, tmp_path: Path) -> None:
        sdk_runner = AsyncMock(return_value="not yaml at all 🤷")
        skill = _make_skill(tmp_path, sdk_runner=sdk_runner)
        result = await skill.execute("/spec_create x", _ctx())
        assert result.success is False
        # No spec file written.
        assert list(tmp_path.glob("*.yaml")) == []

    async def test_sdk_runner_timeout_cancels_runner_and_returns_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.skills.ideation_to_spec.skill import IdeationToSpecSkill

        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def slow_runner(*, system_prompt: str, user_prompt: str) -> str:
            del system_prompt, user_prompt
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return _DRAFT_YAML_TEMPLATE

        monkeypatch.setattr(IdeationToSpecSkill, "max_duration_seconds", 0.01)
        skill = _make_skill(tmp_path, sdk_runner=slow_runner)

        result = await skill.execute("/spec_create добавь /weather", _ctx())

        assert started.is_set()
        assert cancelled.is_set()
        assert result.success is False
        assert "Architect backend не ответил за" in result.response
        assert result.metadata["timeout_seconds"] == 0.01
        assert list(tmp_path.glob("*.yaml")) == []

    async def test_parse_failure_retries_once_with_strict_yaml_prompt(
        self,
        tmp_path: Path,
    ) -> None:
        bad_yaml = """```yaml
slug: channel-visual-pipeline
title: Feature: visual pipeline
created_at: '2026-05-08T18:32:00+03:00'
created_by: zhvusha
tier: 3
goal: добавить visual pipeline: внутренние темы через generated visual
```"""
        sdk_runner = AsyncMock(side_effect=[bad_yaml, _DRAFT_YAML_TEMPLATE])
        skill = _make_skill(tmp_path, sdk_runner=sdk_runner)

        result = await skill.execute("/spec_create visual pipeline", _ctx())

        assert result.success is True
        assert sdk_runner.await_count == 2
        retry_prompt = sdk_runner.await_args.kwargs["user_prompt"]
        assert "Previous Architect output failed YAML parsing" in retry_prompt
        assert "Return exactly one fenced ```yaml" in retry_prompt
        assert len(list(tmp_path.glob("*.yaml"))) == 1

    async def test_validation_failure_returns_helpful_error(
        self, tmp_path: Path
    ) -> None:
        # Draft without required failing_test field.
        bad_draft = """```yaml
slug: bad-spec
title: Missing failing_test
created_at: '2026-04-26T12:00:00+00:00'
created_by: zhvusha
tier: 1
goal: this draft will fail validation because failing_test is missing.
whitelist_paths:
  - src/skills/bad/skill.py
blast_radius: ['x']
rollback_path: ['x']
```"""
        sdk_runner = AsyncMock(return_value=bad_draft)
        skill = _make_skill(tmp_path, sdk_runner=sdk_runner)
        result = await skill.execute("/spec_create x", _ctx())
        assert result.success is False
        assert (
            "validation" in result.response.lower()
            or "failed" in result.response.lower()
        )


@pytest.mark.chain
class TestPrepareDryRun:
    async def test_prepare_returns_plan(self, tmp_path: Path) -> None:
        skill = _make_skill(tmp_path)
        plan = await skill.prepare("/spec_create x", _ctx())
        assert plan.skill_name == "ideation_to_spec"
        assert plan.skill_type == "delegated"
        assert plan.estimated_tokens > 0


@pytest.mark.chain
class TestBlockEventPublishing:
    """Phase 40 — IdeationToSpecSkill emits a PLAN block event after the
    spec is written, so the chat-mode skill can render a 📋 План message."""

    async def test_publishes_plan_event_after_spec_drafted(
        self, tmp_path: Path
    ) -> None:
        from src.skills.chat_self_coding.events import BlockEventType

        publisher = AsyncMock()
        skill = _make_skill(tmp_path, block_publisher=publisher)
        await skill.execute("/spec_create добавь /weather", _ctx())

        publisher.publish.assert_awaited_once()
        evt = publisher.publish.call_args.args[0]
        assert evt.event_type == BlockEventType.PLAN
        assert evt.user_id == 12345
        assert evt.slug == "weather-skill"
        # Payload contains what the renderer needs.
        assert "summary" in evt.payload
        assert "tier" in evt.payload
        assert "files" in evt.payload
        assert evt.payload["verification"] == (
            "tests/skills/weather/test_contract.py::test_returns_temp"
        )
        assert evt.payload["preserve_count"] == 1
        assert evt.payload["risk_count"] == 1
        assert evt.payload["allowed_simplifications"] == []
        assert evt.payload["deliverables"]
        assert evt.payload["safety_notes"]
        assert evt.payload["preserve_items"] == [
            "Existing skills, fallbacks, tests and chat behaviour stay intact."
        ]

    async def test_plan_event_carries_chat_code_task_id(self, tmp_path: Path) -> None:
        publisher = AsyncMock()
        skill = _make_skill(tmp_path, block_publisher=publisher)

        await skill.execute(
            "/spec_create добавь /weather",
            replace(
                _ctx(),
                metadata={"chat_self_coding_code_task_id": "code-task-fixed"},
            ),
        )

        evt = publisher.publish.call_args.args[0]
        assert evt.task_id == "code-task-fixed"

    async def test_default_publisher_is_noop_no_crash(self, tmp_path: Path) -> None:
        """No publisher injected → cycles continue working unchanged."""
        skill = _make_skill(tmp_path)  # no block_publisher kwarg
        result = await skill.execute("/spec_create добавь /weather", _ctx())
        assert result.success is True

    async def test_no_event_published_when_spec_validation_fails(
        self, tmp_path: Path
    ) -> None:
        """If the SDK output doesn't validate, we don't publish a stale PLAN."""
        publisher = AsyncMock()
        bad_yaml = "```yaml\nslug: x\ntier: 1\n```"  # missing required fields
        sdk_runner = AsyncMock(return_value=bad_yaml)
        skill = _make_skill(tmp_path, sdk_runner=sdk_runner, block_publisher=publisher)
        result = await skill.execute("/spec_create x", _ctx())
        assert result.success is False
        publisher.publish.assert_not_awaited()


@pytest.mark.chain
class TestSelfCritique:
    async def test_blocking_self_critique_stops_before_write(
        self, tmp_path: Path
    ) -> None:
        from src.skills.ideation_to_spec.self_critique import SelfCritiqueVerdict

        runner = AsyncMock()
        runner.review = AsyncMock(
            return_value=SelfCritiqueVerdict(
                blocking=True,
                summary="missing hidden dependent scan",
            )
        )
        skill = _make_skill(tmp_path, self_critique_runner=runner)

        result = await skill.execute("/spec_create добавь /weather", _ctx())

        assert not result.success
        assert "self-critique" in result.response.lower()
        assert list(tmp_path.glob("*.yaml")) == []

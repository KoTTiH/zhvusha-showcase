"""Natural-language digital scenario skill contracts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from src.skills.base import AgentContext, SkillResult


def _context(**metadata_overrides: Any) -> AgentContext:
    metadata = {
        "source": "vscode",
        "interface": "vscode",
        "source_actor": "codex",
    }
    metadata.update(metadata_overrides)
    return AgentContext(
        user_id=1291112109,
        chat_id=-7331,
        mode="personal",
        message_id=42,
        metadata=metadata,
    )


@dataclass
class FakeClassifier:
    scenario_id: str = "ai_cto_projects"
    confidence: float = 0.91
    calls: int = 0

    async def classify(self, message: str, context: AgentContext) -> object:
        from src.skills.digital_scenario.skill import DigitalScenarioIntent

        self.calls += 1
        return DigitalScenarioIntent(
            scenario_id=self.scenario_id,
            confidence=self.confidence,
            rationale=f"classified: {message[:30]}",
        )


@dataclass
class FakeActionRunner:
    result: SkillResult
    calls: list[Any] | None = None

    async def __call__(self, action: Any, context: AgentContext) -> SkillResult:
        del context
        if self.calls is None:
            self.calls = []
        self.calls.append(action)
        return self.result


async def test_digital_scenario_skill_routes_natural_language_without_slash() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityGraph,
        CapabilityKind,
        CapabilityNode,
        CapabilityStatus,
    )
    from src.agent_runtime.digital_scenarios import BUILTIN_DIGITAL_SCENARIOS
    from src.skills.digital_scenario.skill import DigitalScenarioSkill

    scenario = next(
        item for item in BUILTIN_DIGITAL_SCENARIOS if item.id == "ai_cto_projects"
    )
    graph = CapabilityGraph(
        capabilities=(
            CapabilityNode(
                id="digital_scenario.ai_cto_projects",
                label="AI-CTO для проектов",
                kind=CapabilityKind.DIGITAL_SCENARIO,
                status=CapabilityStatus.AVAILABLE,
                reason="all required runtime surfaces are available",
            ),
            *(
                CapabilityNode(
                    id=node_id,
                    label=node_id,
                    kind=CapabilityKind.SKILL,
                    status=CapabilityStatus.AVAILABLE,
                )
                for node_id in scenario.required_capability_nodes
            ),
        )
    )
    classifier = FakeClassifier()
    skill = DigitalScenarioSkill(
        admin_user_id=1291112109,
        intent_classifier=classifier,
        capability_graph_provider=lambda: graph,
    )

    assert (
        await skill.can_handle(
            "Жвуша, проверь ZHVUSHA как CTO и найди архитектурный долг",
            _context(),
        )
        == 0.91
    )
    assert (
        await skill.can_handle("/digital_scenarios ai_cto_projects", _context()) == 0.0
    )


async def test_digital_scenario_regular_classifier_score_stays_below_explicit_routes() -> (
    None
):
    from src.agent_runtime.capability_graph import CapabilityGraph
    from src.skills.digital_scenario.skill import DigitalScenarioSkill

    skill = DigitalScenarioSkill(
        admin_user_id=1291112109,
        intent_classifier=FakeClassifier(confidence=0.99),
        capability_graph_provider=CapabilityGraph,
    )

    assert await skill.can_handle("покажи свежие kwork", _context()) == 0.91


async def test_digital_scenario_skill_returns_body_observation_for_zhvusha_loop() -> (
    None
):
    from src.agent_runtime.capability_graph import (
        CapabilityGraph,
        CapabilityKind,
        CapabilityNode,
        CapabilityStatus,
    )
    from src.agent_runtime.digital_scenarios import BUILTIN_DIGITAL_SCENARIOS
    from src.skills.digital_scenario.skill import DigitalScenarioSkill

    scenario = next(
        item for item in BUILTIN_DIGITAL_SCENARIOS if item.id == "ai_cto_projects"
    )
    graph = CapabilityGraph(
        capabilities=(
            CapabilityNode(
                id="digital_scenario.ai_cto_projects",
                label="AI-CTO для проектов",
                kind=CapabilityKind.DIGITAL_SCENARIO,
                status=CapabilityStatus.AVAILABLE,
                reason="all required runtime surfaces are available",
            ),
            *(
                CapabilityNode(
                    id=node_id,
                    label=node_id,
                    kind=CapabilityKind.SKILL,
                    status=CapabilityStatus.AVAILABLE,
                )
                for node_id in scenario.required_capability_nodes
            ),
        )
    )
    skill = DigitalScenarioSkill(
        admin_user_id=1291112109,
        intent_classifier=FakeClassifier(),
        capability_graph_provider=lambda: graph,
    )

    result = await skill.execute(
        "Жвуша, проверь ZHVUSHA как CTO и найди архитектурный долг",
        _context(
            digital_scenario_eval_variant="happy_path",
            digital_scenario_eval_run_id="run-42",
        ),
    )

    assert result.success is True
    assert result.response == ""
    assert result.metadata["requires_zhvusha_response"] is True
    observation = result.metadata["body_observation"]
    assert observation["event"] == "digital_scenario_intent_detected"
    assert observation["source"] == "digital_scenario"
    assert observation["scenario_id"] == "ai_cto_projects"
    assert observation["chat_surface"] == "natural_language_user_flow"
    assert observation["ready_for_live_matrix"] is True
    assert observation["approval_boundaries"] == [
        "write_whitelisted_files_after_approval",
        "commit",
    ]
    assert "happy_path" in observation["eval_variants"]
    assert "codebase_explorer" in " ".join(observation["required_capability_nodes"])
    assert "next_actions" not in str(observation)


async def test_digital_scenario_skill_reports_runtime_gaps_in_observation() -> None:
    from src.agent_runtime.capability_graph import CapabilityGraph
    from src.skills.digital_scenario.skill import DigitalScenarioSkill

    skill = DigitalScenarioSkill(
        admin_user_id=1291112109,
        intent_classifier=FakeClassifier(scenario_id="personal_ops_hq"),
        capability_graph_provider=CapabilityGraph,
    )

    result = await skill.execute(
        "Жвуша, собери мне операционный штаб по проектам и обещаниям",
        _context(),
    )

    observation = result.metadata["body_observation"]
    assert observation["scenario_id"] == "personal_ops_hq"
    assert observation["ready_for_live_matrix"] is False
    assert "skill.morning_digest" in observation["missing_required_nodes"]
    assert observation["instruction"].startswith("Это routing observation")


async def test_digital_scenario_skill_uses_operator_eval_metadata_without_classifier() -> (
    None
):
    from src.agent_runtime.capability_graph import CapabilityGraph
    from src.skills.digital_scenario.skill import DigitalScenarioSkill

    runner = FakeActionRunner(
        result=SkillResult(
            success=True,
            response="Matrix prompt routed by operator metadata.",
            metadata={"skill_name": "codebase_explorer"},
        )
    )
    context = _context(
        digital_scenario_id="ai_cto_projects",
        digital_scenario_eval_variant="happy_path",
    )
    skill = DigitalScenarioSkill(
        admin_user_id=1291112109,
        intent_classifier=None,
        capability_graph_provider=CapabilityGraph,
        action_runner=runner,
    )

    assert await skill.can_handle("Переформулированный matrix prompt", context) == 0.99

    result = await skill.execute("Переформулированный matrix prompt", context)

    assert (
        result.metadata["digital_scenario_intent"]["scenario_id"] == "ai_cto_projects"
    )
    assert result.metadata["digital_scenario_live_evidence"]["variant"] == "happy_path"
    assert runner.calls is not None
    assert runner.calls[0].skill_name == "codebase_explorer"


async def test_digital_scenario_skill_skips_repeated_heavy_action_for_matrix_variants() -> (
    None
):
    from src.agent_runtime.capability_graph import CapabilityGraph
    from src.skills.digital_scenario.skill import DigitalScenarioSkill

    runner = FakeActionRunner(
        result=SkillResult(
            success=True,
            response="sync should not run",
            metadata={"skill_name": "codebase_explorer"},
        )
    )
    context = _context(
        digital_scenario_id="ai_cto_projects",
        digital_scenario_eval_variant="paraphrase",
        digital_scenario_eval_run_id="run-42",
    )
    skill = DigitalScenarioSkill(
        admin_user_id=1291112109,
        intent_classifier=None,
        capability_graph_provider=CapabilityGraph,
        action_runner=runner,
    )

    result = await skill.execute("Переформулированный matrix prompt", context)

    execution = result.metadata["body_observation"]["execution"]
    assert execution["attempted"] is False
    assert execution["success"] is True
    assert "matrix_variant_paraphrase" in execution["reason"]
    constraints = result.metadata["body_observation"]["response_constraints"]
    assert any("Не говори, что запускала tool" in item for item in constraints)
    assert runner.calls is None
    evidence = result.metadata["digital_scenario_live_evidence"]
    assert evidence["variant"] == "paraphrase"
    assert "attempted=false" in evidence["runtime_evidence"]
    assert "eval_run_id=run-42" in evidence["runtime_evidence"]


async def test_digital_scenario_skill_times_out_action_with_structured_evidence() -> (
    None
):
    from src.agent_runtime.capability_graph import CapabilityGraph
    from src.skills.digital_scenario.skill import DigitalScenarioSkill

    class HangingRunner:
        async def __call__(self, action: Any, context: AgentContext) -> SkillResult:
            del action, context
            await asyncio.sleep(60)
            return SkillResult(success=True, response="late")

    skill = DigitalScenarioSkill(
        admin_user_id=1291112109,
        intent_classifier=FakeClassifier(scenario_id="ai_cto_projects"),
        capability_graph_provider=CapabilityGraph,
        action_runner=HangingRunner(),
        action_timeout_seconds=0.01,
    )

    result = await skill.execute(
        "Жвуша, проверь ZHVUSHA как CTO и найди архитектурный долг",
        _context(digital_scenario_eval_variant="happy_path"),
    )

    execution = result.metadata["body_observation"]["execution"]
    assert execution["attempted"] is True
    assert execution["success"] is False
    assert execution["error_type"] == "TimeoutError"
    assert "digital_scenario_action_timed_out" in execution["reason"]
    evidence = result.metadata["digital_scenario_live_evidence"]
    assert "attempted=true" in evidence["runtime_evidence"]
    assert "success=false" in evidence["runtime_evidence"]


async def test_digital_scenario_skill_runs_first_safe_action_for_ai_cto() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityGraph,
        CapabilityKind,
        CapabilityNode,
        CapabilityStatus,
    )
    from src.agent_runtime.digital_scenarios import BUILTIN_DIGITAL_SCENARIOS
    from src.skills.digital_scenario.skill import DigitalScenarioSkill

    scenario = next(
        item for item in BUILTIN_DIGITAL_SCENARIOS if item.id == "ai_cto_projects"
    )
    graph = CapabilityGraph(
        capabilities=(
            CapabilityNode(
                id="digital_scenario.ai_cto_projects",
                label="AI-CTO для проектов",
                kind=CapabilityKind.DIGITAL_SCENARIO,
                status=CapabilityStatus.AVAILABLE,
            ),
            *(
                CapabilityNode(
                    id=node_id,
                    label=node_id,
                    kind=CapabilityKind.SKILL,
                    status=CapabilityStatus.AVAILABLE,
                )
                for node_id in scenario.required_capability_nodes
            ),
        )
    )
    runner = FakeActionRunner(
        result=SkillResult(
            success=True,
            response="Нашла архитектурный долг в dispatcher boundary.",
            metadata={"skill_name": "codebase_explorer"},
        )
    )
    skill = DigitalScenarioSkill(
        admin_user_id=1291112109,
        intent_classifier=FakeClassifier(scenario_id="ai_cto_projects"),
        capability_graph_provider=lambda: graph,
        action_runner=runner,
    )

    result = await skill.execute(
        "Жвуша, проверь ZHVUSHA как CTO и найди архитектурный долг",
        _context(
            digital_scenario_eval_variant="happy_path",
            digital_scenario_eval_run_id="run-42",
        ),
    )

    observation = result.metadata["body_observation"]
    execution = observation["execution"]
    assert execution["attempted"] is True
    assert execution["success"] is True
    assert execution["approval_pending"] is False
    assert execution["action"]["skill_name"] == "codebase_explorer"
    assert execution["action"]["user_visible_command_required"] is False
    assert "dispatcher boundary" in execution["result_response"]
    assert runner.calls is not None
    action = runner.calls[0]
    assert action.skill_name == "codebase_explorer"
    assert not str(action.message).startswith("/")
    evidence = result.metadata["digital_scenario_live_evidence"]
    assert evidence["variant"] == "happy_path"
    assert evidence["chat_message_id"] == "vscode:-7331:42"
    assert "eval_run_id=run-42" in evidence["runtime_evidence"]


async def test_digital_scenario_live_evidence_marks_background_job_pending() -> None:
    from src.agent_runtime.capability_graph import CapabilityGraph
    from src.skills.digital_scenario.skill import DigitalScenarioSkill

    runner = FakeActionRunner(
        result=SkillResult(
            success=True,
            response="взяла в фоновую read-only agent-задачу. Job: `job-test`.",
            metadata={
                "skill_name": "codebase_explorer",
                "agent_job_id": "job-test",
                "agent_job_result_pending": True,
                "agent_job_status": "running",
            },
        )
    )
    skill = DigitalScenarioSkill(
        admin_user_id=1291112109,
        intent_classifier=FakeClassifier(scenario_id="ai_cto_projects"),
        capability_graph_provider=lambda: CapabilityGraph(capabilities=()),
        action_runner=runner,
    )

    result = await skill.execute(
        "Жвуша, проверь ZHVUSHA как CTO",
        _context(
            digital_scenario_eval_variant="happy_path",
            digital_scenario_eval_run_id="run-background",
        ),
    )

    observation = result.metadata["body_observation"]
    execution = observation["execution"]
    assert execution["attempted"] is True
    assert execution["success"] is True
    assert execution["result_pending"] is True
    assert "результат надо смотреть" in " ".join(observation["response_constraints"])
    evidence = result.metadata["digital_scenario_live_evidence"]
    assert "skill_success=true" in evidence["runtime_evidence"]
    assert "result_pending=true" in evidence["runtime_evidence"]
    assert "success=true" not in evidence["runtime_evidence"]
    assert "agent_job_id=job-test" in evidence["runtime_evidence"]
    assert (
        evidence["limitations_or_unknowns"]
        == "Background job started; final Agent Runtime result is still pending."
    )


async def test_digital_scenario_skill_uses_approval_backed_action_without_user_slash() -> (
    None
):
    from src.agent_runtime.capability_graph import CapabilityGraph
    from src.skills.digital_scenario.skill import DigitalScenarioSkill

    runner = FakeActionRunner(
        result=SkillResult(
            success=True,
            response="Нужно решение перед выполнением.",
            metadata={
                "skill_name": "ideation_to_spec",
                "approval_pending": True,
                "decision_id": "skill-approval-test",
            },
        )
    )
    skill = DigitalScenarioSkill(
        admin_user_id=1291112109,
        intent_classifier=FakeClassifier(scenario_id="agent_designer"),
        capability_graph_provider=CapabilityGraph,
        action_runner=runner,
    )

    result = await skill.execute(
        "Жвуша, спроектируй агента-исследователя с evals и approvals",
        _context(),
    )

    observation = result.metadata["body_observation"]
    execution = observation["execution"]
    assert execution["attempted"] is True
    assert execution["approval_pending"] is True
    assert execution["action"]["skill_name"] == "ideation_to_spec"
    assert execution["action"]["user_visible_command_required"] is False
    assert "/spec_create" not in str(observation)
    assert runner.calls is not None
    assert str(runner.calls[0].message).startswith("/spec_create ")


async def test_digital_scenario_skill_runs_personal_ops_digest_without_user_slash() -> (
    None
):
    from src.agent_runtime.capability_graph import CapabilityGraph
    from src.skills.digital_scenario.skill import DigitalScenarioSkill

    runner = FakeActionRunner(
        result=SkillResult(
            success=True,
            response="Операционный штаб: 3 открытых цикла, 1 риск.",
            metadata={"skill_name": "morning_digest"},
        )
    )
    skill = DigitalScenarioSkill(
        admin_user_id=1291112109,
        intent_classifier=FakeClassifier(scenario_id="personal_ops_hq"),
        capability_graph_provider=CapabilityGraph,
        action_runner=runner,
    )

    result = await skill.execute(
        "Жвуша, собери операционный штаб по проектам и обещаниям",
        _context(),
    )

    execution = result.metadata["body_observation"]["execution"]
    assert execution["attempted"] is True
    assert execution["action"]["skill_name"] == "morning_digest"
    assert execution["action"]["user_visible_command_required"] is False
    assert runner.calls is not None
    assert runner.calls[0].skill_name == "morning_digest"
    assert not str(runner.calls[0].message).startswith("/")


def test_goal_polygons_have_non_command_first_safe_actions() -> None:
    from src.agent_runtime.digital_scenarios import BUILTIN_DIGITAL_SCENARIOS
    from src.skills.digital_scenario.skill import _execution_action_for_scenario

    required = {
        "personal_ops_hq",
        "ai_cto_projects",
        "agent_designer",
        "digital_twin_work_style",
        "external_skill_lab",
        "autonomous_niche_researcher",
        "project_archivist_biographer",
        "execution_partner",
    }
    by_id = {scenario.id: scenario for scenario in BUILTIN_DIGITAL_SCENARIOS}

    for scenario_id in required:
        action = _execution_action_for_scenario(
            by_id[scenario_id],
            "Жвуша, выполни этот класс сценариев обычным чатом",
        )

        assert action is not None
        assert action.skill_name
        assert action.summary
        assert action.safety_boundary

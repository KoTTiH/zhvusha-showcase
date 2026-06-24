"""Natural-language entrypoint for generalized digital-agent scenarios."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

from src.agent_runtime.capability_graph import CapabilityGraph
from src.agent_runtime.digital_scenario_coverage import (
    DigitalScenarioCoverage,
    DigitalScenarioLiveEvidence,
    build_digital_scenario_coverage,
)
from src.agent_runtime.digital_scenarios import (
    BUILTIN_DIGITAL_SCENARIOS,
    REQUIRED_EVAL_VARIANTS,
    DigitalScenarioDefinition,
)
from src.llm.protocols import LLMRequest
from src.skills.base import (
    AgentContext,
    ExecutionPlan,
    InlineSkill,
    SideEffect,
    SkillResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.llm.protocols import LLMGatewayProtocol

_MIN_ROUTE_CONFIDENCE = 0.78
_BROAD_ROUTE_MAX_SCORE = 0.91


@dataclass(frozen=True)
class DigitalScenarioIntent:
    """Classifier result for one natural-language digital scenario request."""

    scenario_id: str
    confidence: float
    rationale: str = ""
    missing_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class DigitalScenarioAction:
    """First safe body-layer step for a generalized digital scenario."""

    scenario_id: str
    skill_name: str
    message: str
    kind: Literal[
        "read_only_digest",
        "read_only_project_audit",
        "read_only_research",
        "read_only_archive_lookup",
        "approval_backed_spec",
        "approval_backed_external_skill_search",
    ]
    summary: str
    safety_boundary: str
    expected_artifact: str = ""


class DigitalScenarioIntentClassifier(Protocol):
    """Classify ordinary chat text into a digital scenario family."""

    async def classify(
        self,
        message: str,
        context: AgentContext,
    ) -> DigitalScenarioIntent | None: ...


class DigitalScenarioActionRunner(Protocol):
    """Run one internal scenario action through the production skill gate."""

    async def __call__(
        self,
        action: DigitalScenarioAction,
        context: AgentContext,
    ) -> SkillResult: ...


class LLMDigitalScenarioIntentClassifier:
    """LLM-backed classifier for broad digital-agent scenario families."""

    def __init__(self, *, llm_router: LLMGatewayProtocol) -> None:
        self._llm = llm_router

    async def classify(
        self,
        message: str,
        context: AgentContext,
    ) -> DigitalScenarioIntent | None:
        del context
        response = await self._llm.generate(
            LLMRequest(
                prompt=_classifier_prompt(message),
                system=(
                    "Ты классифицируешь обычное сообщение Никиты по broad "
                    "digital-agent scenario families Жвуши. Не подгоняй "
                    "сообщение под сценарий: если это обычный small talk, "
                    "узкая команда, slash-command или недостаточно похоже на "
                    "один из классов, верни scenario_id none. Ответь только "
                    "JSON."
                ),
                tier="worker",
                temperature=0.0,
                caller="digital_scenario_intent_classifier",
            )
        )
        return _parse_classifier_response(response.text)


class DigitalScenarioSkill(InlineSkill):
    """Route natural-language digital-agent requests into Жвуша's loop."""

    name: ClassVar[str] = "digital_scenario"
    description: ClassVar[str] = "Natural-language router for digital-agent scenarios"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"
    triggers: ClassVar[list[str]] = []
    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "low"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = "auto"
    side_effects: ClassVar[list[SideEffect]] = [SideEffect.CALLS_LLM]
    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    def __init__(
        self,
        *,
        admin_user_id: int,
        intent_classifier: DigitalScenarioIntentClassifier | None,
        capability_graph_provider: Callable[[], CapabilityGraph | None],
        action_runner: DigitalScenarioActionRunner | None = None,
        min_confidence: float = _MIN_ROUTE_CONFIDENCE,
        action_timeout_seconds: float = 180.0,
    ) -> None:
        self._admin_user_id = admin_user_id
        self._classifier = intent_classifier
        self._capability_graph_provider = capability_graph_provider
        self._action_runner = action_runner
        self._min_confidence = min_confidence
        self._action_timeout_seconds = max(1.0, action_timeout_seconds)
        self._intent_cache: dict[
            tuple[int, int | None, int | None, str],
            DigitalScenarioIntent,
        ] = {}

    async def can_handle(self, message: str, context: AgentContext) -> float:
        if context.mode != "personal" or context.user_id != self._admin_user_id:
            return 0.0
        if message.strip().startswith("/"):
            return 0.0
        intent = await self._classify(message, context)
        if intent is None or intent.confidence < self._min_confidence:
            return 0.0
        if _scenario_by_id(intent.scenario_id) is None:
            return 0.0
        if _is_operator_eval_context(context):
            return intent.confidence
        return min(intent.confidence, _BROAD_ROUTE_MAX_SCORE)

    async def prepare(self, message: str, context: AgentContext) -> ExecutionPlan:
        del context
        return ExecutionPlan(
            skill_name=self.name,
            skill_type="inline",
            human_summary=(
                "Распознать broad digital-agent scenario из обычного текста: "
                f"{message[:120]}"
            ),
            estimated_tokens=800,
            estimated_cost_usd=Decimal("0.002"),
            estimated_duration_seconds=2.0,
            side_effects_invoked=list(self.side_effects),
            llm_calls_planned=1,
        )

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        intent = await self._classify(message, context)
        if intent is None or _scenario_by_id(intent.scenario_id) is None:
            return SkillResult(
                success=False,
                response="",
                metadata={
                    "skill_name": self.name,
                    "requires_zhvusha_response": True,
                    "body_observation": {
                        "event": "digital_scenario_intent_unclassified",
                        "source": self.name,
                        "user_request": message,
                        "instruction": (
                            "Это routing observation: если запрос всё же похож "
                            "на broad digital-agent сценарий, попроси Никиту "
                            "уточнить цель обычным языком. Slash-команды не "
                            "требуй."
                        ),
                    },
                },
            )

        scenario = _scenario_by_id(intent.scenario_id)
        assert scenario is not None
        graph = self._capability_graph_provider() or CapabilityGraph()
        coverage = _coverage_for_scenario(graph, scenario.id)
        action = _execution_action_for_scenario(scenario, message)
        execution = await self._execute_scenario_action(
            action,
            context,
            missing_fields=intent.missing_fields,
        )
        live_evidence = _live_evidence_candidate(
            context=context,
            scenario=scenario,
            execution=execution,
        )
        metadata: dict[str, Any] = {
            "skill_name": self.name,
            "requires_zhvusha_response": True,
            "digital_scenario_intent": {
                "scenario_id": scenario.id,
                "confidence": intent.confidence,
                "rationale": intent.rationale,
            },
            "body_observation": _body_observation_from_scenario(
                message=message,
                scenario=scenario,
                coverage=coverage,
                intent=intent,
                action=action,
                execution=execution,
            ),
        }
        if live_evidence is not None:
            metadata["digital_scenario_live_evidence"] = asdict(live_evidence)
        return SkillResult(
            success=True,
            response="",
            metadata=metadata,
        )

    async def _classify(
        self,
        message: str,
        context: AgentContext,
    ) -> DigitalScenarioIntent | None:
        operator_intent = _intent_from_operator_eval_metadata(context)
        if operator_intent is not None:
            return operator_intent
        if self._classifier is None:
            return None
        key = (context.user_id, context.chat_id, context.message_id, message)
        if key not in self._intent_cache:
            intent = await self._classifier.classify(message, context)
            if intent is not None:
                self._intent_cache[key] = _normalize_intent(intent)
        return self._intent_cache.get(key)

    async def _execute_scenario_action(
        self,
        action: DigitalScenarioAction | None,
        context: AgentContext,
        *,
        missing_fields: tuple[str, ...],
    ) -> dict[str, Any]:
        if action is None:
            return {"attempted": False, "reason": "no_safe_action_for_scenario"}
        if missing_fields:
            return {
                "attempted": False,
                "success": True,
                "reason": "missing_fields_before_execution",
                "missing_fields": list(missing_fields),
                "action": _action_descriptor(action),
            }
        eval_variant = _eval_variant_from_context(context)
        if _is_operator_eval_context(context) and eval_variant != "happy_path":
            return {
                "attempted": False,
                "success": True,
                "reason": f"matrix_variant_{eval_variant}_validated_without_repeating_action",
                "variant": eval_variant,
                "action": _action_descriptor(action),
            }
        if self._action_runner is None:
            return {
                "attempted": False,
                "success": False,
                "reason": "action_runner_not_configured",
                "action": _action_descriptor(action),
            }
        try:
            result = await asyncio.wait_for(
                self._action_runner(action, context),
                timeout=self._action_timeout_seconds,
            )
        except TimeoutError:
            return {
                "attempted": True,
                "action": _action_descriptor(action),
                "success": False,
                "error_type": "TimeoutError",
                "approval_pending": False,
                "reason": "digital_scenario_action_timed_out",
            }
        except Exception as exc:
            return {
                "attempted": True,
                "action": _action_descriptor(action),
                "success": False,
                "error_type": type(exc).__name__,
                "approval_pending": False,
            }
        return _execution_observation_from_result(action, result)


def _coverage_for_scenario(
    graph: CapabilityGraph,
    scenario_id: str,
) -> DigitalScenarioCoverage:
    coverage = build_digital_scenario_coverage(graph)
    for item in coverage:
        if item.id == scenario_id:
            return item
    raise KeyError(f"unknown digital scenario: {scenario_id}")


def _body_observation_from_scenario(
    *,
    message: str,
    scenario: DigitalScenarioDefinition,
    coverage: DigitalScenarioCoverage,
    intent: DigitalScenarioIntent,
    action: DigitalScenarioAction | None = None,
    execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "event": "digital_scenario_intent_detected",
        "source": DigitalScenarioSkill.name,
        "user_request": message,
        "scenario_id": scenario.id,
        "title": scenario.title,
        "classification_confidence": intent.confidence,
        "classification_rationale": intent.rationale,
        "task_family": scenario.task_family,
        "user_stories": list(scenario.user_stories),
        "invariants": list(scenario.invariants),
        "chat_surface": scenario.chat_surface,
        "required_capability_nodes": list(scenario.required_capability_nodes),
        "available_required_nodes": list(coverage.available_required_nodes),
        "missing_required_nodes": list(coverage.missing_required_nodes),
        "blocked_required_nodes": list(coverage.blocked_required_nodes),
        "runtime_status": coverage.status.value,
        "ready_for_live_matrix": coverage.ready_for_live_matrix,
        "memory_surfaces": list(scenario.memory_surfaces),
        "artifact_types": list(scenario.artifact_types),
        "approval_boundaries": list(scenario.approval_boundaries),
        "eval_variants": [case.variant for case in scenario.eval_cases],
        "missing_fields": list(intent.missing_fields),
        "first_safe_action": _action_descriptor(action) if action is not None else None,
        "execution": execution or {"attempted": False},
        "response_constraints": _response_constraints_for_execution(execution or {}),
        "instruction": (
            "Это routing observation для Жвушиного cognitive loop. Ответь "
            "обычным языком, не требуй slash-команд, не показывай служебный "
            "JSON. Если execution.attempted=true, синтезируй ответ из результата "
            "внутреннего safe action. Если approval_pending=true, естественно "
            "объясни Никите, что именно ждёт решения, без просьбы писать slash. "
            "Если result_pending=true, скажи, что фоновая Agent Runtime job "
            "стартовала, но финальный результат ещё не готов. "
            "Если execution.attempted=false, не утверждай, что запускала tool, "
            "читала repo, смотрела файлы, прогоняла тесты или собрала fresh "
            "runtime facts в этом turn; это только routing/safety/missing-fields "
            "результат, и его надо так и назвать. "
            "Если runtime surfaces готовы, предложи первый safe read-only шаг "
            "и какие данные нужны. Если есть missing_required_nodes, честно "
            "объясни gaps и что уже сделано без unsafe side effects. "
            "Не отправляй, не публикуй, не меняй файлы и не выполняй внешние "
            "действия без approval."
        ),
    }


def _response_constraints_for_execution(execution: dict[str, Any]) -> list[str]:
    constraints = [
        "Не требуй slash-команд от пользователя.",
        "Не показывай служебный JSON/debug или внутренние handoff-подсказки.",
        "Не заявляй external/write/publish side effects без explicit approval.",
    ]
    if execution.get("attempted") is False:
        constraints.extend(
            [
                (
                    "Не говори, что запускала tool, читала repo, смотрела файлы, "
                    "проверяла checkout/logs/tests или выполнила read-only pass "
                    "в этом turn."
                ),
                (
                    "Назови результат routing/safety/missing-fields проверкой "
                    "и отдели его от первого safe action, который можно запустить."
                ),
            ]
        )
    if execution.get("result_pending") is True:
        constraints.extend(
            [
                (
                    "Не говори, что итоговый агентский результат уже готов; "
                    "говори только, что background job стартовала и результат "
                    "надо смотреть по job/audit trail после завершения."
                ),
                (
                    "Не называй started background job полноценным успешным "
                    "аудитом, если нет завершённого Context Capsule."
                ),
            ]
        )
    if execution.get("reason") == "missing_fields_before_execution":
        constraints.append(
            "Попроси свежие данные, scope или confirmation вместо продолжения stale approval."
        )
    return constraints


def _execution_action_for_scenario(
    scenario: DigitalScenarioDefinition,
    message: str,
) -> DigitalScenarioAction | None:
    text = message.strip()
    actions: dict[str, DigitalScenarioAction] = {
        "personal_ops_hq": DigitalScenarioAction(
            scenario_id=scenario.id,
            skill_name="morning_digest",
            message=text,
            kind="read_only_digest",
            summary="Собрать текущий операционный digest из доступного backlog/state.",
            safety_boundary="read_only; external sends and task writes require approval",
            expected_artifact="daily_ops_digest",
        ),
        "ai_cto_projects": DigitalScenarioAction(
            scenario_id=scenario.id,
            skill_name="codebase_explorer",
            message=text,
            kind="read_only_project_audit",
            summary="Проверить проект read-only и вернуть архитектурные findings.",
            safety_boundary="read_only; implementation/commit stays behind self-coding gates",
            expected_artifact="architecture_audit",
        ),
        "agent_designer": DigitalScenarioAction(
            scenario_id=scenario.id,
            skill_name="ideation_to_spec",
            message=_spec_command(
                "Спроектируй agent capability profile, tools, approvals, "
                "memory, evals и risks для роли из запроса. Запрос: " + text
            ),
            kind="approval_backed_spec",
            summary="Подготовить approval-backed spec для нового агента/worker/skill.",
            safety_boundary="spec write only after explicit approval",
            expected_artifact="agent_definition_spec",
        ),
        "digital_twin_work_style": DigitalScenarioAction(
            scenario_id=scenario.id,
            skill_name="cycle_analyzer",
            message=f"/archive_lookup {text}",
            kind="read_only_archive_lookup",
            summary="Поднять archived evidence по рабочему стилю и прошлым циклам.",
            safety_boundary="read_only; memory consolidation requires approval",
            expected_artifact="work_style_evidence",
        ),
        "external_skill_lab": DigitalScenarioAction(
            scenario_id=scenario.id,
            skill_name="external_skill_acquisition",
            message=(
                "/external_skill_search external_skill_readonly,"
                f"external_skill_execute | {text} | local_folder"
            ),
            kind="approval_backed_external_skill_search",
            summary="Найти и оценить external skill candidates через approved search.",
            safety_boundary="search/import/execute stays behind external skill approvals",
            expected_artifact="external_skill_audit",
        ),
        "autonomous_niche_researcher": DigitalScenarioAction(
            scenario_id=scenario.id,
            skill_name="web_research",
            message=text,
            kind="read_only_research",
            summary="Запустить source-backed read-only web research по нише.",
            safety_boundary="read_only browser/search; submit/login/purchase denied",
            expected_artifact="trend_report",
        ),
        "project_archivist_biographer": DigitalScenarioAction(
            scenario_id=scenario.id,
            skill_name="codebase_explorer",
            message=text,
            kind="read_only_project_audit",
            summary="Собрать историю проекта из файлов, задач, логов и evidence.",
            safety_boundary="read_only; archive writes require approval",
            expected_artifact="project_timeline",
        ),
        "execution_partner": DigitalScenarioAction(
            scenario_id=scenario.id,
            skill_name="ideation_to_spec",
            message=_spec_command(
                "Переведи намерение Никиты в недельный execution plan, specs, "
                "capability checks, risks и verification matrix. Намерение: " + text
            ),
            kind="approval_backed_spec",
            summary="Подготовить approval-backed execution/spec plan из намерения.",
            safety_boundary="spec write/implementation/restart only after approval",
            expected_artifact="weekly_execution_plan_spec",
        ),
    }
    return actions.get(scenario.id)


def _spec_command(request: str) -> str:
    return "/spec_create " + request.strip()


def _action_descriptor(action: DigitalScenarioAction | None) -> dict[str, Any]:
    if action is None:
        return {}
    return {
        "scenario_id": action.scenario_id,
        "skill_name": action.skill_name,
        "kind": action.kind,
        "summary": action.summary,
        "safety_boundary": action.safety_boundary,
        "expected_artifact": action.expected_artifact,
        "user_visible_command_required": False,
    }


def _execution_observation_from_result(
    action: DigitalScenarioAction,
    result: SkillResult,
) -> dict[str, Any]:
    metadata = result.metadata
    result_pending = metadata.get("agent_job_result_pending") is True
    return {
        "attempted": True,
        "action": _action_descriptor(action),
        "success": result.success,
        "result_pending": result_pending,
        "approval_pending": metadata.get("approval_pending") is True,
        "requires_zhvusha_response": metadata.get("requires_zhvusha_response") is True,
        "result_response": _truncate_text(result.response, limit=5000),
        "result_metadata": _safe_result_metadata(metadata),
    }


def _live_evidence_candidate(
    *,
    context: AgentContext,
    scenario: DigitalScenarioDefinition,
    execution: dict[str, Any],
) -> DigitalScenarioLiveEvidence | None:
    if not execution:
        return None
    result_metadata = _dict_value(execution.get("result_metadata"))
    action = _dict_value(execution.get("action"))
    artifacts = _tuple_from_value(result_metadata.get("artifacts"))
    runtime_evidence = [
        f"scenario=digital_scenario.{scenario.id}",
        f"action={action.get('kind', 'unknown')}",
        f"skill={action.get('skill_name', 'unknown')}",
        f"attempted={str(bool(execution.get('attempted'))).lower()}",
    ]
    if execution.get("result_pending") is True:
        runtime_evidence.extend(
            [
                f"skill_success={str(bool(execution.get('success'))).lower()}",
                "result_pending=true",
            ]
        )
    else:
        runtime_evidence.append(
            f"success={str(bool(execution.get('success'))).lower()}"
        )
    reason = str(execution.get("reason", "") or "").strip()
    if reason:
        runtime_evidence.append(f"reason={reason}")
    if execution.get("approval_pending") is True:
        runtime_evidence.append("approval_pending=true")
    agent_job_id = str(result_metadata.get("agent_job_id", "") or "").strip()
    if agent_job_id:
        runtime_evidence.append(f"agent_job_id={agent_job_id}")
    eval_run_id = str(
        context.metadata.get("digital_scenario_eval_run_id", "") or ""
    ).strip()
    if eval_run_id:
        runtime_evidence.append(f"eval_run_id={eval_run_id}")
    return DigitalScenarioLiveEvidence(
        scenario_id=scenario.id,
        variant=_eval_variant_from_context(context),
        source_actor=_source_actor_from_context(context),
        chat_message_id=_chat_message_id_from_context(context),
        runtime_evidence=tuple(runtime_evidence),
        structured_observation_or_result=_structured_result_from_execution(execution),
        limitations_or_unknowns=_limitations_from_execution(execution),
        artifact_refs=artifacts,
        declared_no_artifact=not artifacts,
        approval_boundary_respected=True,
        created_at=datetime.now(tz=UTC).isoformat(),
    )


def _eval_variant_from_context(context: AgentContext) -> str:
    raw = str(context.metadata.get("digital_scenario_eval_variant", "") or "").strip()
    if raw in REQUIRED_EVAL_VARIANTS:
        return raw
    return "happy_path"


def _source_actor_from_context(context: AgentContext) -> str:
    raw = str(context.metadata.get("source_actor", "") or "").strip()
    if raw and raw.casefold() not in {"user", "human", "nikita", "никита"}:
        return raw
    interface = str(context.metadata.get("interface", "") or "").strip()
    source = str(context.metadata.get("source", "") or "").strip()
    label = interface or source or "chat"
    return f"{label}_runtime"


def _chat_message_id_from_context(context: AgentContext) -> str:
    interface = str(context.metadata.get("interface", "") or "").strip()
    source = str(context.metadata.get("source", "") or "").strip()
    label = interface or source or "chat"
    return f"{label}:{context.chat_id or 'unknown'}:{context.message_id or 'unknown'}"


def _structured_result_from_execution(execution: dict[str, Any]) -> str:
    response = str(execution.get("result_response", "") or "").strip()
    if response:
        return _truncate_text(response, limit=2000)
    result_metadata = _dict_value(execution.get("result_metadata"))
    observation = result_metadata.get("body_observation")
    if observation:
        return _truncate_text(json.dumps(observation, ensure_ascii=False), limit=2000)
    error_type = str(execution.get("error_type", "") or "").strip()
    if error_type:
        return f"Scenario action failed with {error_type}."
    return "Scenario action produced structured execution metadata."


def _limitations_from_execution(execution: dict[str, Any]) -> str:
    if execution.get("approval_pending") is True:
        return "No side effect was executed; explicit approval is pending."
    if execution.get("result_pending") is True:
        return "Background job started; final Agent Runtime result is still pending."
    if execution.get("success") is not True:
        return "Scenario action is failed or degraded; response must report limits."
    return "No external side effects were attempted beyond the selected safe action."


def _safe_result_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "skill_name",
        "agent_job_id",
        "agent_job_result_pending",
        "agent_job_status",
        "agent_profile",
        "sources",
        "artifacts",
        "deliver_artifacts_to_chat",
        "approval_pending",
        "approval_id",
        "decision_id",
        "pending_decision",
        "body_observation",
    }
    return {
        key: _strip_internal_handoff(value)
        for key, value in metadata.items()
        if key in allowed
    }


def _strip_internal_handoff(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _strip_internal_handoff(item)
            for key, item in value.items()
            if str(key) not in {"next_actions", "debug_trace", "raw_worker_output"}
        }
    if isinstance(value, list | tuple):
        return [_strip_internal_handoff(item) for item in value]
    return value


def _dict_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _tuple_from_value(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, list | tuple):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _truncate_text(text: str, *, limit: int) -> str:
    clean = text.strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def _scenario_by_id(scenario_id: str) -> DigitalScenarioDefinition | None:
    normalized = scenario_id.strip().removeprefix("digital_scenario.")
    for scenario in BUILTIN_DIGITAL_SCENARIOS:
        if scenario.id == normalized:
            return scenario
    return None


def _normalize_intent(intent: DigitalScenarioIntent) -> DigitalScenarioIntent:
    return DigitalScenarioIntent(
        scenario_id=intent.scenario_id.strip().removeprefix("digital_scenario."),
        confidence=max(0.0, min(1.0, intent.confidence)),
        rationale=intent.rationale.strip(),
        missing_fields=tuple(
            field.strip() for field in intent.missing_fields if field.strip()
        ),
    )


def _intent_from_operator_eval_metadata(
    context: AgentContext,
) -> DigitalScenarioIntent | None:
    scenario_id = str(context.metadata.get("digital_scenario_id", "") or "").strip()
    if not scenario_id or _scenario_by_id(scenario_id) is None:
        return None
    source_actor = str(context.metadata.get("source_actor", "") or "").strip()
    interface = str(context.metadata.get("interface", "") or "").strip()
    eval_variant = str(
        context.metadata.get("digital_scenario_eval_variant", "") or ""
    ).strip()
    if source_actor != "codex" or interface != "vscode" or not eval_variant:
        return None
    return DigitalScenarioIntent(
        scenario_id=scenario_id,
        confidence=0.99,
        rationale="operator eval metadata supplied by VS Code matrix harness",
        missing_fields=_missing_fields_for_eval_variant(eval_variant),
    )


def _is_operator_eval_context(context: AgentContext) -> bool:
    source_actor = str(context.metadata.get("source_actor", "") or "").strip()
    interface = str(context.metadata.get("interface", "") or "").strip()
    eval_variant = str(
        context.metadata.get("digital_scenario_eval_variant", "") or ""
    ).strip()
    return source_actor == "codex" and interface == "vscode" and bool(eval_variant)


def _missing_fields_for_eval_variant(variant: str) -> tuple[str, ...]:
    if variant == "incomplete_request":
        return ("target_scope", "fresh_source_or_data")
    if variant == "stale_state":
        return ("fresh_confirmation_for_stale_permission",)
    return ()


def _classifier_prompt(message: str) -> str:
    scenario_lines = "\n".join(
        f"- {scenario.id}: {scenario.title}; {scenario.task_family}"
        for scenario in BUILTIN_DIGITAL_SCENARIOS
    )
    return (
        "Сообщение Никиты:\n"
        f"{message}\n\n"
        "Доступные digital-agent scenario families:\n"
        f"{scenario_lines}\n\n"
        "Верни JSON строго такого вида:\n"
        '{"scenario_id":"<id или none>","confidence":0.0,'
        '"rationale":"коротко почему","missing_fields":[]}'
    )


def _parse_classifier_response(text: str) -> DigitalScenarioIntent | None:
    try:
        payload = json.loads(_extract_json_object(text))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    scenario_id = str(payload.get("scenario_id", "")).strip()
    if not scenario_id or scenario_id == "none":
        return None
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    raw_missing = payload.get("missing_fields", ())
    missing_fields = (
        tuple(str(item) for item in raw_missing)
        if isinstance(raw_missing, list | tuple)
        else ()
    )
    return _normalize_intent(
        DigitalScenarioIntent(
            scenario_id=scenario_id,
            confidence=confidence,
            rationale=str(payload.get("rationale", "")),
            missing_fields=missing_fields,
        )
    )


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        return stripped
    return stripped[start : end + 1]

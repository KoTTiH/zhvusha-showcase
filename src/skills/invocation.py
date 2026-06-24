"""Central skill invocation gate for Telegram dispatcher calls.

This module is the shared ``can_handle -> prepare -> approval -> execute``
contract for user-facing skill execution. It keeps side-effect approvals out of
individual chat branches while preserving the existing v4 ``BaseSkill`` API.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

import structlog

from src.dialogue.decisions import (
    PendingDecision,
    resolution_from_approval_signal,
    should_defer_to_cognitive_loop,
)
from src.llm.protocols import LLMGatewayProtocol, LLMRequest
from src.skills.base import AgentContext, BaseSkill, ExecutionPlan, SkillResult

ApprovalVerdict = Literal["yes", "no", "later", "ambiguous"]
ApprovalClassifier = Callable[[str], Awaitable[ApprovalVerdict]]
SkillAllowedPredicate = Callable[
    [str, Literal["personal", "assistant", "social"]], bool
]
HIGH_CONFIDENCE_ROUTE_SCORE = 0.92
MIN_CLASSIFIER_ROUTE_CONFIDENCE = 0.70
_ROUTE_PROMPT_MAX_MESSAGE_CHARS = 2_000
_ROUTE_PROMPT_MAX_CONTEXT_CHARS = 5_000
_CODEX_OPERATOR_ACTORS = {"codex", "codex_operator", "operator"}
_CODEX_GOAL_LOOP_OPERATOR_KINDS = {
    "goal_loop_handoff",
    "goal_loop_proof_replay",
}
logger = structlog.get_logger(__name__)


class SkillApprovalStore(Protocol):
    """Storage contract for pending skill decisions."""

    def get(self, context: AgentContext) -> PendingSkillDecision | None: ...
    def put(self, decision: PendingSkillDecision) -> None: ...
    def pop(self, context: AgentContext) -> PendingSkillDecision | None: ...
    def clear(self) -> None: ...


@dataclass(frozen=True)
class PendingSkillDecision:
    """A side-effecting skill invocation waiting for Zhvusha's decision."""

    decision: PendingDecision
    skill_name: str
    message: str
    plan: ExecutionPlan
    user_id: int
    chat_id: int | None
    mode: Literal["personal", "assistant", "social"]
    message_id: int | None = None
    context_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def approval_id(self) -> str:
        """Compatibility bridge for existing ToolGateway approval ids."""
        return self.decision.decision_id


PendingSkillApproval = PendingSkillDecision


@dataclass(frozen=True)
class SkillInvocationOutcome:
    """Result of one dispatcher-level skill invocation attempt."""

    handled: bool
    result: SkillResult | None = None


@dataclass(frozen=True)
class SkillRouteCandidate:
    """Bounded skill description exposed to the worker-tier route classifier."""

    name: str
    description: str
    skill_type: Literal["inline", "delegated", "background"]
    approval_policy: Literal["auto", "required", "mode_dependent"]
    side_effects: tuple[str, ...] = ()
    mode_tags: tuple[str, ...] = ()
    deterministic_score: float = 0.0


@dataclass(frozen=True)
class SkillRouteDecision:
    """Classifier decision that may promote a low-confidence skill route."""

    skill_name: str
    confidence: float
    rationale: str = ""
    normalized_action: dict[str, Any] = field(default_factory=dict)


class SkillRouteClassifier(Protocol):
    """Worker-tier intent router that only selects a skill, never executes it."""

    async def classify(
        self,
        *,
        message: str,
        context: AgentContext,
        candidates: Sequence[SkillRouteCandidate],
    ) -> SkillRouteDecision | None: ...


@dataclass(frozen=True)
class _SelectedSkill:
    skill: BaseSkill
    score: float
    context: AgentContext


class LLMSkillRouteClassifier:
    """LLM-backed broad skill router for ambiguous chat-first tasks."""

    def __init__(self, *, llm_router: LLMGatewayProtocol) -> None:
        self._llm = llm_router

    async def classify(
        self,
        *,
        message: str,
        context: AgentContext,
        candidates: Sequence[SkillRouteCandidate],
    ) -> SkillRouteDecision | None:
        if not candidates:
            return None
        try:
            response = await self._llm.generate(
                LLMRequest(
                    prompt=_route_classifier_prompt(
                        message=message,
                        context=context,
                        candidates=candidates,
                    ),
                    system=_ROUTE_CLASSIFIER_SYSTEM,
                    tier="worker",
                    temperature=0.0,
                    caller="skill_route_classifier",
                )
            )
        except Exception:
            logger.exception("skill_route_classifier_failed")
            return None
        return _parse_route_decision(response.text, candidates)


class InMemorySkillApprovalStore:
    """Process-local pending approval store.

    It is intentionally small and replaceable. The architectural boundary is
    the ``SkillApprovalStore`` protocol; a Redis/DB implementation can be
    dropped in without changing dispatcher semantics.
    """

    def __init__(self) -> None:
        self._pending: dict[tuple[int, int | None, str], PendingSkillDecision] = {}

    def get(self, context: AgentContext) -> PendingSkillDecision | None:
        return self._pending.get(_approval_key(context))

    def put(self, decision: PendingSkillDecision) -> None:
        self._pending[(decision.user_id, decision.chat_id, decision.mode)] = decision

    def pop(self, context: AgentContext) -> PendingSkillDecision | None:
        return self._pending.pop(_approval_key(context), None)

    def clear(self) -> None:
        self._pending.clear()


class SkillInvocationService:
    """Shared dispatcher service for skill routing, approval and execution."""

    def __init__(
        self,
        *,
        approval_store: SkillApprovalStore,
        approval_classifier: ApprovalClassifier,
        is_skill_allowed: SkillAllowedPredicate,
        route_classifier: SkillRouteClassifier | None = None,
    ) -> None:
        self._approval_store = approval_store
        self._approval_classifier = approval_classifier
        self._is_skill_allowed = is_skill_allowed
        self._route_classifier = route_classifier

    def set_route_classifier(
        self,
        route_classifier: SkillRouteClassifier | None,
    ) -> None:
        """Install or clear the worker-tier classifier after LLM init."""
        self._route_classifier = route_classifier

    async def dispatch(
        self,
        message: str,
        context: AgentContext,
        skills: Sequence[BaseSkill],
    ) -> SkillInvocationOutcome:
        """Resolve pending approvals first, otherwise route a fresh message."""

        pending = self._approval_store.get(context)
        if pending is not None:
            return await self._resolve_pending(message, context, skills, pending)
        return await self.invoke_command(message, context, skills)

    async def invoke_command(
        self,
        message: str,
        context: AgentContext,
        skills: Sequence[BaseSkill],
    ) -> SkillInvocationOutcome:
        """Route a command/message through the central skill gate."""

        selected = await self._select_skill_with_score(message, context, skills)
        if selected is None:
            return SkillInvocationOutcome(handled=False)
        skill = selected.skill
        selected_context = selected.context

        if self._requires_approval(skill, message, selected_context):
            return await self._request_approval(skill, message, selected_context)

        return SkillInvocationOutcome(
            handled=True,
            result=await skill.execute(message, selected_context),
        )

    async def invoke_named_skill(
        self,
        message: str,
        context: AgentContext,
        skills: Sequence[BaseSkill],
        skill_name: str,
    ) -> SkillInvocationOutcome:
        """Invoke a specific skill through the same production gate."""
        skill = _find_skill(skills, skill_name)
        if skill is None:
            return SkillInvocationOutcome(handled=False)
        if not self._is_skill_allowed(skill.name, context.mode):
            return SkillInvocationOutcome(handled=False)
        if self._requires_approval(skill, message, context):
            return await self._request_approval(skill, message, context)
        return SkillInvocationOutcome(
            handled=True,
            result=await skill.execute(message, context),
        )

    async def _resolve_pending(
        self,
        message: str,
        context: AgentContext,
        skills: Sequence[BaseSkill],
        pending: PendingSkillDecision,
    ) -> SkillInvocationOutcome:
        if _looks_like_codex_goal_loop_operator_message(message, context):
            return await self.invoke_command(message, context, skills)
        if should_defer_to_cognitive_loop(message):
            revised = await self._try_revise_pending_decision(
                message,
                context,
                skills,
                pending,
            )
            if revised is not None:
                return revised

        superseded = await self._try_supersede_pending_with_new_task(
            message,
            context,
            skills,
            pending,
        )
        if superseded is not None:
            return superseded

        verdict = await self._approval_classifier(message)
        resolution = resolution_from_approval_signal(
            verdict,
            pending.decision,
            user_message=message,
        )

        if resolution.outcome == "ask_more":
            revised = await self._try_revise_pending_decision(
                message,
                context,
                skills,
                pending,
            )
            if revised is not None:
                return revised
            return SkillInvocationOutcome(
                handled=True,
                result=SkillResult(
                    success=True,
                    response=(
                        "Жду решения по действию:\n"
                        f"{pending.plan.human_summary}\n\n"
                        "Можно разрешить, отменить, отложить, попросить правку "
                        "или спросить детали."
                    ),
                    metadata={
                        "approval_pending": True,
                        "approval_id": pending.approval_id,
                        "decision_id": pending.decision.decision_id,
                        "pending_decision": pending.decision.model_dump(mode="json"),
                        "decision_resolution": resolution.model_dump(mode="json"),
                        "skill_name": pending.skill_name,
                    },
                ),
            )

        if resolution.outcome in {"reject", "defer"}:
            self._approval_store.pop(context)
            return SkillInvocationOutcome(
                handled=True,
                result=SkillResult(
                    success=True,
                    response=(
                        "Не выполняю это действие сейчас."
                        if resolution.outcome == "defer"
                        else "Не выполняю это действие."
                    ),
                    metadata={
                        "approval_rejected": True,
                        "approval_id": pending.approval_id,
                        "decision_id": pending.decision.decision_id,
                        "decision_resolution": resolution.model_dump(mode="json"),
                        "skill_name": pending.skill_name,
                        "dialogue_state_patch": {
                            "selected_skill": pending.skill_name,
                            "last_result": f"decision_{resolution.outcome}",
                            "clear_pending_action": True,
                            "source": "skill_approval.rejected",
                        },
                    },
                ),
            )

        self._approval_store.pop(context)
        skill = _find_skill(skills, pending.skill_name)
        if skill is None:
            return SkillInvocationOutcome(
                handled=True,
                result=SkillResult(
                    success=False,
                    response=(
                        "Не могу выполнить подтверждённое действие: "
                        f"skill `{pending.skill_name}` больше не зарегистрирован."
                    ),
                ),
            )

        approved_context = replace(
            context,
            metadata={
                **pending.context_metadata,
                **context.metadata,
                "skill_approval_id": pending.approval_id,
                "skill_approval_granted": True,
                "approved_skill_name": pending.skill_name,
            },
        )
        try:
            result = await skill.execute(pending.message, approved_context)
        except Exception as exc:
            logger.exception(
                "approved_skill_execute_failed",
                approval_id=pending.approval_id,
                skill_name=pending.skill_name,
                error_type=type(exc).__name__,
            )
            return SkillInvocationOutcome(
                handled=True,
                result=_approved_skill_failed_result(
                    pending=pending,
                    approval_message=message,
                    error=exc,
                ),
            )
        return SkillInvocationOutcome(
            handled=True,
            result=_with_approved_skill_result_metadata(
                result,
                pending=pending,
                approval_message=message,
            ),
        )

    async def _try_supersede_pending_with_new_task(
        self,
        message: str,
        context: AgentContext,
        skills: Sequence[BaseSkill],
        pending: PendingSkillDecision,
    ) -> SkillInvocationOutcome | None:
        selected = await self._select_superseding_skill_with_score(
            message,
            context,
            skills,
            pending_skill_name=pending.skill_name,
        )
        if selected is None:
            return None
        skill = selected.skill
        if selected.score < HIGH_CONFIDENCE_ROUTE_SCORE:
            return None

        self._approval_store.pop(context)
        logger.info(
            "skill_approval_superseded_by_new_task",
            approval_id=pending.approval_id,
            pending_skill=pending.skill_name,
            new_skill=skill.name,
            route_score=selected.score,
        )
        outcome = await self.invoke_command(message, context, skills)
        if outcome.result is None:
            return outcome
        return SkillInvocationOutcome(
            handled=outcome.handled,
            result=_with_superseded_pending_metadata(
                outcome.result,
                pending=pending,
                new_skill=skill.name,
                route_score=selected.score,
            ),
        )

    async def _select_superseding_skill_with_score(
        self,
        message: str,
        context: AgentContext,
        skills: Sequence[BaseSkill],
        *,
        pending_skill_name: str,
    ) -> _SelectedSkill | None:
        best_skill: BaseSkill | None = None
        best_score = 0.0
        for skill in skills:
            if skill.name in {pending_skill_name, "chat_response"}:
                continue
            if not self._is_skill_allowed(skill.name, context.mode):
                continue
            score = await skill.can_handle(message, context)
            if score > best_score:
                best_score = score
                best_skill = skill
            if score >= HIGH_CONFIDENCE_ROUTE_SCORE:
                selected_context = await self._normalize_high_confidence_route(
                    message=message,
                    context=context,
                    skill=skill,
                    score=score,
                )
                return _SelectedSkill(
                    skill=skill,
                    score=score,
                    context=selected_context,
                )
        return (
            _SelectedSkill(skill=best_skill, score=best_score, context=context)
            if best_skill is not None
            else None
        )

    async def _try_revise_pending_decision(
        self,
        message: str,
        context: AgentContext,
        skills: Sequence[BaseSkill],
        pending: PendingSkillDecision,
    ) -> SkillInvocationOutcome | None:
        skill = _find_skill(skills, pending.skill_name)
        if skill is None:
            return None
        revision_handler = getattr(skill, "revise_pending_approval", None)
        if not callable(revision_handler):
            return None
        plan = await cast("Any", revision_handler)(
            feedback=message,
            pending_message=pending.message,
            pending_plan=pending.plan,
            context=context,
        )
        if not isinstance(plan, ExecutionPlan):
            return None
        self._approval_store.pop(context)
        return await self._request_approval_for_plan(
            skill,
            message,
            context,
            plan,
        )

    async def _select_skill(
        self,
        message: str,
        context: AgentContext,
        skills: Sequence[BaseSkill],
    ) -> BaseSkill | None:
        selected = await self._select_skill_with_score(message, context, skills)
        if selected is None:
            return None
        return selected.skill

    async def _select_skill_with_score(
        self,
        message: str,
        context: AgentContext,
        skills: Sequence[BaseSkill],
    ) -> _SelectedSkill | None:
        if _looks_like_codex_goal_loop_operator_message(message, context):
            return _codex_goal_loop_operator_chat_selection(
                context=context,
                skills=skills,
                is_skill_allowed=self._is_skill_allowed,
            )

        best_skill: BaseSkill | None = None
        best_score = 0.0
        candidate_scores: list[tuple[BaseSkill, float]] = []
        for skill in skills:
            if not self._is_skill_allowed(skill.name, context.mode):
                continue
            score = await skill.can_handle(message, context)
            candidate_scores.append((skill, score))
            if score > best_score:
                best_score = score
                best_skill = skill
            if score >= HIGH_CONFIDENCE_ROUTE_SCORE:
                selected_context = await self._normalize_high_confidence_route(
                    message=message,
                    context=context,
                    skill=skill,
                    score=score,
                )
                return _SelectedSkill(
                    skill=skill,
                    score=score,
                    context=selected_context,
                )

        classifier_selected = await self._select_with_route_classifier(
            message=message,
            context=context,
            candidate_scores=candidate_scores,
        )
        if classifier_selected is not None:
            return classifier_selected

        return (
            _SelectedSkill(skill=best_skill, score=best_score, context=context)
            if best_skill is not None
            else None
        )

    async def _select_with_route_classifier(
        self,
        *,
        message: str,
        context: AgentContext,
        candidate_scores: Sequence[tuple[BaseSkill, float]],
    ) -> _SelectedSkill | None:
        if self._route_classifier is None:
            return None
        if _prefer_chat_response_only_blocks_classifier(message, context):
            return None
        if _looks_like_codex_goal_loop_operator_message(message, context):
            return None
        if _looks_like_codex_decision_only_goal_handoff(message):
            return None
        if message.strip().startswith("/"):
            return None

        candidate_by_name, candidates = _route_candidates_from_scores(candidate_scores)
        if not candidates:
            return None

        decision = await self._route_classifier.classify(
            message=message,
            context=context,
            candidates=tuple(candidates),
        )
        if decision is None:
            return None
        resolved = candidate_by_name.get(decision.skill_name)
        if resolved is None:
            logger.warning(
                "skill_route_classifier_unknown_skill",
                selected_skill=decision.skill_name,
            )
            return None
        confidence = _clamp_confidence(decision.confidence)
        if confidence < MIN_CLASSIFIER_ROUTE_CONFIDENCE:
            return None
        skill, _deterministic_score = resolved
        routed_context = _with_route_decision_metadata(context, decision, confidence)
        logger.info(
            "skill_route_classifier_selected",
            selected_skill=skill.name,
            confidence=confidence,
            rationale=decision.rationale,
        )
        return _SelectedSkill(skill=skill, score=confidence, context=routed_context)

    async def _normalize_high_confidence_route(
        self,
        *,
        message: str,
        context: AgentContext,
        skill: BaseSkill,
        score: float,
    ) -> AgentContext:
        if self._route_classifier is None:
            return context
        if not bool(getattr(skill, "route_classifier_always_normalize", False)):
            return context
        if _prefer_chat_response_only_blocks_classifier(message, context):
            return context
        if _looks_like_codex_goal_loop_operator_message(message, context):
            return context
        if message.strip().startswith("/"):
            return context

        decision = await self._route_classifier.classify(
            message=message,
            context=context,
            candidates=(_route_candidate_from_skill(skill, score),),
        )
        if decision is None or decision.skill_name != skill.name:
            return context
        confidence = _clamp_confidence(decision.confidence)
        if confidence < MIN_CLASSIFIER_ROUTE_CONFIDENCE:
            return context
        logger.info(
            "skill_route_classifier_normalized_high_confidence_route",
            selected_skill=skill.name,
            confidence=confidence,
            rationale=decision.rationale,
        )
        return _with_route_decision_metadata(context, decision, confidence)

    async def _request_approval(
        self,
        skill: BaseSkill,
        message: str,
        context: AgentContext,
    ) -> SkillInvocationOutcome:
        plan = await skill.prepare(message, context)
        return await self._request_approval_for_plan(skill, message, context, plan)

    async def _request_approval_for_plan(
        self,
        skill: BaseSkill,
        message: str,
        context: AgentContext,
        plan: ExecutionPlan,
    ) -> SkillInvocationOutcome:
        if plan.metadata.get("requires_user_input"):
            decision = _pending_decision_from_plan(
                skill=skill,
                plan=plan,
                context=context,
                kind="missing_required_input",
                required_consent=False,
                allowed_outcomes=("ask_more", "revise", "defer", "new_topic"),
            )
            return SkillInvocationOutcome(
                handled=True,
                result=SkillResult(
                    success=True,
                    response="",
                    metadata={
                        **plan.metadata,
                        "decision_id": decision.decision_id,
                        "pending_decision": decision.model_dump(mode="json"),
                        "body_observation": _body_observation_from_decision(
                            decision,
                            event="missing_required_input",
                        ),
                        "requires_zhvusha_response": True,
                        "skill_name": skill.name,
                    },
                ),
            )

        simulation = await skill.dry_run(plan)
        if not simulation.dependencies_available or not simulation.would_succeed:
            return SkillInvocationOutcome(
                handled=True,
                result=SkillResult(
                    success=False,
                    response="",
                    metadata={
                        "skill_name": skill.name,
                        "requires_zhvusha_response": True,
                        "body_observation": _body_observation_from_dry_run(
                            skill=skill,
                            plan=plan,
                            blockers=simulation.blockers,
                            dependencies_available=simulation.dependencies_available,
                            would_succeed=simulation.would_succeed,
                        ),
                    },
                ),
            )

        decision = _pending_decision_from_plan(
            skill=skill,
            plan=plan,
            context=context,
        )
        pending = PendingSkillDecision(
            decision=decision,
            skill_name=skill.name,
            message=message,
            plan=plan,
            user_id=context.user_id,
            chat_id=context.chat_id,
            mode=context.mode,
            message_id=context.message_id,
            context_metadata=dict(context.metadata),
        )
        self._approval_store.put(pending)
        return SkillInvocationOutcome(
            handled=True,
            result=SkillResult(
                success=True,
                response=_format_pending_decision_request(plan, decision.decision_id),
                metadata={
                    **plan.metadata,
                    "approval_pending": True,
                    "approval_id": pending.approval_id,
                    "decision_id": decision.decision_id,
                    "pending_decision": decision.model_dump(mode="json"),
                    "body_observation": _body_observation_from_decision(
                        decision,
                        event="pending_approval",
                    ),
                    "requires_zhvusha_response": True,
                    "skip_dialogue_assistant_response": True,
                    "skill_name": skill.name,
                },
            ),
        )

    @staticmethod
    def _requires_approval(
        skill: BaseSkill,
        message: str,
        context: AgentContext,
    ) -> bool:
        if context.metadata.get("skill_approval_granted", False):
            return False
        message_policy = getattr(skill, "requires_approval_for_message", None)
        if callable(message_policy) and bool(
            cast("Any", message_policy)(message, context)
        ):
            return True
        modifies = getattr(skill, "modifies", [])
        if modifies:
            return True
        policy = getattr(skill, "approval_policy", "auto")
        if policy == "required":
            return True
        return policy == "mode_dependent" and context.mode != "personal"


_ROUTE_CLASSIFIER_SYSTEM = """\
Ты worker-tier роутер skills Жвуши. Твоя задача — выбрать, какой skill должен
обработать текущее сообщение Никиты, если deterministic can_handle не дал
уверенного результата.

Правила:
- Не выполняй действие и не отвечай пользователю.
- Выбирай только один skill из списка candidates или skill_name "none".
- Не подгоняй обычный разговор, обсуждение идеи или small talk под side-effect
  workflow.
- Учитывай recent/context/body_observation: follow-up может ссылаться на
  предыдущий browser/computer-use результат, артефакт, draft, spec или pending
  workflow без явного URL/слеша.
- Если пользователь просит просто показать/прикрепить уже существующий файл по
  пути, верни none: это должен обработать обычный ответ/доставка артефакта.
- Если пользователь просит получить другой/новый артефакт из уже открытой
  страницы, прокрутить, нажать, продолжить workflow или переснять результат,
  выбирай соответствующий body skill.
- Для body skills, особенно computer_use, normalized_action должен описывать
  цель и требования к артефактам, а не только технический action.
- Для computer_use action=browser_interactive_task всегда указывай стартовый
  http(s) url. Для open-ended discovery используй публичный search URL по
  цели/нику/источникам; не оставляй url пустым, не используй about:blank и
  не используй DuckDuckGo для таких задач.
- Для потенциальных side effects всё равно выбирай skill, но не считай действие
  разрешённым: downstream prepare/dry-run/approval gate обязателен.

Верни только JSON:
{
  "skill_name": "candidate_name|none",
  "confidence": 0.0-1.0,
  "rationale": "короткая причина",
  "normalized_action": {}
}

Для computer_use normalized_action должен быть action payload, например:
{
  "action": "browser_interactive_task",
  "url": "https://example.com/test",
  "text": "пройти тест и вернуть визуальные результаты",
  "goal": "пройти тест и вернуть визуальные результаты",
  "constraints": ["do_not_login", "do_not_submit_without_approval"],
  "artifact_requirements": {
    "screenshots": "all_relevant_result_sections",
    "deliver_to_chat": "true"
  },
  "success_criteria": ["result_visible", "screenshots_attached_or_precise_blocker"],
  "risk_intent": "readonly_existing_session|credential_entry|external_submit|account_mutation|shell_command",
  "approval_scope": {
    "allowed": "read already-open session data",
    "forbidden": "password entry, purchase, send, delete, shell"
  },
  "metadata": {}
}

Примеры computer_use:
- Уже открытая сессия Steam/браузера, найти SteamID без ввода пароля:
  action=browser_interactive_task,
  url=https://steamcommunity.com/search/users/#text=<nick>,
  risk_intent=readonly_existing_session,
  constraints=["use_existing_session_only","do_not_enter_credentials"].
- Ввод пароля/2FA или нажатие sign-in: action=browser_type/browser_click,
  risk_intent=credential_entry; downstream обязан запросить approval capability login.
- Shell/terminal: action=desktop_shell_command, argv=["echo","ok"], cwd=".";
  downstream обязан запросить approval capability desktop.shell.
"""


def _route_classifier_prompt(
    *,
    message: str,
    context: AgentContext,
    candidates: Sequence[SkillRouteCandidate],
) -> str:
    candidates_payload = [
        {
            "name": candidate.name,
            "description": candidate.description,
            "skill_type": candidate.skill_type,
            "approval_policy": candidate.approval_policy,
            "side_effects": list(candidate.side_effects),
            "mode_tags": list(candidate.mode_tags),
            "deterministic_score": candidate.deterministic_score,
        }
        for candidate in candidates
    ]
    return (
        "Текущее сообщение:\n"
        f"{_bounded_prompt_text(message.strip(), _ROUTE_PROMPT_MAX_MESSAGE_CHARS)}"
        "\n\nКонтекст вызова:\n"
        f"{_route_context_for_prompt(context)}"
        "\n\nCandidates:\n"
        f"{json.dumps(candidates_payload, ensure_ascii=False, indent=2)}\n\n"
        "Верни JSON."
    )


def _route_context_for_prompt(context: AgentContext) -> str:
    selected: dict[str, Any] = {
        "user_id": context.user_id,
        "chat_id": context.chat_id,
        "mode": context.mode,
        "message_id": context.message_id,
    }
    keep_keys = {
        "source",
        "interface",
        "dialogue_context",
        "dialogue_state",
        "recent_messages",
        "recent_decision_messages",
        "body_observation",
        "body_observation_synthesis_message",
        "last_result",
        "last_skill_name",
        "skill_name",
        "artifacts",
        "delivered_artifacts",
        "agent_job_id",
        "agent_profile",
    }
    for key, value in context.metadata.items():
        if key in keep_keys or key.startswith(("last_", "pending_", "browser_")):
            selected[key] = _json_safe_prompt_value(value)
    serialized = json.dumps(selected, ensure_ascii=False, default=str)
    return _bounded_prompt_text(serialized, _ROUTE_PROMPT_MAX_CONTEXT_CHARS)


def _json_safe_prompt_value(value: Any) -> Any:
    if isinstance(value, str):
        return _bounded_prompt_text(value, 2_000)
    if isinstance(value, int | float | bool) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(key): _json_safe_prompt_value(item)
            for key, item in list(value.items())[:40]
        }
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [_json_safe_prompt_value(item) for item in list(value)[:40]]
    return _bounded_prompt_text(str(value), 1_000)


def _parse_route_decision(
    text: str,
    candidates: Sequence[SkillRouteCandidate],
) -> SkillRouteDecision | None:
    payload = _extract_json_object(text)
    if payload is None:
        logger.warning("skill_route_classifier_invalid_json")
        return None
    raw_skill = (
        payload.get("skill_name")
        or payload.get("selected_skill")
        or payload.get("skill")
        or ""
    )
    skill_name = str(raw_skill).strip()
    if not skill_name or skill_name.lower() == "none":
        return None
    allowed = {candidate.name for candidate in candidates}
    if skill_name not in allowed:
        logger.warning(
            "skill_route_classifier_rejected_unknown_skill",
            selected_skill=skill_name,
        )
        return None
    confidence = _coerce_confidence(payload.get("confidence"))
    normalized_action = _metadata_dict(payload.get("normalized_action"))
    return SkillRouteDecision(
        skill_name=skill_name,
        confidence=confidence,
        rationale=str(payload.get("rationale") or "").strip(),
        normalized_action=normalized_action,
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        payload = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _route_candidate_from_skill(skill: BaseSkill, score: float) -> SkillRouteCandidate:
    return SkillRouteCandidate(
        name=skill.name,
        description=str(getattr(skill, "description", "")).strip(),
        skill_type=getattr(skill, "skill_type", "inline"),
        approval_policy=getattr(skill, "approval_policy", "auto"),
        side_effects=tuple(
            str(effect) for effect in getattr(skill, "side_effects", [])
        ),
        mode_tags=tuple(str(mode) for mode in getattr(skill, "mode_tags", [])),
        deterministic_score=score,
    )


def _route_candidates_from_scores(
    candidate_scores: Sequence[tuple[BaseSkill, float]],
) -> tuple[dict[str, tuple[BaseSkill, float]], tuple[SkillRouteCandidate, ...]]:
    candidate_by_name: dict[str, tuple[BaseSkill, float]] = {}
    candidates: list[SkillRouteCandidate] = []
    for skill, score in candidate_scores:
        if skill.name == "chat_response":
            continue
        if skill.name in candidate_by_name:
            continue
        candidate_by_name[skill.name] = (skill, score)
        candidates.append(_route_candidate_from_skill(skill, score))
    return candidate_by_name, tuple(candidates)


def _with_route_decision_metadata(
    context: AgentContext,
    decision: SkillRouteDecision,
    confidence: float,
) -> AgentContext:
    return replace(
        context,
        metadata={
            **context.metadata,
            "skill_router_selected_skill": decision.skill_name,
            "skill_router_confidence": confidence,
            "skill_router_rationale": decision.rationale,
            "skill_router_normalized_action": dict(decision.normalized_action),
        },
    )


def _coerce_confidence(value: Any) -> float:
    try:
        return _clamp_confidence(float(value))
    except (TypeError, ValueError):
        return 0.0


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


def _bounded_prompt_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def _approval_key(context: AgentContext) -> tuple[int, int | None, str]:
    return (context.user_id, context.chat_id, context.mode)


def _find_skill(skills: Sequence[BaseSkill], name: str) -> BaseSkill | None:
    for skill in skills:
        if skill.name == name:
            return skill
    return None


def _codex_goal_loop_operator_chat_selection(
    *,
    context: AgentContext,
    skills: Sequence[BaseSkill],
    is_skill_allowed: SkillAllowedPredicate,
) -> _SelectedSkill | None:
    for skill in skills:
        if skill.name == "chat_response" and is_skill_allowed(skill.name, context.mode):
            return _SelectedSkill(skill=skill, score=1.0, context=context)
    return None


def _with_approved_skill_result_metadata(
    result: SkillResult,
    *,
    pending: PendingSkillDecision,
    approval_message: str,
) -> SkillResult:
    """Keep post-approval synthesis grounded in the original requested action."""
    return replace(
        result,
        metadata={
            **result.metadata,
            "approved_original_message": pending.message,
            "approval_response_message": approval_message,
            "body_observation_synthesis_message": pending.message,
        },
    )


def _with_superseded_pending_metadata(
    result: SkillResult,
    *,
    pending: PendingSkillDecision,
    new_skill: str,
    route_score: float,
) -> SkillResult:
    return replace(
        result,
        metadata={
            **result.metadata,
            "approval_superseded": True,
            "superseded_approval_id": pending.approval_id,
            "superseded_skill_name": pending.skill_name,
            "superseding_skill_name": new_skill,
            "superseding_route_score": route_score,
            "dialogue_state_patch": {
                **_metadata_dict(result.metadata.get("dialogue_state_patch")),
                "clear_pending_action": True,
                "source": "skill_approval.superseded_by_new_task",
            },
        },
    )


def _approved_skill_failed_result(
    *,
    pending: PendingSkillDecision,
    approval_message: str,
    error: Exception,
) -> SkillResult:
    error_type = type(error).__name__
    return SkillResult(
        success=False,
        response=(
            "не смогла выполнить подтверждённое действие: "
            f"`{pending.skill_name}` завершился ошибкой `{error_type}`. "
            "Ничего не считаю выполненным."
        ),
        metadata={
            "approved_original_message": pending.message,
            "approval_response_message": approval_message,
            "approved_skill_error": error_type,
            "body_observation_synthesis_message": pending.message,
            "skill_name": pending.skill_name,
            "dialogue_state_patch": {
                "selected_skill": pending.skill_name,
                "last_result": "failure",
                "pending_action": "",
                "recipient_hint": "",
                "executable_chat_id": "",
                "draft_message": "",
                "missing_fields": [],
                "clear_pending_action": True,
                "clear_missing_fields": True,
                "clear_executable_chat_id": True,
                "source": "skill_approval.execute_failed",
            },
        },
    )


def _pending_decision_from_plan(
    *,
    skill: BaseSkill,
    plan: ExecutionPlan,
    context: AgentContext,
    kind: str | None = None,
    required_consent: bool = True,
    allowed_outcomes: tuple[
        Literal["approve", "reject", "revise", "ask_more", "defer", "new_topic"],
        ...,
    ]
    | None = None,
) -> PendingDecision:
    side_effects = tuple(effect.value for effect in plan.side_effects_invoked)
    decision_payload: dict[str, Any] = {
        "decision_id": f"skill-approval-{uuid4().hex}",
        "kind": kind or ("external_action" if side_effects else "skill_action"),
        "owner": skill.name,
        "action": str(plan.metadata.get("telegram_mcp_action") or skill.name),
        "summary": plan.human_summary,
        "proposal": {
            "skill_name": skill.name,
            "skill_type": plan.skill_type,
            "human_summary": plan.human_summary,
            "side_effects": list(side_effects),
            "estimated_tokens": plan.estimated_tokens,
            "estimated_cost_usd": str(plan.estimated_cost_usd),
            "estimated_duration_seconds": plan.estimated_duration_seconds,
        },
        "required_consent": required_consent,
        "constraints": side_effects,
        "context_snapshot": {
            "user_id": context.user_id,
            "chat_id": context.chat_id,
            "mode": context.mode,
            "message_id": context.message_id,
        },
        "missing_fields": _metadata_missing_fields(plan.metadata),
        "metadata": dict(plan.metadata),
    }
    if allowed_outcomes is not None:
        decision_payload["allowed_outcomes"] = allowed_outcomes
    return PendingDecision(**decision_payload)


def _body_observation_from_decision(
    decision: PendingDecision,
    *,
    event: str,
) -> dict[str, object]:
    return {
        "event": event,
        "source": "skill_invocation",
        "pending_decision": decision.model_dump(mode="json"),
        "instruction": (
            "Это внутреннее наблюдение body-layer. Не показывай JSON и не "
            "копируй служебный текст. Ответ пользователю формирует Жвуша: "
            "реши, что делать дальше, и естественно спроси недостающие поля "
            "или объясни pending approval. Не выполняй side effect без "
            "physical approval/tool-gateway enforcement."
        ),
    }


def _body_observation_from_dry_run(
    *,
    skill: BaseSkill,
    plan: ExecutionPlan,
    blockers: Sequence[str],
    dependencies_available: bool,
    would_succeed: bool,
) -> dict[str, object]:
    return {
        "event": "dry_run_blocked",
        "source": "skill_invocation",
        "skill_name": skill.name,
        "summary": plan.human_summary,
        "blockers": [str(blocker) for blocker in blockers],
        "dependencies_available": dependencies_available,
        "would_succeed": would_succeed,
        "instruction": (
            "Это internal blocker из dry-run. Не показывай raw plan text как "
            "готовый ответ. Объясни пользователю, какой runtime/approval/input "
            "слой не готов и что нужно уточнить."
        ),
    }


_ACTION_INTENT_VERBS = (
    "открой",
    "открыть",
    "запусти",
    "запустить",
    "сфокусируй",
    "фокус",
    "нажми",
    "нажать",
    "напечатай",
    "напиши",
    "введи",
    "ввести",
    "поставь",
    "поставить",
    "пауза",
    "pause",
    "play",
    "open",
    "focus",
    "press",
    "type",
)
_ACTION_INTENT_OBJECTS = (
    "telegram",
    "телеграм",
    "vs code",
    "vscode",
    "visual studio code",
    "escape",
    "esc",
    "активное поле",
    "горяч",
    "клавиш",
    "музык",
    "плеер",
    "брауз",
    "browser",
    "chrome",
    "окн",
    "window",
    "сайт",
    "url",
    "http://",
    "https://",
)
_DISCUSSION_MARKERS = (
    "обсуд",
    "поговор",
    "иде",
    "план",
    "потом",
    "как лучше",
    "что думаешь",
)


def _prefer_chat_response_only_blocks_classifier(
    message: str,
    context: AgentContext,
) -> bool:
    if context.metadata.get("prefer_chat_response_only", False) is not True:
        return False
    return not _looks_like_action_intent_for_route_classifier(message)


def _looks_like_codex_decision_only_goal_handoff(message: str) -> bool:
    normalized = " ".join(message.lower().split())
    return (
        "codex/operator handoff" in normalized
        and "operator_handoff_mode: decision_only_existing_agent_evidence" in normalized
    )


def _looks_like_codex_goal_loop_operator_message(
    message: str,
    context: AgentContext,
) -> bool:
    source_actor = str(context.metadata.get("source_actor", "") or "").casefold()
    if source_actor not in _CODEX_OPERATOR_ACTORS:
        return False
    message_kind = str(context.metadata.get("operator_message_kind", "") or "")
    if message_kind in _CODEX_GOAL_LOOP_OPERATOR_KINDS:
        return True
    normalized = " ".join(message.casefold().split())
    return (
        (
            "codex/operator handoff" in normalized
            or "codex/operator proof replay" in normalized
        )
        and "sender=codex" in normalized
        and "не никита" in normalized
    )


def _looks_like_action_intent_for_route_classifier(message: str) -> bool:
    normalized = " ".join(message.lower().split())
    if not normalized:
        return False
    if any(marker in normalized for marker in _DISCUSSION_MARKERS):
        return False
    has_verb = any(verb in normalized for verb in _ACTION_INTENT_VERBS)
    has_object = any(obj in normalized for obj in _ACTION_INTENT_OBJECTS)
    return has_verb and has_object


def _metadata_missing_fields(metadata: dict[str, Any]) -> tuple[str, ...]:
    raw = metadata.get("missing_fields", ())
    if isinstance(raw, str):
        text = raw.strip()
        return (text,) if text else ()
    if isinstance(raw, Sequence):
        return tuple(str(item).strip() for item in raw if str(item).strip())
    return ()


def _metadata_dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _format_pending_decision_request(plan: ExecutionPlan, decision_id: str) -> str:
    side_effects = ", ".join(effect.value for effect in plan.side_effects_invoked)
    side_effects_text = side_effects or "нет заявленных side effects"
    cost = _format_decimal(plan.estimated_cost_usd)
    return (
        "Нужно решение перед выполнением.\n\n"
        f"Действие: {plan.human_summary}\n"
        f"Навык: `{plan.skill_name}`\n"
        f"Side effects: {side_effects_text}\n"
        f"Оценка: ~{plan.estimated_tokens} tokens, ${cost}, "
        f"{plan.estimated_duration_seconds:.0f}s\n"
        f"Decision: `{decision_id}`\n\n"
        "Можно разрешить, отменить, отложить, попросить правку или спросить детали."
    )


def _format_decimal(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.0001")), "f")

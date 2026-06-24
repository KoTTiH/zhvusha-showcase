from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import ClassVar, Literal

from src.llm.protocols import LLMRequest, LLMResponse
from src.skills.base import (
    AgentContext,
    ExecutionPlan,
    InlineSkill,
    SideEffect,
    SimulatedResult,
    SkillResult,
)
from src.skills.invocation import (
    ApprovalVerdict,
    InMemorySkillApprovalStore,
    LLMSkillRouteClassifier,
    SkillInvocationService,
    SkillRouteCandidate,
    SkillRouteClassifier,
    SkillRouteDecision,
)


class _AutoSkill(InlineSkill):
    name: ClassVar[str] = "auto_skill"
    description: ClassVar[str] = "Auto test skill"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = "auto"

    def __init__(self) -> None:
        self.executed: list[str] = []

    async def can_handle(self, message: str, context: AgentContext) -> float:
        del context
        return 1.0 if message.startswith("/auto") else 0.0

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        del context
        self.executed.append(message)
        return SkillResult(success=True, response=f"auto: {message}")


class _RequiredSkill(InlineSkill):
    name: ClassVar[str] = "required_skill"
    description: ClassVar[str] = "Required approval test skill"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = (
        "required"
    )
    side_effects: ClassVar[list[SideEffect]] = [SideEffect.POSTS_TO_CHANNEL]

    def __init__(self) -> None:
        self.executed: list[str] = []
        self.approval_metadata: list[dict[str, object]] = []

    async def can_handle(self, message: str, context: AgentContext) -> float:
        del context
        return 1.0 if message.startswith("/danger") else 0.0

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        self.executed.append(message)
        self.approval_metadata.append(context.metadata)
        return SkillResult(success=True, response=f"required: {message}")

    async def prepare(self, message: str, context: AgentContext) -> ExecutionPlan:
        del message, context
        return ExecutionPlan(
            skill_name=self.name,
            skill_type="inline",
            human_summary="Опасное действие",
            estimated_tokens=100,
            estimated_cost_usd=Decimal("0"),
            estimated_duration_seconds=1.0,
            side_effects_invoked=list(self.side_effects),
            metadata={
                "dialogue_state_patch": {
                    "pending_action": "required_action",
                    "selected_skill": self.name,
                }
            },
        )


class _NaturalRequiredSkill(_RequiredSkill):
    name: ClassVar[str] = "natural_required_skill"

    async def can_handle(self, message: str, context: AgentContext) -> float:
        del context
        return 0.93 if message.startswith("опубликуй пост") else 0.0


class _FailingRequiredSkill(_RequiredSkill):
    name: ClassVar[str] = "failing_required_skill"

    async def can_handle(self, message: str, context: AgentContext) -> float:
        del context
        return 1.0 if message.startswith("/fail") else 0.0

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        del message, context
        raise RuntimeError("catalog exploded")


class _ClarifyingRequiredSkill(_RequiredSkill):
    name: ClassVar[str] = "clarifying_required_skill"

    async def can_handle(self, message: str, context: AgentContext) -> float:
        del context
        return 1.0 if message.startswith("/clarify") else 0.0

    async def prepare(self, message: str, context: AgentContext) -> ExecutionPlan:
        del message, context
        return ExecutionPlan(
            skill_name=self.name,
            skill_type="inline",
            human_summary="Что именно нужно отправить?",
            estimated_tokens=100,
            estimated_cost_usd=Decimal("0"),
            estimated_duration_seconds=0.0,
            side_effects_invoked=list(self.side_effects),
            metadata={
                "requires_user_input": True,
                "missing_fields": ["message"],
            },
        )


class _DryRunBlockingRequiredSkill(_RequiredSkill):
    name: ClassVar[str] = "dry_run_blocking_required_skill"

    async def can_handle(self, message: str, context: AgentContext) -> float:
        del context
        return 1.0 if message.startswith("/dry-block") else 0.0

    async def dry_run(self, plan: ExecutionPlan) -> SimulatedResult:
        return SimulatedResult(
            would_succeed=False,
            would_produce="",
            dependencies_available=False,
            estimated_actual_cost=plan.estimated_cost_usd,
            blockers=["missing scoped runtime adapter", "approval contract incomplete"],
        )


class _RevisableRequiredSkill(_RequiredSkill):
    name: ClassVar[str] = "revisable_required_skill"

    async def can_handle(self, message: str, context: AgentContext) -> float:
        del context
        return 1.0 if message.startswith("/revise") else 0.0

    async def prepare(self, message: str, context: AgentContext) -> ExecutionPlan:
        del message, context
        return ExecutionPlan(
            skill_name=self.name,
            skill_type="inline",
            human_summary="Отправить исходный текст",
            estimated_tokens=100,
            estimated_cost_usd=Decimal("0"),
            estimated_duration_seconds=1.0,
            side_effects_invoked=list(self.side_effects),
        )

    async def revise_pending_approval(
        self,
        *,
        feedback: str,
        pending_message: str,
        pending_plan: ExecutionPlan,
        context: AgentContext,
    ) -> ExecutionPlan | None:
        del pending_message, pending_plan, context
        if "мягче" not in feedback.lower():
            return None
        return ExecutionPlan(
            skill_name=self.name,
            skill_type="inline",
            human_summary="Отправить текст мягче",
            estimated_tokens=100,
            estimated_cost_usd=Decimal("0"),
            estimated_duration_seconds=1.0,
            side_effects_invoked=list(self.side_effects),
        )


class _ScoredAutoSkill(InlineSkill):
    description: ClassVar[str] = "Scored auto skill"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = "auto"

    def __init__(self, *, name: str, score: float) -> None:
        self.name = name
        self._score = score
        self.can_handle_calls = 0
        self.executed = 0

    async def can_handle(self, message: str, context: AgentContext) -> float:
        del message, context
        self.can_handle_calls += 1
        return self._score

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        del message, context
        self.executed += 1
        return SkillResult(success=True, response=self.name)


class _NewTaskSkill(InlineSkill):
    name: ClassVar[str] = "new_task_skill"
    description: ClassVar[str] = "High confidence fresh task skill"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = "auto"

    def __init__(self) -> None:
        self.executed: list[str] = []

    async def can_handle(self, message: str, context: AgentContext) -> float:
        del context
        return 0.96 if message.startswith("/new-task") else 0.0

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        del context
        self.executed.append(message)
        return SkillResult(success=True, response=f"new-task: {message}")


class _PerMessageApprovalSkill(_AutoSkill):
    name: ClassVar[str] = "per_message_approval_skill"

    def requires_approval_for_message(
        self,
        message: str,
        context: AgentContext,
    ) -> bool:
        del context
        return message.startswith("опасно")

    async def can_handle(self, message: str, context: AgentContext) -> float:
        del context
        return 0.93 if message.startswith("опасно") else 0.0


class _ChatFallbackSkill(_ScoredAutoSkill):
    def __init__(self) -> None:
        super().__init__(name="chat_response", score=0.3)


class _RoutedAutoSkill(_ScoredAutoSkill):
    def __init__(self) -> None:
        super().__init__(name="computer_use", score=0.0)
        self.metadata_seen: list[dict[str, object]] = []

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        self.metadata_seen.append(context.metadata)
        return await super().execute(message, context)


class _IntentClassifierSkill(_ScoredAutoSkill):
    def __init__(self) -> None:
        super().__init__(name="telegram_mcp_personal", score=0.0)

    async def can_handle(self, message: str, context: AgentContext) -> float:
        self.can_handle_calls += 1
        raise AssertionError("operator goal-loop replay must not enter classifiers")


class _HighConfidenceRoutedAutoSkill(_RoutedAutoSkill):
    route_classifier_always_normalize: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__()
        self._score = 0.93


class _RecordingRouteClassifier(SkillRouteClassifier):
    def __init__(self, decision: SkillRouteDecision | None) -> None:
        self.decision = decision
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def classify(
        self,
        *,
        message: str,
        context: AgentContext,
        candidates: Sequence[SkillRouteCandidate],
    ) -> SkillRouteDecision | None:
        del context
        self.calls.append((message, tuple(candidate.name for candidate in candidates)))
        return self.decision


class _FakeRouteLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(text=self.text, model="fake-worker")


def _ctx(metadata: dict[str, object] | None = None) -> AgentContext:
    return AgentContext(
        user_id=1,
        chat_id=1,
        mode="personal",
        metadata=metadata or {},
    )


def _service(
    verdict: ApprovalVerdict = "yes",
    *,
    route_classifier: SkillRouteClassifier | None = None,
) -> SkillInvocationService:
    async def classify(text: str) -> ApprovalVerdict:
        del text
        return verdict

    return SkillInvocationService(
        approval_store=InMemorySkillApprovalStore(),
        approval_classifier=classify,
        is_skill_allowed=lambda _name, _mode: True,
        route_classifier=route_classifier,
    )


async def test_auto_skill_executes_immediately() -> None:
    skill = _AutoSkill()
    outcome = await _service().dispatch("/auto go", _ctx(), [skill])

    assert outcome.handled is True
    assert outcome.result is not None
    assert outcome.result.response == "auto: /auto go"
    assert skill.executed == ["/auto go"]


async def test_named_auto_skill_executes_through_gate_even_without_route_match() -> (
    None
):
    skill = _AutoSkill()
    outcome = await _service().invoke_named_skill(
        "/looks-like-command",
        _ctx(),
        [skill],
        "auto_skill",
    )

    assert outcome.handled is True
    assert outcome.result is not None
    assert outcome.result.response == "auto: /looks-like-command"
    assert skill.executed == ["/looks-like-command"]


async def test_high_confidence_skill_short_circuits_later_candidates() -> None:
    first = _ScoredAutoSkill(name="first", score=0.92)
    later = _ScoredAutoSkill(name="later", score=1.0)
    route_classifier = _RecordingRouteClassifier(
        SkillRouteDecision(skill_name="later", confidence=1.0)
    )

    outcome = await _service(route_classifier=route_classifier).dispatch(
        "normal chat",
        _ctx(),
        [first, later],
    )

    assert outcome.result is not None
    assert outcome.result.response == "first"
    assert first.can_handle_calls == 1
    assert first.executed == 1
    assert later.can_handle_calls == 0
    assert later.executed == 0
    assert route_classifier.calls == []


async def test_worker_route_classifier_can_promote_body_skill_follow_up() -> None:
    computer_use = _RoutedAutoSkill()
    chat = _ChatFallbackSkill()
    route_classifier = _RecordingRouteClassifier(
        SkillRouteDecision(
            skill_name="computer_use",
            confidence=0.86,
            rationale="follow-up asks for a new lower screenshot from live page",
            normalized_action={
                "action": "browser_scroll",
                "target": "down",
                "metadata": {"capture_screenshot": "true"},
            },
        )
    )

    outcome = await _service(route_classifier=route_classifier).dispatch(
        "Это тот же скрин, мне нужен скрин нижних результатов",
        _ctx(),
        [computer_use, chat],
    )

    assert outcome.handled is True
    assert outcome.result is not None
    assert outcome.result.response == "computer_use"
    assert computer_use.executed == 1
    assert chat.executed == 0
    assert route_classifier.calls == [
        (
            "Это тот же скрин, мне нужен скрин нижних результатов",
            ("computer_use",),
        )
    ]
    assert computer_use.metadata_seen[0]["skill_router_selected_skill"] == (
        "computer_use"
    )
    assert computer_use.metadata_seen[0]["skill_router_confidence"] == 0.86
    assert computer_use.metadata_seen[0]["skill_router_normalized_action"] == {
        "action": "browser_scroll",
        "target": "down",
        "metadata": {"capture_screenshot": "true"},
    }


async def test_worker_route_classifier_normalizes_high_confidence_body_skill() -> None:
    computer_use = _HighConfidenceRoutedAutoSkill()
    route_classifier = _RecordingRouteClassifier(
        SkillRouteDecision(
            skill_name="computer_use",
            confidence=0.91,
            rationale="browser task requires result screenshots",
            normalized_action={
                "action": "browser_interactive_task",
                "url": "https://example.com/test",
                "text": "пройти тест и вернуть все визуальные результаты",
                "goal": "пройти тест и вернуть все визуальные результаты",
                "artifact_requirements": {
                    "screenshots": "all_relevant_result_sections",
                    "deliver_to_chat": "true",
                },
            },
        )
    )

    outcome = await _service(route_classifier=route_classifier).dispatch(
        "пройди тест и верни все визуальные результаты https://example.com/test",
        _ctx(),
        [computer_use, _ChatFallbackSkill()],
    )

    assert outcome.handled is True
    assert outcome.result is not None
    assert outcome.result.response == "computer_use"
    assert computer_use.can_handle_calls == 1
    assert route_classifier.calls == [
        (
            "пройди тест и верни все визуальные результаты https://example.com/test",
            ("computer_use",),
        )
    ]
    assert computer_use.metadata_seen[0]["skill_router_normalized_action"][
        "artifact_requirements"
    ] == {
        "screenshots": "all_relevant_result_sections",
        "deliver_to_chat": "true",
    }


async def test_action_like_context_budget_preselection_still_runs_route_classifier() -> (
    None
):
    computer_use = _RoutedAutoSkill()
    chat = _ChatFallbackSkill()
    route_classifier = _RecordingRouteClassifier(
        SkillRouteDecision(
            skill_name="computer_use",
            confidence=0.88,
            rationale="desktop app launch action",
            normalized_action={
                "action": "desktop_app_launcher",
                "target": "org.telegram.desktop.desktop",
                "goal": "открыть Telegram",
            },
        )
    )
    context = AgentContext(
        user_id=1,
        chat_id=1,
        mode="personal",
        metadata={"prefer_chat_response_only": True},
    )

    outcome = await _service(route_classifier=route_classifier).dispatch(
        "открой Telegram",
        context,
        [computer_use, chat],
    )

    assert outcome.handled is True
    assert outcome.result is not None
    assert outcome.result.response == "computer_use"
    assert route_classifier.calls == [("открой Telegram", ("computer_use",))]
    assert computer_use.metadata_seen[0]["skill_router_normalized_action"] == {
        "action": "desktop_app_launcher",
        "target": "org.telegram.desktop.desktop",
        "goal": "открыть Telegram",
    }


async def test_discussion_context_budget_preselection_still_stays_chat_response() -> (
    None
):
    computer_use = _RoutedAutoSkill()
    chat = _ChatFallbackSkill()
    route_classifier = _RecordingRouteClassifier(
        SkillRouteDecision(
            skill_name="computer_use",
            confidence=0.95,
            rationale="would be wrong if called",
        )
    )
    context = AgentContext(
        user_id=1,
        chat_id=1,
        mode="personal",
        metadata={"prefer_chat_response_only": True},
    )

    outcome = await _service(route_classifier=route_classifier).dispatch(
        "обсудим Telegram, окна и горячие клавиши потом",
        context,
        [computer_use, chat],
    )

    assert outcome.handled is True
    assert outcome.result is not None
    assert outcome.result.response == "chat_response"
    assert route_classifier.calls == []
    assert computer_use.executed == 0


async def test_codex_decision_only_handoff_bypasses_route_classifier() -> None:
    computer_use = _RoutedAutoSkill()
    chat = _ChatFallbackSkill()
    route_classifier = _RecordingRouteClassifier(
        SkillRouteDecision(
            skill_name="computer_use",
            confidence=0.95,
            rationale="would wrongly turn a decision packet into a body job",
        )
    )

    outcome = await _service(route_classifier=route_classifier).dispatch(
        "\n".join(
            [
                "Codex/operator handoff, sender=codex, не Никита.",
                "operator_handoff_mode: decision_only_existing_agent_evidence",
                "Agent Runtime Job Evidence:",
                "- job_id=job-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa status=done",
            ]
        ),
        _ctx(),
        [computer_use, chat],
    )

    assert outcome.handled is True
    assert outcome.result is not None
    assert outcome.result.response == "chat_response"
    assert route_classifier.calls == []
    assert computer_use.executed == 0
    assert chat.executed == 1


async def test_codex_goal_loop_proof_replay_bypasses_pending_approval() -> None:
    required = _RequiredSkill()
    computer_use = _RoutedAutoSkill()
    chat = _ChatFallbackSkill()
    route_classifier = _RecordingRouteClassifier(
        SkillRouteDecision(
            skill_name="computer_use",
            confidence=0.95,
            rationale="operator proof replay is not a fresh body action",
        )
    )
    store = InMemorySkillApprovalStore()

    async def classify(text: str) -> ApprovalVerdict:
        del text
        return "yes"

    service = SkillInvocationService(
        approval_store=store,
        approval_classifier=classify,
        is_skill_allowed=lambda _name, _mode: True,
        route_classifier=route_classifier,
    )

    pending = await service.dispatch("/danger publish", _ctx(), [required, chat])
    assert pending.handled is True
    assert pending.result is not None
    assert pending.result.metadata["approval_pending"] is True

    proof_replay = "\n".join(
        [
            "Codex/operator proof replay, sender=codex, не Никита.",
            "Это продолжение active goal loop, не новое сообщение пользователя.",
            "No-Write Proof Bundle:",
            "- status=complete",
            "Decision request:",
            "Выбери approve_exact_implementation или blocked_by_dirty_target_or_missing_capability.",
        ]
    )
    replay = await service.dispatch(
        proof_replay,
        _ctx(
            {
                "source": "vscode",
                "interface": "vscode",
                "source_actor": "codex",
                "operator_message_kind": "goal_loop_proof_replay",
            }
        ),
        [required, computer_use, chat],
    )

    assert replay.handled is True
    assert replay.result is not None
    assert replay.result.response == "chat_response"
    assert route_classifier.calls == []
    assert computer_use.executed == 0
    assert required.executed == []

    approved = await service.dispatch("да", _ctx(), [required, chat])

    assert approved.handled is True
    assert required.executed == ["/danger publish"]


async def test_codex_goal_loop_operator_message_skips_skill_can_handle_classifiers() -> (
    None
):
    classifier_skill = _IntentClassifierSkill()
    chat = _ChatFallbackSkill()

    replay = await _service().dispatch(
        "\n".join(
            [
                "Codex/operator proof replay, sender=codex, не Никита.",
                "No-Write Proof Bundle:",
                "- status=complete",
                "Decision request: approve_exact_implementation или blocked_by_dirty.",
            ]
        ),
        _ctx(
            {
                "source": "vscode",
                "interface": "vscode",
                "source_actor": "codex",
                "operator_message_kind": "goal_loop_proof_replay",
            }
        ),
        [classifier_skill, chat],
    )

    assert replay.handled is True
    assert replay.result is not None
    assert replay.result.response == "chat_response"
    assert classifier_skill.can_handle_calls == 0
    assert chat.executed == 1


async def test_worker_route_classifier_none_falls_back_to_chat_response() -> None:
    computer_use = _RoutedAutoSkill()
    chat = _ChatFallbackSkill()
    route_classifier = _RecordingRouteClassifier(None)

    outcome = await _service(route_classifier=route_classifier).dispatch(
        "слушай, обсудим потом этот тест",
        _ctx(),
        [computer_use, chat],
    )

    assert outcome.handled is True
    assert outcome.result is not None
    assert outcome.result.response == "chat_response"
    assert computer_use.executed == 0
    assert chat.executed == 1


async def test_worker_route_classifier_preserves_approval_gate() -> None:
    skill = _RequiredSkill()
    route_classifier = _RecordingRouteClassifier(
        SkillRouteDecision(
            skill_name="required_skill",
            confidence=0.83,
            rationale="publish-like side effect",
            normalized_action={"action": "publish"},
        )
    )

    outcome = await _service(route_classifier=route_classifier).dispatch(
        "надо это опубликовать",
        _ctx(),
        [skill, _ChatFallbackSkill()],
    )

    assert outcome.handled is True
    assert outcome.result is not None
    assert outcome.result.metadata["approval_pending"] is True
    assert skill.executed == []


async def test_llm_skill_route_classifier_uses_worker_tier_json_contract() -> None:
    llm = _FakeRouteLLM(
        """
        {"skill_name":"computer_use","confidence":0.84,
         "rationale":"follow-up needs browser scroll",
         "normalized_action":{"action":"browser_scroll","target":"down"}}
        """
    )
    classifier = LLMSkillRouteClassifier(llm_router=llm)

    decision = await classifier.classify(
        message="нужен скрин ниже",
        context=_ctx(),
        candidates=(
            SkillRouteCandidate(
                name="computer_use",
                description="Live browser and GUI actions",
                skill_type="inline",
                approval_policy="auto",
                side_effects=("network_io_external",),
                mode_tags=("personal",),
                deterministic_score=0.0,
            ),
        ),
    )

    assert decision == SkillRouteDecision(
        skill_name="computer_use",
        confidence=0.84,
        rationale="follow-up needs browser scroll",
        normalized_action={"action": "browser_scroll", "target": "down"},
    )
    assert llm.requests[0].tier == "worker"
    assert llm.requests[0].temperature == 0.0
    assert llm.requests[0].caller == "skill_route_classifier"


async def test_required_skill_requests_approval_before_execute() -> None:
    skill = _RequiredSkill()
    service = _service()

    outcome = await service.dispatch("/danger publish", _ctx(), [skill])

    assert outcome.handled is True
    assert outcome.result is not None
    assert "да" not in outcome.result.response.lower()
    assert "нет" not in outcome.result.response.lower()
    assert outcome.result.metadata["approval_pending"] is True
    assert outcome.result.metadata["pending_decision"]["kind"] == "external_action"
    assert outcome.result.metadata["pending_decision"]["owner"] == "required_skill"
    assert outcome.result.metadata["pending_decision"]["action"] == "required_skill"
    assert outcome.result.metadata["requires_zhvusha_response"] is True
    assert outcome.result.metadata["body_observation"]["event"] == "pending_approval"
    assert outcome.result.metadata["dialogue_state_patch"] == {
        "pending_action": "required_action",
        "selected_skill": "required_skill",
    }
    assert skill.executed == []


async def test_auto_skill_can_require_approval_for_specific_message() -> None:
    skill = _PerMessageApprovalSkill()
    service = _service()

    outcome = await service.dispatch("опасно сделать", _ctx(), [skill])

    assert outcome.handled is True
    assert outcome.result is not None
    assert outcome.result.metadata["approval_pending"] is True
    assert skill.executed == []

    approved = await service.dispatch("да", _ctx(), [skill])

    assert approved.handled is True
    assert approved.result is not None
    assert approved.result.response == "auto: опасно сделать"
    assert skill.executed == ["опасно сделать"]


async def test_natural_required_skill_requests_approval_before_execute() -> None:
    skill = _NaturalRequiredSkill()
    service = _service()

    outcome = await service.dispatch("опубликуй пост hello", _ctx(), [skill])

    assert outcome.handled is True
    assert outcome.result is not None
    assert outcome.result.metadata["approval_pending"] is True
    assert outcome.result.metadata["pending_decision"]["owner"] == (
        "natural_required_skill"
    )
    assert skill.executed == []


async def test_required_skill_executes_original_message_after_approval() -> None:
    skill = _RequiredSkill()
    service = _service("yes")

    await service.dispatch("/danger publish", _ctx(), [skill])
    outcome = await service.dispatch("да", _ctx(), [skill])

    assert outcome.handled is True
    assert outcome.result is not None
    assert outcome.result.response == "required: /danger publish"
    assert outcome.result.metadata["approved_original_message"] == "/danger publish"
    assert outcome.result.metadata["approval_response_message"] == "да"
    assert (
        outcome.result.metadata["body_observation_synthesis_message"]
        == "/danger publish"
    )
    assert skill.executed == ["/danger publish"]
    assert skill.approval_metadata[0]["skill_approval_granted"] is True


async def test_high_confidence_new_task_supersedes_stale_pending_approval() -> None:
    pending_skill = _RequiredSkill()
    fresh_skill = _NewTaskSkill()
    service = _service("yes")

    await service.dispatch("/danger publish", _ctx(), [pending_skill, fresh_skill])
    outcome = await service.dispatch(
        "/new-task open article screenshot",
        _ctx(),
        [pending_skill, fresh_skill],
    )
    later = await service.dispatch("да", _ctx(), [pending_skill, fresh_skill])

    assert outcome.handled is True
    assert outcome.result is not None
    assert outcome.result.response == "new-task: /new-task open article screenshot"
    assert pending_skill.executed == []
    assert fresh_skill.executed == ["/new-task open article screenshot"]
    assert later.handled is False


async def test_required_skill_failure_after_approval_returns_failed_result() -> None:
    skill = _FailingRequiredSkill()
    service = _service("yes")

    await service.dispatch("/fail publish", _ctx(), [skill])
    outcome = await service.dispatch("разрешаю", _ctx(), [skill])

    assert outcome.handled is True
    assert outcome.result is not None
    assert outcome.result.success is False
    assert "не смогла выполнить подтверждённое действие" in outcome.result.response
    assert outcome.result.metadata["approved_original_message"] == "/fail publish"
    assert outcome.result.metadata["approval_response_message"] == "разрешаю"
    assert outcome.result.metadata["approved_skill_error"] == "RuntimeError"
    assert outcome.result.metadata["dialogue_state_patch"] == {
        "selected_skill": "failing_required_skill",
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
    }


async def test_required_skill_rejection_drops_pending_action() -> None:
    skill = _RequiredSkill()
    service = _service("no")

    await service.dispatch("/danger publish", _ctx(), [skill])
    outcome = await service.dispatch("нет", _ctx(), [skill])

    assert outcome.result is not None
    assert outcome.result.metadata["decision_resolution"]["outcome"] == "reject"
    assert skill.executed == []


async def test_required_skill_can_request_clarification_before_approval() -> None:
    skill = _ClarifyingRequiredSkill()
    service = _service()

    outcome = await service.dispatch("/clarify", _ctx(), [skill])

    assert outcome.handled is True
    assert outcome.result is not None
    assert outcome.result.response == ""
    assert outcome.result.metadata["requires_user_input"] is True
    assert outcome.result.metadata["requires_zhvusha_response"] is True
    assert "skip_dialogue_assistant_response" not in outcome.result.metadata
    assert outcome.result.metadata["pending_decision"]["kind"] == (
        "missing_required_input"
    )
    assert outcome.result.metadata["pending_decision"]["missing_fields"] == ["message"]
    body_observation = outcome.result.metadata["body_observation"]
    assert body_observation["event"] == "missing_required_input"
    assert body_observation["source"] == "skill_invocation"
    assert body_observation["pending_decision"]["kind"] == "missing_required_input"
    assert body_observation["pending_decision"]["missing_fields"] == ["message"]
    assert body_observation["pending_decision"]["summary"] == (
        "Что именно нужно отправить?"
    )
    assert outcome.result.metadata["skill_name"] == "clarifying_required_skill"
    assert skill.executed == []

    follow_up = await service.dispatch("да", _ctx(), [skill])

    assert follow_up.handled is False
    assert skill.executed == []


async def test_dry_run_blockers_return_body_observation_for_synthesis() -> None:
    skill = _DryRunBlockingRequiredSkill()
    service = _service()

    outcome = await service.dispatch("/dry-block", _ctx(), [skill])

    assert outcome.handled is True
    assert outcome.result is not None
    assert outcome.result.success is False
    assert outcome.result.response == ""
    assert outcome.result.metadata["requires_zhvusha_response"] is True
    assert "skip_dialogue_assistant_response" not in outcome.result.metadata
    assert outcome.result.metadata["skill_name"] == "dry_run_blocking_required_skill"
    body_observation = outcome.result.metadata["body_observation"]
    assert body_observation["event"] == "dry_run_blocked"
    assert body_observation["source"] == "skill_invocation"
    assert body_observation["skill_name"] == "dry_run_blocking_required_skill"
    assert body_observation["summary"] == "Опасное действие"
    assert body_observation["blockers"] == [
        "missing scoped runtime adapter",
        "approval contract incomplete",
    ]
    assert body_observation["dependencies_available"] is False
    assert body_observation["would_succeed"] is False
    assert skill.executed == []


async def test_pending_decision_revision_runs_before_plain_approval() -> None:
    skill = _RevisableRequiredSkill()
    service = _service("yes")

    first = await service.dispatch("/revise publish", _ctx(), [skill])
    revision = await service.dispatch("да, но мягче", _ctx(), [skill])

    assert first.result is not None
    assert first.result.metadata["pending_decision"]["summary"] == (
        "Отправить исходный текст"
    )
    assert revision.result is not None
    assert revision.result.metadata["pending_decision"]["summary"] == (
        "Отправить текст мягче"
    )
    assert revision.result.metadata["pending_decision"]["owner"] == (
        "revisable_required_skill"
    )
    assert skill.executed == []

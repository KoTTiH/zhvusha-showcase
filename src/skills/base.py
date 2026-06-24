"""Skill contract: v4 hierarchy.

``BaseSkill`` ABC with ``InlineSkill`` / ``DelegatedSkill`` /
``BackgroundSkill`` subclasses, ``CoreCapability`` / ``SideEffect`` StrEnums,
and frozen dataclasses for ``AgentContext`` / ``ExecutionPlan`` /
``SkillResult`` / ``SimulatedResult`` / ``Feedback``. New skills inherit from
one of the three subclasses and declare metadata via a sibling ``skill.yaml``
manifest (see :mod:`src.skills.manifest`).

Phase 7.5 removed the v3 ``LegacyBaseSkill`` / ``LegacyAgentContext`` /
``LegacySkillResult`` / ``LegacyFeedback`` classes and the
``agent_context_v3_to_v4`` helper once every skill had migrated.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# v4 enums
# ---------------------------------------------------------------------------


class CoreCapability(StrEnum):
    """Protected list of core capabilities.

    A skill that modifies any of these is automatically classified as Tier 3
    (recursive self-improvement gate). Extending this enum is itself a Tier 3
    operation that only Никита can perform.
    """

    SKILLS_DISPATCHER = "core.skills_dispatcher"
    PERSONALITY_VOICE = "core.personality_voice"
    PERSONALITY_DECISION_ENGINE = "core.personality_decision_engine"
    SAFETY_MODULE = "core.safety_module"
    TOOL_REGISTRY = "core.tool_registry"
    RESEARCH_CAPABILITY = "core.research_capability"
    SPEC_WORKFLOW = "core.spec_workflow"
    TEST_INFRASTRUCTURE = "core.test_infrastructure"
    LLM_GATEWAY_CONTRACT = "core.llm_gateway_contract"
    MEMORY_CONTRACT = "core.memory_contract"
    KB_CONTRACT = "core.kb_contract"
    DAEMON_CONTRACT = "core.daemon_contract"
    INTERFACES_CONTRACT = "core.interfaces_contract"


class SideEffect(StrEnum):
    """Reserved list of side effects a skill may perform on the outside world."""

    WRITES_TO_KB = "writes_to_kb"
    READS_FROM_KB = "reads_from_kb"
    CALLS_LLM = "calls_llm"
    CALLS_LLM_TIER_STRATEGIST = "calls_llm_tier_strategist"
    SENDS_TELEGRAM_MESSAGE = "sends_telegram_message"
    POSTS_TO_CHANNEL = "posts_to_channel"
    READS_WORKSPACE = "reads_workspace"
    WRITES_WORKSPACE = "writes_workspace"
    READS_FILESYSTEM = "reads_filesystem"
    WRITES_FILESYSTEM = "writes_filesystem"
    SPAWNS_SUBPROCESS = "spawns_subprocess"
    NETWORK_IO_EXTERNAL = "network_io_external"
    DELEGATES_TO_CODE_AGENT = "delegates_to_code_agent"
    DELEGATES_TO_OTHER_AGENT = "delegates_to_other_agent"
    MODIFIES_PERSONALITY_STATE = "modifies_personality_state"
    MODIFIES_MEMORY = "modifies_memory"
    MODIFIES_DESIRES = "modifies_desires"


# ---------------------------------------------------------------------------
# v4 data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentContext:
    """Per-invocation context passed to every skill contract method."""

    user_id: int
    chat_id: int | None
    mode: Literal["personal", "assistant", "social"]
    message_id: int | None = None
    bot: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    is_dry_run: bool = False


@dataclass(frozen=True)
class ExecutionPlan:
    """Plan of what a skill is about to do. Returned by ``prepare()``.

    Contains no side effects — a pure description used for approval UI and
    dry-run simulation.
    """

    skill_name: str
    skill_type: Literal["inline", "delegated", "background"]
    human_summary: str
    estimated_tokens: int
    estimated_cost_usd: Decimal
    estimated_duration_seconds: float
    files_to_read: list[Path] = field(default_factory=list)
    files_to_modify: list[Path] = field(default_factory=list)
    side_effects_invoked: list[SideEffect] = field(default_factory=list)
    llm_calls_planned: int = 0
    delegated_to: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillResult:
    """Result of ``execute()`` — what the skill actually produced."""

    success: bool
    response: str
    metadata: dict[str, Any] = field(default_factory=dict)
    actual_tokens_used: int = 0
    actual_cost_usd: Decimal = Decimal("0")
    actual_duration_seconds: float = 0.0


@dataclass(frozen=True)
class SimulatedResult:
    """Result of ``dry_run()`` — simulation without side effects."""

    would_succeed: bool
    would_produce: str
    dependencies_available: bool
    estimated_actual_cost: Decimal
    blockers: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Feedback:
    """User feedback for the skill learning loop."""

    skill_name: str
    user_id: int
    rating: Literal["positive", "negative", "neutral"]
    comment: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# v4 class hierarchy
# ---------------------------------------------------------------------------


class BaseSkill(ABC):
    """v4 abstract base class.

    Skills do **not** inherit from this directly — they inherit from one of
    ``InlineSkill`` / ``DelegatedSkill`` / ``BackgroundSkill``.
    """

    # === Identity (required on every skill) ===
    name: ClassVar[str]
    description: ClassVar[str]
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]]

    # === Type marker (set by subclasses) ===
    skill_type: ClassVar[Literal["inline", "delegated", "background"]]

    # === Routing ===
    triggers: ClassVar[list[str]] = []
    """Deterministic regex/keyword patterns for fast match."""

    # === Cost & approval ===
    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "low"
    """low: < $0.01, medium: $0.01-$0.50, high: > $0.50"""

    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = "auto"
    """auto: execute without approval (Tier 1 default).
    required: always require approval (any Tier 2+).
    mode_dependent: auto in personal mode, required in assistant mode.
    """

    # === Side effects ===
    side_effects: ClassVar[list[SideEffect]] = []

    # === Mode tags ===
    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    # === Recursive self-improvement gate ===
    modifies: ClassVar[list[CoreCapability]] = []
    """Core capabilities the skill modifies. Usually empty.
    Non-empty value auto-classifies the skill as Tier 3 (human approval).
    """

    # === Contract methods ===

    @abstractmethod
    async def can_handle(self, message: str, context: AgentContext) -> float:
        """Return confidence ``[0.0, 1.0]`` that this skill fits the message."""
        ...

    @abstractmethod
    async def prepare(self, message: str, context: AgentContext) -> ExecutionPlan:
        """Build an ``ExecutionPlan`` without side effects.

        Must be pure: read configs, estimate tokens/cost, formulate
        ``human_summary``. No LLM calls, DB writes, or filesystem operations.
        """
        ...

    @abstractmethod
    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        """Actually execute the skill; may perform any declared side effects.

        Must only be called after approval (if ``approval_policy != "auto"``)
        or when ``approval_policy == "auto"``.
        """
        ...

    @abstractmethod
    async def dry_run(self, plan: ExecutionPlan) -> SimulatedResult:
        """Simulate execution without side effects."""
        ...

    async def on_feedback(self, feedback: Feedback) -> None:
        """Default: no-op. Override to implement a learning loop."""
        del feedback  # unused in default implementation


class InlineSkill(BaseSkill):
    """Skill executed inline in the main chat_response loop.

    Fast, lightweight, does not delegate anywhere.
    """

    skill_type: ClassVar[Literal["inline", "delegated", "background"]] = "inline"

    async def prepare(self, message: str, context: AgentContext) -> ExecutionPlan:
        """Default: trivial plan for an inline skill."""
        del message, context  # unused in default implementation
        return ExecutionPlan(
            skill_name=self.name,
            skill_type="inline",
            human_summary=f"Ответить на запрос через {self.name} (tier: {self.llm_tier})",
            estimated_tokens=2000,
            estimated_cost_usd=Decimal("0.005"),
            estimated_duration_seconds=3.0,
            llm_calls_planned=1,
            side_effects_invoked=list(self.side_effects),
        )

    async def dry_run(self, plan: ExecutionPlan) -> SimulatedResult:
        """Default: assume the LLM tier is reachable."""
        return SimulatedResult(
            would_succeed=True,
            would_produce=f"Inline response via {plan.skill_name}",
            dependencies_available=True,
            estimated_actual_cost=plan.estimated_cost_usd,
        )


class DelegatedSkill(BaseSkill):
    """Skill that delegates heavy work to an external executor.

    Examples: Codex CLI, browser-use. May run for minutes;
    always goes through the approval flow.
    """

    skill_type: ClassVar[Literal["inline", "delegated", "background"]] = "delegated"

    executor: ClassVar[str] = ""  # must be overridden with executor name
    max_duration_seconds: ClassVar[float] = 600

    @abstractmethod
    async def prepare(self, message: str, context: AgentContext) -> ExecutionPlan:
        """Delegated skills **must** override ``prepare`` with real analysis."""
        ...

    async def dry_run(self, plan: ExecutionPlan) -> SimulatedResult:
        """Default: check executor configured, files exist, no blockers."""
        blockers: list[str] = []
        if not self.executor:
            blockers.append("executor not configured")
        for f in plan.files_to_modify:
            if not f.exists():
                blockers.append(f"file does not exist: {f}")
        return SimulatedResult(
            would_succeed=len(blockers) == 0,
            would_produce=plan.human_summary,
            dependencies_available=len(blockers) == 0,
            estimated_actual_cost=plan.estimated_cost_usd,
            blockers=blockers,
        )


class BackgroundSkill(BaseSkill):
    """Skill triggered on a schedule or event by the daemon.

    Not invoked directly by the user through chat.
    """

    skill_type: ClassVar[Literal["inline", "delegated", "background"]] = "background"

    trigger_type: ClassVar[Literal["cron", "event", "interval"]] = "interval"
    trigger_config: ClassVar[dict[str, Any]] = {}

    async def can_handle(self, message: str, context: AgentContext) -> float:
        """Background skills never match user messages."""
        del message, context  # background skills ignore user messages
        return 0.0

    async def prepare(self, message: str, context: AgentContext) -> ExecutionPlan:
        """Default: plan for one background tick."""
        del message, context  # unused in default implementation
        return ExecutionPlan(
            skill_name=self.name,
            skill_type="background",
            human_summary=f"Background iteration of {self.name}",
            estimated_tokens=500,
            estimated_cost_usd=Decimal("0.001"),
            estimated_duration_seconds=10.0,
            side_effects_invoked=list(self.side_effects),
        )

    async def dry_run(self, plan: ExecutionPlan) -> SimulatedResult:
        """Default: simulate one tick without side effects."""
        return SimulatedResult(
            would_succeed=True,
            would_produce=f"Background tick of {plan.skill_name}",
            dependencies_available=True,
            estimated_actual_cost=plan.estimated_cost_usd,
        )

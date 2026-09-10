"""Inline skill that turns backlog topics into spec/proposal candidates."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar, Literal

from src.skills.base import (
    AgentContext,
    ExecutionPlan,
    InlineSkill,
    SideEffect,
    SkillResult,
)
from src.skills.topic_to_spec.builder import (
    build_candidate_from_topic,
    render_candidate,
)

if TYPE_CHECKING:
    from src.skills.topic_to_spec.models import ProposalWriter, TopicProvider

_TRIGGER = "/topic_to_spec"
_NATURAL_TOPIC_PREFIXES: tuple[str, ...] = (
    "создай spec из темы",
    "собери spec из темы",
    "создай proposal из темы",
    "собери proposal из темы",
)


def _normalize_chat_route_text(text: str) -> str:
    return " ".join(text.strip().lower().replace("ё", "е").split())


def _strip_natural_tail(original: str, prefix: str) -> str:
    pattern = r"^\s*" + r"\s+".join(re.escape(part) for part in prefix.split())
    pattern += r"\s*[:\-—]?\s*"
    return re.sub(pattern, "", original, count=1, flags=re.I).strip(" \t\n\r:-—")


def _normalize_message_to_command(message: str) -> str | None:
    text = message.strip()
    lower = text.lower()
    if lower == _TRIGGER or lower.startswith(_TRIGGER + " "):
        return text
    normalized = _normalize_chat_route_text(text)
    for prefix in _NATURAL_TOPIC_PREFIXES:
        if normalized.startswith(prefix + " ") or normalized.startswith(prefix + ":"):
            key = _strip_natural_tail(text, prefix)
            if key:
                return f"{_TRIGGER} {key}"
    return None


def _is_natural_route(message: str) -> bool:
    return (
        not message.strip().lower().startswith(_TRIGGER)
        and _normalize_message_to_command(message) is not None
    )


class TopicToSpecSkill(InlineSkill):
    """Read a ranked topic and produce an approval-ready candidate."""

    name: ClassVar[str] = "topic_to_spec"
    description: ClassVar[str] = (
        "Преобразует topic_clusters в candidate spec/proposal с источниками"
    )
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"
    triggers: ClassVar[list[str]] = [_TRIGGER]
    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "low"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = "auto"
    side_effects: ClassVar[list[SideEffect]] = [
        SideEffect.READS_FILESYSTEM,
        SideEffect.WRITES_FILESYSTEM,
    ]
    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    def __init__(
        self,
        *,
        admin_user_id: int,
        topic_provider: TopicProvider,
        proposal_writer: ProposalWriter | None = None,
    ) -> None:
        self._admin_user_id = admin_user_id
        self._topics = topic_provider
        self._proposal_writer = proposal_writer

    async def can_handle(self, message: str, context: AgentContext) -> float:
        if context.user_id != self._admin_user_id or context.mode != "personal":
            return 0.0
        command = _normalize_message_to_command(message)
        if command is None:
            return 0.0
        return 1.0 if message.strip().lower().startswith(_TRIGGER) else 0.93

    def requires_approval_for_message(
        self,
        message: str,
        context: AgentContext,
    ) -> bool:
        return context.mode == "personal" and _is_natural_route(message)

    async def prepare(self, message: str, context: AgentContext) -> ExecutionPlan:
        del context
        command = _normalize_message_to_command(message) or message.strip()
        return ExecutionPlan(
            skill_name=self.name,
            skill_type="inline",
            human_summary=(
                "Собрать spec/proposal candidate из backlog-темы: "
                f"{command.removeprefix(_TRIGGER).strip() or '<нет>'}"
            ),
            estimated_tokens=0,
            estimated_cost_usd=Decimal("0"),
            estimated_duration_seconds=3.0,
            side_effects_invoked=list(self.side_effects),
            llm_calls_planned=0,
            metadata={"internal_action": command},
        )

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        del context
        command = _normalize_message_to_command(message) or message
        key = command.strip()[len(_TRIGGER) :].strip() or None
        topic = await self._topics.get_topic(key)
        if topic is None:
            return SkillResult(
                success=False,
                response="Не нашла такую тему в backlog.",
            )
        candidate = build_candidate_from_topic(topic)
        proposal_path = None
        response = render_candidate(candidate)
        if candidate.kind == "proposal" and self._proposal_writer is not None:
            proposal_path = self._proposal_writer.write_candidate(candidate)
            response += (
                "\n\n"
                f"Proposal сохранен: `{proposal_path.as_posix()}`. "
                "Код не запускаю автоматически."
            )
        return SkillResult(
            success=True,
            response=response,
            metadata={
                "slug": candidate.slug,
                "tier": candidate.tier,
                "kind": candidate.kind,
                "proposal_path": str(proposal_path) if proposal_path else "",
            },
        )

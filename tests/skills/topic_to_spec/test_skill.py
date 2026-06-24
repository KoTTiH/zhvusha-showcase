"""TopicToSpecSkill contract and behavior tests."""

from __future__ import annotations

from pathlib import Path

from src.skills.base import AgentContext, InlineSkill
from src.skills.manifest import (
    load_manifest_for_skill_class,
    validate_manifest_matches_class,
)
from src.skills.topic_to_spec.models import TopicCandidate, TopicRecord
from src.skills.topic_to_spec.skill import TopicToSpecSkill


class _Provider:
    async def get_topic(self, key: str | None = None) -> TopicRecord | None:
        if key not in {None, "codex-hooks"}:
            return None
        return TopicRecord(
            cluster_key="codex-hooks",
            title="OpenAI Codex hooks update",
            summary="Codex hooks affect self-coding gates.",
            top_terms=("codex", "hooks"),
            final_priority=90.0,
            pillar_alignment={"self_improvement": 1.0},
        )


class _Tier3Provider:
    async def get_topic(self, key: str | None = None) -> TopicRecord | None:
        del key
        return TopicRecord(
            cluster_key="safety-protocol",
            title="Safety protocol",
            summary="Safety protocol affects protected self-improvement gates.",
            top_terms=("safety", "protocol"),
            final_priority=90.0,
        )


class _Writer:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def write_candidate(self, candidate: TopicCandidate) -> Path:
        self.paths.append(candidate.slug)
        return Path("proposals/2026-05-07-safety-protocol.md")


def test_contract_manifest_matches_class() -> None:
    manifest = load_manifest_for_skill_class(TopicToSpecSkill)
    validate_manifest_matches_class(manifest, TopicToSpecSkill)
    assert issubclass(TopicToSpecSkill, InlineSkill)


async def test_execute_returns_candidate_for_topic() -> None:
    skill = TopicToSpecSkill(admin_user_id=1, topic_provider=_Provider())
    context = AgentContext(user_id=1, chat_id=1, mode="personal")

    result = await skill.execute("/topic_to_spec codex-hooks", context)

    assert result.success
    assert result.metadata["tier"] == 2
    assert "Источники" in result.response
    assert result.metadata["proposal_path"] == ""


async def test_natural_topic_to_spec_routes_and_executes() -> None:
    skill = TopicToSpecSkill(admin_user_id=1, topic_provider=_Provider())
    context = AgentContext(user_id=1, chat_id=1, mode="personal")

    assert await skill.can_handle("создай spec из темы codex-hooks", context) >= 0.9
    assert await skill.can_handle("обсудим spec из темы codex-hooks", context) == 0.0

    result = await skill.execute("создай spec из темы codex-hooks", context)

    assert result.success
    assert result.metadata["tier"] == 2
    assert "OpenAI Codex hooks update" in result.response

    spaced = await skill.execute("создай   spec   из   темы: codex-hooks", context)

    assert spaced.success
    assert spaced.metadata["tier"] == 2


async def test_natural_topic_to_spec_uses_skill_approval_gate() -> None:
    from src.skills.invocation import (
        InMemorySkillApprovalStore,
        SkillInvocationService,
    )

    async def _approval_classifier(text: str) -> str:
        del text
        return "yes"

    writer = _Writer()
    skill = TopicToSpecSkill(
        admin_user_id=1,
        topic_provider=_Tier3Provider(),
        proposal_writer=writer,
    )
    context = AgentContext(user_id=1, chat_id=1, mode="personal")
    service = SkillInvocationService(
        approval_store=InMemorySkillApprovalStore(),
        approval_classifier=_approval_classifier,
        is_skill_allowed=lambda _name, _mode: True,
    )

    pending = await service.dispatch(
        "создай proposal из темы safety-protocol",
        context,
        [skill],
    )

    assert pending.result is not None
    assert pending.result.metadata["approval_pending"] is True
    assert writer.paths == []

    approved = await service.dispatch("да", context, [skill])

    assert approved.result is not None
    assert approved.result.success
    assert writer.paths == ["safety-protocol"]


async def test_tier3_topic_writes_proposal_when_writer_is_configured() -> None:
    writer = _Writer()
    skill = TopicToSpecSkill(
        admin_user_id=1,
        topic_provider=_Tier3Provider(),
        proposal_writer=writer,
    )
    context = AgentContext(user_id=1, chat_id=1, mode="personal")

    result = await skill.execute("/topic_to_spec safety-protocol", context)

    assert result.success
    assert result.metadata["kind"] == "proposal"
    assert writer.paths == ["safety-protocol"]
    assert "Proposal сохранен" in result.response


async def test_unknown_topic_returns_failure() -> None:
    skill = TopicToSpecSkill(admin_user_id=1, topic_provider=_Provider())
    context = AgentContext(user_id=1, chat_id=1, mode="personal")

    result = await skill.execute("/topic_to_spec missing", context)

    assert not result.success

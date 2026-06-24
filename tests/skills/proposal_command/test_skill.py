"""ProposalCommandSkill contract and lifecycle tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.skills.base import AgentContext, InlineSkill
from src.skills.manifest import (
    load_manifest_for_skill_class,
    validate_manifest_matches_class,
)
from src.skills.proposal_command.models import ProposalModel
from src.skills.proposal_command.skill import ProposalCommandSkill
from src.skills.proposal_command.store import load_proposal, write_proposal
from src.skills.spec_command.parser import SourceProvenance


def _ctx() -> AgentContext:
    return AgentContext(user_id=1, chat_id=1, mode="personal")


def _proposal(slug: str = "safety-protocol") -> ProposalModel:
    return ProposalModel(
        slug=slug,
        title="Safety protocol",
        created_at=datetime(2026, 5, 7, 12, 0, tzinfo=UTC),
        created_by="zhvusha",
        tier=3,
        summary="Prepare a Tier 3 proposal for safety protocol changes.",
        proposed_change="Describe the architecture change before any code is written.",
        rationale="Safety protocol changes are protected and need human approval.",
        acceptance=["Никита can approve, defer or reject the proposal."],
        files_likely_touched=["src/safety/"],
        risk="Core safety behavior can regress.",
        source_provenance=[
            SourceProvenance(
                url="local:test",
                source_type="local_repo",
                trust_tier="direct",
                claim="Test evidence exists.",
            )
        ],
    )


def test_contract_manifest_matches_class() -> None:
    manifest = load_manifest_for_skill_class(ProposalCommandSkill)
    validate_manifest_matches_class(manifest, ProposalCommandSkill)
    assert issubclass(ProposalCommandSkill, InlineSkill)


async def test_list_and_show_proposal(tmp_path: Path) -> None:
    path = write_proposal(tmp_path, _proposal())
    skill = ProposalCommandSkill(proposals_dir=tmp_path, admin_user_id=1)

    listed = await skill.execute("/proposal list", _ctx())
    shown = await skill.execute("/proposal show safety-protocol", _ctx())

    assert listed.success
    assert "safety-protocol" in listed.response
    assert shown.success
    assert path.name in shown.response
    assert "Safety protocol" in shown.response


async def test_natural_proposal_routes_and_negative(tmp_path: Path) -> None:
    write_proposal(tmp_path, _proposal())
    skill = ProposalCommandSkill(proposals_dir=tmp_path, admin_user_id=1)

    assert await skill.can_handle("покажи proposal safety-protocol", _ctx()) >= 0.9
    assert await skill.can_handle("одобри proposal safety-protocol", _ctx()) >= 0.9
    assert await skill.can_handle("обсудим proposal safety-protocol", _ctx()) == 0.0

    shown = await skill.execute("покажи proposal safety-protocol", _ctx())
    spaced = await skill.execute("покажи   proposal: safety-protocol", _ctx())

    assert shown.success
    assert "Safety protocol" in shown.response
    assert spaced.success
    assert "Safety protocol" in spaced.response


async def test_approve_changes_status_without_autocoding(tmp_path: Path) -> None:
    path = write_proposal(tmp_path, _proposal())
    skill = ProposalCommandSkill(proposals_dir=tmp_path, admin_user_id=1)

    result = await skill.execute("/proposal approve safety-protocol", _ctx())
    reloaded = load_proposal(path)

    assert result.success
    assert reloaded.status.value == "approved"
    assert reloaded.approved_by == "nikita"
    assert "Автокодинг не запускаю" in result.response


async def test_natural_approve_changes_status_without_autocoding(
    tmp_path: Path,
) -> None:
    path = write_proposal(tmp_path, _proposal())
    skill = ProposalCommandSkill(proposals_dir=tmp_path, admin_user_id=1)

    result = await skill.execute("одобри proposal safety-protocol", _ctx())
    reloaded = load_proposal(path)

    assert result.success
    assert reloaded.status.value == "approved"
    assert "Автокодинг не запускаю" in result.response


async def test_natural_approve_uses_skill_approval_gate(tmp_path: Path) -> None:
    from src.skills.invocation import (
        InMemorySkillApprovalStore,
        SkillInvocationService,
    )

    async def _approval_classifier(text: str) -> str:
        del text
        return "yes"

    path = write_proposal(tmp_path, _proposal())
    skill = ProposalCommandSkill(proposals_dir=tmp_path, admin_user_id=1)
    service = SkillInvocationService(
        approval_store=InMemorySkillApprovalStore(),
        approval_classifier=_approval_classifier,
        is_skill_allowed=lambda _name, _mode: True,
    )

    pending = await service.dispatch("одобри proposal safety-protocol", _ctx(), [skill])

    assert pending.result is not None
    assert pending.result.metadata["approval_pending"] is True
    assert load_proposal(path).status.value == "pending_approval"

    approved = await service.dispatch("да", _ctx(), [skill])

    assert approved.result is not None
    assert approved.result.success
    assert load_proposal(path).status.value == "approved"


async def test_defer_and_reject_store_reason(tmp_path: Path) -> None:
    first = write_proposal(tmp_path, _proposal(slug="first"))
    second = write_proposal(tmp_path, _proposal(slug="second"))
    skill = ProposalCommandSkill(proposals_dir=tmp_path, admin_user_id=1)

    deferred = await skill.execute("/proposal defer first too risky", _ctx())
    rejected = await skill.execute("/proposal reject second out of scope", _ctx())

    assert deferred.success
    assert rejected.success
    assert load_proposal(first).deferred_reason == "too risky"
    assert load_proposal(second).rejected_reason == "out of scope"

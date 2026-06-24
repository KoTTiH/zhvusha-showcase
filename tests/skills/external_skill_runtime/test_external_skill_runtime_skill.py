"""External skill runtime skill contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from src.agent_runtime.models import AgentJobStatus, ContextCapsule
from src.skills.base import AgentContext
from src.skills.invocation import (
    ApprovalVerdict,
    InMemorySkillApprovalStore,
    SkillInvocationService,
)


class _Runtime:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.started: list[object] = []
        from src.agent_runtime.storage import InMemoryAgentJobStore

        self.store = InMemoryAgentJobStore()

    async def create_job(self, **kwargs: object) -> object:
        from src.agent_runtime.models import AgentJob

        self.created.append(kwargs)
        job = AgentJob.new(**kwargs)  # type: ignore[arg-type]
        return await self.store.create(job)

    async def start(self, job_id: str) -> object:
        job = await self.store.get(job_id)
        self.started.append(job)
        return SimpleNamespace(
            id=job_id,
            status=AgentJobStatus.DONE,
            result=ContextCapsule(
                summary="external skill capsule",
                markdown_report="external skill report",
            ),
            error="",
        )


def _ctx(
    *,
    approved: bool = False,
    metadata: dict[str, object] | None = None,
) -> AgentContext:
    context_metadata = dict(metadata or {})
    if approved:
        context_metadata.update(
            {
                "skill_approval_granted": True,
                "skill_approval_id": "skill-approval-test",
            }
        )
    return AgentContext(
        user_id=12345,
        chat_id=12345,
        mode="personal",
        message_id=5,
        metadata=context_metadata,
    )


def _service(*verdicts: ApprovalVerdict) -> SkillInvocationService:
    queued_verdicts = list(verdicts or ("yes",))

    async def classify_approval(text: str) -> ApprovalVerdict:
        del text
        if queued_verdicts:
            return queued_verdicts.pop(0)
        return "yes"

    return SkillInvocationService(
        approval_store=InMemorySkillApprovalStore(),
        approval_classifier=classify_approval,
        is_skill_allowed=lambda _name, _mode: True,
    )


def test_manifest_matches_class() -> None:
    from src.skills.external_skill_runtime.skill import ExternalSkillRuntimeSkill
    from src.skills.manifest import (
        load_manifest_for_skill_class,
        validate_manifest_matches_class,
    )

    manifest = load_manifest_for_skill_class(ExternalSkillRuntimeSkill)
    validate_manifest_matches_class(manifest, ExternalSkillRuntimeSkill)


@pytest.mark.asyncio
async def test_readonly_command_runs_only_after_skill_invocation_approval() -> None:
    from src.agent_runtime.profiles import EXTERNAL_SKILL_READONLY
    from src.skills.external_skill_runtime.skill import ExternalSkillRuntimeSkill

    runtime = _Runtime()
    skill = ExternalSkillRuntimeSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        readonly_profile=EXTERNAL_SKILL_READONLY,
        tool_capability_resolver=lambda _tool_name: None,
    )
    service = _service("yes")

    pending = await service.dispatch(
        "/external_skill_readonly kube-debug | Проверь ingress",
        _ctx(),
        [skill],
    )
    assert pending.result is not None
    assert pending.result.metadata["approval_pending"] is True
    assert runtime.created == []

    completed = await service.dispatch("да", _ctx(), [skill])

    assert completed.result is not None
    assert completed.result.success is True
    assert completed.result.metadata["requires_zhvusha_response"] is True
    observation = completed.result.metadata["body_observation"]
    assert observation["event"] == "external_skill_runtime_completed"
    assert observation["external_skill_id"] == "kube-debug"
    assert observation["external_skill_mode"] == "readonly"
    assert observation["capsule"]["summary"] == "external skill capsule"
    assert observation["capsule"]["markdown_report"] == "external skill report"
    assert len(runtime.created) == 1
    created = runtime.created[0]
    assert created["kind"] == "external_skill.readonly"
    assert created["profile"] == EXTERNAL_SKILL_READONLY
    pack = created["context_pack"]
    assert pack.metadata["external_skill_id"] == "kube-debug"
    assert pack.user_request == "Проверь ingress"


@pytest.mark.asyncio
async def test_execution_command_passes_scoped_tool_approval_to_runtime() -> None:
    from src.agent_runtime.approvals import InMemoryAgentToolApprovalGrantStore
    from src.skills.external_skill_runtime.skill import ExternalSkillRuntimeSkill

    runtime = _Runtime()
    approval_grants = InMemoryAgentToolApprovalGrantStore()
    skill = ExternalSkillRuntimeSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        tool_capability_resolver=lambda tool_name: {
            "browser_read_url": "browser_read",
        }.get(tool_name),
        approval_grant_store=approval_grants,
    )
    service = _service("yes")
    command = (
        "/external_skill_execute kube-debug | browser_read_url | "
        '{"url":"https://example.com"} | Открой источник'
    )

    await service.dispatch(command, _ctx(), [skill])
    completed = await service.dispatch("да", _ctx(), [skill])

    assert completed.result is not None
    assert completed.result.success is True
    created = runtime.created[0]
    assert created["kind"] == "external_skill.execution"
    profile = created["profile"]
    assert profile.allows("external_skill_execute")
    assert profile.allows("browser_read")
    started = runtime.started[0]
    pack = started.context_pack
    assert pack.metadata["external_skill_id"] == "kube-debug"
    assert pack.metadata["external_skill_tool_name"] == "browser_read_url"
    assert json.loads(pack.metadata["external_skill_tool_payload"]) == {
        "url": "https://example.com"
    }
    assert pack.metadata["agent_tool_approval_id"].startswith("skill-approval-")
    assert (
        pack.metadata["agent_tool_approval_capabilities"]
        == "browser_read,external_skill_execute"
    )
    assert pack.metadata["agent_tool_approval_id"] in str(created["fingerprint"])
    grant = approval_grants.get(pack.metadata["agent_tool_approval_id"])
    assert grant.job_id == started.id
    assert grant.grants_all(("external_skill_execute", "browser_read"))


@pytest.mark.asyncio
async def test_execution_command_removes_scoped_side_effect_capability_from_denied() -> (
    None
):
    from src.agent_runtime.approvals import InMemoryAgentToolApprovalGrantStore
    from src.skills.external_skill_runtime.skill import ExternalSkillRuntimeSkill

    runtime = _Runtime()
    approval_grants = InMemoryAgentToolApprovalGrantStore()
    skill = ExternalSkillRuntimeSkill(
        admin_user_id=12345,
        runtime=runtime,  # type: ignore[arg-type]
        tool_capability_resolver=lambda tool_name: {
            "write_workspace_file_after_gate": (
                "write_whitelisted_files_after_approval"
            ),
        }.get(tool_name),
        approval_grant_store=approval_grants,
    )
    service = _service("yes")
    command = (
        "/external_skill_execute local-file | write_workspace_file_after_gate | "
        '{"path":"agent_runtime/local_file_tasks/stage-l.txt","content":"ok"} | '
        "Запиши allowlisted artifact"
    )

    await service.dispatch(command, _ctx(), [skill])
    await service.dispatch("да", _ctx(), [skill])

    profile = runtime.created[0]["profile"]
    assert profile.allows("write_whitelisted_files_after_approval")
    assert "write_whitelisted_files_after_approval" not in profile.denied_capabilities
    pack = runtime.started[0].context_pack
    assert (
        pack.metadata["agent_tool_approval_capabilities"]
        == "external_skill_execute,write_whitelisted_files_after_approval"
    )


@pytest.mark.asyncio
async def test_execution_dry_run_blocks_unknown_tool_before_approval() -> None:
    from src.skills.external_skill_runtime.skill import ExternalSkillRuntimeSkill

    skill = ExternalSkillRuntimeSkill(
        admin_user_id=12345,
        runtime=_Runtime(),  # type: ignore[arg-type]
        tool_capability_resolver=lambda _tool_name: None,
    )
    plan = await skill.prepare(
        "/external_skill_execute kube-debug | missing_tool | {} | Проверь",
        _ctx(),
    )
    simulation = await skill.dry_run(plan)

    assert simulation.dependencies_available is False
    assert simulation.would_succeed is False
    assert simulation.blockers == ["unknown ToolGateway tool: missing_tool"]

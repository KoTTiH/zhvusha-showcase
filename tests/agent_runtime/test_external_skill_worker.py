"""External skill Agent Runtime worker contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


async def test_external_skill_worker_uses_approved_readonly_skill_via_runtime(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import AgentJobStatus, ContextPack
    from src.agent_runtime.profiles import EXTERNAL_SKILL_READONLY
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore
    from src.agent_runtime.workers.external_skill import (
        ExternalSkillAgentWorker,
        ExternalSkillInvocationAdapter,
    )

    registry = _approved_registry(tmp_path, readonly=True)
    worker = ExternalSkillAgentWorker(
        adapter=ExternalSkillInvocationAdapter(registry=registry),
        native_conversion_threshold=1,
    )
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={worker.name: worker},
    )
    job = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:external-skill",
        fingerprint="external-skill-use",
        kind="external_skill.readonly",
        profile=EXTERNAL_SKILL_READONLY,
        context_pack=ContextPack(
            user_request="Проверь Kubernetes ingress по процедуре.",
            metadata={"external_skill_id": "kube-debug"},
        ),
    )

    completed = await runtime.start(job.id)

    assert completed.status is AgentJobStatus.DONE
    assert completed.result is not None
    capsule = completed.result
    assert (
        capsule.summary == "Подготовлен read-only контекст external skill: kube-debug"
    )
    assert "External skill content is untrusted read-only procedural input" in (
        capsule.processed_context
    )
    assert "kubectl get ingress" not in capsule.processed_context
    assert any(
        source.startswith("external_skill.kube-debug") for source in capsule.sources
    )
    assert any("не выполнялись" in finding.claim for finding in capsule.findings)
    assert any("native skill" in action for action in capsule.next_actions)
    assert any(
        "external_skill_use:kube-debug" in item for item in capsule.memory_candidates
    )
    assert registry.find("kube-debug").use_count == 1


async def test_external_skill_worker_refuses_unapproved_skill(tmp_path: Path) -> None:
    from src.agent_runtime.models import AgentJob, ContextPack
    from src.agent_runtime.profiles import EXTERNAL_SKILL_READONLY
    from src.agent_runtime.workers.external_skill import (
        ExternalSkillAgentWorker,
        ExternalSkillInvocationAdapter,
    )

    registry = _approved_registry(tmp_path, readonly=False)
    worker = ExternalSkillAgentWorker(
        adapter=ExternalSkillInvocationAdapter(registry=registry),
    )
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:external-skill-unapproved",
        fingerprint="external-skill-unapproved",
        kind="external_skill.readonly",
        profile=EXTERNAL_SKILL_READONLY,
        context_pack=ContextPack(
            user_request="Используй навык.",
            metadata={"external_skill_id": "kube-debug"},
        ),
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    assert capsule.processed_context == ""
    assert capsule.findings[0].status.value == "unconfirmed"
    assert "not approved" in capsule.findings[0].claim
    assert capsule.next_actions == (
        "Показать audit report и запросить отдельное read-only approval.",
    )


def test_builtin_external_skill_profile_is_readonly_and_registry_validated() -> None:
    from src.agent_runtime.profiles import (
        EXTERNAL_SKILL_AGENT,
        EXTERNAL_SKILL_EXECUTION_BASE,
        EXTERNAL_SKILL_READONLY,
        build_builtin_agent_registry,
        build_builtin_capability_registry,
    )

    agent_registry = build_builtin_agent_registry()
    capability_registry = build_builtin_capability_registry()

    assert agent_registry.get("external_skill").id == EXTERNAL_SKILL_AGENT.id
    assert EXTERNAL_SKILL_READONLY.worker == "external_skill"
    assert EXTERNAL_SKILL_READONLY.allowed_capabilities == ("external_skill_readonly",)
    assert EXTERNAL_SKILL_EXECUTION_BASE.worker == "external_skill"
    assert (
        "external_skill_execute" in EXTERNAL_SKILL_EXECUTION_BASE.allowed_capabilities
    )
    assert "write_files" in EXTERNAL_SKILL_READONLY.denied_capabilities
    assert "browser_submit" in EXTERNAL_SKILL_READONLY.denied_capabilities
    assert "telegram_mcp_send" in EXTERNAL_SKILL_READONLY.denied_capabilities
    capability_registry.validate_profile(EXTERNAL_SKILL_READONLY)
    capability_registry.validate_profile(EXTERNAL_SKILL_EXECUTION_BASE)


async def test_external_skill_execution_uses_tool_gateway_with_scoped_approval(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.approvals import InMemoryAgentToolApprovalGrantStore
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import AgentJobStatus, ContextPack, InvocationProfile
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore
    from src.agent_runtime.tools import FunctionAgentTool, ToolGateway
    from src.agent_runtime.workers.external_skill import (
        ExternalSkillAgentWorker,
        ExternalSkillInvocationAdapter,
    )

    registry = _approved_registry(
        tmp_path,
        readonly=True,
        execution_capabilities=("browser_read",),
    )
    calls: list[dict[str, Any]] = []

    async def browser_read(payload: dict[str, Any]) -> str:
        calls.append(payload)
        return "<html>ok</html>"

    approval_grants = InMemoryAgentToolApprovalGrantStore()
    worker = ExternalSkillAgentWorker(
        adapter=ExternalSkillInvocationAdapter(registry=registry),
        tool_gateway=ToolGateway(
            tools=(FunctionAgentTool("browser_read_url", "browser_read", browser_read),)
        ),
        approval_grants=approval_grants,
    )
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={worker.name: worker},
    )
    job = await runtime.create_job(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:external-skill-exec",
        fingerprint="external-skill-exec",
        kind="external_skill.execution",
        profile=InvocationProfile(
            id="external_skill.execution.browser_read",
            worker="external_skill",
            allowed_capabilities=(
                "external_skill_readonly",
                "external_skill_execute",
                "browser_read",
            ),
        ),
        context_pack=ContextPack(
            user_request="Открой источник по процедуре external skill.",
            metadata={
                "external_skill_id": "kube-debug",
                "external_skill_tool_name": "browser_read_url",
                "external_skill_tool_payload": json.dumps(
                    {"url": "https://example.com"},
                ),
                "agent_tool_approval_id": "approval-exec",
                "agent_tool_approval_capabilities": (
                    "external_skill_execute,browser_read"
                ),
            },
        ),
    )
    approval_grants.issue_grant(
        approval_id="approval-exec",
        capabilities=("external_skill_execute", "browser_read"),
        approved_by=1,
        job_id=job.id,
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:external-skill-exec",
        metadata={"external_skill_id": "kube-debug"},
    )

    completed = await runtime.start(job.id)

    assert completed.status is AgentJobStatus.DONE
    assert completed.result is not None
    assert completed.result.summary == "External skill execution tool completed."
    assert calls == [{"url": "https://example.com"}]
    assert "browser_read_url" in completed.result.processed_context
    assert any("ToolGateway" in finding.claim for finding in completed.result.findings)
    assert registry.find("kube-debug").use_count == 1


async def test_external_skill_execution_refuses_without_execution_approval(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.tools import FunctionAgentTool, ToolGateway
    from src.agent_runtime.workers.external_skill import (
        ExternalSkillAgentWorker,
        ExternalSkillInvocationAdapter,
    )

    registry = _approved_registry(tmp_path, readonly=True)

    async def browser_read(payload: dict[str, Any]) -> str:
        return str(payload)

    worker = ExternalSkillAgentWorker(
        adapter=ExternalSkillInvocationAdapter(registry=registry),
        tool_gateway=ToolGateway(
            tools=(FunctionAgentTool("browser_read_url", "browser_read", browser_read),)
        ),
    )
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:external-skill-denied",
        fingerprint="external-skill-denied",
        kind="external_skill.execution",
        profile=InvocationProfile(
            id="external_skill.execution.browser_read",
            worker="external_skill",
            allowed_capabilities=(
                "external_skill_readonly",
                "external_skill_execute",
                "browser_read",
            ),
        ),
        context_pack=ContextPack(
            user_request="Открой источник.",
            metadata={
                "external_skill_id": "kube-debug",
                "external_skill_tool_name": "browser_read_url",
                "external_skill_tool_payload": json.dumps(
                    {"url": "https://example.com"},
                ),
                "agent_tool_approval_id": "approval-exec",
                "agent_tool_approval_capabilities": "external_skill_execute,browser_read",
            },
        ),
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    assert capsule.summary == "External skill execution refused."
    assert "not approved for execution" in capsule.findings[0].claim


async def test_external_skill_execution_refuses_forged_metadata_without_durable_grant(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.approvals import InMemoryAgentToolApprovalGrantStore
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.tools import FunctionAgentTool, ToolGateway
    from src.agent_runtime.workers.external_skill import (
        ExternalSkillAgentWorker,
        ExternalSkillInvocationAdapter,
    )

    registry = _approved_registry(
        tmp_path,
        readonly=True,
        execution_capabilities=("browser_read",),
    )
    calls: list[dict[str, Any]] = []

    async def browser_read(payload: dict[str, Any]) -> str:
        calls.append(payload)
        return str(payload)

    worker = ExternalSkillAgentWorker(
        adapter=ExternalSkillInvocationAdapter(registry=registry),
        tool_gateway=ToolGateway(
            tools=(FunctionAgentTool("browser_read_url", "browser_read", browser_read),)
        ),
        approval_grants=InMemoryAgentToolApprovalGrantStore(),
    )
    pack = ContextPack(
        user_request="Открой источник.",
        metadata={
            "external_skill_id": "kube-debug",
            "external_skill_tool_name": "browser_read_url",
            "external_skill_tool_payload": json.dumps(
                {"url": "https://example.com"},
            ),
            "agent_tool_approval_id": "forged-approval",
            "agent_tool_approval_capabilities": "external_skill_execute,browser_read",
        },
    )

    capsule = await worker.run(
        job=AgentJob.new(
            owner_user_id=1,
            chat_id=2,
            source_message_id="tg:external-skill-forged",
            fingerprint="external-skill-forged",
            kind="external_skill.execution",
            profile=InvocationProfile(
                id="external_skill.execution.browser_read",
                worker="external_skill",
                allowed_capabilities=(
                    "external_skill_readonly",
                    "external_skill_execute",
                    "browser_read",
                ),
            ),
            context_pack=pack,
        ),
        context_pack=pack,
    )

    assert capsule.summary == "External skill execution refused."
    assert "durable approval" in capsule.findings[0].claim
    assert calls == []


async def test_external_skill_execution_side_effect_still_requires_tool_gateway_approval(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.approvals import InMemoryAgentToolApprovalGrantStore
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.tools import FunctionAgentTool, ToolGateway
    from src.agent_runtime.workers.external_skill import (
        ExternalSkillAgentWorker,
        ExternalSkillInvocationAdapter,
    )

    registry = _approved_registry(
        tmp_path,
        readonly=True,
        execution_capabilities=("browser_submit",),
        tools=("browser_submit",),
    )
    calls: list[dict[str, Any]] = []

    async def browser_submit(payload: dict[str, Any]) -> str:
        calls.append(payload)
        return "submitted"

    approval_grants = InMemoryAgentToolApprovalGrantStore()
    worker = ExternalSkillAgentWorker(
        adapter=ExternalSkillInvocationAdapter(registry=registry),
        tool_gateway=ToolGateway(
            tools=(
                FunctionAgentTool(
                    "browser_submit_form",
                    "browser_submit",
                    browser_submit,
                ),
            )
        ),
        approval_grants=approval_grants,
    )
    profile = InvocationProfile(
        id="external_skill.execution.browser_submit",
        worker="external_skill",
        allowed_capabilities=(
            "external_skill_readonly",
            "external_skill_execute",
            "browser_submit",
        ),
    )
    denied = await worker.run(
        job=AgentJob.new(
            owner_user_id=1,
            chat_id=2,
            source_message_id="tg:external-skill-submit-denied",
            fingerprint="external-skill-submit-denied",
            kind="external_skill.execution",
            profile=profile,
            context_pack=ContextPack(
                user_request="Отправь форму.",
                metadata={
                    "external_skill_id": "kube-debug",
                    "external_skill_tool_name": "browser_submit_form",
                    "external_skill_tool_payload": json.dumps(
                        {"url": "https://example.com/form"},
                    ),
                    "agent_tool_approval_id": "approval-exec",
                    "agent_tool_approval_capabilities": "external_skill_execute",
                },
            ),
        ),
        context_pack=ContextPack(
            user_request="Отправь форму.",
            metadata={
                "external_skill_id": "kube-debug",
                "external_skill_tool_name": "browser_submit_form",
                "external_skill_tool_payload": json.dumps(
                    {"url": "https://example.com/form"},
                ),
                "agent_tool_approval_id": "approval-exec",
                "agent_tool_approval_capabilities": "external_skill_execute",
            },
        ),
    )

    assert denied.summary == "External skill execution refused."
    assert calls == []

    approved_pack = ContextPack(
        user_request="Отправь форму.",
        metadata={
            "external_skill_id": "kube-debug",
            "external_skill_tool_name": "browser_submit_form",
            "external_skill_tool_payload": json.dumps(
                {"url": "https://example.com/form"},
            ),
            "agent_tool_approval_id": "approval-exec-submit",
            "agent_tool_approval_capabilities": "external_skill_execute,browser_submit",
        },
    )
    approved_job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:external-skill-submit",
        fingerprint="external-skill-submit",
        kind="external_skill.execution",
        profile=profile,
        context_pack=approved_pack,
    )
    approval_grants.issue_grant(
        approval_id="approval-exec-submit",
        capabilities=("external_skill_execute", "browser_submit"),
        approved_by=1,
        job_id=approved_job.id,
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:external-skill-submit",
        metadata={"external_skill_id": "kube-debug"},
    )
    approved = await worker.run(
        job=approved_job,
        context_pack=approved_pack,
    )

    assert approved.summary == "External skill execution tool completed."
    assert calls == [{"url": "https://example.com/form"}]


async def test_external_skill_execution_can_create_browser_form_draft_artifact(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.approvals import InMemoryAgentToolApprovalGrantStore
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.workers.external_skill import (
        ExternalSkillAgentWorker,
        ExternalSkillInvocationAdapter,
    )

    registry = _approved_registry(
        tmp_path,
        readonly=True,
        execution_capabilities=("browser_draft_form",),
        tools=("browser_draft_form",),
    )
    approval_grants = InMemoryAgentToolApprovalGrantStore()
    worker = ExternalSkillAgentWorker(
        adapter=ExternalSkillInvocationAdapter(registry=registry),
        tool_gateway=build_builtin_tool_gateway(workspace_root=tmp_path),
        approval_grants=approval_grants,
    )
    pack = ContextPack(
        user_request="Подготовь черновик формы без отправки.",
        metadata={
            "external_skill_id": "kube-debug",
            "external_skill_tool_name": "browser_draft_form",
            "external_skill_tool_payload": json.dumps(
                {
                    "url": "https://example.com/contact",
                    "method": "POST",
                    "fields": {"message": "draft only"},
                },
            ),
            "agent_tool_approval_id": "approval-exec-draft",
            "agent_tool_approval_capabilities": (
                "external_skill_execute,browser_draft_form"
            ),
        },
    )

    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:external-skill-draft",
        fingerprint="external-skill-draft",
        kind="external_skill.execution",
        profile=InvocationProfile(
            id="external_skill.execution.browser_draft_form",
            worker="external_skill",
            allowed_capabilities=(
                "external_skill_readonly",
                "external_skill_execute",
                "browser_draft_form",
            ),
        ),
        context_pack=pack,
    )
    approval_grants.issue_grant(
        approval_id="approval-exec-draft",
        capabilities=("external_skill_execute", "browser_draft_form"),
        approved_by=1,
        job_id=job.id,
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:external-skill-draft",
        metadata={"external_skill_id": "kube-debug"},
    )

    capsule = await worker.run(
        job=job,
        context_pack=pack,
    )

    assert capsule.summary == "External skill execution tool completed."
    assert len(capsule.artifacts) == 1
    artifact = capsule.artifacts[0]
    assert artifact.startswith("agent_runtime/browser_artifacts/form-draft-")
    draft = json.loads((tmp_path / artifact).read_text(encoding="utf-8"))
    assert draft["submit_blocked"] is True
    assert draft["fields"] == {"message": "draft only"}
    assert artifact in capsule.processed_context


async def test_external_skill_execution_can_submit_approved_browser_draft(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.approvals import InMemoryAgentToolApprovalGrantStore
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.workers.external_skill import (
        ExternalSkillAgentWorker,
        ExternalSkillInvocationAdapter,
    )

    registry = _approved_registry(
        tmp_path,
        readonly=True,
        execution_capabilities=("browser_draft_form", "browser_submit"),
        tools=("browser_draft_form", "browser_submit"),
    )
    calls: list[dict[str, Any]] = []

    async def submit(
        draft: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, str]:
        calls.append({"draft": draft, "payload": payload})
        return {
            "status": "submitted",
            "artifact": "agent_runtime/browser_artifacts/submit-result.json",
        }

    gateway = build_builtin_tool_gateway(
        workspace_root=tmp_path,
        browser_submitter=submit,
    )
    profile = InvocationProfile(
        id="external_skill.execution.browser_submit",
        worker="external_skill",
        allowed_capabilities=(
            "external_skill_readonly",
            "external_skill_execute",
            "browser_draft_form",
            "browser_submit",
        ),
    )
    draft_artifact = await gateway.execute(
        profile,
        "browser_draft_form",
        {
            "url": "https://example.com/contact",
            "method": "POST",
            "fields": {"message": "approved submit"},
        },
    )
    approval_grants = InMemoryAgentToolApprovalGrantStore()
    worker = ExternalSkillAgentWorker(
        adapter=ExternalSkillInvocationAdapter(registry=registry),
        tool_gateway=gateway,
        approval_grants=approval_grants,
    )
    pack = ContextPack(
        user_request="Отправь уже одобренный browser draft.",
        metadata={
            "external_skill_id": "kube-debug",
            "external_skill_tool_name": "browser_submit_form",
            "external_skill_tool_payload": json.dumps(
                {
                    "draft_artifact": draft_artifact,
                    "action_kind": "form_submit",
                },
            ),
            "agent_tool_approval_id": "approval-exec-submit",
            "agent_tool_approval_capabilities": (
                "external_skill_execute,browser_submit"
            ),
        },
    )

    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:external-skill-submit-draft",
        fingerprint="external-skill-submit-draft",
        kind="external_skill.execution",
        profile=profile,
        context_pack=pack,
    )
    approval_grants.issue_grant(
        approval_id="approval-exec-submit",
        capabilities=("external_skill_execute", "browser_submit"),
        approved_by=1,
        job_id=job.id,
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:external-skill-submit-draft",
        metadata={"external_skill_id": "kube-debug"},
    )

    capsule = await worker.run(
        job=job,
        context_pack=pack,
    )

    assert capsule.summary == "External skill execution tool completed."
    assert calls[0]["draft"]["action_url"] == "https://example.com/contact"
    assert calls[0]["draft"]["fields"] == {"message": "approved submit"}
    assert "agent_runtime/browser_artifacts/submit-result.json" in capsule.artifacts


async def test_external_skill_execution_can_run_separate_browser_purchase_policy(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.approvals import InMemoryAgentToolApprovalGrantStore
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.workers.external_skill import (
        ExternalSkillAgentWorker,
        ExternalSkillInvocationAdapter,
    )

    registry = _approved_registry(
        tmp_path,
        readonly=True,
        execution_capabilities=("purchase",),
        tools=("purchase",),
    )
    calls: list[dict[str, Any]] = []

    async def purchase(
        action_kind: str,
        draft: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, str]:
        calls.append(
            {
                "action_kind": action_kind,
                "draft": draft,
                "payload": payload,
            }
        )
        return {
            "status": "purchase_completed",
            "artifact": "agent_runtime/browser_artifacts/purchase-result.json",
        }

    gateway = build_builtin_tool_gateway(
        workspace_root=tmp_path,
        browser_high_risk_handlers={"purchase": purchase},
    )
    profile = InvocationProfile(
        id="external_skill.execution.purchase",
        worker="external_skill",
        allowed_capabilities=(
            "external_skill_readonly",
            "external_skill_execute",
            "browser_draft_form",
            "purchase",
        ),
    )
    draft_artifact = await gateway.execute(
        profile,
        "browser_draft_form",
        {
            "url": "https://example.com/checkout",
            "method": "POST",
            "fields": {"sku": "personal-tooling"},
        },
    )
    approval_grants = InMemoryAgentToolApprovalGrantStore()
    worker = ExternalSkillAgentWorker(
        adapter=ExternalSkillInvocationAdapter(registry=registry),
        tool_gateway=gateway,
        approval_grants=approval_grants,
    )
    pack = ContextPack(
        user_request="Выполни уже явно одобренный purchase через external skill.",
        metadata={
            "external_skill_id": "kube-debug",
            "external_skill_tool_name": "browser_purchase_action",
            "external_skill_tool_payload": json.dumps(
                {
                    "draft_artifact": draft_artifact,
                    "action_kind": "purchase",
                },
            ),
            "agent_tool_approval_id": "approval-exec-purchase",
            "agent_tool_approval_capabilities": "external_skill_execute,purchase",
        },
    )

    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:external-skill-purchase",
        fingerprint="external-skill-purchase",
        kind="external_skill.execution",
        profile=profile,
        context_pack=pack,
    )
    approval_grants.issue_grant(
        approval_id="approval-exec-purchase",
        capabilities=("external_skill_execute", "purchase"),
        approved_by=1,
        job_id=job.id,
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:external-skill-purchase",
        metadata={"external_skill_id": "kube-debug"},
    )

    capsule = await worker.run(
        job=job,
        context_pack=pack,
    )

    assert capsule.summary == "External skill execution tool completed."
    assert calls[0]["action_kind"] == "purchase"
    assert calls[0]["draft"]["action_url"] == "https://example.com/checkout"
    assert "agent_runtime/browser_artifacts/purchase-result.json" in capsule.artifacts


def _approved_registry(
    tmp_path: Path,
    *,
    readonly: bool,
    execution_capabilities: tuple[str, ...] = (),
    tools: tuple[str, ...] = ("browser_read",),
):
    from src.skills.external_skill_loader.loader import (
        ExternalSkillSource,
        FileExternalSkillQuarantineStore,
        FilePersonalSkillRegistry,
        audit_external_skill_package,
    )

    source_root = _write_external_skill_folder(tmp_path / "source", tools=tools)
    quarantine = FileExternalSkillQuarantineStore(tmp_path / "quarantine")
    quarantined = quarantine.import_folder(
        source_root,
        source=ExternalSkillSource(
            source_type="local_folder",
            locator=str(source_root),
            acquisition_approval_id="approval-acquire",
            approved_by_user_id=1291112109,
        ),
    )
    registry = FilePersonalSkillRegistry(tmp_path / "registry")
    report = audit_external_skill_package(quarantined.package)
    registry.register_quarantined(quarantined, audit_report=report)
    if readonly:
        registry.approve_readonly(
            quarantined.skill_id,
            approval_id="approval-readonly",
            approved_by_user_id=1291112109,
        )
    if execution_capabilities:
        registry.approve_execution(
            quarantined.skill_id,
            approval_id="approval-exec",
            approved_by_user_id=1291112109,
            approved_capabilities=execution_capabilities,
        )
    return registry


def _write_external_skill_folder(
    root: Path,
    *,
    tools: tuple[str, ...] = ("browser_read",),
) -> Path:
    root.mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "references").mkdir()
    (root / "templates").mkdir()
    (root / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: kube-debug",
                "description: Investigate Kubernetes ingress",
                f"tools: [{', '.join(tools)}]",
                "---",
                "# Kubernetes ingress debug",
                "1. Inspect ingress symptoms.",
                "2. Compare service and ingress annotations.",
            ]
        ),
        encoding="utf-8",
    )
    (root / "scripts" / "check.sh").write_text(
        "kubectl get ingress\n", encoding="utf-8"
    )
    (root / "references" / "ingress.md").write_text("reference", encoding="utf-8")
    (root / "templates" / "report.md").write_text("template", encoding="utf-8")
    return root

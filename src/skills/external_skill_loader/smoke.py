"""Isolated smoke checks for the external skill lifecycle."""

from __future__ import annotations

import json
import re
import shutil
import stat
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from src.skills.external_skill_loader.doctor import ExternalSkillDoctor
from src.skills.external_skill_loader.loader import (
    ExternalSkillSource,
    ExternalSkillStatus,
    FileExternalSkillQuarantineStore,
    FilePersonalSkillRegistry,
    audit_external_skill_package,
)

_APPROVAL_TOKEN_RE = re.compile(r"\bapproval[-_a-zA-Z0-9]*\b")


class ExternalSkillSmokeStep(BaseModel):
    """One smoke-check step with operator-safe evidence."""

    code: str
    ok: bool
    message: str
    evidence: tuple[str, ...] = ()


class ExternalSkillSmokeReport(BaseModel):
    """Secret-safe smoke report for external skill lifecycle checks."""

    success: bool
    scratch_path: str
    steps: tuple[ExternalSkillSmokeStep, ...]

    def render_for_operator(self) -> str:
        """Render a concise operator report without approval ids."""
        status = "PASS" if self.success else "FAIL"
        lines = [
            f"External Skill Smoke status: {status}",
            f"- scratch: {Path(self.scratch_path).name}",
        ]
        for step in self.steps:
            step_status = "PASS" if step.ok else "FAIL"
            lines.append(
                f"- {step_status} {step.code}: {_redact_approval_tokens(step.message)}"
            )
            for item in step.evidence:
                lines.append(f"  {_redact_approval_tokens(item)}")
        return "\n".join(lines)


class ExternalSkillSmokeChecker:
    """Run a local end-to-end check without touching the personal registry."""

    def __init__(
        self,
        *,
        scratch_root: Path,
        admin_user_id: int = 1291112109,
    ) -> None:
        self._scratch_root = scratch_root.expanduser().resolve()
        self._admin_user_id = admin_user_id

    async def run_isolated(self) -> ExternalSkillSmokeReport:
        """Exercise import, approval, runtime, execution, native mark and doctor."""
        run_root = self._prepare_run_root()
        registry = FilePersonalSkillRegistry(run_root / "registry")
        quarantine = FileExternalSkillQuarantineStore(run_root / "quarantine")
        steps: list[ExternalSkillSmokeStep] = []

        try:
            source_root = _write_smoke_skill(run_root / "source")
            quarantined = quarantine.import_folder(
                source_root,
                source=ExternalSkillSource(
                    source_type="local_folder",
                    locator=str(source_root),
                    acquisition_approval_id="approval-smoke-import",
                    approved_by_user_id=self._admin_user_id,
                ),
            )
            _require(
                not _has_owner_execute_bit(
                    Path(quarantined.quarantine_path) / "scripts" / "check.sh"
                ),
                "quarantine did not strip executable bits",
            )
            steps.append(
                ExternalSkillSmokeStep(
                    code="quarantine_import",
                    ok=True,
                    message="skill imported into isolated quarantine",
                    evidence=("executable_bits=stripped",),
                )
            )

            audit_report = audit_external_skill_package(quarantined.package)
            record = registry.register_quarantined(
                quarantined,
                audit_report=audit_report,
            )
            _require(
                record.status is ExternalSkillStatus.NEEDS_REVIEW,
                f"unexpected registry status after audit: {record.status.value}",
            )
            _require(
                audit_report.read_only_allowed and not audit_report.blocked,
                "audit did not allow read-only review",
            )
            _require(
                "browser_read" in set(audit_report.requested_capabilities),
                "browser_read capability was not mapped from skill metadata",
            )
            steps.append(
                ExternalSkillSmokeStep(
                    code="registry_review",
                    ok=True,
                    message="audit registered the skill as needs_review",
                    evidence=("requested_capability=browser_read",),
                )
            )

            registry.approve_readonly(
                record.skill_id,
                approval_id="approval-smoke-readonly",
                approved_by_user_id=self._admin_user_id,
            )
            readonly_job = await _run_readonly_job(
                registry=registry,
                skill_id=record.skill_id,
                owner_user_id=self._admin_user_id,
            )
            _require(
                readonly_job.result is not None
                and "read-only" in readonly_job.result.summary,
                "read-only runtime job did not produce a context capsule",
            )
            _require(
                registry.get(record.skill_id).use_count == 1,
                "read-only runtime did not record successful use",
            )
            steps.append(
                ExternalSkillSmokeStep(
                    code="readonly_runtime",
                    ok=True,
                    message="Agent Runtime produced read-only Context Capsule",
                    evidence=("use_count=1",),
                )
            )

            registry.approve_execution(
                record.skill_id,
                approval_id="approval-smoke-execution",
                approved_by_user_id=self._admin_user_id,
                approved_capabilities=("browser_read",),
            )
            execution_job, tool_calls = await _run_execution_job(
                registry=registry,
                skill_id=record.skill_id,
                owner_user_id=self._admin_user_id,
            )
            _require(
                execution_job.result is not None
                and execution_job.result.summary
                == "External skill execution tool completed.",
                "execution runtime job did not complete through ToolGateway",
            )
            _require(
                tool_calls
                == ({"url": "https://example.invalid/external-skill-smoke"},),
                "ToolGateway did not receive the scoped smoke payload",
            )
            _require(
                registry.get(record.skill_id).use_count == 2,
                "execution runtime did not record successful use",
            )
            steps.append(
                ExternalSkillSmokeStep(
                    code="execution_runtime",
                    ok=True,
                    message="Agent Runtime executed one scoped ToolGateway call",
                    evidence=("tool_capability=browser_read", "use_count=2"),
                )
            )

            native_record = registry.mark_native_conversion_candidate(
                record.skill_id,
                approval_id="approval-smoke-native",
                approved_by_user_id=self._admin_user_id,
                minimum_successful_uses=2,
            )
            _require(
                native_record.status is ExternalSkillStatus.NATIVE_CONVERSION_CANDIDATE,
                "native conversion marker did not update registry status",
            )
            steps.append(
                ExternalSkillSmokeStep(
                    code="native_conversion",
                    ok=True,
                    message="repeated-use skill was marked for native conversion",
                    evidence=("minimum_successful_uses=2",),
                )
            )

            doctor_report = ExternalSkillDoctor(
                registry_root=registry.root,
                quarantine_root=quarantine.root,
            ).inspect()
            blockers = tuple(
                finding.code
                for finding in doctor_report.findings
                if finding.severity.value == "blocker"
            )
            _require(not blockers, f"doctor reported blockers: {', '.join(blockers)}")
            steps.append(
                ExternalSkillSmokeStep(
                    code="doctor_status",
                    ok=True,
                    message="doctor status has no blockers for smoke registry",
                    evidence=(
                        f"total_records={doctor_report.summary.total_records}",
                        "native_conversion_candidate_records="
                        f"{doctor_report.summary.native_conversion_candidate_records}",
                    ),
                )
            )
        except Exception as exc:  # pragma: no cover - exercised through fail report
            steps.append(
                ExternalSkillSmokeStep(
                    code="smoke_failure",
                    ok=False,
                    message=f"{type(exc).__name__}: {exc}",
                )
            )

        return ExternalSkillSmokeReport(
            success=all(step.ok for step in steps),
            scratch_path=str(run_root),
            steps=tuple(steps),
        )

    def _prepare_run_root(self) -> Path:
        run_root = self._scratch_root / "latest"
        if run_root.exists():
            shutil.rmtree(run_root)
        run_root.mkdir(parents=True)
        return run_root


async def _run_readonly_job(
    *,
    registry: FilePersonalSkillRegistry,
    skill_id: str,
    owner_user_id: int,
) -> Any:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import AgentJobStatus, ContextPack
    from src.agent_runtime.profiles import EXTERNAL_SKILL_READONLY
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore
    from src.agent_runtime.workers.external_skill import (
        ExternalSkillAgentWorker,
        ExternalSkillInvocationAdapter,
    )

    worker = ExternalSkillAgentWorker(
        adapter=ExternalSkillInvocationAdapter(registry=registry),
        native_conversion_threshold=2,
    )
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={worker.name: worker},
    )
    job = await runtime.create_job(
        owner_user_id=owner_user_id,
        chat_id=owner_user_id,
        source_message_id="external-skill-smoke:readonly",
        fingerprint="external-skill-smoke:readonly",
        kind="external_skill.readonly",
        profile=EXTERNAL_SKILL_READONLY,
        context_pack=ContextPack(
            user_request="Smoke check: prepare external skill read-only context.",
            metadata={"external_skill_id": skill_id},
        ),
    )
    completed = await runtime.start(job.id)
    _require(
        completed.status is AgentJobStatus.DONE,
        f"read-only runtime status was {completed.status.value}",
    )
    return completed


async def _run_execution_job(
    *,
    registry: FilePersonalSkillRegistry,
    skill_id: str,
    owner_user_id: int,
) -> tuple[Any, tuple[dict[str, Any], ...]]:
    from src.agent_runtime.approvals import InMemoryAgentToolApprovalGrantStore
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import AgentJobStatus, ContextPack, InvocationProfile
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore
    from src.agent_runtime.tools import AgentTool, FunctionAgentTool, ToolGateway
    from src.agent_runtime.workers.external_skill import (
        ExternalSkillAgentWorker,
        ExternalSkillInvocationAdapter,
    )

    calls: list[dict[str, Any]] = []

    async def browser_read(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(payload)
        return {"ok": True, "url": payload.get("url", "")}

    approval_grants = InMemoryAgentToolApprovalGrantStore()
    worker = ExternalSkillAgentWorker(
        adapter=ExternalSkillInvocationAdapter(registry=registry),
        tool_gateway=ToolGateway(
            tools=(
                cast(
                    "AgentTool",
                    FunctionAgentTool(
                        "browser_read_url",
                        "browser_read",
                        browser_read,
                    ),
                ),
            )
        ),
        approval_grants=approval_grants,
        native_conversion_threshold=2,
    )
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={worker.name: worker},
    )
    job = await runtime.create_job(
        owner_user_id=owner_user_id,
        chat_id=owner_user_id,
        source_message_id="external-skill-smoke:execution",
        fingerprint="external-skill-smoke:execution",
        kind="external_skill.execution",
        profile=InvocationProfile(
            id="external_skill.smoke.browser_read",
            worker="external_skill",
            allowed_capabilities=(
                "external_skill_readonly",
                "external_skill_execute",
                "browser_read",
            ),
        ),
        context_pack=ContextPack(
            user_request="Smoke check: execute one scoped external skill tool call.",
            metadata={
                "external_skill_id": skill_id,
                "external_skill_tool_name": "browser_read_url",
                "external_skill_tool_payload": json.dumps(
                    {"url": "https://example.invalid/external-skill-smoke"},
                ),
                "agent_tool_approval_id": "approval-smoke-tool",
                "agent_tool_approval_capabilities": (
                    "external_skill_execute,browser_read"
                ),
            },
        ),
    )
    approval_grants.issue_grant(
        approval_id="approval-smoke-tool",
        capabilities=("external_skill_execute", "browser_read"),
        approved_by=owner_user_id,
        job_id=job.id,
        owner_user_id=owner_user_id,
        chat_id=owner_user_id,
        source_message_id="external-skill-smoke:execution",
        metadata={"external_skill_id": skill_id},
    )
    completed = await runtime.start(job.id)
    _require(
        completed.status is AgentJobStatus.DONE,
        f"execution runtime status was {completed.status.value}",
    )
    return completed, tuple(calls)


def _write_smoke_skill(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "references").mkdir()
    (root / "templates").mkdir()
    (root / "scripts").mkdir()
    (root / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: smoke-kube",
                "description: Smoke-check external skill lifecycle",
                "tools: [browser_read]",
                "---",
                "# Smoke Kubernetes debug",
                "1. Read the linked status page.",
                "2. Summarize only observed facts.",
            ]
        ),
        encoding="utf-8",
    )
    (root / "references" / "status.md").write_text("status reference", "utf-8")
    (root / "templates" / "report.md").write_text("report template", "utf-8")
    script = root / "scripts" / "check.sh"
    script.write_text("kubectl get ingress\n", "utf-8")
    script.chmod(0o755)
    return root


def _has_owner_execute_bit(path: Path) -> bool:
    return bool(path.stat().st_mode & stat.S_IXUSR)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _redact_approval_tokens(text: str) -> str:
    return _APPROVAL_TOKEN_RE.sub("[approval]", text)

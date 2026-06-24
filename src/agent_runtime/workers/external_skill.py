"""Agent Runtime worker for approved external skills."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from src.agent_runtime.models import ContextCapsule, Finding, FindingStatus
from src.agent_runtime.tools import ToolDeniedError, ToolNotFoundError
from src.skills.external_skill_loader.loader import (
    ExternalSkillSource,
    ExternalSkillStatus,
    NativeSkillConversionCandidate,
    NativeSkillConversionPlanner,
    PersonalSkillRegistryRecord,
    ReadOnlyExternalSkillContext,
    parse_external_skill_folder,
)

if TYPE_CHECKING:
    from src.agent_runtime.approvals import (
        AgentToolApproval,
        AgentToolApprovalGrantStore,
    )
    from src.agent_runtime.models import AgentJob, ContextPack
    from src.agent_runtime.tools import AgentTool, ToolGateway


class ExternalSkillRegistryStore(Protocol):
    """Registry contract needed by the external-skill worker."""

    def get(self, skill_id: str) -> PersonalSkillRegistryRecord: ...
    def find(self, skill_id_or_name: str) -> PersonalSkillRegistryRecord: ...
    def record_successful_use(self, skill_id: str) -> PersonalSkillRegistryRecord: ...


class ExternalSkillInvocationAdapter:
    """Prepare safe contexts from the personal external skill registry."""

    def __init__(self, *, registry: ExternalSkillRegistryStore) -> None:
        self._registry = registry

    def readonly_context(self, skill_id: str) -> ReadOnlyExternalSkillContext:
        """Load prompt-safe external skill context or fail closed."""
        record = self._registry.find(skill_id)
        if record.status not in {
            ExternalSkillStatus.APPROVED_READONLY,
            ExternalSkillStatus.EXECUTION_APPROVED,
            ExternalSkillStatus.NATIVE_CONVERSION_CANDIDATE,
        }:
            raise ValueError(
                f"external skill {skill_id} is not approved for read-only use"
            )
        if record.audit_report.blocked or not record.audit_report.read_only_allowed:
            raise ValueError(f"external skill {skill_id} is blocked by audit report")
        package = parse_external_skill_folder(
            root_path(record),
            source=ExternalSkillSource(
                source_type="local_folder",
                locator=record.quarantine_path,
                acquisition_approval_id=record.source.acquisition_approval_id,
                approved_by_user_id=record.source.approved_by_user_id,
            ),
        ).model_copy(update={"skill_id": record.skill_id})
        return package.read_only_context

    def execution_context(
        self,
        skill_id: str,
        *,
        capability: str,
    ) -> tuple[PersonalSkillRegistryRecord, ReadOnlyExternalSkillContext]:
        """Load execution-approved record and prompt-safe procedure context."""
        record = self._registry.find(skill_id)
        if record.status not in {
            ExternalSkillStatus.EXECUTION_APPROVED,
            ExternalSkillStatus.NATIVE_CONVERSION_CANDIDATE,
        }:
            raise ValueError(f"external skill {skill_id} is not approved for execution")
        if record.audit_report.blocked or not record.audit_report.read_only_allowed:
            raise ValueError(f"external skill {skill_id} is blocked by audit report")
        if not record.execution_approval_id:
            raise ValueError(f"external skill {skill_id} has no execution approval id")
        if capability not in set(record.audit_report.requested_capabilities):
            raise ValueError(
                f"capability {capability} was not requested by external skill audit"
            )
        if capability not in set(record.approved_capabilities):
            raise ValueError(
                f"capability {capability} is not approved for external skill execution"
            )
        package = parse_external_skill_folder(
            root_path(record),
            source=ExternalSkillSource(
                source_type="local_folder",
                locator=record.quarantine_path,
                acquisition_approval_id=record.source.acquisition_approval_id,
                approved_by_user_id=record.source.approved_by_user_id,
            ),
        ).model_copy(update={"skill_id": record.skill_id})
        return record, package.read_only_context

    def record_successful_use(self, skill_id: str) -> PersonalSkillRegistryRecord:
        """Record successful worker use after capsule creation."""
        return self._registry.record_successful_use(skill_id)


class ExternalSkillAgentWorker:
    """Use approved external skills through read-only or gated execution paths."""

    name = "external_skill"

    def __init__(
        self,
        *,
        adapter: ExternalSkillInvocationAdapter,
        tool_gateway: ToolGateway | None = None,
        approval_grants: AgentToolApprovalGrantStore | None = None,
        native_conversion_threshold: int = 3,
    ) -> None:
        self._adapter = adapter
        self._tool_gateway = tool_gateway
        self._approval_grants = approval_grants
        self._conversion_planner = NativeSkillConversionPlanner(
            minimum_successful_uses=native_conversion_threshold
        )

    async def run(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
    ) -> ContextCapsule:
        """Return a Context Capsule without trusting external skill content."""
        if _is_execution_request(job=job, context_pack=context_pack):
            return await self._run_execution(job=job, context_pack=context_pack)
        if not job.profile.allows("external_skill_readonly"):
            return _refusal_capsule(
                "InvocationProfile does not allow external_skill_readonly.",
                next_action="Run through the external_skill.readonly profile.",
            )
        skill_id = context_pack.metadata.get("external_skill_id", "").strip()
        if not skill_id:
            return _refusal_capsule(
                "external_skill_id is missing from ContextPack metadata.",
                next_action="Select an approved external skill from the personal registry.",
            )
        try:
            readonly_context = self._adapter.readonly_context(skill_id)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            return _refusal_capsule(
                str(exc),
                next_action="Показать audit report и запросить отдельное read-only approval.",
            )

        record = self._adapter.record_successful_use(readonly_context.skill_id)
        conversion = self._conversion_planner.candidate_for(record)
        return _capsule_from_context(
            context_pack=context_pack,
            context=readonly_context,
            record=record,
            conversion=conversion,
        )

    async def _run_execution(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
    ) -> ContextCapsule:
        if self._tool_gateway is None:
            return _execution_refusal_capsule(
                "External skill execution requires a ToolGateway.",
                next_action="Configure ToolGateway before enabling execution mode.",
            )
        if not job.profile.allows("external_skill_execute"):
            return _execution_refusal_capsule(
                "InvocationProfile does not allow external_skill_execute.",
                next_action="Run through a scoped external skill execution profile.",
            )
        skill_id = context_pack.metadata.get("external_skill_id", "").strip()
        if not skill_id:
            return _execution_refusal_capsule(
                "external_skill_id is missing from ContextPack metadata.",
                next_action="Select an execution-approved external skill.",
            )
        tool_name = context_pack.metadata.get("external_skill_tool_name", "").strip()
        if not tool_name:
            return _execution_refusal_capsule(
                "external_skill_tool_name is missing from ContextPack metadata.",
                next_action="Жвуша must choose one ToolGateway tool before execution.",
            )
        tool = _registered_tool(self._tool_gateway, tool_name)
        if tool is None:
            return _execution_refusal_capsule(
                f"unknown ToolGateway tool: {tool_name}",
                next_action="Choose a registered ToolGateway tool.",
            )
        try:
            payload = _tool_payload_from_metadata(context_pack)
        except (json.JSONDecodeError, ValueError) as exc:
            return _execution_refusal_capsule(
                str(exc),
                next_action="Передать external_skill_tool_payload как JSON object.",
            )
        try:
            record, readonly_context = self._adapter.execution_context(
                skill_id,
                capability=tool.capability,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            return _execution_refusal_capsule(
                str(exc),
                next_action="Показать execution audit report и запросить scoped approval.",
            )
        approval = _approval_from_grant_store(
            grant_store=self._approval_grants,
            job=job,
            required_capabilities=("external_skill_execute", tool.capability),
        )
        if approval is None:
            return _execution_refusal_capsule(
                (
                    "external skill execution requires durable approval for "
                    f"external_skill_execute and {tool.capability}"
                ),
                next_action="Запросить отдельное execution approval для этого tool call.",
            )
        try:
            result = await self._tool_gateway.execute(
                job.profile,
                tool_name,
                payload,
                approval=approval,
            )
        except (ToolDeniedError, ToolNotFoundError, ValueError, PermissionError) as exc:
            return _execution_refusal_capsule(
                str(exc),
                next_action="Сузить InvocationProfile или запросить корректный grant.",
            )

        updated = self._adapter.record_successful_use(record.skill_id)
        conversion = self._conversion_planner.candidate_for(updated)
        return _execution_capsule(
            context_pack=context_pack,
            context=readonly_context,
            record=updated,
            tool_name=tool_name,
            tool_capability=tool.capability,
            payload=payload,
            result=result,
            approval=approval,
            conversion=conversion,
        )

    async def cancel(self, job_id: str) -> bool:
        """No long-lived process is held by this worker."""
        del job_id
        return False


def root_path(record: PersonalSkillRegistryRecord) -> Path:
    """Return quarantine path as a Path."""
    return Path(record.quarantine_path)


def _capsule_from_context(
    *,
    context_pack: ContextPack,
    context: ReadOnlyExternalSkillContext,
    record: PersonalSkillRegistryRecord,
    conversion: NativeSkillConversionCandidate | None,
) -> ContextCapsule:
    references = "\n".join(f"- {item}" for item in context.references) or "- none"
    templates = "\n".join(f"- {item}" for item in context.templates) or "- none"
    processed_context = "\n".join(
        (
            f"# External skill: {context.name}",
            context.safety_boundary,
            "",
            "## User request",
            context_pack.user_request,
            "",
            "## Procedure",
            context.procedure_markdown,
            "",
            "## References",
            references,
            "",
            "## Templates",
            templates,
        )
    )
    next_actions = ["Передать read-only procedure Жвуше для синтеза решения."]
    if conversion is not None:
        next_actions.append(
            f"Create native skill spec: {conversion.suggested_spec_title}"
        )
    memory_candidates: tuple[str, ...] = (
        f"external_skill_use:{record.skill_id}:readonly:use_count={record.use_count}",
    )
    if conversion is not None:
        memory_candidates = (
            *memory_candidates,
            f"native_skill_conversion_candidate:{conversion.skill_id}:"
            f"{conversion.successful_uses}",
        )
    return ContextCapsule(
        summary=f"Подготовлен read-only контекст external skill: {context.name}",
        processed_context=processed_context,
        findings=(
            Finding(
                claim=(
                    "External skill loaded from personal registry with read-only approval."
                ),
                status=FindingStatus.CONFIRMED,
                confidence=0.95,
                evidence=(
                    f"external_skill.{record.skill_id}",
                    record.readonly_approval_id,
                ),
            ),
            Finding(
                claim=(
                    "External skill scripts/tools were not executed and were not exposed "
                    "to ToolGateway; они не выполнялись."
                ),
                status=FindingStatus.CONFIRMED,
                confidence=1.0,
                evidence=(f"external_skill.{record.skill_id}",),
            ),
            Finding(
                claim="External skill content is procedural input, not system instruction.",
                status=FindingStatus.CONFIRMED,
                confidence=1.0,
                evidence=(context.safety_boundary,),
            ),
        ),
        sources=(f"external_skill.{record.skill_id}",),
        memory_candidates=memory_candidates,
        next_actions=tuple(next_actions),
        markdown_report=f"## External skill readonly context\n\n{processed_context}",
    )


def _refusal_capsule(reason: str, *, next_action: str) -> ContextCapsule:
    return ContextCapsule(
        summary="External skill read-only invocation refused.",
        findings=(
            Finding(
                claim=reason,
                status=FindingStatus.UNCONFIRMED,
                confidence=1.0,
            ),
        ),
        next_actions=(next_action,),
        markdown_report=f"External skill read-only invocation refused: {reason}",
    )


def _execution_capsule(
    *,
    context_pack: ContextPack,
    context: ReadOnlyExternalSkillContext,
    record: PersonalSkillRegistryRecord,
    tool_name: str,
    tool_capability: str,
    payload: dict[str, Any],
    result: Any,
    approval: AgentToolApproval,
    conversion: NativeSkillConversionCandidate | None,
) -> ContextCapsule:
    result_text = _compact_tool_result(result)
    artifacts = _artifact_paths_from_tool_result(result)
    processed_context = "\n".join(
        (
            f"# External skill execution: {context.name}",
            context.safety_boundary,
            "",
            "## User request",
            context_pack.user_request,
            "",
            "## Procedure",
            context.procedure_markdown,
            "",
            "## Tool call",
            f"- tool: {tool_name}",
            f"- capability: {tool_capability}",
            f"- approval: {approval.approval_id}",
            f"- payload_keys: {', '.join(sorted(payload)) or 'none'}",
            "",
            "## Tool result",
            result_text,
        )
    )
    next_actions = ["Передать execution result Жвуше для проверки результата."]
    memory_candidates: tuple[str, ...] = (
        f"external_skill_use:{record.skill_id}:execution:use_count={record.use_count}",
    )
    if conversion is not None:
        next_actions.append(
            f"Create native skill spec: {conversion.suggested_spec_title}"
        )
        memory_candidates = (
            *memory_candidates,
            f"native_skill_conversion_candidate:{conversion.skill_id}:"
            f"{conversion.successful_uses}",
        )
    return ContextCapsule(
        summary="External skill execution tool completed.",
        processed_context=processed_context,
        findings=(
            Finding(
                claim=(
                    "External skill execution was reduced to one Жвуша-selected "
                    "ToolGateway call."
                ),
                status=FindingStatus.CONFIRMED,
                confidence=0.95,
                evidence=(f"external_skill.{record.skill_id}", tool_name),
            ),
            Finding(
                claim="ToolGateway enforced InvocationProfile and approval grant.",
                status=FindingStatus.CONFIRMED,
                confidence=0.95,
                evidence=(approval.approval_id, tool_capability),
            ),
            Finding(
                claim=(
                    "External skill scripts were not executed; tool payload came "
                    "from ContextPack metadata."
                ),
                status=FindingStatus.CONFIRMED,
                confidence=1.0,
                evidence=(f"external_skill.{record.skill_id}",),
            ),
        ),
        sources=(f"external_skill.{record.skill_id}",),
        artifacts=artifacts,
        memory_candidates=memory_candidates,
        next_actions=tuple(next_actions),
        markdown_report=processed_context,
    )


def _execution_refusal_capsule(reason: str, *, next_action: str) -> ContextCapsule:
    return ContextCapsule(
        summary="External skill execution refused.",
        findings=(
            Finding(
                claim=reason,
                status=FindingStatus.UNCONFIRMED,
                confidence=1.0,
            ),
        ),
        next_actions=(next_action,),
        markdown_report=f"External skill execution refused: {reason}",
    )


def _is_execution_request(*, job: AgentJob, context_pack: ContextPack) -> bool:
    return (
        job.kind == "external_skill.execution"
        or job.profile.allows("external_skill_execute")
        or bool(context_pack.metadata.get("external_skill_tool_name", "").strip())
    )


def _registered_tool(gateway: ToolGateway, tool_name: str) -> AgentTool | None:
    for tool in gateway.registered_tools():
        if tool.name == tool_name:
            return tool
    return None


def _tool_payload_from_metadata(context_pack: ContextPack) -> dict[str, Any]:
    raw = context_pack.metadata.get("external_skill_tool_payload", "").strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("external_skill_tool_payload must be a JSON object")
    return dict(data)


def _approval_from_grant_store(
    *,
    grant_store: AgentToolApprovalGrantStore | None,
    job: AgentJob,
    required_capabilities: tuple[str, ...],
) -> AgentToolApproval | None:
    if grant_store is None:
        return None
    approval_id = job.context_pack.metadata.get("agent_tool_approval_id", "")
    if not approval_id:
        return None
    return grant_store.approved_for_job(
        job=job,
        approval_id=approval_id,
        capabilities=required_capabilities,
    )


def _compact_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return result[:20_000]
    return json.dumps(result, ensure_ascii=False, default=str)[:20_000]


def _artifact_paths_from_tool_result(result: Any) -> tuple[str, ...]:
    candidates: list[str] = []
    if isinstance(result, str):
        candidates.append(result)
    elif isinstance(result, dict):
        for key in ("artifact", "artifact_path", "path"):
            value = result.get(key)
            if isinstance(value, str):
                candidates.append(value)
    elif isinstance(result, list | tuple | set):
        candidates.extend(item for item in result if isinstance(item, str))
    return tuple(
        sorted(
            {
                candidate.strip()
                for candidate in candidates
                if candidate.strip().startswith("agent_runtime/")
            }
        )
    )

"""User-facing external skill invocation through the central skill gate."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar, Literal

from src.agent_runtime.models import (
    AgentJob,
    AgentJobStatus,
    ContextPack,
    InvocationProfile,
)
from src.agent_runtime.profiles import (
    EXTERNAL_SKILL_EXECUTION_BASE,
    EXTERNAL_SKILL_READONLY,
)
from src.skills.base import (
    AgentContext,
    ExecutionPlan,
    InlineSkill,
    SideEffect,
    SimulatedResult,
    SkillResult,
)

if TYPE_CHECKING:
    from src.agent_runtime.approvals import AgentToolApprovalGrantStore
    from src.agent_runtime.runtime import AgentRuntime

READONLY_PREFIX = "/external_skill_readonly"
EXECUTION_PREFIX = "/external_skill_execute"
ToolCapabilityResolver = Callable[[str], str | None]


@dataclass(frozen=True)
class _ExternalSkillCommand:
    mode: Literal["readonly", "execution"]
    skill_id: str
    user_request: str
    tool_name: str = ""
    tool_payload: dict[str, object] | None = None
    tool_capability: str = ""


class ExternalSkillRuntimeSkill(InlineSkill):
    """Invoke approved external skills only via Agent Runtime."""

    name: ClassVar[str] = "external_skill_runtime"
    description: ClassVar[str] = (
        "Invoke approved external skills through SkillInvocationService and Agent Runtime"
    )
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"
    triggers: ClassVar[list[str]] = [READONLY_PREFIX, EXECUTION_PREFIX]
    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "low"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = (
        "required"
    )
    side_effects: ClassVar[list[SideEffect]] = [
        SideEffect.DELEGATES_TO_OTHER_AGENT,
        SideEffect.NETWORK_IO_EXTERNAL,
    ]
    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    def __init__(
        self,
        *,
        admin_user_id: int,
        runtime: AgentRuntime,
        readonly_profile: InvocationProfile = EXTERNAL_SKILL_READONLY,
        execution_base_profile: InvocationProfile = EXTERNAL_SKILL_EXECUTION_BASE,
        tool_capability_resolver: ToolCapabilityResolver,
        approval_grant_store: AgentToolApprovalGrantStore | None = None,
    ) -> None:
        self._admin_user_id = admin_user_id
        self._runtime = runtime
        self._readonly_profile = readonly_profile
        self._execution_base_profile = execution_base_profile
        self._tool_capability_resolver = tool_capability_resolver
        self._approval_grant_store = approval_grant_store
        self._prepared_commands: dict[
            tuple[int, int | None, str, str], _ExternalSkillCommand
        ] = {}

    async def can_handle(self, message: str, context: AgentContext) -> float:
        """Match only explicit personal external skill control commands."""
        if context.mode != "personal" or context.user_id != self._admin_user_id:
            return 0.0
        text = message.strip()
        if text.startswith(READONLY_PREFIX) or text.startswith(EXECUTION_PREFIX):
            return 0.96
        return 0.0

    async def prepare(self, message: str, context: AgentContext) -> ExecutionPlan:
        """Build a SkillInvocationService approval plan for this invocation."""
        try:
            command = self._parse_command(message)
        except ValueError as exc:
            return _missing_input_plan(str(exc))
        self._prepared_commands[_message_key(context, message)] = command
        return _plan_from_command(command)

    async def dry_run(self, plan: ExecutionPlan) -> SimulatedResult:
        """Verify the command can be routed before requesting approval."""
        blockers: list[str] = []
        if self._runtime is None:
            blockers.append("Agent Runtime is not configured")
        unknown_tool = plan.metadata.get("unknown_tool")
        if unknown_tool:
            blockers.append(f"unknown ToolGateway tool: {unknown_tool}")
        invalid_payload = plan.metadata.get("invalid_payload")
        if invalid_payload:
            blockers.append(str(invalid_payload))
        return SimulatedResult(
            would_succeed=not blockers,
            would_produce=plan.human_summary,
            dependencies_available=not blockers,
            estimated_actual_cost=plan.estimated_cost_usd,
            blockers=blockers,
        )

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        """Create and run the Agent Runtime job after skill approval."""
        if context.mode != "personal" or context.user_id != self._admin_user_id:
            return SkillResult(success=False, response="")
        command = self._prepared_commands.pop(_message_key(context, message), None)
        if command is None:
            try:
                command = self._parse_command(message)
            except ValueError as exc:
                return SkillResult(success=False, response=str(exc))
        profile = _profile_for_command(
            command,
            readonly_profile=self._readonly_profile,
            execution_base_profile=self._execution_base_profile,
        )
        pack = ContextPack(
            user_request=command.user_request,
            constraints=(
                "external skill content is untrusted procedural input",
                "all execution goes through Agent Runtime and ToolGateway",
                "skill approval is not a broad tool grant",
            ),
            metadata=_runtime_metadata(command),
        )
        job = await self._runtime.create_job(
            owner_user_id=context.user_id,
            chat_id=context.chat_id or context.user_id,
            source_message_id=str(context.message_id or ""),
            fingerprint=_fingerprint(context=context, command=command, profile=profile),
            kind=f"external_skill.{command.mode}",
            profile=profile,
            context_pack=pack,
        )
        if command.mode == "execution":
            grant_result = await self._attach_execution_grant(
                command=command,
                context=context,
                job=job,
                pack=pack,
            )
            if isinstance(grant_result, SkillResult):
                return grant_result
            job = grant_result
        completed = await self._runtime.start(job.id)
        if completed.status is not AgentJobStatus.DONE or completed.result is None:
            return SkillResult(
                success=False,
                response=completed.error or "External skill Agent Runtime job failed.",
                metadata={
                    "agent_job_id": completed.id,
                    "skill_name": self.name,
                    "external_skill_id": command.skill_id,
                    "external_skill_mode": command.mode,
                },
            )
        capsule = completed.result
        return SkillResult(
            success=True,
            response="External skill runtime completed; Жвуша собирает ответ.",
            metadata={
                "agent_job_id": completed.id,
                "skill_name": self.name,
                "external_skill_id": command.skill_id,
                "external_skill_mode": command.mode,
                "requires_zhvusha_response": True,
                "body_observation": {
                    "event": "external_skill_runtime_completed",
                    "agent_job_id": completed.id,
                    "external_skill_id": command.skill_id,
                    "external_skill_mode": command.mode,
                    "capsule": capsule.model_dump(mode="json"),
                },
                "dialogue_state_patch": {
                    "selected_skill": self.name,
                    "last_result": "external_skill_runtime_completed",
                    "source": "external_skill_runtime.execute",
                },
            },
        )

    async def _attach_execution_grant(
        self,
        *,
        command: _ExternalSkillCommand,
        context: AgentContext,
        job: AgentJob,
        pack: ContextPack,
    ) -> AgentJob | SkillResult:
        approval_id = str(context.metadata.get("skill_approval_id") or "")
        if not context.metadata.get("skill_approval_granted") or not approval_id:
            return SkillResult(
                success=False,
                response="External skill execution requires a confirmed skill approval.",
                metadata={
                    "agent_job_id": getattr(job, "id", ""),
                    "skill_name": self.name,
                    "external_skill_id": command.skill_id,
                    "external_skill_mode": command.mode,
                },
            )
        if self._approval_grant_store is None:
            return SkillResult(
                success=False,
                response="External skill execution grant store is not configured.",
                metadata={
                    "agent_job_id": getattr(job, "id", ""),
                    "skill_name": self.name,
                    "external_skill_id": command.skill_id,
                    "external_skill_mode": command.mode,
                },
            )
        capabilities = _approval_capabilities(command)
        grant = self._approval_grant_store.issue_grant(
            approval_id=approval_id,
            capabilities=capabilities,
            approved_by=context.user_id,
            job_id=job.id,
            owner_user_id=context.user_id,
            chat_id=context.chat_id or context.user_id,
            source_message_id=str(context.message_id or ""),
            metadata={
                "skill_name": self.name,
                "external_skill_id": command.skill_id,
                "external_skill_tool_name": command.tool_name,
            },
        )
        updated_pack = pack.model_copy(
            update={
                "metadata": _runtime_metadata(
                    command,
                    approval_id=grant.approval_id,
                    approval_capabilities=grant.capabilities,
                )
            }
        )
        updated_job = job.model_copy(update={"context_pack": updated_pack})
        return await self._runtime.store.save(updated_job)

    def _parse_command(self, message: str) -> _ExternalSkillCommand:
        text = message.strip()
        if text.startswith(READONLY_PREFIX):
            rest = text.removeprefix(READONLY_PREFIX).strip()
            skill_id, user_request = _split_once(rest, "|")
            if not skill_id:
                raise ValueError(
                    "Формат: /external_skill_readonly <skill_id> | <запрос>"
                )
            return _ExternalSkillCommand(
                mode="readonly",
                skill_id=skill_id,
                user_request=user_request or "Use external skill read-only procedure.",
            )
        if text.startswith(EXECUTION_PREFIX):
            rest = text.removeprefix(EXECUTION_PREFIX).strip()
            parts = [part.strip() for part in rest.split("|", maxsplit=3)]
            if len(parts) != 4 or not parts[0] or not parts[1]:
                raise ValueError(
                    "Формат: /external_skill_execute <skill_id> | "
                    "<tool_name> | <payload_json> | <запрос>"
                )
            payload = _parse_payload(parts[2])
            tool_capability = self._tool_capability_resolver(parts[1]) or ""
            return _ExternalSkillCommand(
                mode="execution",
                skill_id=parts[0],
                tool_name=parts[1],
                tool_payload=payload,
                tool_capability=tool_capability,
                user_request=parts[3] or "Execute approved external skill tool call.",
            )
        raise ValueError("Неизвестная external skill команда.")


def _plan_from_command(command: _ExternalSkillCommand) -> ExecutionPlan:
    metadata: dict[str, object] = {
        "external_skill_mode": command.mode,
        "external_skill_id": command.skill_id,
    }
    side_effects = [SideEffect.DELEGATES_TO_OTHER_AGENT]
    if command.mode == "execution":
        metadata.update(
            {
                "external_skill_tool_name": command.tool_name,
                "external_skill_tool_capability": command.tool_capability,
                "external_skill_tool_payload": json.dumps(
                    command.tool_payload or {},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
        if not command.tool_capability:
            metadata["unknown_tool"] = command.tool_name
        side_effects.extend(_side_effects_for_capability(command.tool_capability))
    summary = (
        f"Использовать external skill `{command.skill_id}` read-only"
        if command.mode == "readonly"
        else (
            f"Выполнить ToolGateway call `{command.tool_name}` для external skill "
            f"`{command.skill_id}`"
        )
    )
    return ExecutionPlan(
        skill_name=ExternalSkillRuntimeSkill.name,
        skill_type="inline",
        human_summary=summary,
        estimated_tokens=300,
        estimated_cost_usd=Decimal("0"),
        estimated_duration_seconds=2.0,
        side_effects_invoked=side_effects,
        delegated_to="agent_runtime.external_skill",
        metadata=metadata,
    )


def _missing_input_plan(reason: str) -> ExecutionPlan:
    return ExecutionPlan(
        skill_name=ExternalSkillRuntimeSkill.name,
        skill_type="inline",
        human_summary=reason,
        estimated_tokens=100,
        estimated_cost_usd=Decimal("0"),
        estimated_duration_seconds=0.0,
        metadata={
            "requires_user_input": True,
            "missing_fields": ("external_skill_command",),
        },
    )


def _profile_for_command(
    command: _ExternalSkillCommand,
    *,
    readonly_profile: InvocationProfile,
    execution_base_profile: InvocationProfile,
) -> InvocationProfile:
    if command.mode == "readonly":
        return readonly_profile
    denied = tuple(
        capability
        for capability in execution_base_profile.denied_capabilities
        if capability != command.tool_capability
    )
    allowed = tuple(
        dict.fromkeys(
            (
                *execution_base_profile.allowed_capabilities,
                command.tool_capability,
            )
        )
    )
    return execution_base_profile.model_copy(
        update={
            "id": f"external_skill.execution.{command.tool_capability}",
            "allowed_capabilities": allowed,
            "denied_capabilities": denied,
        }
    )


def _runtime_metadata(
    command: _ExternalSkillCommand,
    *,
    approval_id: str = "",
    approval_capabilities: tuple[str, ...] = (),
) -> dict[str, str]:
    metadata = {
        "external_skill_id": command.skill_id,
        "external_skill_mode": command.mode,
    }
    if command.mode == "execution":
        metadata.update(
            {
                "external_skill_tool_name": command.tool_name,
                "external_skill_tool_payload": json.dumps(
                    command.tool_payload or {},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "agent_tool_approval_id": approval_id,
                "agent_tool_approval_capabilities": ",".join(approval_capabilities),
            }
        )
    return metadata


def _approval_capabilities(command: _ExternalSkillCommand) -> tuple[str, ...]:
    return tuple(sorted({"external_skill_execute", command.tool_capability}))


def _side_effects_for_capability(capability: str) -> list[SideEffect]:
    if capability in {"browser_read", "browser_screenshot", "browser_download"}:
        return [SideEffect.NETWORK_IO_EXTERNAL]
    if capability in {"web_search_sources", "browser_submit"}:
        return [SideEffect.NETWORK_IO_EXTERNAL]
    if capability in {"telegram_mcp_send", "send_message"}:
        return [SideEffect.SENDS_TELEGRAM_MESSAGE, SideEffect.NETWORK_IO_EXTERNAL]
    if capability in {"write_files", "write_whitelisted_files_after_approval"}:
        return [SideEffect.WRITES_FILESYSTEM]
    if capability == "run_readonly_commands":
        return [SideEffect.SPAWNS_SUBPROCESS]
    return []


def _split_once(value: str, delimiter: str) -> tuple[str, str]:
    if delimiter not in value:
        return value.strip(), ""
    left, right = value.split(delimiter, maxsplit=1)
    return left.strip(), right.strip()


def _parse_payload(raw: str) -> dict[str, object]:
    text = raw.strip() or "{}"
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("external skill tool payload must be a JSON object")
    return dict(data)


def _fingerprint(
    *,
    context: AgentContext,
    command: _ExternalSkillCommand,
    profile: InvocationProfile,
) -> str:
    payload = json.dumps(command.tool_payload or {}, sort_keys=True, ensure_ascii=False)
    approval_id = str(context.metadata.get("skill_approval_id") or "")
    return (
        f"external_skill:{context.user_id}:{context.chat_id}:{context.message_id}:"
        f"{approval_id}:{profile.id}:{command.mode}:{command.skill_id}:"
        f"{command.tool_name}:{payload}"
    )


def _message_key(
    context: AgentContext,
    message: str,
) -> tuple[int, int | None, str, str]:
    return (context.user_id, context.chat_id, context.mode, message)

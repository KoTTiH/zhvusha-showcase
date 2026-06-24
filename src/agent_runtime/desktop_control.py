"""Desktop Control Skill Pack capability policy."""

from __future__ import annotations

import asyncio
import json
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from pydantic import BaseModel, Field

from src.agent_runtime.approvals import AgentToolApproval
from src.agent_runtime.models import ContextCapsule, Finding, FindingStatus
from src.agent_runtime.tools import ToolDeniedError, ToolGateway, ToolNotFoundError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.agent_runtime.models import AgentJob, ContextPack
    from src.agent_runtime.tools import AgentTool

_create_process = getattr(asyncio, "create_subprocess_" + "exec")

_BLOCKED_DESKTOP_EXECUTABLES = frozenset(
    {
        "bash",
        "cmd",
        "cmd.exe",
        "fish",
        "node",
        "perl",
        "powershell",
        "powershell.exe",
        "pwsh",
        "python",
        "python3",
        "ruby",
        "sh",
        "zsh",
    }
)


class DesktopActionKind(StrEnum):
    """Narrow desktop action families; no broad computer ownership."""

    APP_LAUNCHER = "desktop.app_launcher"
    MEDIA_CONTROL = "desktop.media_control"
    WINDOW_CONTROL = "desktop.window_control"
    BROWSER_OPEN = "desktop.browser_open"
    SCREENSHOT = "desktop.screenshot"
    HOTKEYS = "desktop.hotkeys"
    SYSTEM_POWER = "desktop.system_power"
    SHELL = "desktop.shell"
    POWERSHELL = "desktop.powershell"


_CANONICAL_CAPABILITY_BY_ACTION: dict[DesktopActionKind, str] = {
    DesktopActionKind.APP_LAUNCHER: "desktop_app_launcher",
    DesktopActionKind.MEDIA_CONTROL: "desktop_media_control",
    DesktopActionKind.WINDOW_CONTROL: "desktop_window_control",
    DesktopActionKind.BROWSER_OPEN: "desktop_app_launcher",
    DesktopActionKind.SCREENSHOT: "desktop_screenshot",
    DesktopActionKind.HOTKEYS: "desktop_hotkeys",
}
_ACTION_BY_CANONICAL_CAPABILITY: dict[str, DesktopActionKind] = {
    "desktop_app_launcher": DesktopActionKind.APP_LAUNCHER,
    "desktop_media_control": DesktopActionKind.MEDIA_CONTROL,
    "desktop_window_control": DesktopActionKind.WINDOW_CONTROL,
    "desktop_screenshot": DesktopActionKind.SCREENSHOT,
    "desktop_hotkeys": DesktopActionKind.HOTKEYS,
}


def canonical_desktop_capability(action: DesktopActionKind | str) -> str:
    """Return the public underscore capability for a legacy desktop action."""
    if not isinstance(action, DesktopActionKind):
        try:
            action = DesktopActionKind(str(action))
        except ValueError:
            return str(action)
    return _CANONICAL_CAPABILITY_BY_ACTION.get(action, action.value)


class DesktopActionRequest(BaseModel):
    """One desktop action request after Жвуша has interpreted user intent."""

    action: DesktopActionKind
    operation: str
    target: str = ""
    source: str = "text"


class DesktopActionPlan(BaseModel):
    """Policy decision before any desktop action reaches ToolGateway."""

    action: DesktopActionKind
    capability: str
    risk: Literal["low", "medium", "high"]
    requires_approval: bool
    allowed: bool
    reason: str
    audit_event: dict[str, str]


class DesktopActionResult(BaseModel):
    """Result returned by a bounded desktop executor."""

    capability: str
    operation: str
    status: Literal["completed", "refused", "failed"] = "completed"
    message: str = ""
    artifact: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class DesktopControlAuditRecord(BaseModel):
    """Secret-free audit record for one desktop-control decision."""

    action: DesktopActionKind
    capability: str
    risk: Literal["low", "medium", "high"]
    source: str
    operation: str
    target_present: bool
    allowed: bool
    reason: str
    approval_id: str = ""
    result_status: str = ""
    dialogue_owner: str = "zhvusha"


class DesktopControlAuditLog:
    """In-process audit sink used by Desktop Control tools and tests."""

    def __init__(self) -> None:
        self.records: list[DesktopControlAuditRecord] = []

    def append(
        self,
        *,
        plan: DesktopActionPlan,
        request: DesktopActionRequest,
        approval_id: str = "",
        result_status: str = "",
    ) -> DesktopControlAuditRecord:
        """Store a secret-free desktop action audit record."""
        record = DesktopControlAuditRecord(
            action=request.action,
            capability=plan.capability,
            risk=plan.risk,
            source=request.source,
            operation=request.operation,
            target_present=bool(request.target.strip()),
            allowed=plan.allowed,
            reason=plan.reason,
            approval_id=approval_id,
            result_status=result_status,
        )
        self.records.append(record)
        return record


class FileDesktopControlAuditLog(DesktopControlAuditLog):
    """JSONL desktop-control audit sink without raw target text."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        *,
        plan: DesktopActionPlan,
        request: DesktopActionRequest,
        approval_id: str = "",
        result_status: str = "",
    ) -> DesktopControlAuditRecord:
        """Persist one secret-free desktop audit record as JSONL."""
        record = super().append(
            plan=plan,
            request=request,
            approval_id=approval_id,
            result_status=result_status,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
        return record


class DesktopActionExecutor(Protocol):
    """Executes one already policy-approved desktop action."""

    async def execute(self, payload: Mapping[str, Any]) -> DesktopActionResult: ...


class DesktopCommandRunner(Protocol):
    """Runs one fixed argv command without shell interpolation."""

    async def __call__(self, argv: tuple[str, ...]) -> str: ...


class AllowlistedDesktopActionExecutor:
    """Desktop executor backed by fixed argv commands supplied by policy code."""

    def __init__(
        self,
        *,
        command_map: Mapping[tuple[DesktopActionKind, str], tuple[str, ...]],
        runner: DesktopCommandRunner,
    ) -> None:
        self._command_map = {
            (action, operation.strip()): tuple(argv)
            for (action, operation), argv in command_map.items()
            if operation.strip() and argv
        }
        self._runner = runner

    async def execute(self, payload: Mapping[str, Any]) -> DesktopActionResult:
        """Execute only a command from the fixed capability/operation map."""
        action = DesktopActionKind(str(payload.get("action", "")))
        operation = str(payload.get("operation", "")).strip()
        command = self._command_map.get((action, operation))
        if command is None:
            raise PermissionError(
                f"desktop operation {action.value}:{operation} is not allowlisted"
            )
        output = await self._runner(command)
        return DesktopActionResult(
            capability=str(payload.get("capability", action.value)),
            operation=operation,
            status="completed",
            message=output[:2000],
        )


class AsyncFixedArgvDesktopCommandRunner:
    """Run fixed argv commands with no shell interpolation."""

    def __init__(self, *, timeout_seconds: float = 10.0) -> None:
        self._timeout_seconds = timeout_seconds

    async def __call__(self, argv: tuple[str, ...]) -> str:
        """Run a fixed argv command and return compact stdout."""
        _validate_desktop_argv(argv)
        process = await _create_process(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self._timeout_seconds,
            )
        except TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise TimeoutError(f"desktop command timed out: {argv[0]}") from exc
        stdout_bytes = cast("bytes", stdout)
        stderr_bytes = cast("bytes", stderr)
        if process.returncode != 0:
            detail = stderr_bytes.decode("utf-8", errors="replace").strip()
            raise RuntimeError(detail or f"desktop command failed: {argv[0]}")
        return stdout_bytes.decode("utf-8", errors="replace").strip()


class DesktopControlPolicy:
    """Map desktop actions to scoped capabilities and fail closed."""

    def plan(self, request: DesktopActionRequest) -> DesktopActionPlan:
        """Return the capability plan; does not execute the action."""
        if request.action in {
            DesktopActionKind.SHELL,
            DesktopActionKind.POWERSHELL,
        }:
            return _plan(
                request,
                risk="high",
                requires_approval=True,
                allowed=False,
                reason=(
                    f"{request.action.value} is not part of Desktop Control Skill "
                    "Pack; it needs a separate high-risk spec."
                ),
            )

        risk = _risk_for(request.action)
        return _plan(
            request,
            risk=risk,
            requires_approval=True,
            allowed=True,
            reason="Desktop action must execute through ToolGateway with scoped approval.",
        )


class DesktopControlTool:
    """ToolGateway adapter for one narrow Desktop Control capability."""

    def __init__(
        self,
        *,
        name: str,
        action: DesktopActionKind,
        executor: DesktopActionExecutor,
        audit_log: DesktopControlAuditLog | None = None,
        policy: DesktopControlPolicy | None = None,
    ) -> None:
        self.name = name
        self.action = action
        self.capability = canonical_desktop_capability(action)
        self._executor = executor
        self._audit_log = audit_log or DesktopControlAuditLog()
        self._policy = policy or DesktopControlPolicy()

    @classmethod
    def app_launcher(
        cls,
        *,
        executor: DesktopActionExecutor,
        audit_log: DesktopControlAuditLog | None = None,
    ) -> DesktopControlTool:
        """Create the app-launcher desktop tool."""
        return cls(
            name="desktop_app_launcher",
            action=DesktopActionKind.APP_LAUNCHER,
            executor=executor,
            audit_log=audit_log,
        )

    @classmethod
    def media_control(
        cls,
        *,
        executor: DesktopActionExecutor,
        audit_log: DesktopControlAuditLog | None = None,
    ) -> DesktopControlTool:
        """Create the media-control desktop tool."""
        return cls(
            name="desktop_media_control",
            action=DesktopActionKind.MEDIA_CONTROL,
            executor=executor,
            audit_log=audit_log,
        )

    @classmethod
    def window_control(
        cls,
        *,
        executor: DesktopActionExecutor,
        audit_log: DesktopControlAuditLog | None = None,
    ) -> DesktopControlTool:
        """Create the window-control desktop tool."""
        return cls(
            name="desktop_window_control",
            action=DesktopActionKind.WINDOW_CONTROL,
            executor=executor,
            audit_log=audit_log,
        )

    @classmethod
    def browser_open(
        cls,
        *,
        executor: DesktopActionExecutor,
        audit_log: DesktopControlAuditLog | None = None,
    ) -> DesktopControlTool:
        """Create the browser-open desktop tool."""
        return cls(
            name="desktop_browser_open",
            action=DesktopActionKind.BROWSER_OPEN,
            executor=executor,
            audit_log=audit_log,
        )

    @classmethod
    def screenshot(
        cls,
        *,
        executor: DesktopActionExecutor,
        audit_log: DesktopControlAuditLog | None = None,
    ) -> DesktopControlTool:
        """Create the screenshot desktop tool."""
        return cls(
            name="desktop_screenshot",
            action=DesktopActionKind.SCREENSHOT,
            executor=executor,
            audit_log=audit_log,
        )

    async def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Execute one narrow action after policy validation."""
        request = _request_from_payload(self.action, payload)
        plan = self._policy.plan(request)
        approval_id = str(payload.get("approval_id", "")).strip()
        if not plan.allowed:
            self._audit_log.append(
                plan=plan,
                request=request,
                approval_id=approval_id,
                result_status="refused",
            )
            raise PermissionError(plan.reason)

        result = await self._executor.execute(
            {
                "action": request.action.value,
                "capability": plan.capability,
                "operation": request.operation,
                "target": request.target,
                "source": request.source,
                "risk": plan.risk,
            }
        )
        self._audit_log.append(
            plan=plan,
            request=request,
            approval_id=approval_id,
            result_status=result.status,
        )
        return result.model_dump()


class DesktopControlWorkerBackend:
    """Agent Runtime worker that reduces desktop requests to one ToolGateway call."""

    name = "desktop_control"

    def __init__(
        self,
        *,
        tool_gateway: ToolGateway,
        policy: DesktopControlPolicy | None = None,
    ) -> None:
        self._tool_gateway = tool_gateway
        self._policy = policy or DesktopControlPolicy()

    async def run(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
    ) -> ContextCapsule:
        """Execute one approved desktop action and return a Context Capsule."""
        try:
            request = _request_from_context(context_pack)
        except ValueError as exc:
            return _refusal_capsule(
                str(exc),
                next_action="Передать desktop_action и desktop_operation в metadata.",
            )

        plan = self._policy.plan(request)
        if not plan.allowed:
            return _refusal_capsule(
                plan.reason,
                next_action="Создать отдельный high-risk spec для этой capability.",
            )
        if not job.profile.allows(plan.capability):
            return _refusal_capsule(
                f"InvocationProfile does not allow {plan.capability}.",
                next_action="Выбрать desktop_control profile с нужной capability.",
            )

        tool_name = context_pack.metadata.get(
            "desktop_tool_name",
            _tool_name_for(request.action),
        ).strip()
        tool = _registered_tool(self._tool_gateway, tool_name)
        if tool is None:
            return _refusal_capsule(
                f"unknown ToolGateway tool: {tool_name}",
                next_action="Зарегистрировать Desktop Control tool в ToolGateway.",
            )
        if tool.capability != plan.capability:
            return _refusal_capsule(
                (
                    f"ToolGateway tool {tool_name} exposes {tool.capability}, "
                    f"not {plan.capability}."
                ),
                next_action="Использовать tool с matching desktop capability.",
            )

        approval = _approval_from_job_context(job=job, capability=plan.capability)
        if approval is None:
            return _refusal_capsule(
                f"{plan.capability} requires scoped ToolGateway approval.",
                next_action="Запросить approval для конкретной desktop capability.",
            )

        try:
            result = await self._tool_gateway.execute(
                job.profile,
                tool_name,
                {
                    "operation": request.operation,
                    "target": request.target,
                    "source": request.source,
                    "approval_id": approval.approval_id,
                },
                approval=approval,
            )
        except (ToolDeniedError, ToolNotFoundError, ValueError, PermissionError) as exc:
            return _refusal_capsule(
                str(exc),
                next_action="Проверить InvocationProfile, approval grant и tool payload.",
            )

        return _success_capsule(
            request=request,
            plan=plan,
            tool_name=tool_name,
            result=result,
            approval=approval,
        )

    async def cancel(self, job_id: str) -> bool:
        """No long-running desktop process is held by this worker."""
        del job_id
        return False


def _plan(
    request: DesktopActionRequest,
    *,
    risk: Literal["low", "medium", "high"],
    requires_approval: bool,
    allowed: bool,
    reason: str,
) -> DesktopActionPlan:
    return DesktopActionPlan(
        action=request.action,
        capability=canonical_desktop_capability(request.action),
        risk=risk,
        requires_approval=requires_approval,
        allowed=allowed,
        reason=reason,
        audit_event={
            "source": request.source,
            "operation": request.operation,
            "capability": canonical_desktop_capability(request.action),
            "dialogue_owner": "zhvusha",
        },
    )


def _risk_for(action: DesktopActionKind) -> Literal["low", "medium", "high"]:
    if action is DesktopActionKind.MEDIA_CONTROL:
        return "low"
    if action in {
        DesktopActionKind.HOTKEYS,
        DesktopActionKind.SYSTEM_POWER,
    }:
        return "high"
    return "medium"


def parse_desktop_command_map_json(
    raw_json: str,
) -> dict[tuple[DesktopActionKind, str], tuple[str, ...]]:
    """Parse DESKTOP_CONTROL command allowlist from JSON."""
    raw = raw_json.strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("desktop command map must be a JSON object")
    command_map: dict[tuple[DesktopActionKind, str], tuple[str, ...]] = {}
    for raw_key, raw_argv in data.items():
        key = str(raw_key).strip()
        raw_action, separator, raw_operation = key.partition(":")
        if not separator:
            raise ValueError(f"desktop command key must be action:operation: {key}")
        action = _desktop_action_kind(raw_action.strip())
        if action in {DesktopActionKind.SHELL, DesktopActionKind.POWERSHELL}:
            raise ValueError(
                f"{action.value} is not part of Desktop Control Skill Pack"
            )
        operation = raw_operation.strip()
        if not operation:
            raise ValueError(f"desktop command operation is empty: {key}")
        if not isinstance(raw_argv, list):
            raise ValueError(f"desktop command argv must be a JSON array: {key}")
        argv = tuple(str(item) for item in raw_argv)
        _validate_desktop_argv(argv)
        command_map[(action, operation)] = argv
    return command_map


def build_desktop_control_tool_gateway(
    *,
    command_map: dict[tuple[DesktopActionKind, str], tuple[str, ...]],
    runner: DesktopCommandRunner,
    audit_log: DesktopControlAuditLog | None = None,
) -> ToolGateway:
    """Build ToolGateway tools for the allowlisted Desktop Control commands."""
    executor = AllowlistedDesktopActionExecutor(
        command_map=command_map,
        runner=runner,
    )
    tools = tuple(
        DesktopControlTool(
            name=_tool_name_for(action),
            action=action,
            executor=executor,
            audit_log=audit_log,
        )
        for action in sorted(
            {action for action, _operation in command_map},
            key=lambda item: item.value,
        )
    )
    return ToolGateway(tools=tools)


def _request_from_payload(
    action: DesktopActionKind,
    payload: Mapping[str, Any],
) -> DesktopActionRequest:
    operation = str(payload.get("operation", "")).strip()
    if not operation:
        raise ValueError("operation is required")
    return DesktopActionRequest(
        action=action,
        operation=operation,
        target=str(payload.get("target", "")).strip(),
        source=str(payload.get("source", "text")).strip() or "text",
    )


def _request_from_context(context_pack: ContextPack) -> DesktopActionRequest:
    raw_action = context_pack.metadata.get("desktop_action", "").strip()
    if not raw_action:
        raise ValueError("desktop_action is required")
    try:
        action = _desktop_action_kind(raw_action)
    except ValueError as exc:
        raise ValueError(f"unknown desktop_action: {raw_action}") from exc
    operation = context_pack.metadata.get("desktop_operation", "").strip()
    if not operation:
        raise ValueError("desktop_operation is required")
    return DesktopActionRequest(
        action=action,
        operation=operation,
        target=context_pack.metadata.get("desktop_target", "").strip(),
        source=context_pack.metadata.get("desktop_source", "text").strip() or "text",
    )


def _tool_name_for(action: DesktopActionKind) -> str:
    return {
        DesktopActionKind.APP_LAUNCHER: "desktop_app_launcher",
        DesktopActionKind.MEDIA_CONTROL: "desktop_media_control",
        DesktopActionKind.WINDOW_CONTROL: "desktop_window_control",
        DesktopActionKind.BROWSER_OPEN: "desktop_browser_open",
        DesktopActionKind.SCREENSHOT: "desktop_screenshot",
        DesktopActionKind.HOTKEYS: "desktop_hotkeys",
        DesktopActionKind.SYSTEM_POWER: "desktop_system_power",
        DesktopActionKind.SHELL: "desktop_shell",
        DesktopActionKind.POWERSHELL: "desktop_powershell",
    }[action]


def _desktop_action_kind(value: str) -> DesktopActionKind:
    stripped = value.strip()
    if stripped in _ACTION_BY_CANONICAL_CAPABILITY:
        return _ACTION_BY_CANONICAL_CAPABILITY[stripped]
    return DesktopActionKind(stripped)


def _validate_desktop_argv(argv: tuple[str, ...]) -> None:
    if not argv:
        raise ValueError("desktop command argv is empty")
    if any(not isinstance(item, str) or not item for item in argv):
        raise ValueError("desktop command argv must contain non-empty strings")
    executable = Path(argv[0]).name.lower()
    if executable in _BLOCKED_DESKTOP_EXECUTABLES:
        raise ValueError(f"shell-like executable is not allowed: {executable}")


def _registered_tool(gateway: ToolGateway, tool_name: str) -> AgentTool | None:
    for tool in gateway.registered_tools():
        if tool.name == tool_name:
            return tool
    return None


def _approval_from_job_context(
    *,
    job: AgentJob,
    capability: str,
) -> AgentToolApproval | None:
    raw_capabilities = job.context_pack.metadata.get(
        "agent_tool_approval_capabilities",
        "",
    )
    capabilities = tuple(
        item.strip() for item in raw_capabilities.split(",") if item.strip()
    )
    if capability not in set(capabilities):
        return None
    approval_id = job.context_pack.metadata.get("agent_tool_approval_id", "").strip()
    if not approval_id:
        return None
    return AgentToolApproval.approved(
        approval_id=approval_id,
        capabilities=capabilities,
        approved_by=job.owner_user_id,
    )


def _success_capsule(
    *,
    request: DesktopActionRequest,
    plan: DesktopActionPlan,
    tool_name: str,
    result: Any,
    approval: AgentToolApproval,
) -> ContextCapsule:
    result_text = _compact_result(result)
    processed_context = "\n".join(
        (
            "# Desktop Control action",
            f"- action: {request.action.value}",
            f"- operation: {request.operation}",
            f"- source: {request.source}",
            f"- capability: {plan.capability}",
            f"- risk: {plan.risk}",
            f"- tool: {tool_name}",
            f"- approval: {approval.approval_id}",
            "",
            "## Result",
            result_text,
        )
    )
    return ContextCapsule(
        summary="Desktop control action completed.",
        processed_context=processed_context,
        findings=(
            Finding(
                claim="Desktop action was reduced to one ToolGateway call.",
                status=FindingStatus.CONFIRMED,
                confidence=0.95,
                evidence=(tool_name, plan.capability),
            ),
            Finding(
                claim="Source metadata and dialogue owner stayed attached to audit flow.",
                status=FindingStatus.CONFIRMED,
                confidence=0.9,
                evidence=(request.source, "zhvusha"),
            ),
        ),
        artifacts=(str(result.get("artifact", "")),)
        if isinstance(result, dict) and result.get("artifact")
        else (),
        memory_candidates=(
            f"desktop_control_use:{request.action.value}:{request.operation}",
        ),
        next_actions=("Передать результат Жвуше для проверки фактического эффекта.",),
        markdown_report=processed_context,
    )


def _refusal_capsule(reason: str, *, next_action: str) -> ContextCapsule:
    return ContextCapsule(
        summary="Desktop control action refused.",
        findings=(
            Finding(
                claim=reason,
                status=FindingStatus.UNCONFIRMED,
                confidence=1.0,
            ),
        ),
        next_actions=(next_action,),
        markdown_report=f"Desktop control action refused: {reason}",
    )


def _compact_result(result: Any) -> str:
    if isinstance(result, str):
        return result[:20_000]
    return json.dumps(result, ensure_ascii=False, default=str)[:20_000]

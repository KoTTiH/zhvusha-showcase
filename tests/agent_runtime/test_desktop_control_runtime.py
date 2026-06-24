"""Desktop Control runtime tool and worker contracts."""

from __future__ import annotations

from typing import Any


def test_parse_desktop_command_map_requires_fixed_argv_and_rejects_shell() -> None:
    from src.agent_runtime.desktop_control import parse_desktop_command_map_json

    command_map = parse_desktop_command_map_json(
        '{"desktop.media_control:pause":["playerctl","pause"]}'
    )

    assert command_map
    with_shell = '{"desktop.media_control:pause":["sh","-c","playerctl pause"]}'
    try:
        parse_desktop_command_map_json(with_shell)
    except ValueError as exc:
        assert "shell-like executable" in str(exc)
    else:  # pragma: no cover - explicit assertion readability
        raise AssertionError("shell command was accepted for desktop control")


def test_file_desktop_control_audit_log_persists_secret_free_records(tmp_path) -> None:
    from src.agent_runtime.desktop_control import (
        DesktopActionKind,
        DesktopActionRequest,
        DesktopControlPolicy,
        FileDesktopControlAuditLog,
    )

    request = DesktopActionRequest(
        action=DesktopActionKind.MEDIA_CONTROL,
        operation="pause",
        target="player",
        source="voice",
    )
    plan = DesktopControlPolicy().plan(request)
    audit_log = FileDesktopControlAuditLog(tmp_path / "desktop-audit.jsonl")

    audit_log.append(
        plan=plan,
        request=request,
        approval_id="approval-secretish",
        result_status="completed",
    )

    text = (tmp_path / "desktop-audit.jsonl").read_text(encoding="utf-8")
    assert "desktop_media_control" in text
    assert "desktop.media_control" in text
    assert "voice" in text
    assert "player" not in text
    assert "approval-secretish" in text


async def test_allowlisted_desktop_executor_runs_only_fixed_argv_commands() -> None:
    from src.agent_runtime.desktop_control import (
        AllowlistedDesktopActionExecutor,
        DesktopActionKind,
    )

    calls: list[tuple[str, ...]] = []

    async def runner(argv: tuple[str, ...]) -> str:
        calls.append(argv)
        return "ok"

    executor = AllowlistedDesktopActionExecutor(
        command_map={
            (DesktopActionKind.MEDIA_CONTROL, "pause"): ("playerctl", "pause"),
        },
        runner=runner,
    )

    result = await executor.execute(
        {
            "action": "desktop.media_control",
            "capability": "desktop_media_control",
            "operation": "pause",
            "target": "ignored; rm -rf /",
        }
    )

    assert result.status == "completed"
    assert calls == [("playerctl", "pause")]


async def test_allowlisted_desktop_executor_refuses_unknown_operation() -> None:
    from src.agent_runtime.desktop_control import (
        AllowlistedDesktopActionExecutor,
        DesktopActionKind,
    )

    async def runner(argv: tuple[str, ...]) -> str:
        raise AssertionError(f"unexpected command: {argv}")

    executor = AllowlistedDesktopActionExecutor(
        command_map={
            (DesktopActionKind.MEDIA_CONTROL, "pause"): ("playerctl", "pause"),
        },
        runner=runner,
    )

    try:
        await executor.execute(
            {
                "action": "desktop.media_control",
                "capability": "desktop.media_control",
                "operation": "shutdown",
            }
        )
    except PermissionError as exc:
        assert "not allowlisted" in str(exc)
    else:  # pragma: no cover - explicit assertion readability
        raise AssertionError("unknown desktop operation was not rejected")


async def test_desktop_control_tool_executes_allowed_action_and_records_audit() -> None:
    from src.agent_runtime.desktop_control import (
        DesktopActionResult,
        DesktopControlAuditLog,
        DesktopControlTool,
    )

    class FakeExecutor:
        async def execute(self, payload: dict[str, Any]) -> DesktopActionResult:
            return DesktopActionResult(
                capability=str(payload["capability"]),
                operation=str(payload["operation"]),
                status="completed",
                message="paused",
            )

    audit_log = DesktopControlAuditLog()
    tool = DesktopControlTool.media_control(
        executor=FakeExecutor(), audit_log=audit_log
    )

    result = await tool.execute(
        {
            "operation": "pause",
            "source": "voice",
            "target": "player",
            "approval_id": "approval-desktop",
        }
    )

    assert result["status"] == "completed"
    assert result["capability"] == "desktop_media_control"
    assert audit_log.records[0].source == "voice"
    assert audit_log.records[0].approval_id == "approval-desktop"
    assert audit_log.records[0].dialogue_owner == "zhvusha"


async def test_desktop_control_tool_rejects_shell_before_executor() -> None:
    from src.agent_runtime.desktop_control import (
        DesktopActionKind,
        DesktopActionResult,
        DesktopControlAuditLog,
        DesktopControlTool,
    )

    class FailingExecutor:
        async def execute(self, payload: dict[str, Any]) -> DesktopActionResult:
            raise AssertionError("shell must not reach executor")

    audit_log = DesktopControlAuditLog()
    tool = DesktopControlTool(
        name="desktop_shell",
        action=DesktopActionKind.SHELL,
        executor=FailingExecutor(),
        audit_log=audit_log,
    )

    try:
        await tool.execute({"operation": "run", "target": "rm -rf /"})
    except PermissionError as exc:
        assert "not part of Desktop Control Skill Pack" in str(exc)
    else:  # pragma: no cover - explicit assertion readability
        raise AssertionError("shell action was not rejected")

    assert audit_log.records[0].allowed is False
    assert audit_log.records[0].capability == "desktop.shell"


async def test_desktop_control_worker_runs_single_toolgateway_action_with_approval() -> (
    None
):
    from src.agent_runtime.approvals import AgentToolApproval
    from src.agent_runtime.desktop_control import (
        DesktopActionResult,
        DesktopControlAuditLog,
        DesktopControlTool,
        DesktopControlWorkerBackend,
    )
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import ContextPack, InvocationProfile
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore
    from src.agent_runtime.tools import ToolGateway

    class FakeExecutor:
        async def execute(self, payload: dict[str, Any]) -> DesktopActionResult:
            return DesktopActionResult(
                capability=str(payload["capability"]),
                operation=str(payload["operation"]),
                status="completed",
                message="paused",
            )

    audit_log = DesktopControlAuditLog()
    gateway = ToolGateway(
        tools=(
            DesktopControlTool.media_control(
                executor=FakeExecutor(),
                audit_log=audit_log,
            ),
        )
    )
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={
            "desktop_control": DesktopControlWorkerBackend(tool_gateway=gateway),
        },
    )
    profile = InvocationProfile(
        id="desktop_control.convenience",
        worker="desktop_control",
        allowed_capabilities=("desktop_media_control",),
    )
    approval = AgentToolApproval.approved(
        approval_id="approval-desktop",
        capabilities=("desktop_media_control",),
        approved_by=1291112109,
    )
    job = await runtime.create_job(
        owner_user_id=1291112109,
        chat_id=1,
        source_message_id="voice:1",
        fingerprint="desktop-pause",
        kind="desktop_control.action",
        profile=profile,
        context_pack=ContextPack(
            user_request="поставь музыку на паузу",
            metadata={
                "desktop_action": "desktop.media_control",
                "desktop_operation": "pause",
                "desktop_source": "voice",
                "desktop_target": "player",
                "agent_tool_approval_id": approval.approval_id,
                "agent_tool_approval_capabilities": ",".join(approval.capabilities),
            },
        ),
    )

    completed = await runtime.start(job.id)

    assert completed.result is not None
    assert completed.result.summary == "Desktop control action completed."
    assert completed.result.findings[0].status.value == "confirmed"
    assert "desktop_media_control" in completed.result.processed_context
    assert audit_log.records[0].source == "voice"


async def test_desktop_control_worker_refuses_missing_approval() -> None:
    from src.agent_runtime.desktop_control import (
        DesktopActionResult,
        DesktopControlTool,
        DesktopControlWorkerBackend,
    )
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import ContextPack, InvocationProfile
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore
    from src.agent_runtime.tools import ToolGateway

    class FakeExecutor:
        async def execute(self, payload: dict[str, Any]) -> DesktopActionResult:
            raise AssertionError("missing approval must not execute")

    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={
            "desktop_control": DesktopControlWorkerBackend(
                tool_gateway=ToolGateway(
                    tools=(DesktopControlTool.media_control(executor=FakeExecutor()),),
                )
            ),
        },
    )
    job = await runtime.create_job(
        owner_user_id=1291112109,
        chat_id=1,
        source_message_id="voice:2",
        fingerprint="desktop-missing-approval",
        kind="desktop_control.action",
        profile=InvocationProfile(
            id="desktop_control.convenience",
            worker="desktop_control",
            allowed_capabilities=("desktop_media_control",),
        ),
        context_pack=ContextPack(
            user_request="поставь музыку на паузу",
            metadata={
                "desktop_action": "desktop.media_control",
                "desktop_operation": "pause",
                "desktop_source": "voice",
            },
        ),
    )

    completed = await runtime.start(job.id)

    assert completed.result is not None
    assert completed.result.summary == "Desktop control action refused."
    assert "requires scoped ToolGateway approval" in completed.result.findings[0].claim

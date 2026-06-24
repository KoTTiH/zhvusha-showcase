"""Telegram MCP Agent Runtime worker contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextPack
from src.agent_runtime.profiles import TELEGRAM_MCP_PERSONAL_ACTIONS
from src.agent_runtime.tools import FunctionAgentTool, ToolGateway


def _job(context_pack: ContextPack) -> AgentJob:
    return AgentJob.new(
        owner_user_id=12345,
        chat_id=12345,
        source_message_id="msg-1",
        fingerprint="telegram-mcp-test",
        kind="telegram_mcp",
        profile=TELEGRAM_MCP_PERSONAL_ACTIONS,
        context_pack=context_pack,
        status=AgentJobStatus.RUNNING,
    )


def _send_pack(*, approved: bool) -> ContextPack:
    metadata = {}
    if approved:
        metadata = {
            "agent_tool_approval_id": "skill-approval-test",
            "agent_tool_approval_capabilities": "telegram_mcp_send",
        }
    return ContextPack(
        user_request=json.dumps(
            {
                "action": "send_message",
                "chat_id": "@nikita",
                "message": "привет",
            }
        ),
        metadata=metadata,
    )


def test_stdio_client_roots_follow_launch_plan_allowed_roots(tmp_path: Path) -> None:
    from scripts.run_telegram_mcp import LaunchPlan
    from src.agent_runtime.workers.telegram_mcp import _roots_from_launch_plan

    missing = tmp_path / "missing"
    plan = LaunchPlan(
        command="uv",
        args=(
            "uv",
            "--directory",
            "/checkout",
            "run",
            "main.py",
            str(tmp_path),
            str(missing),
        ),
        env={},
    )

    assert _roots_from_launch_plan(plan) == (tmp_path.resolve(),)


@pytest.mark.asyncio
async def test_send_message_requires_runtime_tool_approval() -> None:
    from src.agent_runtime.workers.telegram_mcp import TelegramMCPWorkerBackend

    calls: list[dict[str, Any]] = []

    async def send(payload: dict[str, Any]) -> str:
        calls.append(payload)
        return "Message sent successfully."

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool(
                "telegram_mcp_send_message",
                "telegram_mcp_send",
                send,
            ),
        )
    )
    worker = TelegramMCPWorkerBackend(tool_gateway=gateway, account_label="personal")

    denied = await worker.run(
        job=_job(_send_pack(approved=False)), context_pack=_send_pack(approved=False)
    )

    assert denied.summary == "Telegram MCP action denied."
    assert calls == []

    approved_pack = _send_pack(approved=True)
    capsule = await worker.run(job=_job(approved_pack), context_pack=approved_pack)

    assert capsule.summary == "Telegram MCP action completed."
    assert calls == [
        {
            "chat_id": "@nikita",
            "message": "привет",
            "parse_mode": None,
            "account": "personal",
        }
    ]


@pytest.mark.asyncio
async def test_send_message_capsule_keeps_raw_tool_payload_out_of_chat_report() -> None:
    from src.agent_runtime.workers.telegram_mcp import TelegramMCPWorkerBackend

    async def send(_payload: dict[str, Any]) -> str:
        return '{"result": "Message sent successfully."}'

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool(
                "telegram_mcp_send_message",
                "telegram_mcp_send",
                send,
            ),
        )
    )
    worker = TelegramMCPWorkerBackend(tool_gateway=gateway, account_label="personal")
    pack = _send_pack(approved=True)

    capsule = await worker.run(job=_job(pack), context_pack=pack)

    assert capsule.processed_context == '{"result": "Message sent successfully."}'
    assert capsule.markdown_report == "отправила."


@pytest.mark.asyncio
async def test_read_action_uses_read_capability_without_approval() -> None:
    from src.agent_runtime.workers.telegram_mcp import TelegramMCPWorkerBackend

    calls: list[dict[str, Any]] = []

    async def read(payload: dict[str, Any]) -> str:
        calls.append(payload)
        return "ID: 1 | Nikita | Message: hi"

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool(
                "telegram_mcp_call_read",
                "telegram_mcp_read",
                read,
            ),
        )
    )
    worker = TelegramMCPWorkerBackend(tool_gateway=gateway, account_label="personal")
    pack = ContextPack(
        user_request=json.dumps(
            {
                "action": "read",
                "tool_name": "get_messages",
                "arguments": {"chat_id": "@nikita", "page_size": 1},
            }
        )
    )

    capsule = await worker.run(job=_job(pack), context_pack=pack)

    assert capsule.summary == "Telegram MCP action completed."
    assert calls == [
        {
            "tool_name": "get_messages",
            "arguments": {
                "chat_id": "@nikita",
                "page_size": 1,
                "account": "personal",
            },
        }
    ]

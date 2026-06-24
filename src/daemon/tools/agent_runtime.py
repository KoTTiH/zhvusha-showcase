"""Daemon tool for creating bounded Agent Runtime jobs."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from src.daemon.agent_runtime_requester import (
    AgentRuntimeJobCreator,
    DaemonAgentRuntimeJobRequest,
    DaemonAgentRuntimeJobRequester,
    render_daemon_agent_runtime_enqueue_status,
)
from src.daemon.tools.base import DaemonTool, ToolResult


class AgentRuntimeEnqueueTool(DaemonTool):
    """Create queued read-only/draft Agent Runtime jobs from daemon decisions."""

    name = "agent_runtime_enqueue"
    description = (
        "Поставить bounded read-only/draft Agent Runtime job. "
        "Не запускает side effects и не отправляет сообщения."
    )
    requires_approval = False

    def __init__(
        self,
        *,
        requester: DaemonAgentRuntimeJobRequester,
        runtime: AgentRuntimeJobCreator,
    ) -> None:
        self._requester = requester
        self._runtime = runtime

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Validate params and enqueue a safe Agent Runtime job."""

        try:
            request = DaemonAgentRuntimeJobRequest.model_validate(params)
        except ValidationError:
            return ToolResult(
                success=False,
                message="Daemon Agent Runtime enqueue: invalid_request",
                data={"reason": "invalid_request"},
            )

        result = await self._requester.enqueue(request, self._runtime)
        return ToolResult(
            success=result.allowed,
            message=render_daemon_agent_runtime_enqueue_status(result),
            data={
                "allowed": result.allowed,
                "reason": result.reason,
                "job_id": result.job_id,
                "job_status": result.job_status.value
                if result.job_status is not None
                else "",
                "profile_id": result.plan.request.profile_id,
                "kind": result.plan.request.kind,
            },
        )

"""Tool Gateway for capability-scoped agent invocations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from src.agent_runtime.approvals import AgentToolApproval
    from src.agent_runtime.models import InvocationProfile


SIDE_EFFECT_CAPABILITIES: tuple[str, ...] = (
    "restart",
    "publish",
    "browser_submit",
    "login",
    "purchase",
    "delete",
    "edit_env",
    "send_message",
    "write_files",
    "write_whitelisted_files_after_approval",
    "commit",
    "commit_after_gate",
    "telegram_mcp_send",
    "telegram_mcp_modify",
    "telegram_mcp_admin",
    "telegram_mcp_media_files",
    "telegram_mcp_daivinchik_button",
    "telegram_mcp_daivinchik_reply_button",
    "telegram_mcp_daivinchik_notify",
    "external_skill_execute",
    "desktop_app_launcher",
    "desktop_media_control",
    "desktop_window_control",
    "desktop_input",
    "desktop_hotkeys",
    "desktop.app_launcher",
    "desktop.media_control",
    "desktop.window_control",
    "desktop.browser_open",
    "desktop.screenshot",
    "desktop.hotkeys",
    "desktop.system_power",
    "desktop.shell",
    "desktop.powershell",
)


class ToolDeniedError(PermissionError):
    """Raised when a tool is not available for this invocation profile."""


class ToolNotFoundError(KeyError):
    """Raised when a tool name is unknown to the gateway."""


class AgentTool(Protocol):
    """Runtime tool exposed through the Tool Gateway."""

    name: str
    capability: str

    async def execute(self, payload: Mapping[str, Any]) -> Any: ...


@dataclass(frozen=True)
class FunctionAgentTool:
    """Adapter from an async function to an AgentTool."""

    name: str
    capability: str
    handler: Callable[[dict[str, Any]], Awaitable[Any]]

    async def execute(self, payload: Mapping[str, Any]) -> Any:
        return await self.handler(dict(payload))


class ToolGateway:
    """Return only tools permitted by a concrete InvocationProfile."""

    def __init__(
        self,
        *,
        tools: tuple[AgentTool, ...],
        side_effect_capabilities: tuple[str, ...] = SIDE_EFFECT_CAPABILITIES,
    ) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._side_effect_capabilities = set(side_effect_capabilities)

    def build_toolset(
        self,
        profile: InvocationProfile,
        *,
        approval: AgentToolApproval | None = None,
    ) -> dict[str, AgentTool]:
        """Build the physically available toolset for a worker run."""
        return {
            name: tool
            for name, tool in self._tools.items()
            if self._is_available(profile, tool, approval)
        }

    def registered_tools(self) -> tuple[AgentTool, ...]:
        """Return all physically registered tools for inventory checks."""
        return tuple(self._tools[name] for name in sorted(self._tools))

    async def execute(
        self,
        profile: InvocationProfile,
        tool_name: str,
        payload: Mapping[str, Any],
        *,
        approval: AgentToolApproval | None = None,
    ) -> Any:
        """Execute a tool only when its capability is granted."""
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ToolNotFoundError(tool_name)
        if not self._is_available(profile, tool, approval):
            raise ToolDeniedError(
                f"{tool_name} requires unavailable capability {tool.capability}"
            )
        return await tool.execute(payload)

    def _is_available(
        self,
        profile: InvocationProfile,
        tool: AgentTool,
        approval: AgentToolApproval | None,
    ) -> bool:
        if not profile.allows(tool.capability):
            return False
        if tool.capability not in self._side_effect_capabilities:
            return True
        return approval is not None and approval.allows(tool.capability)


def merge_tool_gateways(
    *gateways: ToolGateway,
    side_effect_capabilities: tuple[str, ...] = SIDE_EFFECT_CAPABILITIES,
) -> ToolGateway:
    """Merge gateways without letting later duplicate tool names override earlier ones."""
    tools: list[AgentTool] = []
    seen: set[str] = set()
    for gateway in gateways:
        for tool in gateway.registered_tools():
            if tool.name in seen:
                continue
            seen.add(tool.name)
            tools.append(tool)
    return ToolGateway(
        tools=tuple(tools),
        side_effect_capabilities=side_effect_capabilities,
    )

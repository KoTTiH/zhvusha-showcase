"""Tool registry for the daemon."""

from __future__ import annotations

from typing import Any

import structlog

from src.daemon.tools.base import DaemonTool, ToolResult

logger = structlog.get_logger()


class ToolRegistry:
    """Registry of available daemon tools."""

    def __init__(self) -> None:
        self._tools: dict[str, DaemonTool] = {}

    def register(self, tool: DaemonTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.info("daemon_tool_registered", name=tool.name)

    def get(self, name: str) -> DaemonTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List registered tool names."""
        return list(self._tools.keys())

    async def execute(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        """Execute a tool by name."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(success=False, message=f"Unknown tool: {tool_name}")

        try:
            return await tool.execute(params)
        except Exception as e:
            logger.exception("daemon_tool_error", tool=tool_name)
            return ToolResult(success=False, message=f"Tool error: {e}")

    def format_for_llm(self) -> str:
        """Format tool descriptions for inclusion in LLM prompt."""
        if not self._tools:
            return "Нет доступных инструментов."

        lines: list[str] = []
        for tool in self._tools.values():
            approval = " [ТРЕБУЕТ ОДОБРЕНИЯ]" if tool.requires_approval else ""
            lines.append(f"- {tool.name}: {tool.description}{approval}")
        return "\n".join(lines)

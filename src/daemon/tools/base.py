"""Base class for daemon tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Result of tool execution."""

    success: bool
    message: str = ""
    data: dict[str, Any] | None = None


class DaemonTool(ABC):
    """Abstract base for daemon actions."""

    name: str
    description: str
    requires_approval: bool = False

    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters."""

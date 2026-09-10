"""Read-only workspace tools for the daemon.

The daemon has NO write access to workspace files.
Write access is restricted to explicit MCP/client tools and chat skills.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.core.file_access import (
    WS_MAX_FILE_SIZE,
    grep_text_files,
    safe_path,
    scan_directory,
)
from src.daemon.tools.base import DaemonTool, ToolResult

if TYPE_CHECKING:
    from pathlib import Path


class WorkspaceListTool(DaemonTool):
    """List files in a workspace directory."""

    name = "workspace_list"
    description = "Показать содержимое директории workspace (только чтение)"
    requires_approval = False

    def __init__(self, workspace_root: Path) -> None:
        self._root = workspace_root

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """List directory. Params: path (str, optional, relative to workspace)."""
        rel_path = params.get("path", "")
        if rel_path:
            target = safe_path(self._root, rel_path)
            if target is None:
                return ToolResult(success=False, message="Path escapes workspace")
        else:
            target = self._root

        if not target.is_dir():
            msg = (
                "Workspace root not found"
                if not rel_path
                else f"Not found or not a directory: {rel_path}"
            )
            return ToolResult(success=False, message=msg)

        try:
            items = await asyncio.to_thread(scan_directory, target, self._root)
        except PermissionError:
            return ToolResult(success=False, message="Permission denied")
        except OSError as exc:
            return ToolResult(success=False, message=str(exc))
        return ToolResult(success=True, data={"items": items})


class WorkspaceReadTool(DaemonTool):
    """Read a file from the workspace."""

    name = "workspace_read"
    description = "Прочитать файл из workspace (только чтение)"
    requires_approval = False

    def __init__(self, workspace_root: Path) -> None:
        self._root = workspace_root

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Read file. Params: path (str, relative to workspace)."""
        rel_path = params.get("path", "")
        if not rel_path:
            return ToolResult(success=False, message="Empty path")

        resolved = safe_path(self._root, rel_path)
        if resolved is None or not resolved.is_file():
            return ToolResult(success=False, message=f"File not found: {rel_path}")

        try:
            if resolved.stat().st_size > WS_MAX_FILE_SIZE:
                return ToolResult(success=False, message=f"File too large: {rel_path}")
        except OSError:
            return ToolResult(success=False, message=f"File not found: {rel_path}")

        try:
            content = await asyncio.to_thread(
                resolved.read_text, encoding="utf-8", errors="replace"
            )
        except OSError as e:
            return ToolResult(success=False, message=f"Read error: {e}")

        return ToolResult(success=True, data={"content": content, "path": rel_path})


class WorkspaceSearchTool(DaemonTool):
    """Search workspace files by text content."""

    name = "workspace_search"
    description = "Поиск по тексту файлов workspace (только чтение)"
    requires_approval = False

    def __init__(self, workspace_root: Path) -> None:
        self._root = workspace_root

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Grep files. Params: query (str), path (str, optional), max_results (int)."""
        query = params.get("query", "")
        if not query:
            return ToolResult(success=False, message="Empty query")

        rel_path = params.get("path", "")
        if rel_path:
            search_root = safe_path(self._root, rel_path)
            if search_root is None or not search_root.is_dir():
                return ToolResult(success=False, message=f"Invalid path: {rel_path}")
        else:
            search_root = self._root

        try:
            limit = max(1, min(int(params.get("max_results", 20)), 100))
        except (ValueError, TypeError):
            limit = 20
        matches = await asyncio.to_thread(
            grep_text_files, search_root, self._root, query.lower(), limit
        )

        return ToolResult(
            success=True,
            message=f"Found {len(matches)} matches",
            data={"matches": matches},
        )

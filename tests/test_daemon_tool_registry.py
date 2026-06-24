"""Tests for daemon tool registry: verify read-only workspace access."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.daemon.tools.registry import ToolRegistry
from src.daemon.tools.workspace_read import (
    WorkspaceListTool,
    WorkspaceReadTool,
    WorkspaceSearchTool,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace with test files."""
    (tmp_path / "diary").mkdir()
    (tmp_path / "diary" / "2025-04-07.md").write_text("Today was productive.")
    (tmp_path / "notes.txt").write_text("Some notes here.")
    return tmp_path


@pytest.fixture
def registry(workspace: Path) -> ToolRegistry:
    """Build a daemon tool registry with workspace read tools only."""
    reg = ToolRegistry()
    reg.register(WorkspaceListTool(workspace))
    reg.register(WorkspaceReadTool(workspace))
    reg.register(WorkspaceSearchTool(workspace))
    return reg


class TestDaemonToolRegistry:
    def test_read_tools_registered(self, registry: ToolRegistry) -> None:
        tools = registry.list_tools()
        assert "workspace_list" in tools
        assert "workspace_read" in tools
        assert "workspace_search" in tools

    def test_write_tools_absent(self, registry: ToolRegistry) -> None:
        tools = registry.list_tools()
        write_tools = {
            "file_write",
            "write_workspace_file",
            "append_workspace",
            "delete_workspace",
        }
        assert write_tools.isdisjoint(set(tools))

    async def test_unknown_tool_returns_error(self, registry: ToolRegistry) -> None:
        result = await registry.execute("file_write", {"path": "x", "content": "y"})
        assert not result.success
        assert "Unknown tool" in result.message


class TestWorkspaceListTool:
    async def test_list_root(self, workspace: Path) -> None:
        tool = WorkspaceListTool(workspace)
        result = await tool.execute({})
        assert result.success
        assert result.data is not None
        names = {item["name"] for item in result.data["items"]}
        assert "diary" in names
        assert "notes.txt" in names

    async def test_list_subdir(self, workspace: Path) -> None:
        tool = WorkspaceListTool(workspace)
        result = await tool.execute({"path": "diary"})
        assert result.success
        assert result.data is not None
        assert len(result.data["items"]) == 1

    async def test_path_traversal(self, workspace: Path) -> None:
        tool = WorkspaceListTool(workspace)
        result = await tool.execute({"path": "../../etc"})
        assert not result.success
        assert "escapes" in result.message.lower()


class TestWorkspaceReadTool:
    async def test_read_file(self, workspace: Path) -> None:
        tool = WorkspaceReadTool(workspace)
        result = await tool.execute({"path": "diary/2025-04-07.md"})
        assert result.success
        assert result.data is not None
        assert "productive" in result.data["content"]

    async def test_read_not_found(self, workspace: Path) -> None:
        tool = WorkspaceReadTool(workspace)
        result = await tool.execute({"path": "nonexistent.md"})
        assert not result.success

    async def test_path_traversal(self, workspace: Path) -> None:
        tool = WorkspaceReadTool(workspace)
        result = await tool.execute({"path": "../../etc/passwd"})
        assert not result.success


class TestWorkspaceSearchTool:
    async def test_search_found(self, workspace: Path) -> None:
        tool = WorkspaceSearchTool(workspace)
        result = await tool.execute({"query": "productive"})
        assert result.success
        assert result.data is not None
        assert len(result.data["matches"]) >= 1

    async def test_search_not_found(self, workspace: Path) -> None:
        tool = WorkspaceSearchTool(workspace)
        result = await tool.execute({"query": "nonexistent xyz"})
        assert result.success
        assert result.data is not None
        assert len(result.data["matches"]) == 0

    async def test_path_traversal(self, workspace: Path) -> None:
        tool = WorkspaceSearchTool(workspace)
        result = await tool.execute({"query": "test", "path": "../../etc"})
        assert not result.success

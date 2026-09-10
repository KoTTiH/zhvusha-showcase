"""Tests for daemon tool registry and tools."""

from __future__ import annotations

from typing import Any

from src.daemon.tools.base import DaemonTool, ToolResult
from src.daemon.tools.registry import ToolRegistry


class MockTool(DaemonTool):
    name = "mock_tool"
    description = "A mock tool for testing"
    requires_approval = False

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, message="done")


class FailingTool(DaemonTool):
    name = "failing_tool"
    description = "Always fails"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        msg = "intentional failure"
        raise RuntimeError(msg)


class TestToolRegistry:
    def test_register_and_get(self) -> None:
        reg = ToolRegistry()
        tool = MockTool()
        reg.register(tool)

        assert reg.get("mock_tool") is tool
        assert reg.get("nonexistent") is None

    def test_list_tools(self) -> None:
        reg = ToolRegistry()
        reg.register(MockTool())
        assert "mock_tool" in reg.list_tools()

    async def test_execute_success(self) -> None:
        reg = ToolRegistry()
        reg.register(MockTool())

        result = await reg.execute("mock_tool", {})
        assert result.success is True
        assert result.message == "done"

    async def test_execute_unknown_tool(self) -> None:
        reg = ToolRegistry()
        result = await reg.execute("ghost", {})
        assert result.success is False
        assert "Unknown tool" in result.message

    async def test_execute_handles_error(self) -> None:
        reg = ToolRegistry()
        reg.register(FailingTool())

        result = await reg.execute("failing_tool", {})
        assert result.success is False
        assert "error" in result.message.lower()

    def test_format_for_llm(self) -> None:
        reg = ToolRegistry()
        reg.register(MockTool())

        formatted = reg.format_for_llm()
        assert "mock_tool" in formatted
        assert "mock tool for testing" in formatted.lower()

    def test_format_for_llm_empty(self) -> None:
        reg = ToolRegistry()
        assert "Нет" in reg.format_for_llm()

    def test_format_approval_indicator(self) -> None:
        reg = ToolRegistry()
        tool = MockTool()
        tool.requires_approval = True
        reg.register(tool)

        formatted = reg.format_for_llm()
        assert "ОДОБРЕНИЯ" in formatted


class TestKnowledgeStoreTool:
    """Tests for KnowledgeStoreTool staging flow."""

    async def test_stages_entry_via_propose_change(self) -> None:
        from unittest.mock import AsyncMock

        from src.daemon.tools.knowledge_store_tool import KnowledgeStoreTool

        store = AsyncMock()
        store.propose_change = AsyncMock(return_value=42)

        tool = KnowledgeStoreTool(store)
        result = await tool.execute(
            {"title": "Test", "content": "Content", "tags": ["python"]}
        )

        assert result.success is True
        assert "42" in result.message
        assert result.data["staging_id"] == 42
        store.propose_change.assert_awaited_once_with(
            operation="add",
            target_entry_id=None,
            proposed_changes={
                "title": "Test",
                "content": "Content",
                "tags": ["python"],
                "source": "daemon",
            },
            reason="Daemon proposed knowledge entry",
            proposed_by="daemon",
        )

    async def test_rejects_empty_title(self) -> None:
        from unittest.mock import AsyncMock

        from src.daemon.tools.knowledge_store_tool import KnowledgeStoreTool

        store = AsyncMock()
        tool = KnowledgeStoreTool(store)
        result = await tool.execute({"title": "", "content": "something"})

        assert result.success is False
        assert "required" in result.message.lower()

    async def test_rejects_empty_content(self) -> None:
        from unittest.mock import AsyncMock

        from src.daemon.tools.knowledge_store_tool import KnowledgeStoreTool

        store = AsyncMock()
        tool = KnowledgeStoreTool(store)
        result = await tool.execute({"title": "T", "content": ""})

        assert result.success is False

    async def test_parses_tags_from_comma_string(self) -> None:
        from unittest.mock import AsyncMock

        from src.daemon.tools.knowledge_store_tool import KnowledgeStoreTool

        store = AsyncMock()
        store.propose_change = AsyncMock(return_value=1)

        tool = KnowledgeStoreTool(store)
        result = await tool.execute(
            {"title": "T", "content": "C", "tags": "python, rust"}
        )

        assert result.success is True
        call_kwargs = store.propose_change.call_args.kwargs
        assert call_kwargs["proposed_changes"]["tags"] == ["python", "rust"]

    async def test_handles_store_error(self) -> None:
        from unittest.mock import AsyncMock

        from src.daemon.tools.knowledge_store_tool import KnowledgeStoreTool

        store = AsyncMock()
        store.propose_change = AsyncMock(side_effect=RuntimeError("DB down"))

        tool = KnowledgeStoreTool(store)
        result = await tool.execute({"title": "T", "content": "C"})

        assert result.success is False
        assert "Staging error" in result.message

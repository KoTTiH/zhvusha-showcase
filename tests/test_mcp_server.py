"""Tests for MCP server tool functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from src.knowledge.store import (
    CategoryInfo,
    FullEntry,
    SearchResult,
    SummaryEntry,
)


@pytest.fixture
def mock_store() -> AsyncMock:
    """Mock KnowledgeStore for MCP server tests."""
    store = AsyncMock()
    store.hybrid_search = AsyncMock(return_value=[])
    store.browse_categories = AsyncMock(return_value=[])
    store.get_summaries = AsyncMock(return_value=[])
    store.get_full_content = AsyncMock(return_value=None)
    store.add_entry = AsyncMock(return_value=1)
    store.propose_change = AsyncMock(return_value=1)
    store.get_pending_staged = AsyncMock(return_value=[])
    store.review_staged = AsyncMock(return_value=True)
    return store


@pytest.fixture(autouse=True)
def patch_get_store(mock_store: AsyncMock) -> Any:
    """Patch _get_store to return mock_store."""
    with patch(
        "src.mcp_server.server._get_store",
        new_callable=AsyncMock,
        return_value=mock_store,
    ) as p:
        yield p


class TestSearchKnowledge:
    async def test_empty_results(self, mock_store: AsyncMock) -> None:
        from src.mcp_server.server import search_knowledge

        result = await search_knowledge("test query")
        assert result == "No results found."

    async def test_with_results(self, mock_store: AsyncMock) -> None:
        from src.mcp_server.server import search_knowledge

        mock_store.hybrid_search.return_value = [
            SearchResult(
                id=1,
                title="Test",
                tags=["python"],
                rrf_score=0.5,
                token_count=100,
            ),
        ]

        result = await search_knowledge("python")
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["title"] == "Test"
        assert data[0]["rrf_score"] == 0.5

    async def test_with_tags_filter(self, mock_store: AsyncMock) -> None:
        from src.mcp_server.server import search_knowledge

        await search_knowledge("test", tags="python,ai")
        call_kwargs = mock_store.hybrid_search.call_args
        assert call_kwargs.kwargs["tags"] == ["python", "ai"]

    async def test_with_category_filter(self, mock_store: AsyncMock) -> None:
        from src.mcp_server.server import search_knowledge

        await search_knowledge("test", category="tools.python")
        call_kwargs = mock_store.hybrid_search.call_args
        assert call_kwargs.kwargs["category"] == "tools.python"


class TestBrowseCategories:
    async def test_root_categories(self, mock_store: AsyncMock) -> None:
        from src.mcp_server.server import browse_categories

        mock_store.browse_categories.return_value = [
            CategoryInfo(
                id=1,
                name="tech",
                name_ru="Технологии",
                path="tech",
                entry_count=10,
            ),
        ]

        result = await browse_categories()
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["name_ru"] == "Технологии"

    async def test_empty(self, mock_store: AsyncMock) -> None:
        from src.mcp_server.server import browse_categories

        result = await browse_categories()
        assert result == "No categories found."

    async def test_with_parent(self, mock_store: AsyncMock) -> None:
        from src.mcp_server.server import browse_categories

        await browse_categories(parent_path="tech")
        mock_store.browse_categories.assert_awaited_with("tech")


class TestGetSummaries:
    async def test_returns_summaries(self, mock_store: AsyncMock) -> None:
        from src.mcp_server.server import get_summaries

        mock_store.get_summaries.return_value = [
            SummaryEntry(id=1, title="Test", summary="A summary"),
        ]

        result = await get_summaries("1")
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["summary"] == "A summary"

    async def test_multiple_ids(self, mock_store: AsyncMock) -> None:
        from src.mcp_server.server import get_summaries

        await get_summaries("1,2,3")
        call_args = mock_store.get_summaries.call_args[0][0]
        assert call_args == [1, 2, 3]

    async def test_not_found(self, mock_store: AsyncMock) -> None:
        from src.mcp_server.server import get_summaries

        result = await get_summaries("999")
        assert "No entries found" in result


class TestGetFullContent:
    async def test_returns_content(self, mock_store: AsyncMock) -> None:
        from src.mcp_server.server import get_full_content

        mock_store.get_full_content.return_value = FullEntry(
            id=1,
            title="Test",
            content="Full text",
            tags=["python"],
            source="manual",
        )

        result = await get_full_content(1)
        data = json.loads(result)
        assert data["content"] == "Full text"

    async def test_not_found(self, mock_store: AsyncMock) -> None:
        from src.mcp_server.server import get_full_content

        result = await get_full_content(999)
        assert "not found" in result


class TestAddKnowledge:
    async def test_add_entry_directly(self, mock_store: AsyncMock) -> None:
        from src.mcp_server.server import add_knowledge

        mock_store.add_entry.return_value = 42

        result = await add_knowledge(title="New Entry", content="Content here")
        assert "42" in result
        assert "New Entry" in result
        mock_store.add_entry.assert_awaited_once()

    async def test_add_with_tags(self, mock_store: AsyncMock) -> None:
        from src.mcp_server.server import add_knowledge

        await add_knowledge(title="T", content="C", tags="python,ai")
        call_kwargs = mock_store.add_entry.call_args.kwargs
        assert call_kwargs["tags"] == ["python", "ai"]

    async def test_add_with_category(self, mock_store: AsyncMock) -> None:
        from src.mcp_server.server import add_knowledge

        await add_knowledge(title="T", content="C", category_path="tools")
        call_kwargs = mock_store.add_entry.call_args.kwargs
        assert call_kwargs["category_path"] == "tools"


class TestListPendingStaging:
    async def test_no_pending(self, mock_store: AsyncMock) -> None:
        from src.mcp_server.server import list_pending_staging

        result = await list_pending_staging()
        assert result == "No pending proposals."

    async def test_returns_pending_items(self, mock_store: AsyncMock) -> None:
        from types import SimpleNamespace

        from src.mcp_server.server import list_pending_staging

        mock_store.get_pending_staged.return_value = [
            SimpleNamespace(
                id=1,
                operation="add",
                target_entry_id=None,
                proposed_changes={"title": "Test", "content": "C"},
                reason="Added via MCP",
                proposed_by="mcp_server",
                created_at=None,
            ),
        ]

        result = await list_pending_staging()
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["id"] == 1
        assert data[0]["operation"] == "add"


class TestReviewStaging:
    async def test_approve(self, mock_store: AsyncMock) -> None:
        from src.mcp_server.server import review_staging

        result = await review_staging(staging_id=1, approve=True)
        assert "Approved" in result
        mock_store.review_staged.assert_awaited_once_with(1, approve=True)

    async def test_reject(self, mock_store: AsyncMock) -> None:
        from src.mcp_server.server import review_staging

        result = await review_staging(staging_id=1, approve=False)
        assert "Rejected" in result
        mock_store.review_staged.assert_awaited_once_with(1, approve=False)

    async def test_not_found(self, mock_store: AsyncMock) -> None:
        from src.mcp_server.server import review_staging

        mock_store.review_staged.return_value = False
        result = await review_staging(staging_id=999, approve=True)
        assert "not found" in result


class TestGetDatabaseUrl:
    def test_raises_when_database_url_empty(self) -> None:
        from src.mcp_server.server import _get_database_url

        with (
            patch.dict("os.environ", {"DATABASE_URL": ""}, clear=False),
            pytest.raises(RuntimeError, match="DATABASE_URL"),
        ):
            _get_database_url()

    def test_raises_when_database_url_missing(self) -> None:
        from src.mcp_server.server import _get_database_url

        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(RuntimeError, match="DATABASE_URL"),
        ):
            _get_database_url()

    def test_returns_url_when_set(self) -> None:
        from src.mcp_server.server import _get_database_url

        url = "postgresql+asyncpg://u:p@localhost/db"
        with patch.dict("os.environ", {"DATABASE_URL": url}, clear=False):
            assert _get_database_url() == url


# ------------------------------------------------------------------ #
#  Workspace tool tests                                                #
# ------------------------------------------------------------------ #


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace with test files."""
    (tmp_path / "diary").mkdir()
    (tmp_path / "diary" / "2025-04-07.md").write_text(
        "Today I built MCP tools.\nIt was great."
    )
    (tmp_path / "personality").mkdir()
    (tmp_path / "personality" / "core.md").write_text("I am Zhvusha.")
    (tmp_path / "empty_dir").mkdir()
    (tmp_path / "readme.txt").write_text("workspace root file")
    return tmp_path


@pytest.fixture(autouse=False)
def patch_ws_root(workspace: Path) -> Any:
    """Patch _ws_root to return the temp workspace."""
    with patch(
        "src.mcp_server.server._ws_root",
        return_value=workspace,
    ):
        yield


class TestListWorkspace:
    async def test_list_root(self, patch_ws_root: Any, workspace: Path) -> None:
        from src.mcp_server.server import list_workspace

        result = await list_workspace()
        data = json.loads(result)
        names = {item["name"] for item in data}
        assert "diary" in names
        assert "personality" in names
        assert "readme.txt" in names

    async def test_list_subdir(self, patch_ws_root: Any, workspace: Path) -> None:
        from src.mcp_server.server import list_workspace

        result = await list_workspace(path="diary")
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["name"] == "2025-04-07.md"
        assert data[0]["type"] == "file"
        assert "size" in data[0]

    async def test_list_nonexistent(self, patch_ws_root: Any) -> None:
        from src.mcp_server.server import list_workspace

        result = await list_workspace(path="nonexistent")
        assert "not found" in result.lower()

    async def test_list_path_traversal(self, patch_ws_root: Any) -> None:
        from src.mcp_server.server import list_workspace

        result = await list_workspace(path="../../etc")
        assert "Invalid path" in result

    async def test_list_empty_dir(self, patch_ws_root: Any) -> None:
        from src.mcp_server.server import list_workspace

        result = await list_workspace(path="empty_dir")
        assert "empty" in result.lower()


class TestReadWorkspaceFile:
    async def test_read_file(self, patch_ws_root: Any) -> None:
        from src.mcp_server.server import read_workspace_file

        result = await read_workspace_file("diary/2025-04-07.md")
        assert "Today I built MCP tools" in result

    async def test_read_not_found(self, patch_ws_root: Any) -> None:
        from src.mcp_server.server import read_workspace_file

        result = await read_workspace_file("nonexistent.md")
        assert "not found" in result.lower()

    async def test_read_path_traversal(self, patch_ws_root: Any) -> None:
        from src.mcp_server.server import read_workspace_file

        result = await read_workspace_file("../../etc/passwd")
        assert "not found" in result.lower()

    async def test_read_too_large(self, patch_ws_root: Any, workspace: Path) -> None:
        from src.mcp_server.server import read_workspace_file

        big_file = workspace / "big.txt"
        big_file.write_bytes(b"x" * (3 * 1024 * 1024))
        result = await read_workspace_file("big.txt")
        assert "too large" in result.lower()


class TestSearchWorkspace:
    async def test_search_found(self, patch_ws_root: Any) -> None:
        from src.mcp_server.server import search_workspace

        result = await search_workspace("MCP tools")
        data = json.loads(result)
        assert len(data) >= 1
        assert data[0]["file"] == "diary/2025-04-07.md"
        assert data[0]["line"] == 1

    async def test_search_not_found(self, patch_ws_root: Any) -> None:
        from src.mcp_server.server import search_workspace

        result = await search_workspace("nonexistent query xyz")
        assert "No matches" in result

    async def test_search_in_subdir(self, patch_ws_root: Any) -> None:
        from src.mcp_server.server import search_workspace

        result = await search_workspace("Zhvusha", path="personality")
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["file"] == "personality/core.md"

    async def test_search_path_traversal(self, patch_ws_root: Any) -> None:
        from src.mcp_server.server import search_workspace

        result = await search_workspace("test", path="../../etc")
        assert "Invalid search path" in result

    async def test_search_case_insensitive(self, patch_ws_root: Any) -> None:
        from src.mcp_server.server import search_workspace

        result = await search_workspace("mcp TOOLS")
        data = json.loads(result)
        assert len(data) >= 1


class TestWriteWorkspaceFile:
    async def test_create_new_file(self, patch_ws_root: Any, workspace: Path) -> None:
        from src.mcp_server.server import write_workspace_file

        result = await write_workspace_file("notes/test.md", "hello", mode="create")
        assert "Written" in result
        assert (workspace / "notes" / "test.md").read_text() == "hello"

    async def test_create_fails_if_exists(self, patch_ws_root: Any) -> None:
        from src.mcp_server.server import write_workspace_file

        result = await write_workspace_file("diary/2025-04-07.md", "new", mode="create")
        assert "already exists" in result

    async def test_overwrite(self, patch_ws_root: Any, workspace: Path) -> None:
        from src.mcp_server.server import write_workspace_file

        result = await write_workspace_file(
            "diary/2025-04-07.md", "overwritten", mode="overwrite"
        )
        assert "Written" in result
        assert (workspace / "diary" / "2025-04-07.md").read_text() == "overwritten"

    async def test_append_mode_rejected(self, patch_ws_root: Any) -> None:
        from src.mcp_server.server import write_workspace_file

        result = await write_workspace_file(
            "diary/2025-04-07.md", "\nextra", mode="append"
        )
        assert "Invalid mode" in result

    async def test_invalid_mode(self, patch_ws_root: Any) -> None:
        from src.mcp_server.server import write_workspace_file

        result = await write_workspace_file("test.md", "x", mode="bad")
        assert "Invalid mode" in result

    async def test_path_traversal(self, patch_ws_root: Any) -> None:
        from src.mcp_server.server import write_workspace_file

        result = await write_workspace_file("../../etc/evil", "x")
        assert "traversal" in result.lower()

    async def test_size_limit(self, patch_ws_root: Any) -> None:
        from src.mcp_server.server import write_workspace_file

        big = "x" * (3 * 1024 * 1024)
        result = await write_workspace_file("big.md", big)
        assert "too large" in result.lower()


class TestAppendToWorkspaceFile:
    async def test_append_with_separator(
        self, patch_ws_root: Any, workspace: Path
    ) -> None:
        from src.mcp_server.server import append_to_workspace_file

        result = await append_to_workspace_file("diary/2025-04-07.md", "new section")
        assert "Appended" in result
        content = (workspace / "diary" / "2025-04-07.md").read_text()
        assert "\n\nnew section" in content

    async def test_create_new_file(self, patch_ws_root: Any, workspace: Path) -> None:
        from src.mcp_server.server import append_to_workspace_file

        result = await append_to_workspace_file("new/file.md", "first entry")
        assert "Appended" in result
        assert (workspace / "new" / "file.md").read_text() == "first entry"

    async def test_path_traversal(self, patch_ws_root: Any) -> None:
        from src.mcp_server.server import append_to_workspace_file

        result = await append_to_workspace_file("../../etc/evil", "x")
        assert "traversal" in result.lower()

    async def test_custom_separator(self, patch_ws_root: Any, workspace: Path) -> None:
        from src.mcp_server.server import append_to_workspace_file

        result = await append_to_workspace_file(
            "diary/2025-04-07.md", "entry", separator="\n---\n"
        )
        assert "Appended" in result
        content = (workspace / "diary" / "2025-04-07.md").read_text()
        assert "\n---\nentry" in content

    async def test_append_exceeds_size_limit(
        self, patch_ws_root: Any, workspace: Path
    ) -> None:
        from src.mcp_server.server import append_to_workspace_file

        # Create a file near the limit
        big_file = workspace / "big.md"
        big_file.write_text("x" * (2 * 1024 * 1024 - 100))
        result = await append_to_workspace_file("big.md", "y" * 200)
        assert "exceed" in result.lower()


class TestDeleteWorkspaceFile:
    async def test_delete_without_confirm(self, patch_ws_root: Any) -> None:
        from src.mcp_server.server import delete_workspace_file

        result = await delete_workspace_file("diary/2025-04-07.md")
        assert "confirm" in result.lower()

    async def test_delete_with_confirm(
        self, patch_ws_root: Any, workspace: Path
    ) -> None:
        from src.mcp_server.server import delete_workspace_file

        assert (workspace / "diary" / "2025-04-07.md").exists()
        result = await delete_workspace_file("diary/2025-04-07.md", confirm=True)
        assert "Deleted" in result
        assert not (workspace / "diary" / "2025-04-07.md").exists()

    async def test_delete_not_found(self, patch_ws_root: Any) -> None:
        from src.mcp_server.server import delete_workspace_file

        result = await delete_workspace_file("nonexistent.md", confirm=True)
        assert "not found" in result.lower()

    async def test_delete_path_traversal(self, patch_ws_root: Any) -> None:
        from src.mcp_server.server import delete_workspace_file

        result = await delete_workspace_file("../../etc/passwd", confirm=True)
        assert "traversal" in result.lower()

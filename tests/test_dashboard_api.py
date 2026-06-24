"""Tests for dashboard API helpers and route handlers."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.mcp_server.dashboard_api import _safe_path, _workspace_path


class TestSafePath:
    """Path traversal prevention tests."""

    def test_valid_subpath(self, tmp_path: Path) -> None:
        subdir = tmp_path / "data"
        subdir.mkdir()
        result = _safe_path(tmp_path, "data")
        assert result is not None
        assert result == subdir

    def test_traversal_blocked(self, tmp_path: Path) -> None:
        result = _safe_path(tmp_path, "../../../etc/passwd")
        assert result is None

    def test_prefix_spoof_blocked(self, tmp_path: Path) -> None:
        # Create sibling dir with similar name
        evil_dir = tmp_path.parent / (tmp_path.name + "-evil")
        evil_dir.mkdir(exist_ok=True)
        evil_file = evil_dir / "secret.txt"
        evil_file.write_text("secret")
        try:
            result = _safe_path(tmp_path, f"../{evil_dir.name}/secret.txt")
            assert result is None
        finally:
            evil_file.unlink()
            evil_dir.rmdir()

    def test_base_itself_allowed(self, tmp_path: Path) -> None:
        result = _safe_path(tmp_path, ".")
        assert result is not None

    def test_nested_valid(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        result = _safe_path(tmp_path, "a/b/c")
        assert result == nested


class TestWorkspacePath:
    def test_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WORKSPACE_PATH", None)
            p = _workspace_path()
            assert str(p).endswith("zhvusha-workspace")

    def test_custom(self, tmp_path: Path) -> None:
        ws = str(tmp_path / "custom-ws")
        with patch.dict(os.environ, {"WORKSPACE_PATH": ws}):
            assert _workspace_path() == Path(ws)


class TestApiEpisodesLimitValidation:
    """Ensure invalid limit param returns 400 instead of 500."""

    async def test_invalid_limit_returns_400(self) -> None:
        from src.mcp_server.dashboard_api import api_episodes

        request = MagicMock()
        request.query_params = {"limit": "abc"}

        response = await api_episodes(request)
        assert response.status_code == 400

    async def test_valid_limit(self) -> None:
        from src.mcp_server.dashboard_api import api_episodes

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_session_maker = MagicMock(return_value=mock_session)

        request = MagicMock()
        request.query_params = {"limit": "10"}

        with patch(
            "src.mcp_server.server._get_session_maker",
            new_callable=AsyncMock,
            return_value=mock_session_maker,
        ):
            response = await api_episodes(request)
            assert response.status_code == 200

    async def test_negative_limit_clamped(self) -> None:
        from src.mcp_server.dashboard_api import api_episodes

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_session_maker = MagicMock(return_value=mock_session)

        request = MagicMock()
        request.query_params = {"limit": "-5"}

        with patch(
            "src.mcp_server.server._get_session_maker",
            new_callable=AsyncMock,
            return_value=mock_session_maker,
        ):
            response = await api_episodes(request)
            assert response.status_code == 200


class TestApiWorkspaceFile:
    """File serving security tests."""

    async def test_missing_path_returns_400(self) -> None:
        from src.mcp_server.dashboard_api import api_workspace_file

        request = MagicMock()
        request.query_params = {}

        response = await api_workspace_file(request)
        assert response.status_code == 400

    async def test_traversal_returns_404(self) -> None:
        from src.mcp_server.dashboard_api import api_workspace_file

        request = MagicMock()
        request.query_params = {"path": "../../../etc/passwd"}

        with patch(
            "src.mcp_server.dashboard_api._workspace_path",
            return_value=Path("/nonexistent/workspace"),
        ):
            response = await api_workspace_file(request)
            assert response.status_code == 404

    async def test_file_too_large_returns_413(self, tmp_path: Path) -> None:
        from src.mcp_server.dashboard_api import api_workspace_file

        big_file = tmp_path / "big.bin"
        big_file.write_bytes(b"x" * (3 * 1024 * 1024))  # 3 MB

        request = MagicMock()
        request.query_params = {"path": "big.bin"}

        with patch(
            "src.mcp_server.dashboard_api._workspace_path",
            return_value=tmp_path,
        ):
            response = await api_workspace_file(request)
            assert response.status_code == 413

    async def test_valid_file_returned(self, tmp_path: Path) -> None:
        from src.mcp_server.dashboard_api import api_workspace_file

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        request = MagicMock()
        request.query_params = {"path": "test.txt"}

        with patch(
            "src.mcp_server.dashboard_api._workspace_path",
            return_value=tmp_path,
        ):
            response = await api_workspace_file(request)
            assert response.status_code == 200


class TestApiWorkspaceTree:
    """Workspace tree scanning tests."""

    async def test_missing_workspace_returns_empty(self) -> None:
        from src.mcp_server.dashboard_api import api_workspace_tree

        request = MagicMock()

        with patch(
            "src.mcp_server.dashboard_api._workspace_path",
            return_value=Path("/nonexistent/path"),
        ):
            response = await api_workspace_tree(request)
            assert response.status_code == 200

    async def test_scans_directory(self, tmp_path: Path) -> None:
        from src.mcp_server.dashboard_api import api_workspace_tree

        (tmp_path / "file.txt").write_text("content")
        (tmp_path / "subdir").mkdir()

        request = MagicMock()

        with patch(
            "src.mcp_server.dashboard_api._workspace_path",
            return_value=tmp_path,
        ):
            response = await api_workspace_tree(request)
            assert response.status_code == 200


class TestDashboardPage:
    async def test_serves_html(self) -> None:
        from src.mcp_server.dashboard_api import dashboard_page

        request = MagicMock()

        with patch(
            "src.mcp_server.dashboard_api._get_dashboard_html",
            return_value="<html>test</html>",
        ):
            response = await dashboard_page(request)
            assert response.status_code == 200


class TestJsonResponses:
    def test_json_responses_are_not_cached(self) -> None:
        from src.mcp_server.dashboard_api import _json

        response = _json({"ok": True})

        assert response.headers["Cache-Control"] == "no-store"


# ---------------------------------------------------------------------------
#  DB-dependent endpoint tests
# ---------------------------------------------------------------------------


def _mock_session(execute_results: list[MagicMock]) -> MagicMock:
    """Create a mock async session with given execute results."""
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=execute_results)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


def _fetchall_result(rows: list[tuple[object, ...]]) -> MagicMock:
    r = MagicMock()
    r.fetchall.return_value = rows
    return r


def _scalar_result(value: object) -> MagicMock:
    r = MagicMock()
    r.scalar.return_value = value
    return r


class TestApiKnowledgeGraph:
    """Knowledge graph endpoint with mocked DB."""

    async def test_returns_nodes_and_edges(self) -> None:
        from src.mcp_server.dashboard_api import api_knowledge_graph

        entries = [
            (
                1,
                "Python basics",
                ["python"],
                "article",
                "active",
                500,
                10,
                datetime(2024, 1, 1, tzinfo=UTC),
            ),
            (
                2,
                "AI overview",
                ["ai"],
                "note",
                "active",
                200,
                None,
                datetime(2024, 1, 2, tzinfo=UTC),
            ),
            (
                3,
                "Python libs",
                ["python"],
                "note",
                "active",
                150,
                10,
                datetime(2024, 1, 1, tzinfo=UTC),
            ),
        ]
        categories = [(10, "tools.languages", "Языки")]
        relations = [(1, 2, "related")]

        session = _mock_session(
            [
                _fetchall_result(entries),
                _fetchall_result(categories),
                _fetchall_result(relations),
            ]
        )
        store = MagicMock()
        store.session.return_value = session

        with patch(
            "src.mcp_server.server._get_store",
            new_callable=AsyncMock,
            return_value=store,
        ):
            response = await api_knowledge_graph(MagicMock())

        assert response.status_code == 200
        data = json.loads(response.body)
        assert len(data["nodes"]) == 3
        # 1 explicit (1→2) + 1 tag overlap (1↔3, shared "python")
        assert len(data["edges"]) == 2
        assert data["nodes"][0]["title"] == "Python basics"
        assert data["nodes"][0]["link_count"] == 2  # explicit + tag overlap

    async def test_empty_db(self) -> None:
        from src.mcp_server.dashboard_api import api_knowledge_graph

        session = _mock_session(
            [
                _fetchall_result([]),
                _fetchall_result([]),
                _fetchall_result([]),
            ]
        )
        store = MagicMock()
        store.session.return_value = session

        with patch(
            "src.mcp_server.server._get_store",
            new_callable=AsyncMock,
            return_value=store,
        ):
            response = await api_knowledge_graph(MagicMock())

        data = json.loads(response.body)
        assert data == {"nodes": [], "edges": []}


class TestApiKnowledgeTree:
    """Category tree endpoint."""

    async def test_builds_tree(self) -> None:
        from src.mcp_server.dashboard_api import api_knowledge_tree

        rows = [
            (1, "tools", "Инструменты", "tools", 5, None),
            (2, "languages", "Языки", "tools.languages", 3, 1),
        ]
        entry_rows = [
            (10, "Python guide", 2, 500, "article"),
        ]
        session = _mock_session(
            [
                _fetchall_result(rows),
                _fetchall_result(entry_rows),
            ]
        )
        store = MagicMock()
        store.session.return_value = session

        with patch(
            "src.mcp_server.server._get_store",
            new_callable=AsyncMock,
            return_value=store,
        ):
            response = await api_knowledge_tree(MagicMock())

        assert response.status_code == 200
        data = json.loads(response.body)
        assert len(data) == 1
        assert data[0]["name"] == "tools"
        assert len(data[0]["children"]) == 1
        child = data[0]["children"][0]
        assert child["name"] == "languages"
        assert len(child["entries"]) == 1
        assert child["entries"][0]["title"] == "Python guide"

    async def test_empty_categories(self) -> None:
        from src.mcp_server.dashboard_api import api_knowledge_tree

        session = _mock_session(
            [
                _fetchall_result([]),
                _fetchall_result([]),
            ]
        )
        store = MagicMock()
        store.session.return_value = session

        with patch(
            "src.mcp_server.server._get_store",
            new_callable=AsyncMock,
            return_value=store,
        ):
            response = await api_knowledge_tree(MagicMock())

        assert json.loads(response.body) == []


class TestApiKnowledgeEntry:
    """Single entry endpoint with links."""

    async def test_returns_entry_with_links(self) -> None:
        from src.mcp_server.dashboard_api import api_knowledge_entry

        entry = MagicMock()
        entry.id = 1
        entry.title = "Test Entry"
        entry.content = "Body"
        entry.summary = "Short"
        entry.tags = ["python"]
        entry.source = "manual"
        entry.source_url = None
        entry.content_type = "article"
        entry.status = "active"
        entry.token_count = 100
        entry.category.path = "tools"
        entry.category.name_ru = "Инструменты"
        entry.created_at = datetime(2024, 1, 1, tzinfo=UTC)
        entry.updated_at = datetime(2024, 1, 2, tzinfo=UTC)

        r_entry = MagicMock()
        r_entry.scalar_one_or_none.return_value = entry

        session = _mock_session(
            [
                r_entry,
                _fetchall_result([(2, "related")]),  # forward links
                _fetchall_result([]),  # backlinks
                _fetchall_result([(2, "Linked")]),  # titles
            ]
        )
        store = MagicMock()
        store.session.return_value = session

        request = MagicMock()
        request.path_params = {"entry_id": "1"}

        with patch(
            "src.mcp_server.server._get_store",
            new_callable=AsyncMock,
            return_value=store,
        ):
            response = await api_knowledge_entry(request)

        assert response.status_code == 200
        data = json.loads(response.body)
        assert data["title"] == "Test Entry"
        assert len(data["forward_links"]) == 1
        assert data["forward_links"][0]["title"] == "Linked"

    async def test_not_found(self) -> None:
        from src.mcp_server.dashboard_api import api_knowledge_entry

        r_entry = MagicMock()
        r_entry.scalar_one_or_none.return_value = None
        session = _mock_session([r_entry])
        store = MagicMock()
        store.session.return_value = session

        request = MagicMock()
        request.path_params = {"entry_id": "999"}

        with patch(
            "src.mcp_server.server._get_store",
            new_callable=AsyncMock,
            return_value=store,
        ):
            response = await api_knowledge_entry(request)

        assert response.status_code == 404


class TestApiStagingEndpoint:
    """Staging proposals endpoint."""

    async def test_returns_items(self) -> None:
        from src.mcp_server.dashboard_api import api_staging

        item = MagicMock()
        item.id = 1
        item.operation = "create"
        item.target_entry_id = None
        item.proposed_changes = {"title": "New"}
        item.reason = "auto"
        item.proposed_by = "sleep_agent"
        item.created_at = datetime(2024, 1, 1, tzinfo=UTC)

        store = MagicMock()
        store.get_pending_staged = AsyncMock(return_value=[item])

        with patch(
            "src.mcp_server.server._get_store",
            new_callable=AsyncMock,
            return_value=store,
        ):
            response = await api_staging(MagicMock())

        assert response.status_code == 200
        data = json.loads(response.body)
        assert len(data) == 1
        assert data[0]["operation"] == "create"
        assert data[0]["proposed_by"] == "sleep_agent"

    async def test_empty(self) -> None:
        from src.mcp_server.dashboard_api import api_staging

        store = MagicMock()
        store.get_pending_staged = AsyncMock(return_value=[])

        with patch(
            "src.mcp_server.server._get_store",
            new_callable=AsyncMock,
            return_value=store,
        ):
            response = await api_staging(MagicMock())

        assert json.loads(response.body) == []


class TestApiDaemonStatus:
    """Daemon status endpoint — tests graceful degradation."""

    async def test_degrades_gracefully(self) -> None:
        import src.mcp_server.dashboard_api as dapi
        from src.mcp_server.dashboard_api import api_daemon_status

        old_redis = dapi._redis_client
        dapi._redis_client = None

        settings = MagicMock()
        settings.daemon_enabled = False
        settings.daemon_max_llm_cost_per_day_usd = 5.0
        settings.redis_url = "redis://localhost:6379/0"
        settings.workspace_path = "/tmp/nonexistent-ws"  # noqa: S108

        try:
            with (
                patch("src.core.config.get_settings", return_value=settings),
                patch("redis.asyncio.from_url", side_effect=Exception("no redis")),
                patch(
                    "src.mcp_server.server._get_store",
                    new_callable=AsyncMock,
                    side_effect=Exception("no db"),
                ),
            ):
                response = await api_daemon_status(MagicMock())
        finally:
            dapi._redis_client = old_redis

        assert response.status_code == 200
        data = json.loads(response.body)
        assert data["daemon_enabled"] is False
        assert data["queues"] == {"critical": 0, "normal": 0, "background": 0}
        assert data["budget_spent_usd"] == 0

    async def test_with_redis_and_db(self) -> None:
        import src.mcp_server.dashboard_api as dapi
        from src.mcp_server.dashboard_api import api_daemon_status

        settings = MagicMock()
        settings.daemon_enabled = True
        settings.daemon_max_llm_cost_per_day_usd = 10.0
        settings.redis_url = "redis://localhost:6379/0"
        settings.workspace_path = "/tmp/nonexistent-ws"  # noqa: S108

        mock_redis = AsyncMock()
        mock_redis.xlen = AsyncMock(return_value=3)

        session = _mock_session(
            [
                _scalar_result(42),  # total entries
                _scalar_result(5),  # staging pending
                _scalar_result(100),  # total episodes
                _fetchall_result([("Инструменты", 10)]),  # categories
            ]
        )
        store = MagicMock()
        store.session.return_value = session

        old_redis = dapi._redis_client
        dapi._redis_client = mock_redis

        try:
            with (
                patch("src.core.config.get_settings", return_value=settings),
                patch(
                    "src.mcp_server.server._get_store",
                    new_callable=AsyncMock,
                    return_value=store,
                ),
            ):
                response = await api_daemon_status(MagicMock())
        finally:
            dapi._redis_client = old_redis

        assert response.status_code == 200
        data = json.loads(response.body)
        assert data["daemon_enabled"] is True
        assert data["queues"]["critical"] == 3
        assert data["total_entries"] == 42
        assert data["staging_pending"] == 5
        assert data["total_episodes"] == 100

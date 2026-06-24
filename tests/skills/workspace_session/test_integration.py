"""Integration tests for WorkspaceSessionSkill.

Ported from tests/test_workspace_session_skill.py in phase 7.3. Contexts use
the v4 ``AgentContext`` frozen dataclass; ``SkillResult.metadata`` replaces
the legacy ``data`` field.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.skills.base import AgentContext
from src.skills.workspace_session.skill import WorkspaceSessionSkill

if TYPE_CHECKING:
    from pathlib import Path


def _settings(**overrides: Any) -> SimpleNamespace:
    defaults: dict[str, Any] = {
        "workspace_path": "",
        "morning_session_model": "gpt-5.5",
        "morning_session_reasoning_effort": "xhigh",
        "morning_session_hour": 8,
        "morning_session_enabled": False,
        "codex_cli_path": "codex",
        "code_agent_model": "",
        "redis_url": "redis://localhost:6379/0",
        "admin_user_id": 12345,
        "channel_id": "@zhvusha",
    }
    return SimpleNamespace(**{**defaults, **overrides})


def _context(**overrides: Any) -> AgentContext:
    defaults: dict[str, Any] = {
        "user_id": 12345,
        "chat_id": 100,
        "mode": "personal",
        "message_id": 50,
    }
    return AgentContext(**{**defaults, **overrides})


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    ws = tmp_path / "zhvusha-workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def skill(workspace_root: Path) -> WorkspaceSessionSkill:
    return WorkspaceSessionSkill(workspace_path=str(workspace_root))


def _make_codex_backend(text: str = "Session complete.") -> MagicMock:
    """Create a mock Codex backend."""
    from src.skills.code_agent.protocols import CodeAgentResult

    backend = MagicMock()
    backend.run_delegate = AsyncMock(
        return_value=CodeAgentResult(text=text, backend="codex_cli")
    )
    return backend


class TestCanHandle:
    async def test_can_handle_morning(self, skill: WorkspaceSessionSkill) -> None:
        score = await skill.can_handle("/morning", _context())
        assert score == 0.9

    async def test_can_handle_irrelevant(self, skill: WorkspaceSessionSkill) -> None:
        score = await skill.can_handle("hello", _context())
        assert score == 0.0

    async def test_can_handle_morning_with_args(
        self, skill: WorkspaceSessionSkill
    ) -> None:
        score = await skill.can_handle("/morning force", _context())
        assert score == 0.9

    async def test_can_handle_natural_morning_session(
        self, skill: WorkspaceSessionSkill
    ) -> None:
        score = await skill.can_handle("собери утро", _context())
        assert score >= 0.9

    async def test_can_handle_natural_morning_discussion_as_irrelevant(
        self, skill: WorkspaceSessionSkill
    ) -> None:
        score = await skill.can_handle("обсудим утренние привычки", _context())
        assert score == 0.0


class TestExecute:
    async def test_full_flow(
        self, skill: WorkspaceSessionSkill, workspace_root: Path
    ) -> None:
        bot = AsyncMock()
        ctx = _context(bot=bot)

        backend = _make_codex_backend("Morning session done.")

        settings = _settings(workspace_path=str(workspace_root))

        with (
            patch(
                "src.skills.workspace_session.skill.get_settings",
                return_value=settings,
            ),
            patch(
                "src.skills.workspace_session.skill.CodexCliBackend",
                return_value=backend,
            ) as backend_factory,
        ):
            result = await skill.execute("/morning", ctx)

        assert result.success is True
        backend_factory.assert_called_once()
        backend.run_delegate.assert_awaited_once()
        request = backend.run_delegate.await_args.args[0]
        assert request.cwd == workspace_root

    async def test_creates_workspace(
        self, skill: WorkspaceSessionSkill, workspace_root: Path
    ) -> None:
        import shutil

        shutil.rmtree(workspace_root)

        bot = AsyncMock()
        ctx = _context(bot=bot)

        backend = _make_codex_backend()
        settings = _settings(workspace_path=str(workspace_root))

        with (
            patch(
                "src.skills.workspace_session.skill.get_settings",
                return_value=settings,
            ),
            patch(
                "src.skills.workspace_session.skill.CodexCliBackend",
                return_value=backend,
            ),
        ):
            await skill.execute("/morning", ctx)

        assert workspace_root.is_dir()
        assert (workspace_root / "inbox").is_dir()
        assert (workspace_root / "outbox").is_dir()

    async def test_collects_inbox(
        self, skill: WorkspaceSessionSkill, workspace_root: Path
    ) -> None:
        bot = AsyncMock()
        ctx = _context(bot=bot)

        backend = _make_codex_backend()
        settings = _settings(workspace_path=str(workspace_root))

        with (
            patch(
                "src.skills.workspace_session.skill.get_settings",
                return_value=settings,
            ),
            patch(
                "src.skills.workspace_session.skill.CodexCliBackend",
                return_value=backend,
            ),
            patch(
                "src.skills.workspace_session.collector.collect_phase3_sources",
                return_value=[],
            ),
        ):
            await skill.execute("/morning", ctx)

        processed_files = list((workspace_root / "inbox" / ".processed").glob("*.md"))
        # Two files expected: the daily inbox collection + the mood snapshot
        # written in Phase 3b to anchor the diary register.
        processed_names = {f.name for f in processed_files}
        assert len(processed_files) == 2
        assert "current_mood.md" in processed_names
        inbox_files = list((workspace_root / "inbox").glob("*.md"))
        assert len(inbox_files) == 0

    async def test_morning_includes_recent_self_coding_archive(
        self, skill: WorkspaceSessionSkill, workspace_root: Path
    ) -> None:
        from datetime import UTC, datetime

        bot = AsyncMock()
        ctx = _context(bot=bot)
        today = datetime.now(UTC).date().isoformat()
        node_dir = workspace_root / "self_coding_archive" / "codex-parity-abc123"
        node_dir.mkdir(parents=True)
        (node_dir / "metadata.yaml").write_text(
            "\n".join(
                [
                    "slug: codex-parity-abc123",
                    "spec_slug: codex-parity",
                    "status: committed",
                    f"created_at: '{today}T09:00:00+00:00'",
                    "commit_sha: abc1234567890",
                    "metadata:",
                    "  self_coding_actor: zhvusha",
                    "  agent_backend: codex_cli",
                ]
            ),
            encoding="utf-8",
        )
        (node_dir / "insight.md").write_text(
            "# codex-parity\n\n## Вывод\nCodex thread continuity restored.\n",
            encoding="utf-8",
        )

        backend = _make_codex_backend()
        settings = _settings(workspace_path=str(workspace_root))

        with (
            patch(
                "src.skills.workspace_session.skill.get_settings",
                return_value=settings,
            ),
            patch(
                "src.skills.workspace_session.skill.CodexCliBackend",
                return_value=backend,
            ),
            patch(
                "src.skills.workspace_session.collector.collect_phase3_sources",
                return_value=[],
            ),
        ):
            await skill.execute("/morning", ctx)

        processed = list((workspace_root / "inbox" / ".processed").glob("*.md"))
        daily = next(path for path in processed if path.name != "current_mood.md")
        text = daily.read_text(encoding="utf-8")
        assert "Self-Coding Archive" in text
        assert "codex-parity" in text
        assert "backend=codex_cli" in text

    async def test_successful_consolidation_updates_last_run_marker(
        self, workspace_root: Path
    ) -> None:
        from src.memory import ConsolidationLock

        bot = AsyncMock()
        ctx = _context(bot=bot)
        backend = _make_codex_backend()
        engine = AsyncMock()
        engine.run_consolidation = AsyncMock(
            return_value=SimpleNamespace(
                summary="Consolidated.",
                episodes_consolidated=0,
            )
        )
        settings = _settings(workspace_path=str(workspace_root))
        skill = WorkspaceSessionSkill(
            workspace_path=str(workspace_root),
            consolidation_engine=engine,
        )

        with (
            patch(
                "src.skills.workspace_session.skill.get_settings",
                return_value=settings,
            ),
            patch(
                "src.skills.workspace_session.skill.CodexCliBackend",
                return_value=backend,
            ),
            patch(
                "src.skills.workspace_session.collector.collect_phase3_sources",
                return_value=[],
            ),
        ):
            await skill.execute("/morning", ctx)

        lock = ConsolidationLock(workspace_root / "personality")
        assert await lock.read_last_consolidated_at() > 0

    async def test_codex_error_returns_failure(
        self, skill: WorkspaceSessionSkill, workspace_root: Path
    ) -> None:
        bot = AsyncMock()
        ctx = _context(bot=bot)

        from src.skills.code_agent.protocols import CodeAgentExecutionError

        backend = _make_codex_backend()
        backend.run_delegate = AsyncMock(
            side_effect=CodeAgentExecutionError("codex_cli", "Codex error")
        )

        settings = _settings(workspace_path=str(workspace_root))

        with (
            patch(
                "src.skills.workspace_session.skill.get_settings",
                return_value=settings,
            ),
            patch(
                "src.skills.workspace_session.skill.CodexCliBackend",
                return_value=backend,
            ),
        ):
            result = await skill.execute("/morning", ctx)

        assert result.success is False

    async def test_reads_outbox_into_metadata(
        self, skill: WorkspaceSessionSkill, workspace_root: Path
    ) -> None:
        bot = AsyncMock()
        ctx = _context(bot=bot)

        from datetime import UTC, datetime

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        (workspace_root / "outbox" / "reports").mkdir(parents=True, exist_ok=True)
        (workspace_root / "outbox" / "channel_posts").mkdir(parents=True, exist_ok=True)
        (workspace_root / "outbox" / "reports" / f"{today}.md").write_text(
            "# Morning Report\nAll good."
        )
        (workspace_root / "outbox" / "channel_posts" / f"{today}.md").write_text(
            "Channel post draft."
        )

        backend = _make_codex_backend()
        settings = _settings(workspace_path=str(workspace_root))

        with (
            patch(
                "src.skills.workspace_session.skill.get_settings",
                return_value=settings,
            ),
            patch(
                "src.skills.workspace_session.skill.CodexCliBackend",
                return_value=backend,
            ),
        ):
            result = await skill.execute("/morning", ctx)

        assert result.success is True
        assert "outbox_items" in result.metadata
        assert len(result.metadata["outbox_items"]) == 2

    async def test_sends_outbox_to_user(
        self, skill: WorkspaceSessionSkill, workspace_root: Path
    ) -> None:
        bot = AsyncMock()
        sent_msg = AsyncMock()
        sent_msg.message_id = 200
        bot.send_message = AsyncMock(return_value=sent_msg)

        ctx = _context(bot=bot)

        from datetime import UTC, datetime

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        (workspace_root / "outbox" / "reports").mkdir(parents=True, exist_ok=True)
        (workspace_root / "outbox" / "reports" / f"{today}.md").write_text(
            "# Report\nDone."
        )

        backend = _make_codex_backend()
        settings = _settings(workspace_path=str(workspace_root))

        with (
            patch(
                "src.skills.workspace_session.skill.get_settings",
                return_value=settings,
            ),
            patch(
                "src.skills.workspace_session.skill.CodexCliBackend",
                return_value=backend,
            ),
        ):
            await skill.execute("/morning", ctx)

        assert bot.send_message.await_count >= 1


class TestCodexBackend:
    async def test_launch_session_uses_codex_backend_settings(
        self, skill: WorkspaceSessionSkill, workspace_root: Path
    ) -> None:
        bot = AsyncMock()
        ctx = _context(bot=bot)
        backend = _make_codex_backend()
        settings = _settings(
            workspace_path=str(workspace_root),
            codex_cli_path="/usr/bin/codex",
            morning_session_model="gpt-5.5",
            code_agent_model="gpt-5.4",
        )

        with (
            patch(
                "src.skills.workspace_session.skill.get_settings",
                return_value=settings,
            ),
            patch(
                "src.skills.workspace_session.skill.CodexCliBackend",
                return_value=backend,
            ) as backend_factory,
        ):
            await skill.execute("/morning", ctx)

        backend_factory.assert_called_once_with(
            codex_path="/usr/bin/codex",
            model="gpt-5.5",
            reasoning_effort="xhigh",
        )
        backend.run_delegate.assert_awaited_once()
        request = backend.run_delegate.await_args.args[0]
        assert request.cwd == workspace_root
        assert request.model == "gpt-5.5"
        assert request.reasoning_effort == "xhigh"
        assert "утреннюю сессию" in request.task


class TestOutboxPartition:
    """Stale outbox items (e.g. yesterday's report that failed to archive)
    must not re-appear in today's chat. They get archived silently instead.
    """

    def _make_skill(self, tmp_path: Path) -> WorkspaceSessionSkill:
        return WorkspaceSessionSkill(workspace_path=str(tmp_path))

    def test_fresh_today_items_not_stale(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        outbox = tmp_path / "outbox"
        (outbox / "reports").mkdir(parents=True)
        (outbox / "reports" / f"{today}.md").write_text("today")
        (outbox / "channel_posts").mkdir()
        (outbox / "channel_posts" / f"{today}.md").write_text("today post")

        skill = self._make_skill(tmp_path)
        fresh, stale = skill._partition_outbox(outbox)
        assert len(fresh) == 2
        assert stale == []

    def test_yesterday_items_marked_stale(self, tmp_path: Path) -> None:
        outbox = tmp_path / "outbox"
        (outbox / "reports").mkdir(parents=True)
        # A date clearly in the past
        (outbox / "reports" / "2020-01-01.md").write_text("old")
        (outbox / "reports" / "2020-01-01-topic.md").write_text("old topic")

        skill = self._make_skill(tmp_path)
        fresh, stale = skill._partition_outbox(outbox)
        assert fresh == []
        assert {s["filename"] for s in stale} == {
            "2020-01-01.md",
            "2020-01-01-topic.md",
        }

    def test_no_date_prefix_treated_as_fresh(self, tmp_path: Path) -> None:
        outbox = tmp_path / "outbox"
        (outbox / "kwork_drafts").mkdir(parents=True)
        (outbox / "kwork_drafts" / "3141313.md").write_text("kwork draft")

        skill = self._make_skill(tmp_path)
        fresh, stale = skill._partition_outbox(outbox)
        assert len(fresh) == 1
        assert stale == []

    def test_archive_items_moves_to_processed(self, tmp_path: Path) -> None:
        outbox = tmp_path / "outbox"
        (outbox / "reports").mkdir(parents=True)
        stale_file = outbox / "reports" / "2020-01-01.md"
        stale_file.write_text("old report")

        skill = self._make_skill(tmp_path)
        skill._archive_items(
            [
                {
                    "type": "reports",
                    "filename": stale_file.name,
                    "content": "old report",
                    "path": str(stale_file),
                }
            ],
            outbox / ".processed",
        )

        assert not stale_file.exists()
        assert (outbox / ".processed" / "2020-01-01.md").exists()


class TestChannelPostPreviewVisuals:
    async def test_preview_sends_ready_visual_and_strips_frontmatter(
        self,
        tmp_path: Path,
    ) -> None:
        workspace = tmp_path
        outbox = workspace / "outbox" / "channel_posts"
        outbox.mkdir(parents=True)
        asset = workspace / "agent_runtime" / "browser_artifacts" / "source.png"
        asset.parent.mkdir(parents=True)
        asset.write_bytes(b"\x89PNG\r\n\x1a\n")
        post = outbox / "2026-05-14.md"
        content = (
            "---\n"
            "visual:\n"
            "  needed: true\n"
            "  type: screenshot\n"
            "  status: ready\n"
            "  asset_path: agent_runtime/browser_artifacts/source.png\n"
            "  source_url: https://openai.com/index/work-with-codex-from-anywhere/\n"
            "  caption: Codex mobile\n"
            "---\n\n"
            "Post body."
        )
        post.write_text(content, encoding="utf-8")
        bot = AsyncMock()
        bot.send_message = AsyncMock(return_value=MagicMock(message_id=10))
        bot.send_photo = AsyncMock(return_value=MagicMock(message_id=9))
        skill = WorkspaceSessionSkill(workspace_path=str(workspace))

        await skill._send_results(
            bot,
            1,
            [
                {
                    "type": "channel_posts",
                    "filename": post.name,
                    "content": content,
                    "path": str(post),
                }
            ],
        )

        bot.send_photo.assert_awaited_once()
        assert bot.send_photo.await_args.kwargs["caption"] == "Codex mobile"
        sent_text = bot.send_message.await_args.kwargs["text"]
        assert "Post body." in sent_text
        assert "visual: готов" in sent_text
        assert "---" not in sent_text

    async def test_preview_warns_when_required_visual_is_not_ready(
        self,
        tmp_path: Path,
    ) -> None:
        workspace = tmp_path
        outbox = workspace / "outbox" / "channel_posts"
        outbox.mkdir(parents=True)
        post = outbox / "2026-05-14.md"
        content = (
            "---\n"
            "visual:\n"
            "  needed: true\n"
            "  type: screenshot\n"
            "  source_url: https://openai.com/index/work-with-codex-from-anywhere/\n"
            "---\n\n"
            "Post body."
        )
        post.write_text(content, encoding="utf-8")
        bot = AsyncMock()
        bot.send_message = AsyncMock(return_value=MagicMock(message_id=10))
        bot.send_photo = AsyncMock()
        skill = WorkspaceSessionSkill(workspace_path=str(workspace))

        await skill._send_results(
            bot,
            1,
            [
                {
                    "type": "channel_posts",
                    "filename": post.name,
                    "content": content,
                    "path": str(post),
                }
            ],
        )

        bot.send_photo.assert_not_awaited()
        sent_text = bot.send_message.await_args.kwargs["text"]
        assert "Approve заблокирован" in sent_text
        assert "Post body." in sent_text
        assert "---" not in sent_text

    async def test_stale_reports_never_sent_to_chat(self, tmp_path: Path) -> None:
        """Golden path: a stale yesterday report + a fresh today report →
        only the fresh one is sent to chat, stale is archived."""
        from datetime import UTC, datetime

        workspace_root = tmp_path / "zhvusha-workspace"
        workspace_root.mkdir()

        # Seed outbox with one stale and one fresh file BEFORE execute()
        # so we can verify `_launch_session` doesn't need to produce them.
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        reports = workspace_root / "outbox" / "reports"
        reports.mkdir(parents=True)
        (reports / "2020-01-01.md").write_text("stale yesterday")
        (reports / f"{today}.md").write_text("fresh today")

        bot = AsyncMock()
        ctx = _context(bot=bot)
        backend = _make_codex_backend()
        settings = _settings(workspace_path=str(workspace_root))
        skill = WorkspaceSessionSkill(workspace_path=str(workspace_root))

        with (
            patch(
                "src.skills.workspace_session.skill.get_settings",
                return_value=settings,
            ),
            patch(
                "src.skills.workspace_session.skill.CodexCliBackend",
                return_value=backend,
            ),
            patch(
                "src.skills.workspace_session.collector.collect_phase3_sources",
                return_value=[],
            ),
        ):
            await skill.execute("/morning", ctx)

        # The stale file must have been archived silently, not re-sent.
        assert not (reports / "2020-01-01.md").exists()
        assert (workspace_root / "outbox" / ".processed" / "2020-01-01.md").exists()
        # Count how many times the stale content was sent in chat —
        # it must be zero.
        sent_texts = [
            c.kwargs.get("text", "") for c in bot.send_message.await_args_list
        ]
        assert not any("stale yesterday" in t for t in sent_texts)
        # The fresh report did get archived through the normal path
        # after being sent.
        assert (workspace_root / "outbox" / ".processed" / f"{today}.md").exists()

"""Temporary worktree isolation for self-coding cycles."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.contract


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


class TestIsolatedWorkspaceManager:
    def test_create_workspace_does_not_checkout_self_branch(
        self, tmp_git_repo: Path, tmp_path: Path
    ) -> None:
        from src.skills.implement_spec.workspace import IsolatedWorkspaceManager

        manager = IsolatedWorkspaceManager(
            repo_root=tmp_git_repo,
            worktrees_root=tmp_path.parent / "worktrees",
        )
        workspace = manager.create_workspace("weather-skill")

        current = _git(tmp_git_repo, "branch", "--show-current").stdout.strip()
        self_branches = _git(
            tmp_git_repo, "branch", "--list", "zhvusha/weather-skill*"
        ).stdout
        assert current == "main"
        assert self_branches.strip() == ""
        assert workspace.path.exists()
        assert workspace.base_branch == "main"

        manager.cleanup(workspace)
        assert not workspace.path.exists()

    def test_create_workspace_allows_dirty_main_without_copying_dirty_state(
        self, tmp_git_repo: Path, tmp_path: Path
    ) -> None:
        from src.skills.implement_spec.workspace import IsolatedWorkspaceManager

        (tmp_git_repo / "README.md").write_text("operator draft\n", encoding="utf-8")
        (tmp_git_repo / "scratch.md").write_text("local note\n", encoding="utf-8")
        dirty_before = _git(tmp_git_repo, "status", "--porcelain").stdout

        manager = IsolatedWorkspaceManager(
            repo_root=tmp_git_repo,
            worktrees_root=tmp_path.parent / "worktrees",
        )
        workspace = manager.create_workspace("weather-skill")

        try:
            assert workspace.path.exists()
            assert (workspace.path / "README.md").read_text(encoding="utf-8") == (
                "init\n"
            )
            assert not (workspace.path / "scratch.md").exists()
            assert _git(tmp_git_repo, "status", "--porcelain").stdout == dirty_before
        finally:
            manager.cleanup(workspace)

    def test_apply_commit_cherry_picks_back_to_current_branch(
        self, tmp_git_repo: Path, tmp_path: Path
    ) -> None:
        from src.skills.implement_spec.workspace import IsolatedWorkspaceManager

        manager = IsolatedWorkspaceManager(
            repo_root=tmp_git_repo,
            worktrees_root=tmp_path.parent / "worktrees",
        )
        workspace = manager.create_workspace("weather-skill")
        try:
            (workspace.path / "feature.txt").write_text("ok\n", encoding="utf-8")
            _git(workspace.path, "add", "feature.txt")
            _git(
                workspace.path,
                "-c",
                "user.email=zhvusha@local",
                "-c",
                "user.name=zhvusha-coder",
                "commit",
                "-m",
                "feat(self_coding): isolated",
            )
            worktree_sha = _git(workspace.path, "rev-parse", "HEAD").stdout.strip()

            applied = manager.apply_commit(worktree_sha)

            assert applied.sha
            assert (tmp_git_repo / "feature.txt").read_text(encoding="utf-8") == "ok\n"
            assert (
                _git(tmp_git_repo, "branch", "--show-current").stdout.strip() == "main"
            )
        finally:
            manager.cleanup(workspace)

    def test_apply_commit_allows_unrelated_dirty_main_state(
        self, tmp_git_repo: Path, tmp_path: Path
    ) -> None:
        from src.skills.implement_spec.workspace import IsolatedWorkspaceManager

        manager = IsolatedWorkspaceManager(
            repo_root=tmp_git_repo,
            worktrees_root=tmp_path.parent / "worktrees",
        )
        workspace = manager.create_workspace("weather-skill")
        try:
            (tmp_git_repo / "README.md").write_text(
                "operator draft\n", encoding="utf-8"
            )
            (tmp_git_repo / "scratch.md").write_text("local note\n", encoding="utf-8")
            dirty_before = _git(tmp_git_repo, "status", "--porcelain").stdout

            (workspace.path / "feature.txt").write_text("ok\n", encoding="utf-8")
            _git(workspace.path, "add", "feature.txt")
            _git(
                workspace.path,
                "-c",
                "user.email=zhvusha@local",
                "-c",
                "user.name=zhvusha-coder",
                "commit",
                "-m",
                "feat(self_coding): isolated",
            )
            worktree_sha = _git(workspace.path, "rev-parse", "HEAD").stdout.strip()

            applied = manager.apply_commit(worktree_sha)

            assert applied.sha
            assert (tmp_git_repo / "feature.txt").read_text(encoding="utf-8") == "ok\n"
            assert (tmp_git_repo / "README.md").read_text(encoding="utf-8") == (
                "operator draft\n"
            )
            assert (tmp_git_repo / "scratch.md").read_text(encoding="utf-8") == (
                "local note\n"
            )
            committed_files = _git(
                tmp_git_repo, "show", "--name-only", "--pretty=", applied.sha
            ).stdout.splitlines()
            assert committed_files == ["feature.txt"]
            status = _git(tmp_git_repo, "status", "--porcelain").stdout
            assert "feature.txt" not in status
            assert dirty_before.strip()
            assert " M README.md" in status
            assert "?? scratch.md" in status
        finally:
            manager.cleanup(workspace)

    def test_apply_commit_blocks_staged_main_state(
        self, tmp_git_repo: Path, tmp_path: Path
    ) -> None:
        from src.skills.implement_spec.workspace import (
            IsolatedWorkspaceError,
            IsolatedWorkspaceManager,
        )

        manager = IsolatedWorkspaceManager(
            repo_root=tmp_git_repo,
            worktrees_root=tmp_path.parent / "worktrees",
        )
        workspace = manager.create_workspace("weather-skill")
        try:
            (tmp_git_repo / "staged.txt").write_text("staged\n", encoding="utf-8")
            _git(tmp_git_repo, "add", "staged.txt")
            head_before = _git(tmp_git_repo, "rev-parse", "HEAD").stdout.strip()

            (workspace.path / "feature.txt").write_text("ok\n", encoding="utf-8")
            _git(workspace.path, "add", "feature.txt")
            _git(
                workspace.path,
                "-c",
                "user.email=zhvusha@local",
                "-c",
                "user.name=zhvusha-coder",
                "commit",
                "-m",
                "feat(self_coding): isolated",
            )
            worktree_sha = _git(workspace.path, "rev-parse", "HEAD").stdout.strip()

            with pytest.raises(IsolatedWorkspaceError, match="Staged changes"):
                manager.apply_commit(worktree_sha)

            assert _git(tmp_git_repo, "rev-parse", "HEAD").stdout.strip() == head_before
            assert not (tmp_git_repo / "feature.txt").exists()
            assert "A  staged.txt" in _git(tmp_git_repo, "status", "--porcelain").stdout
        finally:
            manager.cleanup(workspace)

    def test_apply_commit_blocks_dirty_target_collision(
        self, tmp_git_repo: Path, tmp_path: Path
    ) -> None:
        from src.skills.implement_spec.workspace import (
            IsolatedWorkspaceError,
            IsolatedWorkspaceManager,
        )

        manager = IsolatedWorkspaceManager(
            repo_root=tmp_git_repo,
            worktrees_root=tmp_path.parent / "worktrees",
        )
        workspace = manager.create_workspace("weather-skill")
        try:
            (tmp_git_repo / "feature.txt").write_text(
                "operator draft\n", encoding="utf-8"
            )
            head_before = _git(tmp_git_repo, "rev-parse", "HEAD").stdout.strip()

            (workspace.path / "feature.txt").write_text("ok\n", encoding="utf-8")
            _git(workspace.path, "add", "feature.txt")
            _git(
                workspace.path,
                "-c",
                "user.email=zhvusha@local",
                "-c",
                "user.name=zhvusha-coder",
                "commit",
                "-m",
                "feat(self_coding): isolated",
            )
            worktree_sha = _git(workspace.path, "rev-parse", "HEAD").stdout.strip()

            with pytest.raises(IsolatedWorkspaceError, match="target files"):
                manager.apply_commit(worktree_sha)

            assert _git(tmp_git_repo, "rev-parse", "HEAD").stdout.strip() == head_before
            assert (tmp_git_repo / "feature.txt").read_text(encoding="utf-8") == (
                "operator draft\n"
            )
        finally:
            manager.cleanup(workspace)

    def test_refuses_to_start_from_self_coding_branch(
        self, tmp_git_repo: Path, tmp_path: Path
    ) -> None:
        from src.skills.implement_spec.workspace import (
            IsolatedWorkspaceError,
            IsolatedWorkspaceManager,
        )

        _git(tmp_git_repo, "checkout", "-b", "zhvusha/stale")
        manager = IsolatedWorkspaceManager(
            repo_root=tmp_git_repo,
            worktrees_root=tmp_path.parent / "worktrees",
        )

        with pytest.raises(IsolatedWorkspaceError, match="self-coding branch"):
            manager.create_workspace("weather-skill")

    def test_preserve_failure_keeps_worktree_and_writes_artifacts(
        self, tmp_git_repo: Path, tmp_path: Path
    ) -> None:
        from src.skills.implement_spec.workspace import IsolatedWorkspaceManager

        manager = IsolatedWorkspaceManager(
            repo_root=tmp_git_repo,
            worktrees_root=tmp_path.parent / "worktrees",
        )
        workspace = manager.create_workspace("weather-skill")
        try:
            (workspace.path / "feature.txt").write_text("draft\n", encoding="utf-8")
            _git(workspace.path, "add", "feature.txt")

            artifact = manager.preserve_failure(
                workspace,
                reason="pre-commit failed",
            )

            assert workspace.path.exists()
            assert artifact.path == workspace.path
            assert artifact.status_path.exists()
            assert artifact.diff_path.exists()
            assert "feature.txt" in artifact.status
            assert "pre-commit failed" in artifact.status_path.read_text(
                encoding="utf-8"
            )
            assert "feature.txt" in artifact.diff_path.read_text(encoding="utf-8")
        finally:
            manager.cleanup(workspace)

    def test_reopen_preserved_workspace_returns_same_worktree(
        self, tmp_git_repo: Path, tmp_path: Path
    ) -> None:
        from src.skills.implement_spec.workspace import IsolatedWorkspaceManager

        manager = IsolatedWorkspaceManager(
            repo_root=tmp_git_repo,
            worktrees_root=tmp_path.parent / "worktrees",
        )
        workspace = manager.create_workspace("weather-skill")
        try:
            artifact = manager.preserve_failure(
                workspace,
                reason="reviewer rejected",
            )

            reopened = manager.reopen_preserved_workspace(
                path=artifact.path,
                label=workspace.label,
                base_branch=workspace.base_branch,
                base_sha=workspace.base_sha,
            )

            assert reopened == workspace
            assert reopened.path.exists()
        finally:
            manager.cleanup(workspace)

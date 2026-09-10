"""Temporary worktree isolation for self-coding implementation cycles."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog

from src.utils.subprocess_env import clean_env_for_git_subprocess

logger = structlog.get_logger()

_SELF_BRANCH_PREFIX = "zhvusha/"


class IsolatedWorkspaceError(RuntimeError):
    """Raised when a temporary self-coding workspace cannot be managed."""


@dataclass(frozen=True)
class IsolatedWorkspace:
    """Detached worktree used for one self-coding cycle."""

    path: Path
    label: str
    base_branch: str
    base_sha: str


@dataclass(frozen=True)
class AppliedCommit:
    """Commit that was atomically applied back to the live repo."""

    sha: str


@dataclass(frozen=True)
class PreservedFailureWorkspace:
    """Failed worktree left on disk for manual inspection."""

    path: Path
    status_path: Path
    diff_path: Path
    status: str


class IsolatedWorkspaceManager:
    """Run Editor work in a detached worktree, then cherry-pick green diffs."""

    def __init__(self, *, repo_root: Path, worktrees_root: Path | None = None) -> None:
        self._repo_root = repo_root
        self._root = worktrees_root or _default_worktrees_root(repo_root)
        self._counter = 0

    def create_workspace(self, slug: str) -> IsolatedWorkspace:
        base_branch = self._current_branch()
        if base_branch.startswith(_SELF_BRANCH_PREFIX):
            raise IsolatedWorkspaceError(
                f"Refusing to start self-coding from self-coding branch {base_branch}."
            )
        self._log_dirty_tree_for_workspace_creation(slug=slug)
        base_sha = self._git("rev-parse", "HEAD", check=True).stdout.strip()
        self._counter += 1
        label = f"isolated:{slug}:{os.getpid()}:{self._counter}"
        path = self._root / f"{slug}-{os.getpid()}-{self._counter}"
        if path.exists():
            shutil.rmtree(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._git("worktree", "add", "--detach", str(path), "HEAD", check=True)
        return IsolatedWorkspace(
            path=path,
            label=label,
            base_branch=base_branch,
            base_sha=base_sha,
        )

    def apply_commit(self, commit_sha: str) -> AppliedCommit:
        branch = self._current_branch()
        if branch.startswith(_SELF_BRANCH_PREFIX):
            raise IsolatedWorkspaceError(
                f"Refusing to apply self-coding commit onto {branch}."
            )
        self._assert_clean_index_for_apply()
        changed_paths = self._commit_changed_paths(commit_sha)
        self._assert_apply_targets_clean(changed_paths)
        result = self._git("cherry-pick", commit_sha, check=False)
        if result.returncode != 0:
            self._git("cherry-pick", "--abort", check=False)
            raise IsolatedWorkspaceError(
                "git cherry-pick failed: "
                + (result.stderr or result.stdout or "unknown error").strip()
            )
        sha = self._git("rev-parse", "HEAD", check=True).stdout.strip()
        return AppliedCommit(sha=sha)

    def cleanup(self, workspace: IsolatedWorkspace) -> None:
        result = self._git(
            "worktree",
            "remove",
            "--force",
            str(workspace.path),
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "isolated_workspace_remove_failed",
                path=str(workspace.path),
                stderr=result.stderr,
            )
            shutil.rmtree(workspace.path, ignore_errors=True)
        self._git("worktree", "prune", check=False)

    def preserve_failure(
        self, workspace: IsolatedWorkspace, *, reason: str
    ) -> PreservedFailureWorkspace:
        """Leave a failed worktree inspectable and write diagnostic artifacts."""
        if not workspace.path.exists():
            raise IsolatedWorkspaceError(
                f"Cannot preserve missing worktree: {workspace.path}"
            )

        status_result = self._git_at(
            workspace.path,
            "status",
            "--short",
            "--untracked-files=all",
            check=False,
        )
        status = (
            status_result.stdout.strip()
            or status_result.stderr.strip()
            or "(clean worktree)"
        )
        staged = self._git_at(
            workspace.path,
            "diff",
            "--cached",
            "--binary",
            "HEAD",
            check=False,
        ).stdout.strip()
        base_diff = self._git_at(
            workspace.path,
            "diff",
            "--binary",
            workspace.base_sha,
            "HEAD",
            check=False,
        ).stdout.strip()
        unstaged = self._git_at(
            workspace.path,
            "diff",
            "--binary",
            check=False,
        ).stdout.strip()
        diff_parts: list[str] = []
        if base_diff:
            diff_parts.append("# Committed diff since isolated base\n" + base_diff)
        if staged:
            diff_parts.append("# Staged diff\n" + staged)
        if unstaged:
            diff_parts.append("# Unstaged diff\n" + unstaged)
        diff_text = "\n\n".join(diff_parts) or (
            "# No tracked diff captured.\n"
            "# Inspect the preserved worktree for untracked files from git status.\n"
        )

        status_path = workspace.path / ".zhvusha-failed-worktree.md"
        diff_path = workspace.path / ".zhvusha-failed-worktree.diff"
        status_path.write_text(
            "\n".join(
                [
                    "# Failed self-coding worktree",
                    "",
                    f"Preserved at: {datetime.now(tz=UTC).isoformat()}",
                    f"Label: {workspace.label}",
                    f"Base branch: {workspace.base_branch}",
                    f"Base sha: {workspace.base_sha}",
                    "",
                    "## Reason",
                    "",
                    reason.strip() or "(no reason recorded)",
                    "",
                    "## Git status",
                    "",
                    "```",
                    status,
                    "```",
                    "",
                    f"Diff artifact: {diff_path}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        diff_path.write_text(diff_text + "\n", encoding="utf-8")
        logger.warning(
            "isolated_workspace_preserved_after_failure",
            path=str(workspace.path),
            status_path=str(status_path),
            diff_path=str(diff_path),
        )
        return PreservedFailureWorkspace(
            path=workspace.path,
            status_path=status_path,
            diff_path=diff_path,
            status=status,
        )

    def reopen_preserved_workspace(
        self,
        *,
        path: Path,
        label: str,
        base_branch: str,
        base_sha: str,
    ) -> IsolatedWorkspace:
        """Reattach to a preserved failed worktree for a resumed Editor run."""
        resolved_path = path.resolve()
        root = self._root.resolve()
        if not resolved_path.is_relative_to(root):
            raise IsolatedWorkspaceError(
                f"Refusing to reopen worktree outside managed root: {path}"
            )
        if not path.exists():
            raise IsolatedWorkspaceError(f"Preserved worktree is missing: {path}")
        top = self._git_at(
            path,
            "rev-parse",
            "--show-toplevel",
            check=True,
        ).stdout.strip()
        if Path(top).resolve() != resolved_path:
            raise IsolatedWorkspaceError(
                f"Preserved worktree root mismatch: expected {path}, got {top}"
            )
        if not label.strip() or not base_branch.strip() or not base_sha.strip():
            raise IsolatedWorkspaceError("Preserved worktree metadata is incomplete.")
        return IsolatedWorkspace(
            path=path,
            label=label.strip(),
            base_branch=base_branch.strip(),
            base_sha=base_sha.strip(),
        )

    def _assert_clean_tree(self) -> None:
        result = self._git("status", "--porcelain", check=True)
        if result.stdout.strip():
            raise IsolatedWorkspaceError(
                "Working tree is not clean — commit or stash before "
                "starting a Жвуша cycle:\n" + result.stdout
            )

    def _assert_clean_index_for_apply(self) -> None:
        result = self._git("diff", "--cached", "--name-status", check=True)
        if result.stdout.strip():
            raise IsolatedWorkspaceError(
                "Staged changes are present; refusing scoped dirty apply:\n"
                + result.stdout
            )

    def _commit_changed_paths(self, commit_sha: str) -> tuple[str, ...]:
        result = self._git(
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "--root",
            "-r",
            commit_sha,
            check=True,
        )
        paths = tuple(
            line.strip() for line in result.stdout.splitlines() if line.strip()
        )
        if not paths:
            raise IsolatedWorkspaceError(
                f"Self-coding commit {commit_sha} has no changed files to apply."
            )
        return paths

    def _assert_apply_targets_clean(self, paths: tuple[str, ...]) -> None:
        result = self._git(
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *paths,
            check=True,
        )
        if result.stdout.strip():
            raise IsolatedWorkspaceError(
                "Self-coding target files are not clean in main; refusing "
                "scoped dirty apply:\n" + result.stdout
            )

    def _log_dirty_tree_for_workspace_creation(self, *, slug: str) -> None:
        result = self._git("status", "--porcelain", check=True)
        dirty_lines = [line for line in result.stdout.splitlines() if line.strip()]
        if dirty_lines:
            logger.warning(
                "isolated_workspace_create_from_dirty_repo",
                slug=slug,
                dirty_paths=dirty_lines,
            )

    def _current_branch(self) -> str:
        branch = self._git(
            "rev-parse", "--abbrev-ref", "HEAD", check=True
        ).stdout.strip()
        return branch or "HEAD"

    def _git(self, *args: str, check: bool) -> subprocess.CompletedProcess[str]:
        return self._git_at(self._repo_root, *args, check=check)

    def _git_at(
        self, cwd: Path, *args: str, check: bool
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(  # noqa: S603
                ["git", *args],  # noqa: S607
                cwd=cwd,
                env=clean_env_for_git_subprocess(),
                check=check,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise IsolatedWorkspaceError(
                f"git {' '.join(args)} failed: {exc.stderr or exc.stdout}"
            ) from exc


def _default_worktrees_root(repo_root: Path) -> Path:
    digest = hashlib.sha256(str(repo_root.resolve()).encode("utf-8")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / "zhvusha-self-coding-worktrees" / digest

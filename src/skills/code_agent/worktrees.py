"""Isolated git worktrees for concurrent Codex runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

from src.utils.subprocess_env import clean_env_for_git_subprocess


class GitRunner(Protocol):
    def __call__(self, *args: str) -> None: ...


@dataclass(frozen=True)
class WorktreeLease:
    path: Path
    branch: str


class WorktreeManager:
    """Allocate unique worktrees under a controlled temp root."""

    def __init__(
        self,
        *,
        repo_root: Path,
        worktrees_root: Path,
        git_runner: GitRunner | None = None,
    ) -> None:
        self._repo_root = repo_root
        self._root = worktrees_root
        self._git_runner = git_runner or self._run_git
        self._counter = 0

    def allocate(self, slug: str) -> WorktreeLease:
        self._counter += 1
        suffix = f"{os.getpid()}-{self._counter}"
        branch = f"zhvusha/{slug}-{suffix}"
        path = self._root / f"{slug}-{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._git_runner("worktree", "add", "-b", branch, str(path), "HEAD")
        return WorktreeLease(path=path, branch=branch)

    def _run_git(self, *args: str) -> None:
        import subprocess

        subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=self._repo_root,
            check=True,
            capture_output=True,
            env=clean_env_for_git_subprocess(),
            text=True,
        )

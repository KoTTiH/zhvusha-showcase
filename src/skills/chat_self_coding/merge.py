"""Safe fast-forward merge helper for chat self-coding DONE stage."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.utils.subprocess_env import clean_env_for_git_subprocess

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class MergeResult:
    success: bool
    reason: str
    branch: str


def merge_done_spec(*, repo_root: Path, branch: str) -> MergeResult:
    """Fast-forward merge an already done self-coding branch."""
    dirty = _git(repo_root, "status", "--porcelain", check=True).stdout.strip()
    if dirty:
        return MergeResult(
            success=False,
            reason="Working tree is not clean; refusing merge.",
            branch=branch,
        )
    exists = _git(
        repo_root,
        "rev-parse",
        "--verify",
        "--quiet",
        f"refs/heads/{branch}",
        check=False,
    )
    if exists.returncode != 0:
        return MergeResult(
            success=False, reason=f"Branch {branch} not found.", branch=branch
        )
    merged = _git(repo_root, "merge", "--ff-only", branch, check=False)
    if merged.returncode != 0:
        return MergeResult(
            success=False,
            reason=(merged.stderr or merged.stdout or "merge failed").strip(),
            branch=branch,
        )
    return MergeResult(success=True, reason="Merged with --ff-only.", branch=branch)


def _git(repo_root: Path, *args: str, check: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=repo_root,
        check=check,
        capture_output=True,
        env=clean_env_for_git_subprocess(),
        text=True,
    )

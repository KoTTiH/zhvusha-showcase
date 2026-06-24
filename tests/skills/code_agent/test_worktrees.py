"""Isolated worktree allocation for concurrent Codex runs."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_two_concurrent_codex_runs_isolate_in_separate_worktrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.skills.code_agent.worktrees import WorktreeManager

    calls: list[tuple[str, ...]] = []

    def fake_git(*args: str) -> None:
        calls.append(args)

    monkeypatch.setattr(os, "getpid", lambda: 4242)
    manager = WorktreeManager(
        repo_root=tmp_path / "repo",
        worktrees_root=tmp_path / "worktrees",
        git_runner=fake_git,
    )

    first = manager.allocate("same-spec")
    second = manager.allocate("same-spec")

    assert first.path != second.path
    assert first.branch != second.branch
    assert first.path.parent == second.path.parent
    assert len(calls) == 2
    assert all(call[:2] == ("worktree", "add") for call in calls)

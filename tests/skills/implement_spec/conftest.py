"""Shared fixtures for implement_spec tests.

``tmp_git_repo`` initialises a fresh repo at ``tmp_path`` with a single
``main`` commit and a stable local user/email config. No remote is
configured, so anything that does ``git pull`` must skip when none is
present.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — args literal, repo is tmp_path
        ["git", *args],  # noqa: S607 — git resolved via PATH
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def tmp_git_repo(tmp_path: Path) -> Path:
    """Initialise a fresh repo with one commit on ``main``."""
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "config", "user.email", "test@local")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "README.md").write_text("init\n", encoding="utf-8")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-m", "init")
    return tmp_path

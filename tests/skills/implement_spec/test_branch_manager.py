"""Contract tests for ``BranchManager`` (Phase 13).

Uses a real ``git`` binary against a tmp repo (no remote, no network).
Pull is silently skipped when no ``origin`` remote is configured — the
production manager handles that case the same way these tests rely on.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.contract


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — args literal, repo is tmp_path
        ["git", *args],  # noqa: S607 — git resolved via PATH
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _bm(repo: Path):  # type: ignore[no-untyped-def]
    from src.skills.implement_spec.branch_manager import BranchManager

    return BranchManager(repo_root=repo, base_branch="main")


class TestCreateBranchHappy:
    def test_creates_zhvusha_prefixed_branch(self, tmp_git_repo: Path) -> None:
        result = _bm(tmp_git_repo).create_branch("weather-skill")
        assert result.name == "zhvusha/weather-skill"

    def test_branch_is_checked_out(self, tmp_git_repo: Path) -> None:
        _bm(tmp_git_repo).create_branch("weather-skill")
        current = _git(tmp_git_repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        assert current == "zhvusha/weather-skill"

    def test_branch_starts_at_main_head(self, tmp_git_repo: Path) -> None:
        main_sha = _git(tmp_git_repo, "rev-parse", "main").stdout.strip()
        _bm(tmp_git_repo).create_branch("weather-skill")
        new_sha = _git(tmp_git_repo, "rev-parse", "HEAD").stdout.strip()
        assert new_sha == main_sha


class TestCreateBranchRefuses:
    def test_refuses_dirty_working_tree(self, tmp_git_repo: Path) -> None:
        from src.skills.implement_spec.branch_manager import BranchManagerError

        (tmp_git_repo / "dirty.txt").write_text("uncommitted")
        with pytest.raises(BranchManagerError, match=r"dirty|clean"):
            _bm(tmp_git_repo).create_branch("weather-skill")

    def test_refuses_existing_branch(self, tmp_git_repo: Path) -> None:
        from src.skills.implement_spec.branch_manager import BranchManagerError

        _git(tmp_git_repo, "branch", "zhvusha/weather-skill")
        with pytest.raises(BranchManagerError, match=r"already exists|существует"):
            _bm(tmp_git_repo).create_branch("weather-skill")

    def test_refuses_invalid_slug(self, tmp_git_repo: Path) -> None:
        from src.skills.implement_spec.branch_manager import BranchManagerError

        with pytest.raises(BranchManagerError, match="slug"):
            _bm(tmp_git_repo).create_branch("Bad Slug With Spaces")

    def test_refuses_empty_slug(self, tmp_git_repo: Path) -> None:
        from src.skills.implement_spec.branch_manager import BranchManagerError

        with pytest.raises(BranchManagerError, match="slug"):
            _bm(tmp_git_repo).create_branch("")


class TestPullBehaviour:
    def test_skips_pull_when_no_remote(self, tmp_git_repo: Path) -> None:
        # No origin → pull silently skipped, branch still created.
        result = _bm(tmp_git_repo).create_branch("weather-skill")
        assert result.name == "zhvusha/weather-skill"


# ----------------------------------------------------------- regression
#
# 2026-04-27 — Editor cycle ran on stale code because BranchManager
# hard-coded ``base_branch="main"`` and checked out main before
# branching, even when the operator was on a feature branch with newer
# commits. The fix: ``base_branch=None`` (the new default) means
# "branch from current HEAD" — no checkout, no pull, just
# ``git checkout -b zhvusha/<slug>`` against where we already are.


def _bm_current_head(repo: Path):  # type: ignore[no-untyped-def]
    """BranchManager that branches from current HEAD (base_branch=None)."""
    from src.skills.implement_spec.branch_manager import BranchManager

    return BranchManager(repo_root=repo)


class TestBaseBranchNoneUsesCurrentHead:
    """``base_branch=None`` (default) branches from wherever HEAD is —
    not from a hard-coded base. Required for working on feature
    branches whose commits aren't in main yet."""

    def test_default_init_does_not_require_base_branch(
        self, tmp_git_repo: Path
    ) -> None:
        from src.skills.implement_spec.branch_manager import BranchManager

        bm = BranchManager(repo_root=tmp_git_repo)
        result = bm.create_branch("weather-skill")
        assert result.name == "zhvusha/weather-skill"

    def test_branches_from_current_feature_branch_not_main(
        self, tmp_git_repo: Path
    ) -> None:
        # Make a feature branch with a commit ahead of main; switch to it.
        _git(tmp_git_repo, "checkout", "-b", "feature-x")
        (tmp_git_repo / "feature_only.txt").write_text("only on feature-x")
        _git(tmp_git_repo, "add", "feature_only.txt")
        _git(
            tmp_git_repo,
            "-c",
            "user.email=t@local",
            "-c",
            "user.name=t",
            "commit",
            "-m",
            "add feature file",
        )
        feature_sha = _git(tmp_git_repo, "rev-parse", "feature-x").stdout.strip()
        main_sha = _git(tmp_git_repo, "rev-parse", "main").stdout.strip()
        assert feature_sha != main_sha, "fixture must have feature ahead of main"

        # Default BranchManager (no base_branch) must branch from feature-x.
        _bm_current_head(tmp_git_repo).create_branch("weather-skill")
        new_sha = _git(tmp_git_repo, "rev-parse", "HEAD").stdout.strip()
        assert new_sha == feature_sha, (
            "branch must start at current HEAD (feature-x), not main"
        )

    def test_does_not_checkout_main_when_base_branch_is_none(
        self, tmp_git_repo: Path
    ) -> None:
        # Sitting on a feature branch — BranchManager must not silently
        # detour through main, otherwise stale code slips into the cycle.
        _git(tmp_git_repo, "checkout", "-b", "feature-x")
        # Add a "main-only" change to make a divergence visible.
        _git(tmp_git_repo, "checkout", "main")
        (tmp_git_repo / "main_only.txt").write_text("only on main")
        _git(tmp_git_repo, "add", "main_only.txt")
        _git(
            tmp_git_repo,
            "-c",
            "user.email=t@local",
            "-c",
            "user.name=t",
            "commit",
            "-m",
            "add main file",
        )
        _git(tmp_git_repo, "checkout", "feature-x")

        _bm_current_head(tmp_git_repo).create_branch("weather-skill")
        # New branch should NOT contain main_only.txt — proves it didn't
        # detour through main.
        assert not (tmp_git_repo / "main_only.txt").exists(), (
            "Branch contains main-only file → BranchManager checked out main"
        )


class TestBaseBranchExplicitStillWorks:
    """Backwards-compat: passing ``base_branch="main"`` still works the
    way it used to — checkout + maybe_pull + branch from main HEAD."""

    def test_explicit_main_checks_out_main_first(self, tmp_git_repo: Path) -> None:
        # Sitting on feature-x with a divergent commit; explicit base_branch=main
        # must still branch from main, not from feature-x.
        _git(tmp_git_repo, "checkout", "-b", "feature-x")
        (tmp_git_repo / "feature_only.txt").write_text("only on feature-x")
        _git(tmp_git_repo, "add", "feature_only.txt")
        _git(
            tmp_git_repo,
            "-c",
            "user.email=t@local",
            "-c",
            "user.name=t",
            "commit",
            "-m",
            "add feature file",
        )
        main_sha = _git(tmp_git_repo, "rev-parse", "main").stdout.strip()

        _bm(tmp_git_repo).create_branch("weather-skill")
        new_sha = _git(tmp_git_repo, "rev-parse", "HEAD").stdout.strip()
        assert new_sha == main_sha, (
            "explicit base_branch='main' must branch from main HEAD"
        )

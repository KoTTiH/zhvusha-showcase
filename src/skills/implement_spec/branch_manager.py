"""Git branch lifecycle for ImplementSpecSkill (Phase 13).

Wraps the commands the Editor needs before delegating to the SDK:

1. Refuse a dirty working tree — Жвуша's commit must be a clean diff,
   not a mixture of her changes with whatever Никита was editing.
2. Optionally checkout an explicit ``base_branch`` and refresh it from
   ``origin`` — used when the cycle should run against the canonical
   ``main`` HEAD regardless of where the operator is sitting.
3. When ``base_branch`` is ``None`` (the default), branch from the
   *current* HEAD. This is the right behaviour during feature
   development: working on ``v4-refactor`` should branch from
   ``v4-refactor``, not silently detour through ``main`` and lose
   today's commits.

The slug pattern matches :class:`SpecModel.slug` (``[a-z0-9-]+``) so we
can fail early on bad input without invoking ``git`` at all.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from src.utils.subprocess_env import clean_env_for_git_subprocess

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger()

_SLUG_RE = re.compile(r"^[a-z0-9-]+$")
_BRANCH_PREFIX = "zhvusha/"


class BranchManagerError(RuntimeError):
    """Raised when the branch cannot be created safely."""


@dataclass(frozen=True)
class BranchCreated:
    """The branch we just checked out — name is ``zhvusha/<slug>``."""

    name: str


class BranchManager:
    """Create ``zhvusha/<slug>`` from a clean ``main`` head."""

    def __init__(self, *, repo_root: Path, base_branch: str | None = None) -> None:
        self._repo_root = repo_root
        self._base_branch = base_branch

    def create_branch(self, slug: str) -> BranchCreated:
        if not slug or not _SLUG_RE.match(slug):
            raise BranchManagerError(
                f"Invalid slug {slug!r} — must match ^[a-z0-9-]+$ (non-empty)."
            )
        self._assert_clean_tree()
        branch_name = f"{_BRANCH_PREFIX}{slug}"
        self._assert_branch_does_not_exist(branch_name)
        if self._base_branch is not None:
            self._checkout(self._base_branch)
            self._maybe_pull()
        self._checkout_new(branch_name)
        logger.info(
            "branch_created", branch=branch_name, base=self._base_branch or "HEAD"
        )
        return BranchCreated(name=branch_name)

    # ------------------------------------------------------------- helpers

    def _assert_clean_tree(self) -> None:
        result = self._git("status", "--porcelain", check=True)
        if result.stdout.strip():
            raise BranchManagerError(
                "Working tree is not clean — commit or stash before "
                "starting a Жвуша cycle:\n" + result.stdout
            )

    def _assert_branch_does_not_exist(self, branch: str) -> None:
        result = self._git(
            "rev-parse",
            "--verify",
            "--quiet",
            f"refs/heads/{branch}",
            check=False,
        )
        if result.returncode == 0:
            raise BranchManagerError(f"Branch {branch} already exists.")

    def _checkout(self, ref: str) -> None:
        self._git("checkout", ref, check=True)

    def _checkout_new(self, branch: str) -> None:
        self._git("checkout", "-b", branch, check=True)

    def _maybe_pull(self) -> None:
        if self._base_branch is None:
            return
        remote = self._git("remote", check=False)
        if not remote.stdout.strip():
            logger.info("branch_manager_skip_pull_no_remote")
            return
        try:
            self._git("pull", "--ff-only", "origin", self._base_branch, check=True)
        except BranchManagerError:
            # A failed pull (no internet, diverged remote) shouldn't kill
            # the cycle — surface it but let the local snapshot stand.
            logger.warning("branch_manager_pull_failed", exc_info=True)

    def _git(self, *args: str, check: bool) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", *args],  # noqa: S607 — git resolved via PATH
                cwd=self._repo_root,
                check=check,
                capture_output=True,
                env=clean_env_for_git_subprocess(),
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise BranchManagerError(
                f"git {' '.join(args)} failed: {exc.stderr or exc.stdout}"
            ) from exc

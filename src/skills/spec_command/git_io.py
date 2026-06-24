"""Atomic git commit for ``tasks/<slug>.yaml`` mutations (Phase 19).

``/spec approve`` and ``/spec reject`` mutate the spec yaml in place
(status, approved_at, rejected_reason, etc.). Without auto-committing
those mutations, the worktree stays dirty and the next ``/spec_run``
trips ``BranchManager._assert_clean_tree``.

This helper is the single-purpose tool that lands one yaml-only commit
under the repo's existing git author identity. Three guarantees:

1. **Atomic** — only the spec.yaml is staged and committed. If anything
   else is already staged, the helper refuses (returns ``False``)
   rather than mixing unrelated changes into a self-coding-flavoured
   commit.
2. **Idempotent** — when the yaml is identical to HEAD, returns
   ``False`` and does not produce an empty commit.
3. **Tolerant** — when ``repo_root`` is not a git repo (most
   ``SpecCommandSkill`` test setups), returns ``False`` without
   raising. Production sets ``repo_root=project_root`` and the helper
   commits; tests on tmp_path keep their old fast no-IO behaviour.

No ``zhvusha-coder`` author override here — these commits represent
Никита's manual approve/reject actions, run under his identity, and
``check_whitelist.sh`` skips for non-zhvusha authors. The helper is
distinct from ``implement_spec.commit_runner.commit_yaml_update``,
which lands the *Editor cycle's* status mutation under the
``zhvusha-coder`` author.
"""

from __future__ import annotations

import contextlib
import subprocess
from typing import TYPE_CHECKING

import structlog

from src.utils.subprocess_env import clean_env_for_git_subprocess

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger()


def commit_yaml_mutation(
    *,
    spec_path: Path,
    repo_root: Path,
    subject: str,
) -> bool:
    """Land a single yaml-only commit under the repo's git author.

    Returns ``True`` if a commit was created, ``False`` for any of the
    no-op cases (not a git repo, spec outside repo, no diff vs HEAD,
    other files already staged). Never raises on the no-op cases — the
    caller's flow continues whether or not the commit lands.
    """
    if not (repo_root / ".git").exists():
        return False
    try:
        rel = spec_path.relative_to(repo_root).as_posix()
    except ValueError:
        return False

    # Refuse if pre-existing staging is non-empty — atomicity.
    pre_staged = _staged_files(repo_root)
    if pre_staged:
        logger.info(
            "commit_yaml_mutation_refused_due_to_pre_staged",
            staged=pre_staged,
            spec=rel,
        )
        return False

    try:
        _run_git(repo_root, "add", "--", rel)
    except subprocess.CalledProcessError:
        logger.exception("commit_yaml_mutation_add_failed", spec=rel)
        return False

    staged = _staged_files(repo_root)
    if not staged:
        # Nothing changed vs HEAD — idempotent no-op.
        return False
    if staged != [rel]:
        # ``git add -- rel`` somehow brought in other files — defence in
        # depth, should not happen with explicit rel argument.
        logger.warning(
            "commit_yaml_mutation_unexpected_staging",
            staged=staged,
            spec=rel,
        )
        # Reset to a clean staging so we don't smuggle anything in.
        _run_git(repo_root, "reset", "HEAD")
        return False

    try:
        _run_git(repo_root, "commit", "-m", subject)
    except subprocess.CalledProcessError as exc:
        logger.exception(
            "commit_yaml_mutation_commit_failed",
            spec=rel,
            stderr=exc.stderr,
        )
        # Best-effort cleanup of the staging we created.
        with contextlib.suppress(subprocess.CalledProcessError):
            _run_git(repo_root, "reset", "HEAD")
        return False

    logger.info("commit_yaml_mutation_committed", spec=rel, subject=subject)
    return True


def _staged_files(repo_root: Path) -> list[str]:
    out = _run_git(repo_root, "diff", "--cached", "--name-only").stdout
    return [line for line in out.splitlines() if line.strip()]


def _run_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — args literal, repo_root caller-provided
        ["git", *args],  # noqa: S607 — git on PATH
        cwd=repo_root,
        check=True,
        capture_output=True,
        env=clean_env_for_git_subprocess(),
        text=True,
    )

"""Whitelist-only commit creation for ImplementSpecSkill (Phase 13).

After the SDK exits, the runner has to land Жвуша's diff in a single
``zhvusha-coder``-authored commit that touches *only* the spec's
``whitelist_paths``. Two guard rails:

1. **Pre-staged check** — anything already in the index that isn't on the
   whitelist aborts the commit. That blocks both Никита-staged hand-edits
   and the (unlikely) case of an Editor SDK side effect bypassing the
   PreToolUse hook.
2. **Post-stage sanity** — after ``git add``, the index must contain
   *only* whitelist paths. If new entries appeared, abort.

The author identity ``zhvusha-coder <zhvusha@local>`` lets every Жвуша
commit be filtered with ``git log --author=zhvusha`` and routes through
``check_tier3_protection.sh`` and (Phase 13) ``check_whitelist.sh``.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from src.utils.subprocess_env import clean_env_for_git_subprocess

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger()

_DEFAULT_AUTHOR_NAME = "zhvusha-coder"
_DEFAULT_AUTHOR_EMAIL = "zhvusha@local"
_AGENT_BACKEND_LINE = "Agent-Backend: codex_cli"
_CO_AUTHORED_LINE = "Co-Authored-By: Codex <noreply@openai.com>"
_COMMIT_RETRY_LIMIT = 3
_PROTECTED_DELETION_EXACT_PATHS = frozenset({"AGENTS.md", "CLAUDE.md"})
_PROTECTED_DELETION_MARKERS = (
    "prompt",
    "personality",
    "context",
)


class CommitRunnerError(RuntimeError):
    """Raised when the runner refuses to land a commit."""


@dataclass(frozen=True)
class CommitResult:
    """Final commit metadata — SHA points at HEAD after ``git commit``."""

    sha: str


class CommitRunner:
    """Stage whitelist + commit under the ``zhvusha-coder`` identity."""

    def __init__(
        self,
        *,
        repo_root: Path,
        author_name: str = _DEFAULT_AUTHOR_NAME,
        author_email: str = _DEFAULT_AUTHOR_EMAIL,
    ) -> None:
        self._repo_root = repo_root
        self._author_name = author_name
        self._author_email = author_email

    # ------------------------------------------------------------- public

    def commit(
        self,
        *,
        spec_slug: str,
        spec_title: str,
        spec_tier: int,
        spec_path: Path,
        whitelist_paths: list[str],
        existing_tests_to_update_paths: list[str] | None = None,
        allowed_simplifications: list[str] | None = None,
    ) -> CommitResult:
        # Phase 16 — ``existing_tests_to_update_paths`` are paths the spec
        # has explicitly declared as legitimate test mutations. They join
        # the surgical ``whitelist_paths`` for the duration of this
        # commit: pre-staged check, post-stage extras check, and the
        # auto-fix retry all treat the union as "allowed".
        commit_paths = list(whitelist_paths) + list(
            existing_tests_to_update_paths or []
        )
        commit_set = frozenset(commit_paths)
        self._assert_no_pre_staged_extras(commit_set)

        existing = [p for p in commit_paths if (self._repo_root / p).exists()]
        if existing:
            self._git("add", "--", *existing, check=True)

        staged = self._staged_files()
        if not staged:
            raise CommitRunnerError(
                "no whitelist file changed — nothing to commit. "
                "Did the SDK actually edit anything in the spec's whitelist?"
            )
        extras = [f for f in staged if f not in commit_set]
        if extras:
            raise CommitRunnerError(
                "staged files outside whitelist after add: " + ", ".join(extras)
            )

        message = self._build_message(
            spec_slug=spec_slug,
            spec_title=spec_title,
            spec_tier=spec_tier,
            spec_path=spec_path,
        )
        self._commit_with_retry(
            message,
            whitelist=commit_set,
            allowed_simplifications=list(allowed_simplifications or []),
        )

        sha = self._git("rev-parse", "HEAD", check=True).stdout.strip()
        logger.info(
            "commit_runner_committed",
            slug=spec_slug,
            tier=spec_tier,
            sha=sha,
            files=staged,
        )
        return CommitResult(sha=sha)

    def commit_yaml_update(
        self,
        *,
        spec_slug: str,
        spec_path: Path,
        subject: str,
    ) -> CommitResult:
        """Commit a spec.yaml-only mutation as a second ``zhvusha-coder``
        commit.

        Used by ImplementSpecSkill after the Editor's main commit + the
        post-cycle ``save_spec_raw`` (status/branch/commit_sha/iterations).
        Without this, the yaml change sits modified-but-uncommitted and
        the next cycle's BranchManager refuses to start.

        Refuses if the yaml is unchanged or if anything else is staged
        alongside — keeps audit guarantees consistent with ``commit()``.
        """
        try:
            rel = spec_path.relative_to(self._repo_root).as_posix()
        except ValueError as exc:
            raise CommitRunnerError(
                f"spec_path {spec_path} is not inside repo_root {self._repo_root}"
            ) from exc

        self._git("add", "--", rel, check=True)

        staged = self._staged_files()
        if not staged:
            raise CommitRunnerError(
                f"no spec.yaml change to commit for {spec_slug} — "
                f"was save_spec_raw a no-op?"
            )
        extras = [f for f in staged if f != rel]
        if extras:
            raise CommitRunnerError(
                "extra files staged alongside spec.yaml: " + ", ".join(extras)
            )

        message = (
            f"chore(self_coding): {subject}\n"
            f"\n"
            f"Spec: {rel}\n"
            f"Slug: {spec_slug}\n"
            f"{_AGENT_BACKEND_LINE}\n"
            f"\n"
            f"{_CO_AUTHORED_LINE}\n"
        )
        self._commit_with_retry(
            message, whitelist=frozenset({rel}), allowed_simplifications=[]
        )
        sha = self._git("rev-parse", "HEAD", check=True).stdout.strip()
        logger.info(
            "commit_runner_yaml_update_committed",
            slug=spec_slug,
            sha=sha,
            file=rel,
        )
        return CommitResult(sha=sha)

    # ------------------------------------------------------------- helpers

    def _assert_no_pre_staged_extras(self, whitelist: frozenset[str]) -> None:
        pre_staged = self._staged_files()
        extras = [f for f in pre_staged if f not in whitelist]
        if extras:
            raise CommitRunnerError(
                "files outside whitelist were already staged before the runner "
                "started: " + ", ".join(extras)
            )

    def _staged_files(self) -> list[str]:
        out = self._git("diff", "--cached", "--name-only", check=True).stdout
        return [line for line in out.splitlines() if line.strip()]

    def _build_message(
        self,
        *,
        spec_slug: str,
        spec_title: str,
        spec_tier: int,
        spec_path: Path,
    ) -> str:
        try:
            spec_rel = spec_path.relative_to(self._repo_root)
        except ValueError:
            spec_rel = spec_path
        subject = f"feat(self_coding): {spec_title}".rstrip()
        body = (
            f"Spec: {spec_rel.as_posix()}\n"
            f"Slug: {spec_slug}\n"
            f"Tier: {spec_tier}\n"
            f"{_AGENT_BACKEND_LINE}\n"
            f"\n"
            f"{_CO_AUTHORED_LINE}\n"
        )
        return f"{subject}\n\n{body}"

    def _commit_with_retry(
        self,
        message: str,
        *,
        whitelist: frozenset[str],
        allowed_simplifications: list[str],
    ) -> None:
        """Commit, retrying after auto-modifying pre-commit hooks.

        Hooks like ``ruff-format`` reformat files in place, fail the
        commit, and require the operator to re-stage. We do that
        automatically up to ``_COMMIT_RETRY_LIMIT`` times. Beyond that
        the failure is real (test failure, lint issue not auto-fixable,
        etc.) and we surface the original ``CommitRunnerError``.

        Refuses to retry if the hook left modifications **outside**
        ``whitelist`` — re-staging those would smuggle non-whitelisted
        content into the commit, breaking audit guarantees.
        """
        last_error: CommitRunnerError | None = None
        for attempt in range(_COMMIT_RETRY_LIMIT):
            self._assert_no_protected_deletions(
                allowed_simplifications=allowed_simplifications
            )
            try:
                self._commit_once(message)
                return
            except CommitRunnerError as exc:
                last_error = exc
                modified = self._modified_in_working_tree()
                if not modified:
                    raise
                outside = [m for m in modified if m not in whitelist]
                if outside:
                    raise CommitRunnerError(
                        "pre-commit hook modified files outside whitelist: "
                        + ", ".join(outside)
                    ) from exc
                logger.info(
                    "commit_runner_retry_after_auto_fix",
                    attempt=attempt + 1,
                    modified=modified,
                )
                self._git("add", "--", *modified, check=True)
        raise CommitRunnerError(
            f"git commit failed after {_COMMIT_RETRY_LIMIT} attempts: {last_error}"
        ) from last_error

    def _assert_no_protected_deletions(
        self, *, allowed_simplifications: list[str]
    ) -> None:
        """Block prompt/personality/context deletions unless spec allowed them."""
        if allowed_simplifications:
            return
        staged = self._staged_files()
        protected = [path for path in staged if _is_protected_deletion_path(path)]
        if not protected:
            return
        out = self._git(
            "diff", "--cached", "--numstat", "--", *protected, check=True
        ).stdout
        issues: list[str] = []
        for line in out.splitlines():
            parts = line.split("\t", maxsplit=2)
            if len(parts) != 3:
                continue
            _additions_raw, deletions_raw, path = parts
            if deletions_raw == "-":
                continue
            try:
                deletions = int(deletions_raw)
            except ValueError:
                continue
            if deletions > 0:
                issues.append(f"{path} (-{deletions})")
        if issues:
            raise CommitRunnerError(
                "protected prompt/personality/context deletions require "
                "spec.allowed_simplifications: " + ", ".join(issues)
            )

    def _commit_once(self, message: str) -> None:
        env = clean_env_for_git_subprocess()
        env.update(
            {
                "GIT_AUTHOR_NAME": self._author_name,
                "GIT_AUTHOR_EMAIL": self._author_email,
                "GIT_COMMITTER_NAME": self._author_name,
                "GIT_COMMITTER_EMAIL": self._author_email,
            }
        )
        try:
            subprocess.run(
                ["git", "commit", "-F", "-"],  # noqa: S607 — git on PATH
                cwd=self._repo_root,
                input=message,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise CommitRunnerError(
                f"git commit failed: {exc.stderr or exc.stdout}"
            ) from exc

    def _modified_in_working_tree(self) -> list[str]:
        """Files a failed hook left unstaged or untracked.

        ``git diff --name-only`` misses untracked whitelist files created by
        auto-fixing hooks. ``status --porcelain`` lets us retry those while
        ignoring staged-only paths from the original commit attempt, so a
        persistent hook failure is still surfaced immediately.
        """
        out = self._git(
            "status",
            "--porcelain",
            "--untracked-files=all",
            check=True,
        ).stdout
        paths: list[str] = []
        for line in out.splitlines():
            if not line.strip():
                continue
            status = line[:2]
            if status == "??" or status[1] != " ":
                path = line[3:].strip()
                if path:
                    paths.append(path)
        return paths

    def _git(self, *args: str, check: bool) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", *args],  # noqa: S607 — git on PATH
                cwd=self._repo_root,
                env=clean_env_for_git_subprocess(),
                check=check,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise CommitRunnerError(
                f"git {' '.join(args)} failed: {exc.stderr or exc.stdout}"
            ) from exc


def _is_protected_deletion_path(path: str) -> bool:
    if path in _PROTECTED_DELETION_EXACT_PATHS:
        return True
    if path.endswith("/AGENTS.md") or path.endswith("/CLAUDE.md"):
        return True
    lower = path.lower()
    return any(marker in lower for marker in _PROTECTED_DELETION_MARKERS)

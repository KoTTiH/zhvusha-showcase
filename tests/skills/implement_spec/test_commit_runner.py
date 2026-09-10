"""Contract tests for ``CommitRunner`` (Phase 13).

Real ``git`` against a tmp repo: each test reproduces the post-SDK state
(some files modified inside the whitelist, optionally something dirty
outside it) and verifies that the runner commits exactly the whitelist —
no more, no less — under the ``zhvusha-coder`` author identity, with the
``Spec:`` footer pointing back to the spec file.
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


def _runner(repo: Path):  # type: ignore[no-untyped-def]
    from src.skills.implement_spec.commit_runner import CommitRunner

    return CommitRunner(repo_root=repo)


def _write(repo: Path, rel: str, content: str = "x") -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_spec_file(repo: Path, slug: str = "weather") -> Path:
    spec = repo / "tasks" / f"2026-04-26-{slug}.yaml"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(f"slug: {slug}\n", encoding="utf-8")
    return spec


class TestCommitHappyPath:
    def test_commits_whitelist_files(self, tmp_git_repo: Path) -> None:
        _write(tmp_git_repo, "src/skills/weather/skill.py", "code\n")
        _write(tmp_git_repo, "tests/skills/weather/test_contract.py", "test\n")
        spec_path = _make_spec_file(tmp_git_repo)
        result = _runner(tmp_git_repo).commit(
            spec_slug="weather",
            spec_title="Add weather skill",
            spec_tier=1,
            spec_path=spec_path,
            whitelist_paths=[
                "src/skills/weather/skill.py",
                "tests/skills/weather/test_contract.py",
                "tasks/2026-04-26-weather.yaml",
            ],
        )
        assert result.sha
        # Verify those files landed in HEAD.
        files = (
            _git(tmp_git_repo, "show", "--name-only", "--pretty=", "HEAD")
            .stdout.strip()
            .splitlines()
        )
        assert "src/skills/weather/skill.py" in files
        assert "tests/skills/weather/test_contract.py" in files

    def test_uses_zhvusha_coder_author(self, tmp_git_repo: Path) -> None:
        _write(tmp_git_repo, "src/skills/weather/skill.py", "code\n")
        spec_path = _make_spec_file(tmp_git_repo)
        _runner(tmp_git_repo).commit(
            spec_slug="weather",
            spec_title="Add weather skill",
            spec_tier=1,
            spec_path=spec_path,
            whitelist_paths=["src/skills/weather/skill.py"],
        )
        author = _git(tmp_git_repo, "log", "-1", "--pretty=%an <%ae>").stdout.strip()
        assert "zhvusha-coder" in author
        assert "zhvusha@local" in author

    def test_message_includes_spec_footer_and_tier(self, tmp_git_repo: Path) -> None:
        _write(tmp_git_repo, "src/skills/weather/skill.py", "code\n")
        spec_path = _make_spec_file(tmp_git_repo)
        _runner(tmp_git_repo).commit(
            spec_slug="weather",
            spec_title="Add weather skill",
            spec_tier=2,
            spec_path=spec_path,
            whitelist_paths=["src/skills/weather/skill.py"],
        )
        msg = _git(tmp_git_repo, "log", "-1", "--pretty=%B").stdout
        assert "weather" in msg.lower() or "Add weather skill" in msg
        assert "tasks/2026-04-26-weather.yaml" in msg
        assert "Tier: 2" in msg
        assert "Agent-Backend: codex_cli" in msg
        assert "Co-Authored-By: Codex" in msg

    def test_returns_short_sha(self, tmp_git_repo: Path) -> None:
        _write(tmp_git_repo, "src/skills/weather/skill.py", "code\n")
        spec_path = _make_spec_file(tmp_git_repo)
        result = _runner(tmp_git_repo).commit(
            spec_slug="weather",
            spec_title="Add weather skill",
            spec_tier=1,
            spec_path=spec_path,
            whitelist_paths=["src/skills/weather/skill.py"],
        )
        head = _git(tmp_git_repo, "rev-parse", "HEAD").stdout.strip()
        assert result.sha == head


class TestCommitYamlUpdate:
    """Second commit, used by ImplementSpecSkill after Editor's main commit
    to land the spec.yaml status mutation. Without this, save_spec_raw
    leaves the worktree dirty and the next cycle's branch_manager refuses
    to start (regression observed on Phase 15 first live cycle)."""

    def test_commits_only_the_spec_yaml(self, tmp_git_repo: Path) -> None:
        spec_path = _make_spec_file(tmp_git_repo, slug="weather")
        # Simulate skill._run_live_cycle: spec.yaml just got rewritten
        # with status=done.
        spec_path.write_text("slug: weather\nstatus: done\n", encoding="utf-8")

        runner = _runner(tmp_git_repo)
        result = runner.commit_yaml_update(
            spec_slug="weather",
            spec_path=spec_path,
            subject="mark weather done",
        )

        files = (
            _git(tmp_git_repo, "show", "--name-only", "--pretty=", "HEAD")
            .stdout.strip()
            .splitlines()
        )
        assert files == ["tasks/2026-04-26-weather.yaml"]
        assert result.sha

    def test_uses_zhvusha_coder_author(self, tmp_git_repo: Path) -> None:
        spec_path = _make_spec_file(tmp_git_repo, slug="weather")
        spec_path.write_text("slug: weather\nstatus: done\n", encoding="utf-8")

        _runner(tmp_git_repo).commit_yaml_update(
            spec_slug="weather",
            spec_path=spec_path,
            subject="mark weather done",
        )
        author = _git(tmp_git_repo, "log", "-1", "--pretty=%an <%ae>").stdout.strip()
        assert "zhvusha-coder" in author
        assert "zhvusha@local" in author

    def test_message_contains_subject_and_slug(self, tmp_git_repo: Path) -> None:
        spec_path = _make_spec_file(tmp_git_repo, slug="weather")
        spec_path.write_text("slug: weather\nstatus: done\n", encoding="utf-8")

        _runner(tmp_git_repo).commit_yaml_update(
            spec_slug="weather",
            spec_path=spec_path,
            subject="mark weather done",
        )
        msg = _git(tmp_git_repo, "log", "-1", "--pretty=%B").stdout
        assert "mark weather done" in msg
        assert "weather" in msg
        assert "Agent-Backend: codex_cli" in msg
        assert "Co-Authored-By: Codex" in msg

    def test_yaml_unchanged_raises(self, tmp_git_repo: Path) -> None:
        from src.skills.implement_spec.commit_runner import CommitRunnerError

        spec_path = _make_spec_file(tmp_git_repo, slug="weather")
        # Stage the spec file so the working tree is clean against HEAD.
        _git(tmp_git_repo, "add", str(spec_path.relative_to(tmp_git_repo)))
        _git(tmp_git_repo, "commit", "-m", "init spec")

        with pytest.raises(CommitRunnerError, match=r"no.*chang|nothing"):
            _runner(tmp_git_repo).commit_yaml_update(
                spec_slug="weather",
                spec_path=spec_path,
                subject="mark weather done",
            )

    def test_other_modified_files_are_not_committed(self, tmp_git_repo: Path) -> None:
        """Only the spec.yaml goes in this commit. Pre-staged or modified
        files outside the yaml stay where they are."""
        spec_path = _make_spec_file(tmp_git_repo, slug="weather")
        spec_path.write_text("slug: weather\nstatus: done\n", encoding="utf-8")
        # Modify another file (not staged).
        _write(tmp_git_repo, "scratch.md", "edited\n")

        _runner(tmp_git_repo).commit_yaml_update(
            spec_slug="weather",
            spec_path=spec_path,
            subject="mark weather done",
        )
        files = (
            _git(tmp_git_repo, "show", "--name-only", "--pretty=", "HEAD")
            .stdout.strip()
            .splitlines()
        )
        assert "scratch.md" not in files
        # scratch.md still untracked
        status = _git(tmp_git_repo, "status", "--porcelain").stdout
        assert "scratch.md" in status


class TestCommitGuards:
    def test_no_whitelist_changes_raises(self, tmp_git_repo: Path) -> None:
        from src.skills.implement_spec.commit_runner import CommitRunnerError

        spec_path = _make_spec_file(tmp_git_repo)
        with pytest.raises(CommitRunnerError, match=r"no.*chang|нет"):
            _runner(tmp_git_repo).commit(
                spec_slug="weather",
                spec_title="Add weather skill",
                spec_tier=1,
                spec_path=spec_path,
                whitelist_paths=["src/skills/weather/skill.py"],
            )

    def test_pre_staged_extra_file_raises(self, tmp_git_repo: Path) -> None:
        """Something already staged outside whitelist must abort the commit."""
        from src.skills.implement_spec.commit_runner import CommitRunnerError

        # Stage a file outside the whitelist before the runner starts.
        _write(tmp_git_repo, "src/llm/router.py", "secret\n")
        _git(tmp_git_repo, "add", "src/llm/router.py")
        # Also write a whitelist file so there *is* something to commit.
        _write(tmp_git_repo, "src/skills/weather/skill.py", "code\n")
        spec_path = _make_spec_file(tmp_git_repo)
        with pytest.raises(CommitRunnerError, match=r"whitelist|src/llm"):
            _runner(tmp_git_repo).commit(
                spec_slug="weather",
                spec_title="Add weather skill",
                spec_tier=1,
                spec_path=spec_path,
                whitelist_paths=["src/skills/weather/skill.py"],
            )

    def test_unstaged_outside_whitelist_is_ignored(self, tmp_git_repo: Path) -> None:
        """Untracked / modified files outside the whitelist stay untouched."""
        _write(tmp_git_repo, "scratchpad.md", "draft\n")  # untracked, outside
        _write(tmp_git_repo, "src/skills/weather/skill.py", "code\n")  # whitelist
        spec_path = _make_spec_file(tmp_git_repo)
        _runner(tmp_git_repo).commit(
            spec_slug="weather",
            spec_title="Add weather skill",
            spec_tier=1,
            spec_path=spec_path,
            whitelist_paths=["src/skills/weather/skill.py"],
        )
        # scratchpad.md still untracked, not in HEAD
        files = _git(tmp_git_repo, "show", "--name-only", "--pretty=", "HEAD").stdout
        assert "scratchpad.md" not in files


# =====================================================================
# Phase 16: existing_tests_to_update legitimate-mutation channel
# =====================================================================


class TestCommitWithExistingTestsToUpdate:
    """``commit`` must accept paths from ``spec.existing_tests_to_update``
    as legitimate edit targets, alongside ``whitelist_paths``.

    Phase 16 — Architect declares specific existing tests that the spec
    legitimately needs to mutate (extending a finite collection behind
    a fixed-set assertion). These paths are NOT in the surgical
    ``whitelist_paths``, but the commit gate must let them through:
    pre-staged extras coming from these paths must not abort, the
    post-stage extras check must accept them, and the auto-fix retry
    must allow hooks to mutate them too.
    """

    def test_listed_test_path_is_committed(self, tmp_git_repo: Path) -> None:
        _write(tmp_git_repo, "src/skills/weather/skill.py", "code\n")
        _write(
            tmp_git_repo,
            "tests/research/test_research_service.py",
            "test\n",
        )
        spec_path = _make_spec_file(tmp_git_repo)
        result = _runner(tmp_git_repo).commit(
            spec_slug="weather",
            spec_title="Add weather skill + extend presets",
            spec_tier=1,
            spec_path=spec_path,
            whitelist_paths=["src/skills/weather/skill.py"],
            existing_tests_to_update_paths=[
                "tests/research/test_research_service.py",
            ],
        )
        assert result.sha
        files = _git(tmp_git_repo, "show", "--name-only", "--pretty=", "HEAD").stdout
        assert "tests/research/test_research_service.py" in files
        assert "src/skills/weather/skill.py" in files

    def test_listed_test_path_pre_staged_is_not_treated_as_extra(
        self, tmp_git_repo: Path
    ) -> None:
        """Editor staged ``tests/research/test_research_service.py``
        legitimately via the existing_tests_to_update channel. The
        pre-staged check must accept it, not abort the commit."""
        _write(
            tmp_git_repo,
            "tests/research/test_research_service.py",
            "test\n",
        )
        _git(tmp_git_repo, "add", "tests/research/test_research_service.py")
        _write(tmp_git_repo, "src/skills/weather/skill.py", "code\n")
        spec_path = _make_spec_file(tmp_git_repo)
        # Must not raise — listed test path is allowed even pre-staged.
        result = _runner(tmp_git_repo).commit(
            spec_slug="weather",
            spec_title="Add weather skill + extend presets",
            spec_tier=1,
            spec_path=spec_path,
            whitelist_paths=["src/skills/weather/skill.py"],
            existing_tests_to_update_paths=[
                "tests/research/test_research_service.py",
            ],
        )
        assert result.sha

    def test_unlisted_test_path_pre_staged_still_aborts(
        self, tmp_git_repo: Path
    ) -> None:
        """A test path NOT in either list pre-staged still aborts the
        commit — the legitimate channel must be explicit, not catch-all."""
        from src.skills.implement_spec.commit_runner import CommitRunnerError

        _write(tmp_git_repo, "tests/skills/research/test_other.py", "test\n")
        _git(tmp_git_repo, "add", "tests/skills/research/test_other.py")
        _write(tmp_git_repo, "src/skills/weather/skill.py", "code\n")
        spec_path = _make_spec_file(tmp_git_repo)
        with pytest.raises(CommitRunnerError, match=r"whitelist|tests/skills"):
            _runner(tmp_git_repo).commit(
                spec_slug="weather",
                spec_title="Add weather skill",
                spec_tier=1,
                spec_path=spec_path,
                whitelist_paths=["src/skills/weather/skill.py"],
                existing_tests_to_update_paths=[
                    "tests/research/test_research_service.py",
                ],
            )

    def test_default_empty_list_keeps_old_behaviour(self, tmp_git_repo: Path) -> None:
        """Without the kwarg, a non-whitelisted test path is still
        rejected — full backward compat with pre-Phase-16 callers."""
        from src.skills.implement_spec.commit_runner import CommitRunnerError

        _write(
            tmp_git_repo,
            "tests/research/test_research_service.py",
            "test\n",
        )
        _git(tmp_git_repo, "add", "tests/research/test_research_service.py")
        _write(tmp_git_repo, "src/skills/weather/skill.py", "code\n")
        spec_path = _make_spec_file(tmp_git_repo)
        with pytest.raises(CommitRunnerError, match=r"whitelist"):
            _runner(tmp_git_repo).commit(
                spec_slug="weather",
                spec_title="Add weather skill",
                spec_tier=1,
                spec_path=spec_path,
                whitelist_paths=["src/skills/weather/skill.py"],
            )


class TestCommitNoDowngradeDeletionGate:
    def _track_prompt_file(self, repo: Path) -> str:
        target = "src/skills/chat_response/prompts.py"
        _write(repo, target, "RULE_A\nRULE_B\n")
        _git(repo, "add", target)
        _git(
            repo,
            "-c",
            "user.email=t@local",
            "-c",
            "user.name=t",
            "commit",
            "-m",
            "track prompt",
        )
        return target

    def test_blocks_prompt_deletion_without_allowed_simplification(
        self, tmp_git_repo: Path
    ) -> None:
        from src.skills.implement_spec.commit_runner import CommitRunnerError

        target = self._track_prompt_file(tmp_git_repo)
        _write(tmp_git_repo, target, "RULE_A\n")
        spec_path = _make_spec_file(tmp_git_repo)

        with pytest.raises(
            CommitRunnerError, match=r"allowed_simplifications|protected"
        ):
            _runner(tmp_git_repo).commit(
                spec_slug="chat-prompt",
                spec_title="Calibrate chat prompt",
                spec_tier=2,
                spec_path=spec_path,
                whitelist_paths=[target],
            )

    def test_allows_prompt_deletion_when_spec_declares_simplification(
        self, tmp_git_repo: Path
    ) -> None:
        target = self._track_prompt_file(tmp_git_repo)
        _write(tmp_git_repo, target, "RULE_A\n")
        spec_path = _make_spec_file(tmp_git_repo)

        result = _runner(tmp_git_repo).commit(
            spec_slug="chat-prompt",
            spec_title="Calibrate chat prompt",
            spec_tier=2,
            spec_path=spec_path,
            whitelist_paths=[target],
            allowed_simplifications=[
                "Remove duplicate prompt rule after preserving coverage.",
            ],
        )

        assert result.sha

    def test_non_protected_deletion_still_uses_normal_whitelist_gate(
        self, tmp_git_repo: Path
    ) -> None:
        target = "src/skills/weather/skill.py"
        _write(tmp_git_repo, target, "a\nb\n")
        _git(tmp_git_repo, "add", target)
        _git(
            tmp_git_repo,
            "-c",
            "user.email=t@local",
            "-c",
            "user.name=t",
            "commit",
            "-m",
            "track weather",
        )
        _write(tmp_git_repo, target, "a\n")
        spec_path = _make_spec_file(tmp_git_repo)

        result = _runner(tmp_git_repo).commit(
            spec_slug="weather",
            spec_title="Calibrate weather",
            spec_tier=1,
            spec_path=spec_path,
            whitelist_paths=[target],
        )

        assert result.sha


# ----------------------------------------------------------- regression
#
# 2026-04-27 — first refactor live cycle on v4-refactor failed at the
# commit gate: ``ruff-format`` auto-fixed one of the touched files
# during pre-commit and ``CommitRunner.commit`` did not retry. The
# Editor's diff was correct, but the commit silently failed and the
# spec was marked failed. Fix: when a post-commit failure leaves
# whitelist files modified in the working tree (the signature of an
# auto-fixing hook), re-stage them and retry the commit. Capped at
# 3 attempts so a real failure still surfaces.


def _install_auto_fix_hook(repo: Path, target_rel: str) -> None:
    """Install a pre-commit hook that auto-fixes ``target_rel`` once.

    First commit attempt: hook appends an "AUTO_FIX_MARKER" line to the
    target file and exits 1 (mimicking ruff-format's behaviour after
    ``files were modified by this hook``).

    Subsequent attempts: marker is already present, hook exits 0 and the
    commit goes through. This reproduces the real ruff-format pattern.
    """
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(
        f"""#!/bin/bash
file="{target_rel}"
if [ -f "$file" ] && ! grep -q "AUTO_FIX_MARKER" "$file"; then
    echo "AUTO_FIX_MARKER" >> "$file"
    echo "files were modified by this hook" >&2
    exit 1
fi
exit 0
""",
        encoding="utf-8",
    )
    hook.chmod(0o755)


def _install_persistent_fail_hook(repo: Path) -> None:
    """Install a pre-commit hook that always fails — to verify the
    retry loop has a finite cap and surfaces the real error."""
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text('#!/bin/bash\necho "always fails" >&2\nexit 1\n', encoding="utf-8")
    hook.chmod(0o755)


def _install_untracked_auto_fix_hook(repo: Path, target_rel: str) -> None:
    """Install a pre-commit hook that creates an untracked whitelist file once."""
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(
        f"""#!/bin/bash
file="{target_rel}"
if [ ! -f "$file" ]; then
    mkdir -p "$(dirname "$file")"
    echo "generated" > "$file"
    echo "files were modified by this hook" >&2
    exit 1
fi
exit 0
""",
        encoding="utf-8",
    )
    hook.chmod(0o755)


class TestCommitRetryAfterAutoFix:
    """``commit`` retries once after a hook auto-fixes a whitelist file."""

    def test_retries_after_auto_fix_and_succeeds(self, tmp_git_repo: Path) -> None:
        target = "src/skills/weather/skill.py"
        _write(tmp_git_repo, target, "code\n")
        spec_path = _make_spec_file(tmp_git_repo)
        _install_auto_fix_hook(tmp_git_repo, target)

        result = _runner(tmp_git_repo).commit(
            spec_slug="weather",
            spec_title="Add weather skill",
            spec_tier=1,
            spec_path=spec_path,
            whitelist_paths=[target, "tasks/2026-04-26-weather.yaml"],
        )

        assert result.sha
        # Working tree is clean — no leftover hook-modified file.
        status = _git(tmp_git_repo, "status", "--porcelain").stdout.strip()
        assert status == "", f"working tree dirty after retry-success: {status!r}"
        # The auto-fix made it into the commit (the marker is in HEAD).
        committed = _git(tmp_git_repo, "show", f"HEAD:{target}").stdout
        assert "AUTO_FIX_MARKER" in committed

    def test_retries_after_hook_creates_untracked_whitelist_file(
        self, tmp_git_repo: Path
    ) -> None:
        target = "src/skills/weather/skill.py"
        generated = "tests/skills/weather/test_generated_contract.py"
        _write(tmp_git_repo, target, "code\n")
        spec_path = _make_spec_file(tmp_git_repo)
        _install_untracked_auto_fix_hook(tmp_git_repo, generated)

        result = _runner(tmp_git_repo).commit(
            spec_slug="weather",
            spec_title="Add weather skill",
            spec_tier=1,
            spec_path=spec_path,
            whitelist_paths=[target, generated, "tasks/2026-04-26-weather.yaml"],
        )

        assert result.sha
        files = _git(tmp_git_repo, "show", "--name-only", "--pretty=", "HEAD").stdout
        assert generated in files
        status = _git(tmp_git_repo, "status", "--porcelain").stdout.strip()
        assert status == "", f"working tree dirty after retry-success: {status!r}"

    def test_does_not_retry_when_no_whitelist_file_was_modified(
        self, tmp_git_repo: Path
    ) -> None:
        """A failure that doesn't leave whitelist files modified isn't
        an auto-fix scenario — surface the original error immediately."""
        from src.skills.implement_spec.commit_runner import CommitRunnerError

        target = "src/skills/weather/skill.py"
        _write(tmp_git_repo, target, "code\n")
        spec_path = _make_spec_file(tmp_git_repo)
        _install_persistent_fail_hook(tmp_git_repo)

        with pytest.raises(CommitRunnerError, match=r"git commit failed"):
            _runner(tmp_git_repo).commit(
                spec_slug="weather",
                spec_title="Add weather skill",
                spec_tier=1,
                spec_path=spec_path,
                whitelist_paths=[target, "tasks/2026-04-26-weather.yaml"],
            )

    def test_retry_refuses_when_hook_modifies_outside_whitelist(
        self, tmp_git_repo: Path
    ) -> None:
        """If the post-fail working-tree mutation includes paths
        outside the whitelist, we cannot blindly re-stage — abort
        rather than smuggle non-whitelisted content into the commit."""
        from src.skills.implement_spec.commit_runner import CommitRunnerError

        target = "src/skills/weather/skill.py"
        outside = "src/llm/router.py"
        _write(tmp_git_repo, target, "code\n")
        _write(tmp_git_repo, outside, "secret\n")
        # Track outside file in HEAD so it shows as modified, not untracked.
        _git(tmp_git_repo, "add", outside)
        _git(
            tmp_git_repo,
            "-c",
            "user.email=t@local",
            "-c",
            "user.name=t",
            "commit",
            "-m",
            "track outside",
        )
        # Now have the hook modify the outside file on first commit attempt.
        hook = tmp_git_repo / ".git" / "hooks" / "pre-commit"
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_text(
            f"""#!/bin/bash
if ! grep -q "MARKER" "{outside}"; then
    echo "MARKER" >> "{outside}"
    echo "modified" >&2
    exit 1
fi
exit 0
""",
            encoding="utf-8",
        )
        hook.chmod(0o755)

        spec_path = _make_spec_file(tmp_git_repo)
        with pytest.raises(CommitRunnerError, match=r"outside whitelist|outside"):
            _runner(tmp_git_repo).commit(
                spec_slug="weather",
                spec_title="Add weather skill",
                spec_tier=1,
                spec_path=spec_path,
                whitelist_paths=[target, "tasks/2026-04-26-weather.yaml"],
            )

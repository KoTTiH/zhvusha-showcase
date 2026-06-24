"""Contract tests for ``spec_command.git_io.commit_yaml_mutation``.

The helper is what lets ``/spec approve`` and ``/spec reject`` (and
future yaml-mutating commands) leave the worktree clean — without it,
every approve produces a modified-but-uncommitted spec file that
``BranchManager`` then refuses on the next ``/spec_run``.

Behaviour pinned here:

* Atomic — only the spec.yaml is staged and committed; pre-existing
  staged files outside the spec abort with a return value indicating
  no commit was created.
* Author-agnostic — uses the repo's existing git config (Никита's
  identity in production, "Test" in fixtures). No ``zhvusha-coder``
  override; ``check_whitelist.sh`` thereby skips, so this commit doesn't
  trip on its own gate.
* Idempotent — running twice with the same content (after the first
  commit lands) is a no-op (returns False) instead of an empty commit.
* Tolerates absent git — if ``repo_root/.git`` doesn't exist, returns
  False without raising. Lets tests using non-git tmp_path keep working.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603 — args literal, repo tmp_path
        ["git", *args],  # noqa: S607 — git on PATH
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _make_spec(tmp_git_repo: Path, slug: str = "weather-skill") -> Path:
    spec_dir = tmp_git_repo / "tasks"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec = spec_dir / f"2026-04-26-{slug}.yaml"
    spec.write_text(f"slug: {slug}\nstatus: pending_approval\n", encoding="utf-8")
    return spec


class TestCommitYamlMutationHappyPath:
    def test_creates_commit_with_subject(self, tmp_git_repo: Path) -> None:
        from src.skills.spec_command.git_io import commit_yaml_mutation

        spec = _make_spec(tmp_git_repo)
        # Modify spec — would otherwise leave worktree dirty.
        spec.write_text(
            "slug: weather-skill\nstatus: approved\n",
            encoding="utf-8",
        )
        result = commit_yaml_mutation(
            spec_path=spec,
            repo_root=tmp_git_repo,
            subject="approve weather-skill",
        )
        assert result is True
        msg = _git(tmp_git_repo, "log", "-1", "--pretty=%B")
        assert "approve weather-skill" in msg

    def test_only_yaml_lands_in_commit(self, tmp_git_repo: Path) -> None:
        from src.skills.spec_command.git_io import commit_yaml_mutation

        spec = _make_spec(tmp_git_repo)
        spec.write_text(
            "slug: weather-skill\nstatus: approved\n",
            encoding="utf-8",
        )
        # Touch an unrelated file in the working tree (untracked).
        (tmp_git_repo / "scratch.md").write_text("draft", encoding="utf-8")
        commit_yaml_mutation(
            spec_path=spec,
            repo_root=tmp_git_repo,
            subject="approve weather-skill",
        )
        files = (
            _git(tmp_git_repo, "show", "--name-only", "--pretty=", "HEAD")
            .strip()
            .splitlines()
        )
        assert files == ["tasks/2026-04-26-weather-skill.yaml"]

    def test_source_does_not_set_author_env_vars(self) -> None:
        """No ``zhvusha-coder`` author override — these commits represent
        Никита's manual approve / reject actions, not Editor-cycle
        mutations. ``check_whitelist.sh`` skips for non-zhvusha authors,
        so the gate doesn't trip on its own commits.

        This is a *source-inspection* test, not a behavioural one: the
        previous version asserted on the actual git author after a
        commit, but inside an Editor cycle the parent process exports
        ``GIT_AUTHOR_NAME=zhvusha-coder`` and that env var leaks into
        every subprocess — including this test's own ``git commit``,
        making the behavioural assert flake.

        The semantic invariant is unambiguous either way: the helper's
        source must not export ``GIT_AUTHOR_*`` / ``GIT_COMMITTER_*``
        vars or call ``--author=`` / ``-c user.name=``. If somebody
        adds an override later, this test catches it.
        """
        import inspect

        from src.skills.spec_command import git_io

        source = inspect.getsource(git_io)
        forbidden_overrides = (
            "GIT_AUTHOR_NAME",
            "GIT_AUTHOR_EMAIL",
            "GIT_COMMITTER_NAME",
            "GIT_COMMITTER_EMAIL",
            "--author",
            "user.name=",
            "user.email=",
        )
        for needle in forbidden_overrides:
            assert needle not in source, (
                f"git_io.py must not override author identity, but found "
                f"{needle!r} in source — these commits should inherit the "
                f"system git config, not impersonate zhvusha-coder."
            )


class TestCommitYamlMutationGuards:
    def test_no_diff_returns_false(self, tmp_git_repo: Path) -> None:
        """If the spec.yaml is identical to HEAD, do nothing — no empty commit."""
        from src.skills.spec_command.git_io import commit_yaml_mutation

        spec = _make_spec(tmp_git_repo)
        # Stage + commit the spec first to bring HEAD to the same content.
        _git(
            tmp_git_repo,
            "add",
            str(spec.relative_to(tmp_git_repo)),
        )
        _git(tmp_git_repo, "commit", "-m", "init spec")
        # Now no diff — calling commit_yaml_mutation must not produce an
        # empty commit.
        head_before = _git(tmp_git_repo, "rev-parse", "HEAD").strip()
        result = commit_yaml_mutation(
            spec_path=spec,
            repo_root=tmp_git_repo,
            subject="approve weather-skill",
        )
        head_after = _git(tmp_git_repo, "rev-parse", "HEAD").strip()
        assert result is False
        assert head_before == head_after

    def test_pre_staged_other_file_aborts(self, tmp_git_repo: Path) -> None:
        """Atomicity — if something else is already staged, refuse to add
        the yaml on top. Either the operator finishes their commit first,
        or aborts; auto-mixing is a footgun."""
        from src.skills.spec_command.git_io import commit_yaml_mutation

        # Pre-stage an unrelated file.
        (tmp_git_repo / "other.md").write_text("staged\n", encoding="utf-8")
        _git(tmp_git_repo, "add", "other.md")

        spec = _make_spec(tmp_git_repo)
        spec.write_text(
            "slug: weather-skill\nstatus: approved\n",
            encoding="utf-8",
        )
        result = commit_yaml_mutation(
            spec_path=spec,
            repo_root=tmp_git_repo,
            subject="approve weather-skill",
        )
        # Refuses → False, no commit created on top of init.
        assert result is False

    def test_returns_false_for_non_git_repo(self, tmp_path: Path) -> None:
        """Tests outside an initialised git repo (most existing
        SpecCommandSkill tests) must keep working — the function is a
        no-op on a non-git directory."""
        from src.skills.spec_command.git_io import commit_yaml_mutation

        spec_dir = tmp_path / "tasks"
        spec_dir.mkdir()
        spec = spec_dir / "2026-04-26-x.yaml"
        spec.write_text("slug: x\n", encoding="utf-8")
        result = commit_yaml_mutation(
            spec_path=spec,
            repo_root=tmp_path,
            subject="x",
        )
        assert result is False

    def test_spec_outside_repo_root_returns_false(
        self, tmp_git_repo: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """Sanity: if spec_path isn't under repo_root, refuse rather than
        committing something nonsensical."""
        from src.skills.spec_command.git_io import commit_yaml_mutation

        # Spec lives in a different tmp dir.
        outside = tmp_path_factory.mktemp("outside")
        spec = outside / "x.yaml"
        spec.write_text("slug: x\n", encoding="utf-8")
        result = commit_yaml_mutation(
            spec_path=spec,
            repo_root=tmp_git_repo,
            subject="x",
        )
        assert result is False

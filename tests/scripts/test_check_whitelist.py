"""Contract tests for ``scripts/check_whitelist.sh`` (Phase 13).

The hook is a pre-commit gate: when the current branch is
``zhvusha/<slug>`` and the configured git author contains "zhvusha", the
hook reads ``tasks/<slug>.yaml`` and refuses to commit any staged file
that isn't on the spec's ``whitelist_paths``. For any other author or
branch the hook is a no-op (exit 0). Override via the
``WHITELIST_OVERRIDE`` env var.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from collections.abc import Iterable

pytestmark = pytest.mark.contract

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_whitelist.sh"


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    out = subprocess.run(  # noqa: S603 — args literal, repo is tmp_path
        ["git", *args],  # noqa: S607 — git on PATH
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return out.stdout


def _init_repo(repo: Path, *, author_email: str = "test@local") -> None:
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", author_email)
    _git(repo, "config", "user.name", author_email.split("@")[0])
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "README.md").write_text("init\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")


def _write_spec(
    repo: Path,
    *,
    slug: str,
    whitelist: Iterable[str],
    existing_tests_to_update: Iterable[dict[str, str]] | None = None,
) -> Path:
    spec_dir = repo / "tasks"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / f"2026-04-26-{slug}.yaml"
    payload: dict[str, object] = {
        "slug": slug,
        "title": f"Add {slug}",
        "tier": 1,
        "whitelist_paths": list(whitelist),
    }
    if existing_tests_to_update is not None:
        payload["existing_tests_to_update"] = list(existing_tests_to_update)
    spec_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return spec_path


def _stage(repo: Path, rel: str, content: str = "x") -> None:
    full = repo / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    _git(repo, "add", rel)


def _run_script(
    repo: Path, *, env_override: str | None = None
) -> subprocess.CompletedProcess[str]:
    import os

    # Pass through PATH/HOME so git, uv, and python3 (with PyYAML in the
    # project venv) all resolve.
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),  # noqa: S108
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
    }
    if env_override is not None:
        env["WHITELIST_OVERRIDE"] = env_override
    return subprocess.run(  # noqa: S603 — args literal
        ["bash", str(SCRIPT)],  # noqa: S607 — bash on PATH
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


# ---------------------------------------------------------------- skips


class TestSkip:
    def test_non_zhvusha_author_skips(self, tmp_path: Path) -> None:
        _init_repo(tmp_path, author_email="nikita@local")
        _git(tmp_path, "checkout", "-b", "feature/x")
        _stage(tmp_path, "anything.py", "code")
        result = _run_script(tmp_path)
        assert result.returncode == 0, result.stderr

    def test_not_on_zhvusha_branch_skips(self, tmp_path: Path) -> None:
        _init_repo(tmp_path, author_email="zhvusha-coder@local")
        # Stay on main; no zhvusha/ branch
        _stage(tmp_path, "anything.py", "code")
        result = _run_script(tmp_path)
        assert result.returncode == 0, result.stderr

    def test_no_staged_changes_passes(self, tmp_path: Path) -> None:
        _init_repo(tmp_path, author_email="zhvusha-coder@local")
        _git(tmp_path, "checkout", "-b", "zhvusha/weather-skill")
        _write_spec(
            tmp_path,
            slug="weather-skill",
            whitelist=["src/skills/weather/skill.py"],
        )
        result = _run_script(tmp_path)
        assert result.returncode == 0

    def test_override_env_skips_check(self, tmp_path: Path) -> None:
        _init_repo(tmp_path, author_email="zhvusha-coder@local")
        _git(tmp_path, "checkout", "-b", "zhvusha/weather-skill")
        _write_spec(
            tmp_path,
            slug="weather-skill",
            whitelist=["src/skills/weather/skill.py"],
        )
        # File outside whitelist staged.
        _stage(tmp_path, "src/llm/router.py", "secret")
        result = _run_script(tmp_path, env_override="manual ack")
        assert result.returncode == 0


# ---------------------------------------------------------------- enforcement


class TestEnforce:
    def test_pass_when_only_whitelist_staged(self, tmp_path: Path) -> None:
        _init_repo(tmp_path, author_email="zhvusha-coder@local")
        _git(tmp_path, "checkout", "-b", "zhvusha/weather-skill")
        _write_spec(
            tmp_path,
            slug="weather-skill",
            whitelist=[
                "src/skills/weather/skill.py",
                "tasks/2026-04-26-weather-skill.yaml",
            ],
        )
        _stage(tmp_path, "src/skills/weather/skill.py", "code")
        result = _run_script(tmp_path)
        assert result.returncode == 0, result.stderr

    def test_block_when_extra_file_staged(self, tmp_path: Path) -> None:
        _init_repo(tmp_path, author_email="zhvusha-coder@local")
        _git(tmp_path, "checkout", "-b", "zhvusha/weather-skill")
        _write_spec(
            tmp_path,
            slug="weather-skill",
            whitelist=["src/skills/weather/skill.py"],
        )
        _stage(tmp_path, "src/skills/weather/skill.py", "code")
        _stage(tmp_path, "src/llm/router.py", "secret")
        result = _run_script(tmp_path)
        assert result.returncode != 0
        assert (
            "src/llm/router.py" in result.stderr or "src/llm/router.py" in result.stdout
        )

    def test_block_when_spec_missing(self, tmp_path: Path) -> None:
        _init_repo(tmp_path, author_email="zhvusha-coder@local")
        _git(tmp_path, "checkout", "-b", "zhvusha/no-spec")
        _stage(tmp_path, "src/skills/foo/skill.py", "code")
        result = _run_script(tmp_path)
        assert result.returncode != 0


class TestEnforceExistingTestsToUpdate:
    """Phase 16: ``existing_tests_to_update[*].path`` joins
    ``whitelist_paths`` for the duration of the gate.

    The bash hook reads the spec yaml directly via PyYAML — when the
    field is present, each entry's ``path`` must be added to the
    allowed set, otherwise the legitimately-mutated test would be
    blocked at the second gate (defence-in-depth alongside the
    runtime PreToolUse hook and ``CommitRunner``).
    """

    def test_pass_when_only_listed_test_staged(self, tmp_path: Path) -> None:
        _init_repo(tmp_path, author_email="zhvusha-coder@local")
        _git(tmp_path, "checkout", "-b", "zhvusha/weather-skill")
        _write_spec(
            tmp_path,
            slug="weather-skill",
            whitelist=["src/skills/weather/skill.py"],
            existing_tests_to_update=[
                {
                    "path": "tests/research/test_research_service.py",
                    "test_name": "TestResearchPresets.test_four_presets_defined",
                    "reason": "extending PRESETS breaks fixed-set assertion",
                    "allowed_changes": "add the new entry to the asserted set",
                }
            ],
        )
        _stage(
            tmp_path,
            "tests/research/test_research_service.py",
            "test\n",
        )
        result = _run_script(tmp_path)
        assert result.returncode == 0, result.stderr

    def test_pass_when_whitelist_and_listed_test_staged(self, tmp_path: Path) -> None:
        _init_repo(tmp_path, author_email="zhvusha-coder@local")
        _git(tmp_path, "checkout", "-b", "zhvusha/weather-skill")
        _write_spec(
            tmp_path,
            slug="weather-skill",
            whitelist=["src/skills/weather/skill.py"],
            existing_tests_to_update=[
                {
                    "path": "tests/research/test_research_service.py",
                    "test_name": "TestResearchPresets.test_four_presets_defined",
                    "reason": "extending PRESETS breaks fixed-set assertion",
                    "allowed_changes": "add the new entry to the asserted set",
                }
            ],
        )
        _stage(tmp_path, "src/skills/weather/skill.py", "code")
        _stage(
            tmp_path,
            "tests/research/test_research_service.py",
            "test\n",
        )
        result = _run_script(tmp_path)
        assert result.returncode == 0, result.stderr

    def test_block_when_unlisted_test_staged(self, tmp_path: Path) -> None:
        _init_repo(tmp_path, author_email="zhvusha-coder@local")
        _git(tmp_path, "checkout", "-b", "zhvusha/weather-skill")
        _write_spec(
            tmp_path,
            slug="weather-skill",
            whitelist=["src/skills/weather/skill.py"],
            existing_tests_to_update=[
                {
                    "path": "tests/research/test_research_service.py",
                    "test_name": "TestResearchPresets.test_four_presets_defined",
                    "reason": "extending PRESETS breaks fixed-set assertion",
                    "allowed_changes": "add the new entry to the asserted set",
                }
            ],
        )
        _stage(tmp_path, "src/skills/weather/skill.py", "code")
        # Different test path, NOT in either list — must be blocked.
        _stage(tmp_path, "tests/skills/research/test_other.py", "test\n")
        result = _run_script(tmp_path)
        assert result.returncode != 0
        assert (
            "tests/skills/research/test_other.py" in result.stderr
            or "tests/skills/research/test_other.py" in result.stdout
        )

    def test_empty_existing_tests_to_update_acts_like_old_behaviour(
        self, tmp_path: Path
    ) -> None:
        _init_repo(tmp_path, author_email="zhvusha-coder@local")
        _git(tmp_path, "checkout", "-b", "zhvusha/weather-skill")
        _write_spec(
            tmp_path,
            slug="weather-skill",
            whitelist=["src/skills/weather/skill.py"],
            existing_tests_to_update=[],
        )
        _stage(tmp_path, "src/skills/weather/skill.py", "code")
        _stage(
            tmp_path,
            "tests/research/test_research_service.py",
            "test\n",
        )
        result = _run_script(tmp_path)
        assert result.returncode != 0

"""Tests for GitChangesCollector."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from src.collectors.git import GitChangesCollector

GIT_AVAILABLE = shutil.which("git") is not None
pytestmark = pytest.mark.skipif(not GIT_AVAILABLE, reason="git binary required")


def _git_env(repo: Path) -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(repo),
        "GIT_AUTHOR_NAME": "Nikita",
        "GIT_AUTHOR_EMAIL": "nikita@example.com",
        "GIT_COMMITTER_NAME": "Nikita",
        "GIT_COMMITTER_EMAIL": "nikita@example.com",
    }


def _sh(cwd: Path, *args: str, env: dict[str, str]) -> None:
    subprocess.run(  # noqa: S603
        args, cwd=str(cwd), check=True, capture_output=True, env=env
    )


def _init_repo(repo: Path) -> dict[str, str]:
    repo.mkdir(parents=True, exist_ok=True)
    env = _git_env(repo)
    _sh(repo, "git", "init", "-q", "-b", "main", env=env)
    _sh(repo, "git", "config", "user.email", "nikita@example.com", env=env)
    _sh(repo, "git", "config", "user.name", "Nikita", env=env)
    _sh(repo, "git", "config", "commit.gpgsign", "false", env=env)
    return env


def _commit(repo: Path, env: dict[str, str], fname: str, content: str, msg: str) -> str:
    (repo / fname).write_text(content, encoding="utf-8")
    _sh(repo, "git", "add", fname, env=env)
    _sh(repo, "git", "commit", "-q", "-m", msg, env=env)
    rev_args = ["git", "rev-parse", "HEAD"]
    rev = subprocess.run(  # noqa: S603
        rev_args,
        cwd=str(repo),
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return rev.stdout.strip()


@pytest.fixture
def git_settings(tmp_path: Path) -> SimpleNamespace:
    project = tmp_path / "project"
    workspace = tmp_path / "workspace"
    (workspace / "inbox").mkdir(parents=True)
    return SimpleNamespace(
        project_path=str(project),
        workspace_path=str(workspace),
        admin_user_id=12345,
        git_max_commits=100,
    )


async def test_first_run_no_state_uses_since(git_settings: SimpleNamespace):
    """First run (no state file) uses --since and returns recent commits."""
    project = Path(git_settings.project_path)
    env = _init_repo(project)
    _commit(project, env, "a.py", "print(1)\n", "feat: add a")
    _commit(project, env, "b.py", "print(2)\n", "feat: add b")

    collector = GitChangesCollector(git_settings)
    result = await collector.collect(since=datetime.now(tz=UTC) - timedelta(hours=24))

    assert len(result.commits) == 2
    assert result.range_mode == "first_run"
    assert result.branch == "main"
    # Newest first
    assert result.commits[0].subject == "feat: add b"
    assert result.commits[0].insertions >= 1


async def test_second_run_uses_sha_range(git_settings: SimpleNamespace):
    """After first run, state is saved; second run uses last_sha..HEAD range."""
    project = Path(git_settings.project_path)
    env = _init_repo(project)
    _commit(project, env, "a.py", "1\n", "c1")

    collector = GitChangesCollector(git_settings)
    summary1 = await collector.collect_and_save(episodic=None)
    assert "1 коммит" in summary1

    state_path = Path(git_settings.workspace_path) / "state" / "git_collector.json"
    assert state_path.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "last_head_sha" in state

    _commit(project, env, "b.py", "2\n", "c2")

    result = await collector.collect()
    assert result.range_mode == "sha_range"
    assert len(result.commits) == 1
    assert result.commits[0].subject == "c2"


async def test_forced_lookback_ignores_saved_sha_range(
    git_settings: SimpleNamespace,
):
    """Explicit /morning lookback must cover the requested time window."""
    project = Path(git_settings.project_path)
    env = _init_repo(project)
    _commit(project, env, "a.py", "1\n", "c1")
    _commit(project, env, "b.py", "2\n", "c2")

    collector = GitChangesCollector(git_settings)
    await collector.collect_and_save(episodic=None)

    _commit(project, env, "c.py", "3\n", "c3")

    since = datetime.now(tz=UTC) - timedelta(hours=24)
    state_range = await collector.collect(since=since)
    forced = await collector.collect(since=since, force_since=True)

    assert state_range.range_mode == "sha_range"
    assert [commit.subject for commit in state_range.commits] == ["c3"]
    assert forced.range_mode == "lookback"
    assert [commit.subject for commit in forced.commits] == ["c3", "c2", "c1"]


async def test_empty_range_writes_no_new_commits(git_settings: SimpleNamespace):
    """When HEAD == last_head_sha, summary is 'Нет новых коммитов'."""
    project = Path(git_settings.project_path)
    env = _init_repo(project)
    _commit(project, env, "a.py", "1\n", "c1")

    collector = GitChangesCollector(git_settings)
    await collector.collect_and_save(episodic=None)

    summary = await collector.collect_and_save(episodic=None)
    assert summary == "Нет новых коммитов"

    inbox = Path(git_settings.workspace_path) / "inbox"
    md = next(inbox.glob("git_*.md"))
    assert "Нет новых коммитов" in md.read_text(encoding="utf-8")


async def test_stale_sha_falls_back_to_since(git_settings: SimpleNamespace):
    """When saved SHA no longer exists, fallback to --since."""
    project = Path(git_settings.project_path)
    state_dir = Path(git_settings.workspace_path) / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "git_collector.json").write_text(
        json.dumps(
            {
                "last_head_sha": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
                "last_head_short": "deadbee",
                "last_branch": "main",
                "last_run_at": "2026-04-01T08:00:00+00:00",
                "last_commit_count": 0,
            }
        ),
        encoding="utf-8",
    )

    env = _init_repo(project)
    _commit(project, env, "a.py", "1\n", "c1")

    collector = GitChangesCollector(git_settings)
    result = await collector.collect(since=datetime.now(tz=UTC) - timedelta(hours=24))
    assert result.range_mode in ("since", "first_run")
    assert len(result.commits) == 1


async def test_not_a_git_repo(git_settings: SimpleNamespace):
    """Path exists but is not a git repo → error."""
    project = Path(git_settings.project_path)
    project.mkdir(parents=True)
    (project / "readme.txt").write_text("not a repo", encoding="utf-8")

    collector = GitChangesCollector(git_settings)
    result = await collector.collect()
    assert "Не git" in result.error


async def test_project_path_missing(git_settings: SimpleNamespace):
    """project_path doesn't exist → error."""
    collector = GitChangesCollector(git_settings)
    result = await collector.collect()
    assert "не найден" in result.error


async def test_format_markdown_structure(git_settings: SimpleNamespace):
    """Markdown has all required sections."""
    project = Path(git_settings.project_path)
    env = _init_repo(project)
    _commit(project, env, "a.py", "print('hello')\n", "feat(core): add greeting")
    _commit(project, env, "b.py", "print('world')\n", "fix(core): add world")

    collector = GitChangesCollector(git_settings)
    await collector.collect_and_save(episodic=None)

    md_path = next((Path(git_settings.workspace_path) / "inbox").glob("git_*.md"))
    md = md_path.read_text(encoding="utf-8")
    assert "# Git изменения" in md
    assert "**Репозиторий**" in md
    assert "**Период**" in md
    assert "## Коммиты" in md
    assert "feat(core): add greeting" in md
    assert "fix(core): add world" in md
    assert "Никита" in md
    assert "## Итого" in md


async def test_max_commits_truncation(git_settings: SimpleNamespace):
    """When commit count exceeds git_max_commits, truncated flag is set."""
    git_settings.git_max_commits = 2
    project = Path(git_settings.project_path)
    env = _init_repo(project)
    for i in range(4):
        _commit(project, env, f"f{i}.py", f"{i}\n", f"c{i}")

    collector = GitChangesCollector(git_settings)
    result = await collector.collect(since=datetime.now(tz=UTC) - timedelta(hours=24))
    assert result.truncated is True
    assert len(result.commits) == 2


async def test_records_episodes_per_commit(git_settings: SimpleNamespace):
    """Each commit is recorded as an episode with source=git."""
    project = Path(git_settings.project_path)
    env = _init_repo(project)
    _commit(project, env, "a.py", "1\n", "c1")
    _commit(project, env, "b.py", "2\n", "c2")

    mock_episodic = AsyncMock()
    mock_episodic.record = AsyncMock(return_value=1)

    collector = GitChangesCollector(git_settings)
    await collector.collect_and_save(episodic=mock_episodic)

    assert mock_episodic.record.await_count == 2
    call_kwargs = mock_episodic.record.call_args_list[0].kwargs
    assert call_kwargs["source"] == "git"
    assert call_kwargs["person_name"] == "Никита"
    assert call_kwargs["domain"] == "chat"
    assert call_kwargs["chat_type"] == "personal"
    assert "Никита commit" in call_kwargs["content"]


async def test_zhvusha_commits_are_marked_separately(git_settings: SimpleNamespace):
    """Self-coding commits stay distinguishable in /morning git context."""
    project = Path(git_settings.project_path)
    env = _init_repo(project)
    _commit(project, env, "a.py", "1\n", "fix: human change")
    zhvusha_env = {
        **env,
        "GIT_AUTHOR_NAME": "zhvusha-coder",
        "GIT_AUTHOR_EMAIL": "zhvusha@local",
        "GIT_COMMITTER_NAME": "zhvusha-coder",
        "GIT_COMMITTER_EMAIL": "zhvusha@local",
    }
    _commit(
        project,
        zhvusha_env,
        "b.py",
        "2\n",
        "feat(self_coding): calibrate greeting",
    )

    mock_episodic = AsyncMock()
    mock_episodic.record = AsyncMock(return_value=1)

    collector = GitChangesCollector(git_settings)
    await collector.collect_and_save(episodic=mock_episodic)

    md_path = next((Path(git_settings.workspace_path) / "inbox").glob("git_*.md"))
    md = md_path.read_text(encoding="utf-8")
    assert "Жвуша self-coding" in md
    assert "zhvusha-coder <zhvusha@local>" in md
    assert "Никита: 1 коммит" in md
    assert "Жвуша self-coding: 1 коммит" in md

    first_call = mock_episodic.record.call_args_list[0].kwargs
    assert first_call["person_name"] == "Жвуша"
    assert first_call["metadata"]["actor"] == "zhvusha_self_coding"
    assert first_call["metadata"]["self_coding_commit"] == "true"
    assert "Жвуша self-coding commit" in first_call["content"]


def test_parse_shortstat():
    """Shortstat parser handles all three cases."""
    f, i, d = GitChangesCollector._parse_shortstat(
        "5 files changed, 66 insertions(+), 92 deletions(-)"
    )
    assert (f, i, d) == (5, 66, 92)
    f, i, d = GitChangesCollector._parse_shortstat("1 file changed, 3 insertions(+)")
    assert (f, i, d) == (1, 3, 0)
    f, i, d = GitChangesCollector._parse_shortstat("2 files changed, 5 deletions(-)")
    assert (f, i, d) == (2, 0, 5)


async def test_state_persisted_on_empty_range(git_settings: SimpleNamespace):
    """Even when no new commits, state last_run_at is bumped."""
    project = Path(git_settings.project_path)
    env = _init_repo(project)
    _commit(project, env, "a.py", "1\n", "c1")

    collector = GitChangesCollector(git_settings)
    await collector.collect_and_save(episodic=None)
    state_path = Path(git_settings.workspace_path) / "state" / "git_collector.json"
    first_state = json.loads(state_path.read_text(encoding="utf-8"))
    await asyncio.sleep(0.01)
    await collector.collect_and_save(episodic=None)
    second_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert second_state["last_run_at"] != first_state["last_run_at"]
    assert second_state["last_head_sha"] == first_state["last_head_sha"]


async def test_run_git_collector_wrapper_success(git_settings: SimpleNamespace):
    """The orchestrator wrapper returns a success status."""
    from src.skills.workspace_session.collector import _run_git_collector

    project = Path(git_settings.project_path)
    env = _init_repo(project)
    _commit(project, env, "a.py", "1\n", "c1")

    status = await _run_git_collector(git_settings, episodic=None)
    assert status.name == "Git"
    assert status.success is True
    assert "коммит" in status.message.lower()


@pytest.mark.parametrize(
    ("lookback_hours", "expected_force_since"),
    [(24, True), (70, True)],
)
async def test_phase3_forces_git_since_for_recovery_window(
    git_settings: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    lookback_hours: int,
    expected_force_since: bool,
):
    """/morning must make git honor the requested recovery time window."""
    from src.skills.workspace_session import collector as workspace_collector

    seen_force_since: bool | None = None

    async def fake_run_git_collector(
        settings: SimpleNamespace,
        episodic: object | None,
        since: datetime | None = None,
        *,
        force_since: bool = False,
    ):
        nonlocal seen_force_since
        del settings, episodic, since
        seen_force_since = force_since
        return SimpleNamespace(
            name="Git",
            success=True,
            message="ok",
            format_line=lambda: "- Git: ok",
        )

    monkeypatch.setattr(
        workspace_collector,
        "_run_git_collector",
        fake_run_git_collector,
    )

    await workspace_collector.collect_phase3_sources(
        Path(git_settings.workspace_path) / "inbox",
        git_settings,
        lookback_hours=lookback_hours,
    )

    assert seen_force_since is expected_force_since


async def test_run_git_collector_wrapper_missing_path(tmp_path: Path):
    """The orchestrator wrapper reports failure with error details."""
    from src.skills.workspace_session.collector import _run_git_collector

    (tmp_path / "ws" / "inbox").mkdir(parents=True)
    settings = SimpleNamespace(
        project_path=str(tmp_path / "does-not-exist"),
        workspace_path=str(tmp_path / "ws"),
        admin_user_id=12345,
        git_max_commits=100,
    )
    status = await _run_git_collector(settings, episodic=None)
    assert status.name == "Git"
    assert status.success is False

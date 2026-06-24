"""Chat-mode DONE merge handler."""

from __future__ import annotations

import subprocess


def _git(repo, *args: str) -> str:  # type: ignore[no-untyped-def]
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_merge_done_spec_fast_forwards_and_requires_clean_tree(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.skills.chat_self_coding.merge import merge_done_spec

    repo = tmp_path
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "zhvusha/my-spec")
    (repo / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(repo, "add", "feature.txt")
    _git(repo, "commit", "-m", "feature")
    _git(repo, "checkout", "main")

    result = merge_done_spec(repo_root=repo, branch="zhvusha/my-spec")

    assert result.success
    assert (repo / "feature.txt").read_text(encoding="utf-8") == "feature\n"


def test_merge_refuses_dirty_tree(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.skills.chat_self_coding.merge import merge_done_spec

    repo = tmp_path
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")

    result = merge_done_spec(repo_root=repo, branch="zhvusha/missing")

    assert not result.success
    assert "clean" in result.reason.lower() or "dirty" in result.reason.lower()

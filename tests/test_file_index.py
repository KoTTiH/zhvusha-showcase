"""Tests for file_index utility."""

from __future__ import annotations

from pathlib import Path

from src.utils.file_index import clear_cache, list_files


def _make_tree(root: Path) -> None:
    """Create a small directory tree for testing."""
    (root / "personality").mkdir()
    (root / "personality" / "core.md").write_text("core")
    (root / "personality" / "genes.md").write_text("genes")
    (root / "diary").mkdir()
    (root / "diary" / "2026-04-02.md").write_text("day")
    (root / "knowledge").mkdir()
    (root / "knowledge" / "python").mkdir()
    (root / "knowledge" / "python" / "asyncio.md").write_text("async")


def test_lists_files(tmp_path: Path) -> None:
    clear_cache()
    _make_tree(tmp_path)

    result = list_files(tmp_path)

    assert "personality/" in result
    assert "personality/core.md" in result
    assert "diary/2026-04-02.md" in result
    assert "knowledge/python/asyncio.md" in result


def test_caches_result(tmp_path: Path) -> None:
    clear_cache()
    _make_tree(tmp_path)

    result1 = list_files(tmp_path)
    # Add a new file — should NOT appear because cache is active
    (tmp_path / "new_file.txt").write_text("new")
    result2 = list_files(tmp_path)

    assert result1 == result2
    assert "new_file.txt" not in result2


def test_max_depth_limits_traversal(tmp_path: Path) -> None:
    clear_cache()
    _make_tree(tmp_path)

    result = list_files(tmp_path, max_depth=1)

    # depth 0: personality/, diary/, knowledge/
    # depth 1: personality/core.md, personality/genes.md, diary/2026-04-02.md, knowledge/python/
    assert "personality/core.md" in result
    assert "diary/2026-04-02.md" in result
    # depth 2 should be cut off
    assert "knowledge/python/asyncio.md" not in result


def test_skips_git_and_pycache(tmp_path: Path) -> None:
    clear_cache()
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("git")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "mod.pyc").write_text("pyc")
    (tmp_path / "real.txt").write_text("real")

    result = list_files(tmp_path)

    assert ".git" not in result
    assert "__pycache__" not in result
    assert "real.txt" in result


def test_skips_processed_dir(tmp_path: Path) -> None:
    clear_cache()
    (tmp_path / ".processed").mkdir()
    (tmp_path / ".processed" / "old.md").write_text("old")
    (tmp_path / "active.md").write_text("active")

    result = list_files(tmp_path)

    assert ".processed" not in result
    assert "active.md" in result


def test_skips_hidden_files(tmp_path: Path) -> None:
    clear_cache()
    (tmp_path / ".env").write_text("secret")
    (tmp_path / "visible.txt").write_text("ok")

    result = list_files(tmp_path)

    assert ".env" not in result
    assert "visible.txt" in result


def test_empty_dir_returns_empty(tmp_path: Path) -> None:
    clear_cache()
    empty = tmp_path / "empty"
    empty.mkdir()

    result = list_files(empty)

    assert result == ""


def test_nonexistent_dir_returns_empty(tmp_path: Path) -> None:
    clear_cache()
    result = list_files(tmp_path / "nonexistent")
    assert result == ""


def test_clear_cache_invalidates(tmp_path: Path) -> None:
    clear_cache()
    (tmp_path / "first.txt").write_text("first")
    result1 = list_files(tmp_path)

    (tmp_path / "second.txt").write_text("second")
    clear_cache()
    result2 = list_files(tmp_path)

    assert "first.txt" in result1
    assert "second.txt" not in result1
    assert "second.txt" in result2


def test_symlink_to_dir_not_followed(tmp_path: Path) -> None:
    """Symlinks starting with . are skipped by the hidden-file rule."""
    clear_cache()
    target = tmp_path / "target_dir"
    target.mkdir()
    (target / "secret.txt").write_text("secret")
    link = tmp_path / ".link"
    link.symlink_to(target)

    result = list_files(tmp_path)

    assert ".link" not in result
    assert "target_dir/" in result
    assert "target_dir/secret.txt" in result

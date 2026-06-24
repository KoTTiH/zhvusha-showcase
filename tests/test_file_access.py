"""Tests for FileAccessService."""

from __future__ import annotations

from pathlib import Path

from src.core.file_access import (
    MAX_CODE_FILES,
    MAX_FILE_CHARS,
    MAX_WORKSPACE_FILES,
    FileAccessService,
)


def _make_workspace(root: Path) -> Path:
    ws = root / "workspace"
    (ws / "personality").mkdir(parents=True)
    (ws / "personality" / "core.md").write_text("I am Zhvusha.", encoding="utf-8")
    (ws / "diary").mkdir()
    (ws / "diary" / "2026-04-02.md").write_text("Good day", encoding="utf-8")
    return ws


def _make_project(root: Path) -> Path:
    proj = root / "project"
    (proj / "src").mkdir(parents=True)
    (proj / "src" / "main.py").write_text("print('hi')", encoding="utf-8")
    return proj


def test_reads_workspace_file(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    proj = _make_project(tmp_path)
    svc = FileAccessService(ws, proj)

    result = svc.read_files(workspace_files=["personality/core.md"])

    assert "personality/core.md" in result.workspace_contents
    assert result.workspace_contents["personality/core.md"] == "I am Zhvusha."


def test_reads_code_file(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    proj = _make_project(tmp_path)
    svc = FileAccessService(ws, proj)

    result = svc.read_files(code_files=["src/main.py"])

    assert "src/main.py" in result.code_contents
    assert result.code_contents["src/main.py"] == "print('hi')"


def test_truncates_at_max_chars(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    proj = _make_project(tmp_path)
    long_content = "x" * (MAX_FILE_CHARS + 5000)
    (ws / "big.txt").write_text(long_content, encoding="utf-8")
    svc = FileAccessService(ws, proj)

    result = svc.read_files(workspace_files=["big.txt"])

    assert len(result.workspace_contents["big.txt"]) == MAX_FILE_CHARS


def test_rejects_path_traversal(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    proj = _make_project(tmp_path)
    svc = FileAccessService(ws, proj)

    result = svc.read_files(workspace_files=["../../../etc/passwd"])

    assert result.workspace_contents == {}


def test_enforces_max_workspace_files(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    proj = _make_project(tmp_path)
    # Create more than MAX_WORKSPACE_FILES files
    for i in range(MAX_WORKSPACE_FILES + 5):
        (ws / f"file_{i}.txt").write_text(f"content {i}", encoding="utf-8")
    svc = FileAccessService(ws, proj)

    files = [f"file_{i}.txt" for i in range(MAX_WORKSPACE_FILES + 5)]
    result = svc.read_files(workspace_files=files)

    assert len(result.workspace_contents) == MAX_WORKSPACE_FILES


def test_enforces_max_code_files(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    proj = _make_project(tmp_path)
    for i in range(MAX_CODE_FILES + 5):
        (proj / f"mod_{i}.py").write_text(f"# mod {i}", encoding="utf-8")
    svc = FileAccessService(ws, proj)

    files = [f"mod_{i}.py" for i in range(MAX_CODE_FILES + 5)]
    result = svc.read_files(code_files=files)

    assert len(result.code_contents) == MAX_CODE_FILES


def test_rejects_symlink(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    proj = _make_project(tmp_path)
    target = tmp_path / "secret.txt"
    target.write_text("secret", encoding="utf-8")
    link = ws / "link.txt"
    link.symlink_to(target)
    svc = FileAccessService(ws, proj)

    result = svc.read_files(workspace_files=["link.txt"])

    assert result.workspace_contents == {}


def test_nonexistent_file_returns_empty(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    proj = _make_project(tmp_path)
    svc = FileAccessService(ws, proj)

    result = svc.read_files(workspace_files=["nonexistent.md"])

    assert result.workspace_contents == {}


def test_none_file_lists(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    proj = _make_project(tmp_path)
    svc = FileAccessService(ws, proj)

    result = svc.read_files()

    assert result.workspace_contents == {}
    assert result.code_contents == {}


def test_get_workspace_index(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    proj = _make_project(tmp_path)
    svc = FileAccessService(ws, proj)

    index = svc.get_workspace_index()

    assert "personality/" in index
    assert "personality/core.md" in index


def test_get_project_index(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    proj = _make_project(tmp_path)
    svc = FileAccessService(ws, proj)

    index = svc.get_project_index()

    assert "src/" in index
    assert "src/main.py" in index

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from src.skills.workspace_session.workspace import (
    MANAGED_FILES,
    SEED_FILES,
    WORKSPACE_DIRS,
    ensure_workspace,
    get_workspace_path,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    return tmp_path / "zhvusha-workspace"


async def test_get_workspace_path_expands_tilde():
    path = get_workspace_path("~/zhvusha-workspace")
    assert "~" not in str(path)
    assert str(path).endswith("zhvusha-workspace")


async def test_ensure_workspace_creates_directories(workspace_root: Path):
    await ensure_workspace(workspace_root)

    for dir_rel in WORKSPACE_DIRS:
        assert (workspace_root / dir_rel).is_dir(), f"Missing dir: {dir_rel}"


async def test_ensure_workspace_creates_files(workspace_root: Path):
    await ensure_workspace(workspace_root)

    for file_rel in {**MANAGED_FILES, **SEED_FILES}:
        full = workspace_root / file_rel
        assert full.is_file(), f"Missing file: {file_rel}"
        assert full.stat().st_size > 0, f"Empty file: {file_rel}"


async def test_ensure_workspace_does_not_overwrite_seed_files(workspace_root: Path):
    await ensure_workspace(workspace_root)

    # Write custom content to core.md
    core_md = workspace_root / "personality" / "core.md"
    custom = "I have evolved beyond my initial state."
    core_md.write_text(custom)

    # Run again — must NOT overwrite seed files
    await ensure_workspace(workspace_root)
    assert core_md.read_text() == custom


async def test_ensure_workspace_always_overwrites_managed_files(workspace_root: Path):
    await ensure_workspace(workspace_root)

    agents_md = workspace_root / "AGENTS.md"
    agents_md.write_text("stale content")

    # Run again — must overwrite managed files
    await ensure_workspace(workspace_root)
    assert agents_md.read_text() != "stale content"
    assert "Жвуша" in agents_md.read_text()


async def test_ensure_workspace_idempotent(workspace_root: Path):
    await ensure_workspace(workspace_root)
    await ensure_workspace(workspace_root)

    for dir_rel in WORKSPACE_DIRS:
        assert (workspace_root / dir_rel).is_dir()


async def test_workspace_agents_md_content(workspace_root: Path):
    await ensure_workspace(workspace_root)

    agents_md = workspace_root / "AGENTS.md"
    text = agents_md.read_text()

    # Must contain key sections
    assert "Phase 1" in text or "Фаза 1" in text
    assert "Phase 2" in text or "Фаза 2" in text
    assert "Phase 3" in text or "Фаза 3" in text
    assert "inbox" in text
    assert "outbox" in text
    assert "diary" in text
    assert "Self-Coding Archive" in text
    assert "не запускать на её основе новый `/код`" in text
    assert "Kwork-проекты" not in text


async def test_workspace_core_md_content(workspace_root: Path):
    await ensure_workspace(workspace_root)

    core_md = workspace_root / "personality" / "core.md"
    text = core_md.read_text()
    assert "Жвуша" in text or "Zhvusha" in text


async def test_workspace_genes_md_content(workspace_root: Path):
    await ensure_workspace(workspace_root)

    genes_md = workspace_root / "personality" / "genes.md"
    text = genes_md.read_text()
    assert "curiosity" in text.lower() or "любопытство" in text.lower()
    assert "honesty" in text.lower() or "честность" in text.lower()

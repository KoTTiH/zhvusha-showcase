"""Personality substitution: non-personal modes must load public_core.md /
public_identity.md instead of core.md / identity.md.

This keeps private details (Nikita's name, intimate context, relationship
language) from leaking into prompts served to non-owner users — even if
those files grow new private content over time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.skills.chat_response.context_loader import ContextLoader

if TYPE_CHECKING:
    from pathlib import Path


def _write_personality(
    root: Path,
    *,
    core: str | None = "PRIVATE_CORE",
    identity: str | None = "PRIVATE_IDENTITY",
    public_core: str | None = "PUBLIC_CORE",
    public_identity: str | None = "PUBLIC_IDENTITY",
    genes: str = "Curiosity: HIGH",
) -> None:
    p = root / "personality"
    p.mkdir(parents=True, exist_ok=True)
    if core is not None:
        (p / "core.md").write_text(core, encoding="utf-8")
    if identity is not None:
        (p / "identity.md").write_text(identity, encoding="utf-8")
    if public_core is not None:
        (p / "public_core.md").write_text(public_core, encoding="utf-8")
    if public_identity is not None:
        (p / "public_identity.md").write_text(public_identity, encoding="utf-8")
    (p / "genes.md").write_text(genes, encoding="utf-8")


def test_personal_mode_loads_private_core_and_identity(tmp_path: Path) -> None:
    _write_personality(tmp_path)
    loader = ContextLoader(tmp_path)

    result = loader.load_personality(mode="personal")

    assert "PRIVATE_CORE" in result
    assert "PRIVATE_IDENTITY" in result
    assert "PUBLIC_CORE" not in result
    assert "PUBLIC_IDENTITY" not in result


def test_assistant_mode_substitutes_public_core_and_identity(tmp_path: Path) -> None:
    _write_personality(tmp_path)
    loader = ContextLoader(tmp_path)

    result = loader.load_personality(mode="assistant")

    assert "PRIVATE_CORE" not in result
    assert "PRIVATE_IDENTITY" not in result
    assert "PUBLIC_CORE" in result
    assert "PUBLIC_IDENTITY" in result
    assert "Curiosity: HIGH" in result


def test_social_mode_substitutes_public_core_and_identity(tmp_path: Path) -> None:
    _write_personality(tmp_path)
    loader = ContextLoader(tmp_path)

    result = loader.load_personality(mode="social")

    assert "PRIVATE_CORE" not in result
    assert "PRIVATE_IDENTITY" not in result
    assert "PUBLIC_CORE" in result
    assert "PUBLIC_IDENTITY" in result


def test_assistant_fallback_stub_when_public_core_missing(tmp_path: Path) -> None:
    """If public_core.md is absent, a hard-coded stub replaces core.md —
    private file must never leak just because the public copy wasn't written."""
    _write_personality(tmp_path, public_core=None, public_identity=None)
    loader = ContextLoader(tmp_path)

    result = loader.load_personality(mode="assistant")

    assert "PRIVATE_CORE" not in result
    assert "PRIVATE_IDENTITY" not in result
    # Stub must still identify Zhvusha so the prompt isn't empty
    assert "Жвуша" in result


def test_assistant_missing_private_file_still_loads_public(tmp_path: Path) -> None:
    """Missing private files are fine; public ones should still be loaded
    so the assistant prompt is not accidentally empty."""
    _write_personality(tmp_path, core=None, identity=None)
    loader = ContextLoader(tmp_path)

    result = loader.load_personality(mode="assistant")

    assert "PUBLIC_CORE" in result
    assert "PUBLIC_IDENTITY" in result


def test_non_priority_private_files_skipped_in_assistant(tmp_path: Path) -> None:
    """dreams.md / reinforcements.md are already gated by _PERSONAL_ONLY_FILES.
    This test anchors that gate alongside the new core/identity substitution."""
    _write_personality(tmp_path)
    (tmp_path / "personality" / "dreams.md").write_text("MY_DREAM", encoding="utf-8")
    (tmp_path / "personality" / "reinforcements.md").write_text(
        "MY_REINF", encoding="utf-8"
    )
    loader = ContextLoader(tmp_path)

    result = loader.load_personality(mode="assistant")

    assert "MY_DREAM" not in result
    assert "MY_REINF" not in result
    assert "PUBLIC_CORE" in result

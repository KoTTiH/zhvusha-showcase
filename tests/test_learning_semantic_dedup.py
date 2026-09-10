"""Tests for semantic dedup of learning signals before staging.

Covers:
- _extract_statements parses **Statement:** lines from staging markdown
- Empty/missing file returns empty list
- Substring dedup continues to work
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.memory.learning_staging import StagingWriter, _extract_statements
from src.memory.sonnet_enricher import LearningSignal

if TYPE_CHECKING:
    from pathlib import Path


def _make_signal(**overrides: object) -> LearningSignal:
    base: dict[str, object] = {
        "type": "rule",
        "statement": "не писать формально в personal mode",
        "scope": "tone",
        "confidence": 0.92,
        "apply_immediately": True,
        "original_claim": None,
    }
    base.update(overrides)
    return LearningSignal(**base)  # type: ignore[arg-type]


# --- _extract_statements ---


def test_extract_statements_from_staging_file(tmp_path: Path) -> None:
    """Parses **Statement:** lines from staging markdown."""
    content = (
        "## [rule] tone — 2026-04-08 09:53\n"
        "**Statement:** Не писать красивые посты без проверки фактов.\n"
        "**Confidence:** 0.95\n"
        "\n"
        "## [rule] work — 2026-04-08 09:54\n"
        "**Statement:** Всегда проверять источники перед публикацией.\n"
        "**Confidence:** 0.85\n"
    )
    path = tmp_path / "learnings_immediate.md"
    path.write_text(content, encoding="utf-8")

    statements = _extract_statements(path)

    assert len(statements) == 2
    assert "Не писать красивые посты без проверки фактов." in statements
    assert "Всегда проверять источники перед публикацией." in statements


def test_extract_statements_empty_file(tmp_path: Path) -> None:
    """Empty file returns empty list."""
    path = tmp_path / "learnings_immediate.md"
    path.write_text("", encoding="utf-8")

    assert _extract_statements(path) == []


def test_extract_statements_missing_file(tmp_path: Path) -> None:
    """Non-existent file returns empty list."""
    path = tmp_path / "nonexistent.md"

    assert _extract_statements(path) == []


# --- Substring dedup still works ---


def test_substring_dedup_still_works(tmp_path: Path) -> None:
    """Original substring dedup continues to catch exact matches."""
    writer = StagingWriter(tmp_path / ".staging")
    signal = _make_signal(statement="одно и то же правило")

    first = writer.append(signal, episode_id=1)
    second = writer.append(signal, episode_id=2)

    assert first is not None
    assert second is None  # dedup caught it

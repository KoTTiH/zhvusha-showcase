"""Tests for StagingWriter — routes LearningSignal to markdown staging files.

Covers: directory creation, routing by confidence and apply_immediately flags,
correction-specific fields, append vs dedup, size cap warning, chat_id
annotation, and graceful handling of permission errors."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from src.memory.learning_staging import (
    _DEDUP_SCAN_BYTES,
    _MAX_STAGING_BYTES,
    StagingWriter,
)
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


# --- Directory + routing ---


def test_staging_writer_creates_directory(tmp_path: Path) -> None:
    staging_dir = tmp_path / "personality" / ".staging"
    assert not staging_dir.exists()

    StagingWriter(staging_dir)

    assert staging_dir.is_dir()


def test_staging_writer_routes_strong_signal_to_immediate(tmp_path: Path) -> None:
    writer = StagingWriter(tmp_path / ".staging")
    signal = _make_signal()  # default confidence=0.92, apply_immediately=True

    target = writer.append(signal, episode_id=42)

    assert target is not None
    assert target.name == "learnings_immediate.md"
    assert target.exists()
    content = target.read_text(encoding="utf-8")
    assert "не писать формально" in content
    assert "[rule] tone" in content
    assert "**Confidence:** 0.92" in content
    assert "**Trigger episode:** 42" in content


def test_staging_writer_routes_weak_signal_to_pending(tmp_path: Path) -> None:
    writer = StagingWriter(tmp_path / ".staging")
    # low confidence → pending even if apply_immediately=True
    signal = _make_signal(confidence=0.7, apply_immediately=True)

    target = writer.append(signal, episode_id=1)

    assert target is not None
    assert target.name == "learnings_pending.md"


def test_staging_writer_routes_non_immediate_signal_to_pending(tmp_path: Path) -> None:
    writer = StagingWriter(tmp_path / ".staging")
    # high confidence but apply_immediately=False → pending
    signal = _make_signal(confidence=0.95, apply_immediately=False)

    target = writer.append(signal, episode_id=2)

    assert target is not None
    assert target.name == "learnings_pending.md"


def test_staging_writer_includes_original_claim_for_correction(tmp_path: Path) -> None:
    writer = StagingWriter(tmp_path / ".staging")
    signal = _make_signal(
        type="correction",
        statement="Kwork — единственный доход",
        scope="personal_facts",
        original_claim="Никита упоминал какую-то основную работу",
    )

    target = writer.append(signal, episode_id=7)

    assert target is not None
    content = target.read_text(encoding="utf-8")
    assert "**Original claim:** Никита упоминал какую-то основную работу" in content


def test_staging_writer_omits_original_claim_line_for_non_correction(
    tmp_path: Path,
) -> None:
    writer = StagingWriter(tmp_path / ".staging")
    signal = _make_signal()  # rule, original_claim=None

    target = writer.append(signal, episode_id=3)

    assert target is not None
    content = target.read_text(encoding="utf-8")
    assert "**Original claim:**" not in content


def test_staging_writer_appends_multiple_distinct_entries(tmp_path: Path) -> None:
    writer = StagingWriter(tmp_path / ".staging")
    writer.append(
        _make_signal(statement="не писать формально в личном чате"), episode_id=1
    )
    writer.append(
        _make_signal(statement="всегда проверять факты перед публикацией в канал"),
        episode_id=2,
    )

    target = tmp_path / ".staging" / "learnings_immediate.md"
    content = target.read_text(encoding="utf-8")
    assert "не писать формально" in content
    assert "проверять факты" in content
    assert content.count("[rule] tone") == 2


# --- Dedup ---


def test_staging_writer_dedups_identical_statement(tmp_path: Path) -> None:
    writer = StagingWriter(tmp_path / ".staging")
    first = writer.append(_make_signal(statement="одно правило"), episode_id=10)
    second = writer.append(_make_signal(statement="одно правило"), episode_id=11)

    assert first is not None
    assert second is None  # dedup skipped the write

    content = (tmp_path / ".staging" / "learnings_immediate.md").read_text(
        encoding="utf-8"
    )
    assert content.count("одно правило") == 1


def test_staging_writer_semantic_dedup_catches_distant_duplicate(
    tmp_path: Path,
) -> None:
    """Semantic dedup catches identical statements even beyond substring scan window."""
    writer = StagingWriter(tmp_path / ".staging")
    target = tmp_path / ".staging" / "learnings_immediate.md"

    # First append: the statement we'll want to re-append later
    writer.append(_make_signal(statement="долгое правило"), episode_id=1)
    # Pad the file with lots of unrelated entries to push "долгое правило"
    # outside the substring dedup scan window
    filler = "x" * (_DEDUP_SCAN_BYTES + 1024)
    with target.open("a", encoding="utf-8") as f:
        f.write(f"\n## [rule] tone — PAD\n**Statement:** {filler}\n")

    second = writer.append(_make_signal(statement="долгое правило"), episode_id=2)

    # Semantic dedup catches it globally (unlike old substring-only scan)
    assert second is None


# --- Size cap ---


def test_staging_writer_logs_warning_on_oversize(tmp_path: Path) -> None:
    writer = StagingWriter(tmp_path / ".staging")
    target = tmp_path / ".staging" / "learnings_immediate.md"
    # Pre-fill beyond cap
    target.write_text("x" * (_MAX_STAGING_BYTES + 100), encoding="utf-8")

    with patch("src.memory.learning_staging.logger") as mock_logger:
        writer.append(_make_signal(statement="new after oversize"), episode_id=5)

    warning_events = [
        call.args[0] for call in mock_logger.warning.call_args_list if call.args
    ]
    assert "staging_file_oversize" in warning_events


# --- Metadata fields ---


def test_staging_writer_includes_chat_id_when_provided(tmp_path: Path) -> None:
    writer = StagingWriter(tmp_path / ".staging")

    target = writer.append(_make_signal(), episode_id=42, chat_id=12345)

    assert target is not None
    content = target.read_text(encoding="utf-8")
    assert "**Chat:** 12345" in content


def test_staging_writer_omits_chat_id_line_when_none(tmp_path: Path) -> None:
    writer = StagingWriter(tmp_path / ".staging")

    target = writer.append(_make_signal(), episode_id=42, chat_id=None)

    assert target is not None
    content = target.read_text(encoding="utf-8")
    assert "**Chat:**" not in content


# --- Error handling ---


def test_staging_writer_logs_warning_on_permission_error(tmp_path: Path) -> None:
    """OSError during write must not raise — writer logs and returns None."""
    writer = StagingWriter(tmp_path / ".staging")

    with (
        patch(
            "src.memory.learning_staging.Path.open",
            side_effect=PermissionError("denied"),
        ),
        patch("src.memory.learning_staging.logger") as mock_logger,
    ):
        result = writer.append(_make_signal(), episode_id=99)

    assert result is None
    warning_events = [
        call.args[0] for call in mock_logger.warning.call_args_list if call.args
    ]
    assert "staging_write_failed" in warning_events

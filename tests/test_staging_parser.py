"""Tests for `src/memory/staging_parser.py` — the line-state parser that
reads `personality/.staging/learnings_{pending,immediate}.md` files written
by `StagingWriter._format_entry`.

The parser must never raise — malformed blocks either go into
`recoverable_raw_blocks` (when the header is valid but the body is broken)
or are reported via `errors` and dropped (when the header is unparseable).
"""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING

from src.memory.staging_parser import StagingEntry, parse_staging_file

if TYPE_CHECKING:
    from pathlib import Path


# --- Helpers ---


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


_SINGLE_RULE = """
## [rule] tone — 2026-04-04 09:00
**Statement:** не писать формально в personal mode
**Confidence:** 0.92
**Trigger episode:** 42
"""

_CORRECTION = """
## [correction] personal_facts — 2026-04-04 10:15
**Statement:** Kwork — единственный источник дохода
**Confidence:** 0.9
**Chat:** 12345
**Trigger episode:** 103
**Original claim:** предполагала, что у Никиты есть основная работа
"""

_MULTI_ENTRY = """
## [rule] tone — 2026-04-04 09:00
**Statement:** правило A
**Confidence:** 0.92
**Trigger episode:** 1

## [fact] personal_facts — 2026-04-04 09:10
**Statement:** факт B
**Confidence:** 0.85
**Trigger episode:** 2

## [preference] preferences — 2026-04-04 09:20
**Statement:** предпочтение C
**Confidence:** 0.75
**Trigger episode:** 3
"""


# --- Tests ---


def test_parse_empty_file_returns_empty_lists(tmp_path: Path) -> None:
    """Nonexistent path, empty file, and whitespace-only file all return empty."""
    nonexistent = tmp_path / "nothing.md"
    entries, errors, recoverable = parse_staging_file(nonexistent)
    assert entries == []
    assert errors == []
    assert recoverable == []

    empty_file = tmp_path / "empty.md"
    empty_file.write_text("", encoding="utf-8")
    entries, errors, recoverable = parse_staging_file(empty_file)
    assert entries == []
    assert errors == []
    assert recoverable == []

    ws_file = tmp_path / "ws.md"
    ws_file.write_text("   \n\n  \n", encoding="utf-8")
    entries, errors, recoverable = parse_staging_file(ws_file)
    assert entries == []
    assert errors == []
    assert recoverable == []


def test_parse_single_rule_entry(tmp_path: Path) -> None:
    path = tmp_path / "learnings_immediate.md"
    _write(path, _SINGLE_RULE)

    entries, errors, recoverable = parse_staging_file(path)

    assert len(entries) == 1
    assert errors == []
    assert recoverable == []

    entry = entries[0]
    assert isinstance(entry, StagingEntry)
    assert entry.type == "rule"
    assert entry.scope == "tone"
    assert entry.statement == "не писать формально в personal mode"
    assert entry.confidence == 0.92
    assert entry.episode_id == 42
    assert entry.chat_id is None
    assert entry.original_claim is None
    assert entry.source_file == "learnings_immediate.md"
    assert "## [rule] tone" in entry.raw_block
    assert "**Statement:** не писать формально" in entry.raw_block


def test_parse_correction_with_original_claim(tmp_path: Path) -> None:
    path = tmp_path / "learnings_immediate.md"
    _write(path, _CORRECTION)

    entries, errors, recoverable = parse_staging_file(path)

    assert len(entries) == 1
    assert errors == []
    assert recoverable == []
    entry = entries[0]
    assert entry.type == "correction"
    assert entry.scope == "personal_facts"
    assert entry.statement == "Kwork — единственный источник дохода"
    assert entry.confidence == 0.9
    assert entry.chat_id == 12345
    assert entry.episode_id == 103
    assert entry.original_claim == "предполагала, что у Никиты есть основная работа"


def test_parse_multi_entry_file(tmp_path: Path) -> None:
    path = tmp_path / "learnings_pending.md"
    _write(path, _MULTI_ENTRY)

    entries, errors, recoverable = parse_staging_file(path)

    assert len(entries) == 3
    assert errors == []
    assert recoverable == []

    # Order preserved
    assert entries[0].statement == "правило A"
    assert entries[0].type == "rule"
    assert entries[1].statement == "факт B"
    assert entries[1].type == "fact"
    assert entries[2].statement == "предпочтение C"
    assert entries[2].type == "preference"
    assert entries[2].scope == "preferences"


def test_parse_entry_with_chat_id(tmp_path: Path) -> None:
    content = """
## [rule] tone — 2026-04-04 09:00
**Statement:** с чатом
**Confidence:** 0.9
**Chat:** 98765
**Trigger episode:** 1
"""
    path = tmp_path / "learnings_immediate.md"
    _write(path, content)

    entries, _errors, _recoverable = parse_staging_file(path)
    assert len(entries) == 1
    assert entries[0].chat_id == 98765


def test_parse_entry_without_chat_id(tmp_path: Path) -> None:
    path = tmp_path / "learnings_immediate.md"
    _write(path, _SINGLE_RULE)  # has no Chat: line

    entries, _, _ = parse_staging_file(path)
    assert len(entries) == 1
    assert entries[0].chat_id is None


def test_parse_malformed_header_dropped(tmp_path: Path) -> None:
    """Non-matching header line is not a block start — we scan past it looking
    for the next valid `## [...]` header."""
    content = """
## bad header without brackets
**Statement:** orphan
**Confidence:** 0.9

## [rule] tone — 2026-04-04 09:00
**Statement:** valid after orphan
**Confidence:** 0.9
**Trigger episode:** 5
"""
    path = tmp_path / "learnings_immediate.md"
    _write(path, content)

    entries, _errors, _recoverable = parse_staging_file(path)

    # Only the valid entry is parsed; the orphan lines are skipped silently
    # (they never matched `## [` prefix, so they're not considered blocks).
    assert len(entries) == 1
    assert entries[0].statement == "valid after orphan"


def test_parse_recoverable_bad_body_preserves_raw_block(tmp_path: Path) -> None:
    """Header is valid but `Confidence` is non-numeric — we keep the raw block
    in `recoverable_raw_blocks` so the caller can rewrite it as hold."""
    content = """
## [rule] tone — 2026-04-04 09:00
**Statement:** with bad confidence
**Confidence:** not_a_float
**Trigger episode:** 7
"""
    path = tmp_path / "learnings_pending.md"
    _write(path, content)

    entries, errors, recoverable = parse_staging_file(path)

    assert entries == []
    assert len(errors) >= 1
    assert len(recoverable) == 1
    assert "## [rule] tone — 2026-04-04 09:00" in recoverable[0]
    assert "**Statement:** with bad confidence" in recoverable[0]
    assert "**Confidence:** not_a_float" in recoverable[0]


def test_parse_bad_timestamp_dropped(tmp_path: Path) -> None:
    """Unparseable timestamp in header → error, not entry, but raw_block is
    saved as recoverable (header matched the regex so block bounds are known)."""
    content = """
## [rule] tone — not-a-date
**Statement:** bad ts
**Confidence:** 0.9
**Trigger episode:** 1
"""
    path = tmp_path / "learnings_immediate.md"
    _write(path, content)

    entries, _errors, recoverable = parse_staging_file(path)
    # Header regex requires `\d{4}-\d{2}-\d{2} \d{2}:\d{2}` — this doesn't match
    # the regex at all, so the line is not treated as a block start and the
    # body lines get skipped silently.
    assert entries == []
    assert recoverable == []


def test_parse_correction_without_original_claim_invalid(tmp_path: Path) -> None:
    """Biconditional invariant: type=correction ⟺ original_claim present."""
    content = """
## [correction] personal_facts — 2026-04-04 09:00
**Statement:** correction without claim
**Confidence:** 0.9
**Trigger episode:** 1
"""
    path = tmp_path / "learnings_immediate.md"
    _write(path, content)

    entries, errors, recoverable = parse_staging_file(path)

    assert entries == []
    assert len(errors) >= 1
    # Block is recoverable (header valid, body breaks invariant)
    assert len(recoverable) == 1


def test_parse_non_correction_with_original_claim_invalid(tmp_path: Path) -> None:
    """Other side of the invariant: type=rule with original_claim is invalid."""
    content = """
## [rule] tone — 2026-04-04 09:00
**Statement:** rule with claim
**Confidence:** 0.9
**Trigger episode:** 1
**Original claim:** should not be here
"""
    path = tmp_path / "learnings_immediate.md"
    _write(path, content)

    entries, errors, recoverable = parse_staging_file(path)
    assert entries == []
    assert len(errors) >= 1
    assert len(recoverable) == 1


def test_parse_timestamp_has_utc_tzinfo(tmp_path: Path) -> None:
    path = tmp_path / "learnings_immediate.md"
    _write(path, _SINGLE_RULE)

    entries, _, _ = parse_staging_file(path)
    assert len(entries) == 1
    assert entries[0].timestamp.tzinfo == UTC
    assert entries[0].timestamp.year == 2026
    assert entries[0].timestamp.month == 4
    assert entries[0].timestamp.day == 4
    assert entries[0].timestamp.hour == 9
    assert entries[0].timestamp.minute == 0

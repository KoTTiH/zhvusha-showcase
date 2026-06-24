"""Parser for `personality/.staging/learnings_{pending,immediate}.md` files.

Reads the markdown format produced by
`src.memory.learning_staging.StagingWriter._format_entry` and returns a list
of `StagingEntry` dataclasses plus two diagnostic lists:

* `errors`: human-readable messages for blocks that couldn't be parsed.
* `recoverable_raw_blocks`: verbatim raw text of blocks that had a valid
  header but a malformed body — callers (Phase 4 morning review) rewrite
  these back to `learnings_pending.md` as "hold" to avoid data loss.

Never raises: any I/O error is captured into `errors`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger()


_HEADER_RE = re.compile(
    r"^## \[(?P<type>[a-z_]+)\] (?P<scope>[a-z_]+) — "
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s*$"
)
_KV_RE = re.compile(r"^\*\*(?P<key>[A-Za-z][\w ]*):\*\*\s*(?P<val>.+?)\s*$")

_VALID_TYPES = {"rule", "preference", "correction", "fact", "boundary"}
_VALID_SCOPES = {"tone", "work", "personal_facts", "boundaries", "preferences"}


@dataclass
class StagingEntry:
    """A single parsed learning-signal entry from a staging file."""

    type: str
    scope: str
    statement: str
    confidence: float
    timestamp: datetime
    episode_id: int
    chat_id: int | None = None
    original_claim: str | None = None
    source_file: str = ""
    raw_block: str = ""
    line_offset: int = 0


@dataclass
class _BlockResult:
    """Outcome of parsing a single staging block."""

    entry: StagingEntry | None = None
    error: str | None = None
    recoverable: str | None = None


def _parse_kv_lines(lines: list[str]) -> dict[str, str]:
    """Extract `**Key:** value` lines into a lowercase-keyed dict."""
    kv: dict[str, str] = {}
    for line in lines:
        match = _KV_RE.match(line)
        if match:
            kv[match.group("key").lower()] = match.group("val").strip()
    return kv


def _build_entry(
    *,
    entry_type: str,
    scope: str,
    ts: datetime,
    kv: dict[str, str],
    source_file: str,
    raw_block: str,
    block_start: int,
) -> _BlockResult:
    """Validate KV fields and construct a StagingEntry, or return error."""
    if "statement" not in kv or "confidence" not in kv or "trigger episode" not in kv:
        return _BlockResult(
            error=f"missing required fields at line {block_start + 1}",
            recoverable=raw_block,
        )

    try:
        confidence = float(kv["confidence"])
        episode_id = int(kv["trigger episode"])
    except ValueError:
        return _BlockResult(
            error=f"bad numeric fields at line {block_start + 1}",
            recoverable=raw_block,
        )

    chat_id: int | None = None
    if "chat" in kv:
        try:
            chat_id = int(kv["chat"])
        except ValueError:
            # Silently ignore a malformed chat_id; the rest is still usable.
            chat_id = None

    original_claim = kv.get("original claim")

    # Biconditional invariant: type=correction ⟺ original_claim is set.
    if (entry_type == "correction") != (original_claim is not None):
        return _BlockResult(
            error=f"correction/original_claim mismatch at line {block_start + 1}",
            recoverable=raw_block,
        )

    return _BlockResult(
        entry=StagingEntry(
            type=entry_type,
            scope=scope,
            statement=kv["statement"],
            confidence=confidence,
            timestamp=ts,
            episode_id=episode_id,
            chat_id=chat_id,
            original_claim=original_claim,
            source_file=source_file,
            raw_block=raw_block,
            line_offset=block_start + 1,
        ),
    )


def _parse_block(
    lines: list[str],
    block_start: int,
    block_end: int,
    source_file: str,
) -> _BlockResult:
    """Parse one block (header at block_start, body up to block_end-1)."""
    header_line = lines[block_start]
    header_match = _HEADER_RE.match(header_line)
    if header_match is None:
        # Line starts with "## [" but doesn't match the header regex — we
        # don't know block bounds reliably; caller drops it with no recovery.
        return _BlockResult(
            error=f"bad header at line {block_start + 1}: {header_line[:80]}",
        )

    raw_block = "\n".join(lines[block_start:block_end])
    entry_type = header_match.group("type")
    scope = header_match.group("scope")
    ts_str = header_match.group("ts")

    if entry_type not in _VALID_TYPES or scope not in _VALID_SCOPES:
        return _BlockResult(
            error=(
                f"unknown type/scope at line {block_start + 1}: {entry_type}/{scope}"
            ),
            recoverable=raw_block,
        )

    try:
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
    except ValueError:
        return _BlockResult(
            error=f"bad timestamp at line {block_start + 1}: {ts_str}",
            recoverable=raw_block,
        )

    kv = _parse_kv_lines(lines[block_start + 1 : block_end])
    return _build_entry(
        entry_type=entry_type,
        scope=scope,
        ts=ts,
        kv=kv,
        source_file=source_file,
        raw_block=raw_block,
        block_start=block_start,
    )


def _iter_block_ranges(lines: list[str]) -> list[tuple[int, int, bool]]:
    """Walk lines and return `(start, end, header_valid)` for each block.

    `header_valid=False` means the `## [` prefix is present but the header
    regex doesn't match — such stray lines advance by 1 (not by block)
    so they don't swallow subsequent valid blocks.
    """
    ranges: list[tuple[int, int, bool]] = []
    i = 0
    n = len(lines)
    while i < n:
        if not lines[i].startswith("## ["):
            i += 1
            continue
        if _HEADER_RE.match(lines[i]) is None:
            ranges.append((i, i + 1, False))
            i += 1
            continue
        j = i + 1
        while j < n and not lines[j].startswith("## ["):
            j += 1
        ranges.append((i, j, True))
        i = j
    return ranges


def parse_staging_file(
    path: Path,
) -> tuple[list[StagingEntry], list[str], list[str]]:
    """Parse a staging markdown file.

    Returns `(entries, errors, recoverable_raw_blocks)`.

    * `entries`: all successfully parsed `StagingEntry` objects in document
      order.
    * `errors`: one string per block that failed validation.
    * `recoverable_raw_blocks`: verbatim block text for blocks whose header
      regex matched (so bounds are known) but whose body failed validation.
      Caller may re-append these to `learnings_pending.md` to avoid losing
      data.

    Never raises — I/O errors are captured in `errors`.
    """
    if not path.exists():
        return [], [], []

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [], [f"read failed: {exc}"], []

    if not text.strip():
        return [], [], []

    source_file = path.name
    entries: list[StagingEntry] = []
    errors: list[str] = []
    recoverable: list[str] = []

    lines = text.splitlines()
    for start, end, header_valid in _iter_block_ranges(lines):
        if not header_valid:
            errors.append(f"bad header at line {start + 1}: {lines[start][:80]}")
            continue

        result = _parse_block(lines, start, end, source_file)
        if result.entry is not None:
            entries.append(result.entry)
        if result.error is not None:
            errors.append(result.error)
        if result.recoverable is not None:
            recoverable.append(result.recoverable)

    return entries, errors, recoverable

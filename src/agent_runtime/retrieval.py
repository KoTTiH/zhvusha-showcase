"""Read-only relevant file discovery for Agent Runtime context packs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

_PATH_RE = re.compile(
    r"(?:^|[\s`'\"(])"
    r"(?P<path>(?:src|tests|docs|tasks|scripts|alembic)/[A-Za-z0-9_./-]+"
    r"\.[A-Za-z0-9_]+)"
)
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9_]{3,}")
_DEFAULT_SUFFIXES = (
    ".py",
    ".md",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".txt",
)
_EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}
_TOKEN_ALIASES: dict[str, tuple[str, ...]] = {
    "агент": ("agent", "agent_runtime", "runtime"),
    "агенты": ("agent", "agent_runtime", "runtime"),
    "агентный": ("agent", "agent_runtime", "runtime"),
    "вложение": ("attachment", "attachments", "uploads"),
    "вложения": ("attachment", "attachments", "uploads"),
    "кодинг": ("self_coding", "chat_self_coding", "implement_spec"),
    "самокодинг": ("self_coding", "chat_self_coding", "implement_spec"),
    "пост": ("post", "source_compare", "codebase_explorer"),
    "посты": ("post", "source_compare", "codebase_explorer"),
    "браузер": ("browser", "web", "web_research"),
    "интернет": ("web", "web_research", "browser"),
    "рендер": ("render", "rendering", "renderer"),
    "статус": ("status", "events", "rendering"),
}


class MemorySourceKind(StrEnum):
    """Source domain for source-aware personal recall."""

    WORKSPACE = "workspace"
    KB = "kb"
    TELEGRAM_MCP = "telegram_mcp"
    EXTERNAL_SKILL = "external_skill"
    SELF_CODING_ARCHIVE = "self_coding_archive"
    DIALOGUE_STATE = "dialogue_state"
    LIFE_RUNTIME = "life_runtime"


@dataclass(frozen=True)
class SourceAwareMemoryRecord:
    """One recallable memory item with its source and evidence preserved."""

    source_kind: MemorySourceKind
    text: str
    evidence: tuple[str, ...]
    confidence: float = 0.6
    sensitive: bool = False
    stale: bool = False


@dataclass(frozen=True)
class SourceAwareRecallHit:
    """Ranked source-aware recall result."""

    record: SourceAwareMemoryRecord
    score: int


class SourceAwareMemoryRecallProvider(Protocol):
    """Read-only source-aware recall provider used by ContextPackBuilder."""

    def recall(
        self,
        query: str,
        *,
        allowed_sources: tuple[MemorySourceKind, ...] = (),
        include_sensitive: bool = False,
        max_results: int | None = None,
    ) -> tuple[SourceAwareRecallHit, ...]: ...

    def render_for_context(
        self,
        hits: tuple[SourceAwareRecallHit, ...],
        *,
        max_text_chars: int = 220,
    ) -> str: ...


@dataclass(frozen=True)
class SourceAwareMemoryRecall:
    """Deterministic read-only recall over source-tagged memory records."""

    records: tuple[SourceAwareMemoryRecord, ...]
    max_results: int = 8

    def recall(
        self,
        query: str,
        *,
        allowed_sources: tuple[MemorySourceKind, ...] = (),
        include_sensitive: bool = False,
        max_results: int | None = None,
    ) -> tuple[SourceAwareRecallHit, ...]:
        """Return ranked records without leaking sensitive items by default."""
        wanted = set(_query_tokens(query))
        if not wanted:
            return ()
        allowed = set(allowed_sources)
        scored: list[SourceAwareRecallHit] = []
        for record in self.records:
            if allowed and record.source_kind not in allowed:
                continue
            if record.sensitive and not include_sensitive:
                continue
            score = _score_memory_record(wanted, record)
            if score <= 0:
                continue
            scored.append(SourceAwareRecallHit(record=record, score=score))
        scored.sort(
            key=lambda hit: (
                -hit.score,
                hit.record.stale,
                -hit.record.confidence,
                hit.record.source_kind.value,
                hit.record.text,
            )
        )
        return tuple(scored[: max_results or self.max_results])

    def render_for_context(
        self,
        hits: tuple[SourceAwareRecallHit, ...],
        *,
        max_text_chars: int = 220,
    ) -> str:
        """Render a prompt-safe recall block with source/evidence labels."""
        if not hits:
            return ""
        lines = ["## Source-aware memory recall"]
        for hit in hits:
            record = hit.record
            text = " ".join(record.text.split())[:max_text_chars]
            evidence = ", ".join(record.evidence) or "none"
            lines.append(
                "- "
                f"source={record.source_kind.value} "
                f"confidence={record.confidence:.2f} "
                f"stale={str(record.stale).lower()} "
                f"evidence={evidence} :: {text}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class FileSourceAwareMemoryRecall:
    """Source-aware recall backed by memory staging files and refreshed per query."""

    staging_dir: Path
    extra_records: tuple[SourceAwareMemoryRecord, ...] = ()
    max_results: int = 8

    def recall(
        self,
        query: str,
        *,
        allowed_sources: tuple[MemorySourceKind, ...] = (),
        include_sensitive: bool = False,
        max_results: int | None = None,
    ) -> tuple[SourceAwareRecallHit, ...]:
        """Load current staging records and return ranked recall hits."""
        records = (
            *self.extra_records,
            *load_source_aware_staging_records(self.staging_dir),
        )
        return SourceAwareMemoryRecall(
            records=records,
            max_results=self.max_results,
        ).recall(
            query,
            allowed_sources=allowed_sources,
            include_sensitive=include_sensitive,
            max_results=max_results,
        )

    def render_for_context(
        self,
        hits: tuple[SourceAwareRecallHit, ...],
        *,
        max_text_chars: int = 220,
    ) -> str:
        """Render hits using the same prompt-safe format as in-memory recall."""
        return SourceAwareMemoryRecall(records=()).render_for_context(
            hits,
            max_text_chars=max_text_chars,
        )


@dataclass(frozen=True)
class RelevantFileFinder:
    """Find likely relevant repo files from user/chat context."""

    project_root: Path
    max_files: int = 12
    max_candidates: int = 2_000
    suffixes: tuple[str, ...] = _DEFAULT_SUFFIXES

    def find(
        self,
        *,
        query_parts: tuple[str, ...],
        explicit_files: tuple[str, ...] = (),
        max_files: int | None = None,
    ) -> tuple[str, ...]:
        """Return ranked repo-relative file paths for a context pack."""
        limit = max_files or self.max_files
        root = self.project_root.expanduser().resolve()
        query = "\n".join(part for part in query_parts if part.strip())
        explicit = _existing_relative_paths(
            root=root,
            paths=(*explicit_files, *_extract_path_mentions(query)),
        )
        tokens = _query_tokens(query)
        scored: list[tuple[int, str]] = []
        for rel_path in self._candidate_files(root):
            if rel_path in explicit:
                continue
            score = _score_path(rel_path, tokens)
            if score <= 0:
                continue
            scored.append((score, rel_path))

        scored.sort(key=lambda item: (-item[0], item[1]))
        ranked = [path for _, path in scored]
        return tuple(_dedupe((*explicit, *ranked))[:limit])

    def _candidate_files(self, root: Path) -> tuple[str, ...]:
        candidates: list[str] = []
        for path in root.rglob("*"):
            if len(candidates) >= self.max_candidates:
                break
            if not path.is_file() or path.suffix not in self.suffixes:
                continue
            rel_parts = path.relative_to(root).parts
            if any(part in _EXCLUDED_DIRS for part in rel_parts):
                continue
            candidates.append(path.relative_to(root).as_posix())
        return tuple(candidates)


def load_source_aware_staging_records(
    staging_dir: Path,
) -> tuple[SourceAwareMemoryRecord, ...]:
    """Load source-aware recall records from memory staging markdown files."""
    records: list[SourceAwareMemoryRecord] = []
    root = staging_dir.expanduser().resolve()
    for filename in ("learnings_immediate.md", "learnings_pending.md"):
        path = root / filename
        if not path.exists():
            continue
        records.extend(_source_aware_records_from_staging_file(path))
    return tuple(records)


def _source_aware_records_from_staging_file(
    path: Path,
) -> tuple[SourceAwareMemoryRecord, ...]:
    records: list[SourceAwareMemoryRecord] = []
    current_statement = ""
    current_confidence = 0.6
    current_entry_line = 0
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if line.startswith("## "):
            if current_statement:
                records.append(
                    _record_from_staging_statement(
                        current_statement,
                        current_confidence,
                        evidence=f"{path.name}:{current_entry_line}",
                    )
                )
            current_statement = ""
            current_confidence = 0.6
            current_entry_line = line_number
            continue
        if line.startswith("**Statement:** "):
            current_statement = line.removeprefix("**Statement:** ").strip()
            if current_entry_line == 0:
                current_entry_line = line_number
            continue
        if line.startswith("**Confidence:** "):
            current_confidence = _safe_confidence(
                line.removeprefix("**Confidence:** ").strip()
            )
    if current_statement:
        records.append(
            _record_from_staging_statement(
                current_statement,
                current_confidence,
                evidence=f"{path.name}:{current_entry_line}",
            )
        )
    return tuple(records)


def _record_from_staging_statement(
    statement: str,
    confidence: float,
    *,
    evidence: str,
) -> SourceAwareMemoryRecord:
    return SourceAwareMemoryRecord(
        source_kind=_infer_memory_source(statement),
        text=statement,
        evidence=(evidence,),
        confidence=confidence,
    )


def _extract_path_mentions(text: str) -> tuple[str, ...]:
    return tuple(match.group("path") for match in _PATH_RE.finditer(text))


def _existing_relative_paths(*, root: Path, paths: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    for raw_path in paths:
        rel_path = raw_path.strip().removeprefix("./")
        if not rel_path:
            continue
        target = (root / rel_path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            continue
        result.append(target.relative_to(root).as_posix())
    return tuple(_dedupe(tuple(result)))


def _query_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(text.lower()):
        tokens.append(match)
        tokens.extend(_TOKEN_ALIASES.get(match, ()))
    return tuple(_dedupe(tuple(tokens)))


def _score_path(rel_path: str, tokens: tuple[str, ...]) -> int:
    haystack = rel_path.lower().replace("/", " ")
    score = 0
    for token in tokens:
        normalized = token.lower()
        if normalized in haystack:
            score += 4 if "/" in normalized or "_" in normalized else 1
    filename = Path(rel_path).name.lower()
    stem = Path(rel_path).stem.lower()
    for token in tokens:
        if token == stem:
            score += 8
        elif token in filename:
            score += 3
    if rel_path.startswith("src/"):
        score += 1
    return score


def _score_memory_record(wanted: set[str], record: SourceAwareMemoryRecord) -> int:
    haystack = " ".join((record.source_kind.value, record.text, *record.evidence))
    available = set(_query_tokens(haystack))
    score = len(wanted & available) * 3
    if record.source_kind.value in wanted:
        score += 4
    if record.stale:
        score -= 1
    return max(score, 0)


def _infer_memory_source(statement: str) -> MemorySourceKind:
    normalized = statement.lower()
    if "source=external_skill" in normalized or normalized.startswith(
        "external_skill_"
    ):
        return MemorySourceKind.EXTERNAL_SKILL
    if "source=telegram_mcp" in normalized or "telegram_mcp" in normalized:
        return MemorySourceKind.TELEGRAM_MCP
    if "source=life_runtime" in normalized or "liferuntime" in normalized:
        return MemorySourceKind.LIFE_RUNTIME
    if "source=dialogue_state" in normalized or "dialogue_state" in normalized:
        return MemorySourceKind.DIALOGUE_STATE
    if "source=kb" in normalized or "source=knowledge" in normalized:
        return MemorySourceKind.KB
    if (
        "source=self_coding_archive" in normalized
        or "self-coding" in normalized
        or "self_coding" in normalized
    ):
        return MemorySourceKind.SELF_CODING_ARCHIVE
    return MemorySourceKind.WORKSPACE


def _safe_confidence(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        return 0.6
    return max(0.0, min(value, 1.0))


def _dedupe(values: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result

"""Stage LearningSignal entries to markdown files for feedback into system prompt.

Strong signals (`apply_immediately AND confidence > 0.8`) go to
`learnings_immediate.md`, which `ContextLoader.load_personality()` reads into
the system prompt of the next response in the same chat. Weak signals go to
`learnings_pending.md` for /morning manual review (Phase 4 will drain).

Sync file I/O via `Path.open("a", encoding="utf-8")` — matches codebase
convention. Small entries (<4KB) are append-atomic on Linux, safe under
concurrent background tasks.

Dedup is a cheap tail scan of the last `_DEDUP_SCAN_BYTES` bytes of the target
file — good enough to prevent back-to-back duplicates on one topic, not global.

Size cap: if the file exceeds `_MAX_STAGING_BYTES` after a write, a warning
is logged. Phase 2 does not truncate — that is Phase 4's /morning drain.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 — runtime access via patch in tests
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.memory.types import LearningSignal

logger = structlog.get_logger()


_MAX_STAGING_BYTES = 10 * 1024  # soft cap — warn, don't truncate
_DEDUP_SCAN_BYTES = 10 * 1024  # scan full file for duplicates (match soft cap)
_SEMANTIC_DEDUP_THRESHOLD = 0.85  # cosine similarity above this = duplicate

_STATEMENT_PREFIX = "**Statement:** "


def _extract_statements(target: Path) -> list[str]:
    """Extract all **Statement:** values from a staging markdown file.

    Used by semantic dedup to compare new signals against existing ones.
    Returns empty list if file doesn't exist or can't be read.
    """
    if not target.exists():
        return []
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return []
    statements: list[str] = []
    for line in text.splitlines():
        if line.startswith(_STATEMENT_PREFIX):
            statements.append(line[len(_STATEMENT_PREFIX) :].strip())
    return statements


class StagingWriter:
    """Append LearningSignal entries to staging markdown files.

    Routes strong signals to `learnings_immediate.md` (read into next prompt)
    and weak/deferred signals to `learnings_pending.md` (review later).
    All writes are best-effort: OSErrors are logged and swallowed so that
    background enrichment never crashes the event loop.
    """

    def __init__(self, staging_dir: Path) -> None:
        self._dir = staging_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._immediate = self._dir / "learnings_immediate.md"
        self._pending = self._dir / "learnings_pending.md"

    def append(
        self,
        signal: LearningSignal,
        episode_id: int,
        chat_id: int | None = None,
    ) -> Path | None:
        """Write a single LearningSignal entry to the appropriate file.

        Returns the target path on successful write, or `None` if the write
        was skipped (dedup) or failed (OSError — logged).
        """
        target = (
            self._immediate
            if (signal.apply_immediately and signal.confidence > 0.8)
            else self._pending
        )

        if self._is_duplicate(target, signal.statement):
            logger.info(
                "staging_dedup_skipped",
                target=target.name,
                statement=signal.statement[:60],
                method="substring",
            )
            return None

        if self._is_semantic_duplicate(target, signal.statement):
            logger.info(
                "staging_dedup_skipped",
                target=target.name,
                statement=signal.statement[:60],
                method="semantic",
            )
            return None

        entry = self._format_entry(signal, episode_id, chat_id)

        try:
            with target.open("a", encoding="utf-8") as f:
                f.write(entry)
        except OSError as exc:
            logger.warning(
                "staging_write_failed",
                target=str(target),
                error=str(exc),
            )
            return None

        try:
            size = target.stat().st_size
            if size > _MAX_STAGING_BYTES:
                logger.warning(
                    "staging_file_oversize",
                    target=target.name,
                    bytes=size,
                    cap=_MAX_STAGING_BYTES,
                )
        except OSError:
            pass

        logger.info(
            "learning_signal_staged",
            target=target.name,
            signal_type=signal.type,
            scope=signal.scope,
            confidence=signal.confidence,
            episode_id=episode_id,
        )

        return target

    @staticmethod
    def _is_semantic_duplicate(target: Path, statement: str) -> bool:
        """Check if statement is semantically similar to existing ones via embeddings.

        Uses embed_batch for all texts in a single call (O(1) model invocations
        instead of O(N)). Caller should run this off the event loop via
        asyncio.to_thread since embedding is CPU-bound.
        """
        existing = _extract_statements(target)
        if not existing:
            return False
        try:
            from src.embeddings import EmbeddingService

            all_embeddings = EmbeddingService.embed_batch([statement, *existing])
            new_emb = all_embeddings[0]
            for i, existing_stmt in enumerate(existing):
                sim = EmbeddingService.cosine_similarity(new_emb, all_embeddings[i + 1])
                if sim >= _SEMANTIC_DEDUP_THRESHOLD:
                    logger.debug(
                        "semantic_dedup_match",
                        new=statement[:60],
                        existing=existing_stmt[:60],
                        similarity=round(sim, 3),
                    )
                    return True
        except Exception:
            logger.warning("semantic_dedup_failed", exc_info=True)
        return False

    @staticmethod
    def _is_duplicate(target: Path, statement: str) -> bool:
        """Cheap substring check against the last _DEDUP_SCAN_BYTES of target."""
        if not target.exists():
            return False
        try:
            size = target.stat().st_size
        except OSError:
            return False
        start = max(0, size - _DEDUP_SCAN_BYTES)
        try:
            with target.open("rb") as f:
                f.seek(start)
                tail = f.read().decode("utf-8", errors="ignore")
        except OSError:
            return False
        return statement in tail

    @staticmethod
    def _format_entry(
        signal: LearningSignal,
        episode_id: int,
        chat_id: int | None,
    ) -> str:
        timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M")
        lines = [
            "",  # blank line separator from previous entry
            f"## [{signal.type}] {signal.scope} — {timestamp}",
            f"**Statement:** {signal.statement}",
            f"**Confidence:** {signal.confidence}",
        ]
        if chat_id is not None:
            lines.append(f"**Chat:** {chat_id}")
        lines.append(f"**Trigger episode:** {episode_id}")
        if signal.original_claim is not None:
            lines.append(f"**Original claim:** {signal.original_claim}")
        lines.append("")  # trailing newline
        return "\n".join(lines)

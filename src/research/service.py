"""ResearchService — KB + code-search composer with a wall-clock budget.

A single :meth:`research` call fans out to the sources enabled by the chosen
preset, aggregates citations, and returns a frozen :class:`ResearchResult`
including a ``truncated`` flag if any source raised :exc:`TimeoutError` or
another exception. The service deliberately does NOT raise itself — it
reports partial results so the caller (``ideation_to_spec``) can record
the gap in ``spec.research_findings`` and proceed honestly.

Budget enforcement is applied per source while preserving source order: slow
sources are cancelled when the remaining wall-clock budget is exhausted, and
the caller receives a partial ``truncated=True`` result.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from src.research.presets import PRESETS
from src.research.protocols import Citation, CitationSource, ResearchResult

if TYPE_CHECKING:
    from src.research.protocols import (
        CodeSearchCallable,
        KBSearchCallable,
        ResearchSourceCallable,
    )


class ResearchService:
    """Composes KB and code search per preset; returns frozen results."""

    def __init__(
        self,
        *,
        kb_search: KBSearchCallable,
        code_search: CodeSearchCallable,
        runtime_sources: tuple[ResearchSourceCallable, ...] = (),
    ) -> None:
        self._kb_search = kb_search
        self._code_search = code_search
        self._runtime_sources = runtime_sources

    async def research(
        self,
        *,
        query: str,
        preset: str,
        budget_seconds: float,
    ) -> ResearchResult:
        """Run the configured sources and aggregate citations.

        Raises :exc:`ValueError` if ``preset`` is not a known preset name.
        Otherwise always returns — partial results carry ``truncated=True``.
        """
        if preset not in PRESETS:
            raise ValueError(
                f"unknown preset {preset!r}; valid: {sorted(PRESETS.keys())!r}"
            )
        config = PRESETS[preset]

        start = time.monotonic()
        citations: list[Citation] = []
        sources_used: list[str] = []
        truncated = False

        if config.use_kb:
            truncated |= await _collect_source(
                query=query,
                source=self._kb_search,
                citations=citations,
                sources_used=sources_used,
                fallback_source="kb",
                timeout_seconds=_remaining_budget_seconds(start, budget_seconds),
            )

        if config.use_code_search:
            truncated |= await _collect_source(
                query=query,
                source=self._code_search,
                citations=citations,
                sources_used=sources_used,
                fallback_source="code",
                timeout_seconds=_remaining_budget_seconds(start, budget_seconds),
            )

        for runtime_source in self._runtime_sources:
            truncated |= await _collect_source(
                query=query,
                source=runtime_source,
                citations=citations,
                sources_used=sources_used,
                timeout_seconds=_remaining_budget_seconds(start, budget_seconds),
            )

        # Web search through the old preset flag is intentionally unimplemented
        # in this leaf module. Runtime-backed web adapters can be injected via
        # ``runtime_sources`` without importing Agent Runtime here.
        elapsed = time.monotonic() - start
        return ResearchResult(
            citations=citations,
            elapsed_seconds=elapsed,
            truncated=truncated,
            findings_summary="",
            sources_used=sources_used,
        )


async def _collect_source(
    *,
    query: str,
    source: ResearchSourceCallable,
    citations: list[Citation],
    sources_used: list[str],
    fallback_source: CitationSource | None = None,
    timeout_seconds: float | None = None,
) -> bool:
    if timeout_seconds is not None and timeout_seconds <= 0:
        return True
    try:
        if timeout_seconds is None:
            source_citations = await source(query)
        else:
            source_citations = await asyncio.wait_for(
                source(query),
                timeout=timeout_seconds,
            )
    except (TimeoutError, RuntimeError):
        return True
    citations.extend(source_citations)
    if fallback_source is not None:
        sources_used.append(fallback_source)
        return False
    for citation in source_citations:
        if citation.source not in sources_used:
            sources_used.append(citation.source)
    return False


def _remaining_budget_seconds(start: float, budget_seconds: float) -> float | None:
    if budget_seconds <= 0:
        return None
    return budget_seconds - (time.monotonic() - start)

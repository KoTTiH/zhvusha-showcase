"""Public contract for the research leaf module.

Frozen dataclasses (no Pydantic — keeps the leaf dependency-light) and two
callable Protocols injected into :class:`ResearchService`. The callables are
the only seam through which the leaf reaches into capability modules
(KB / code-grep) without importing them directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

CitationSource = Literal[
    "kb",
    "code",
    "web",
    "telegram_mcp",
    "workspace",
    "news",
    "unknown",
]


@dataclass(frozen=True)
class Citation:
    """A single source pointer surfaced during research.

    ``source`` distinguishes the origin in the KB→code→web/runtime hierarchy
    (KB #72). ``ref`` is the canonical identifier (``kb_71``,
    ``src/skills/base.py:155``, full URL); ``excerpt`` is a short quote
    that justifies the citation's relevance to the research query.
    """

    source: CitationSource
    ref: str
    excerpt: str


@dataclass(frozen=True)
class ResearchResult:
    """Aggregate of a single :meth:`ResearchService.research` call.

    ``truncated`` is set when at least one source raised :exc:`TimeoutError`
    or another budgeted-failure exception. Callers are expected to react —
    e.g. ``ideation_to_spec`` records ``truncated=True`` in
    ``spec.research_findings`` so the spec carries forward an honest note
    about the gap.
    """

    citations: list[Citation]
    elapsed_seconds: float
    truncated: bool
    findings_summary: str = ""
    sources_used: list[str] = field(default_factory=list)


class KBSearchCallable(Protocol):
    """Callable that runs a knowledge-base lookup and returns citations."""

    async def __call__(self, query: str) -> list[Citation]: ...


class CodeSearchCallable(Protocol):
    """Callable that runs a repository-wide code search (e.g. ripgrep)."""

    async def __call__(self, query: str) -> list[Citation]: ...


class ResearchSourceCallable(Protocol):
    """Additional injected research source with the same citation contract."""

    async def __call__(self, query: str) -> list[Citation]: ...

"""Public contract for Knowledge Base capability module.

All other modules import from this file (or from the parent package
``src.knowledge``, which re-exports everything here). The concrete
implementation ``KnowledgeStore`` is hidden behind this protocol — but
the concrete class remains accessible via
``from src.knowledge import KnowledgeStore`` for clients that need
methods NOT in the protocol (e.g. ORM-leaking queries used by
``daemon.sleep_agent``).

Side effects of implementations:
    * reads/writes PostgreSQL via SQLAlchemy async
    * generates embeddings via sentence-transformers (lazy load)
    * writes to pgvector indices
    * interacts with hybrid BM25 + semantic search

Errors (forward-looking — current ``KnowledgeStore`` uses ``None`` /
``False`` returns instead of raising; these are reserved for future
implementations and for tests that want to assert contract-level error
semantics):
    * ``KnowledgeBaseError`` — base for all module-specific exceptions
    * ``EntryNotFoundError`` — requested entry id does not exist
    * ``CategoryNotFoundError`` — requested category path does not exist
    * ``StagingError`` — staging operation failed

Notes on design choices:

1. **Pydantic BaseModel, not frozen dataclass.** KB #81 template 4 uses
   ``@dataclass(frozen=True)``. We deviate because existing tests and
   MCP server handlers depend on Pydantic-specific APIs
   (``model_dump``). Changing to dataclasses would ripple through
   ``tests/test_knowledge_store.py`` and ``tests/test_mcp_server.py``
   and violate the phase 4 invariant of unchanged behaviour.

2. **Protocol covers only "clean" methods.** ORM-leaking methods on
   ``KnowledgeStore`` (``get_entry``, ``get_untagged``,
   ``get_pending_staged``, ``update_entry``, ``recalculate_entry_counts``
   at the concrete class level, plus the ``session()`` escape hatch)
   return SQLAlchemy models and remain on the concrete class only.
   Clients that need them import the concrete class directly.

See also: KB records #69 (v4 architecture), #70 (test strategy),
#81 (code templates), #82 (enforcement), #97 (lessons learned).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

# --------------------------------------------------------------------- #
#  Data types (progressive disclosure + categories)                      #
# --------------------------------------------------------------------- #


class SearchResult(BaseModel):
    """Level 1 search result with RRF score."""

    id: int
    title: str
    tags: list[str]
    category_name_ru: str | None = None
    token_count: int | None = None
    rrf_score: float


class IndexEntry(BaseModel):
    """Level 1: ~20 tokens per entry."""

    id: int
    title: str
    tags: list[str]
    token_count: int | None = None


class SummaryEntry(BaseModel):
    """Level 2: ~100 tokens per entry."""

    id: int
    title: str
    summary: str | None = None


class FullEntry(BaseModel):
    """Level 3: full text of a single entry."""

    id: int
    title: str
    content: str
    summary: str | None = None
    tags: list[str] = []
    source: str | None = None
    source_url: str | None = None
    content_type: str = "fact"
    status: str = "raw"
    category_name_ru: str | None = None
    # Use Any here to keep the shape matching the current KnowledgeStore
    # return type (which passes through SQLAlchemy's datetime values).
    created_at: Any = None
    updated_at: Any = None


class CategoryInfo(BaseModel):
    """Category tree node."""

    id: int
    name: str
    name_ru: str
    path: str
    entry_count: int = 0
    summary: str | None = None


# --------------------------------------------------------------------- #
#  Errors                                                                #
# --------------------------------------------------------------------- #


class KnowledgeBaseError(Exception):
    """Base exception for Knowledge Base capability module."""


class EntryNotFoundError(KnowledgeBaseError):
    """Requested entry id does not exist."""


class CategoryNotFoundError(KnowledgeBaseError):
    """Requested category path does not exist."""


class StagingError(KnowledgeBaseError):
    """Staging operation failed."""


# --------------------------------------------------------------------- #
#  Protocol                                                              #
# --------------------------------------------------------------------- #


@runtime_checkable
class KnowledgeStoreProtocol(Protocol):
    """Public contract for the Knowledge Base capability module.

    Covers the clean, Pydantic-typed portion of the ``KnowledgeStore``
    API. ORM-leaking methods (``get_entry``, ``get_untagged``, …) and the
    ``session()`` escape hatch are NOT part of this contract — they
    remain accessible on the concrete ``KnowledgeStore`` class for
    clients that need them.
    """

    async def hybrid_search(
        self,
        query: str,
        *,
        category: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Hybrid BM25 + semantic + RRF search over knowledge entries."""
        ...

    async def get_index(self, entry_ids: list[int]) -> list[IndexEntry]:
        """Level 1: index entries (~20 tokens each)."""
        ...

    async def get_summaries(self, entry_ids: list[int]) -> list[SummaryEntry]:
        """Level 2: summaries (~100 tokens each)."""
        ...

    async def get_full_content(self, entry_id: int) -> FullEntry | None:
        """Level 3: full text of a single entry. Returns ``None`` if missing."""
        ...

    async def browse_categories(
        self,
        parent_path: str | None = None,
    ) -> list[CategoryInfo]:
        """List categories. Root if ``parent_path is None``, children otherwise."""
        ...

    async def add_entry(
        self,
        title: str,
        content: str,
        *,
        category_path: str | None = None,
        tags: list[str] | None = None,
        source: str = "manual",
        source_url: str | None = None,
        content_type: str = "fact",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a new knowledge entry directly. Returns its ID.

        For manual tools (MCP, bot). Autonomous processes must use
        ``propose_change`` instead (staging area, per AGENTS.md
        principle 2).
        """
        ...

    async def archive_entry(self, entry_id: int) -> bool:
        """Soft-delete: set entry status to ``archived``. Returns ``True`` on change."""
        ...

    async def get_or_create_category(self, path: str, name_ru: str) -> int:
        """Get category by path or atomically create it. Returns category ID."""
        ...

    async def propose_change(
        self,
        operation: str,
        target_entry_id: int | None,
        proposed_changes: dict[str, Any],
        reason: str,
        proposed_by: str,
    ) -> int:
        """Add a staging proposal for review. Returns staging item ID.

        For autonomous processes (sleep agent, daemon). Manual tools use
        ``add_entry`` directly.
        """
        ...

    async def review_staged(self, staging_id: int, approve: bool) -> bool:
        """Approve or reject a staging proposal. Approved items are applied."""
        ...

    async def add_relation(
        self,
        source_id: int,
        target_id: int,
        relation_type: str,
    ) -> int:
        """Add a directed relation between two entries."""
        ...

    async def recalculate_entry_counts(self) -> None:
        """Recalculate all category ``entry_count`` values from actual data."""
        ...

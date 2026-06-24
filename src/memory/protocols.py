"""Public contract for the Memory capability module (v4).

All other modules import types, errors, protocols and free functions from
THIS FILE ONLY. Concrete implementations (``EpisodicMemory``, ``PeopleManager``,
``DesireProcessor``) live in sibling modules and are exported from
``src.memory`` package ``__init__`` together with the protocol surface.
``import-linter`` enforces this isolation via the ``memory_isolation`` rule.

Side effects the module performs
--------------------------------
- Reads/writes PostgreSQL episodes table (via SQLAlchemy async session).
- Reads/writes pgvector embeddings (Vector(384)).
- Generates embeddings via :class:`src.embeddings.EmbeddingService`.
- Reads/writes markdown profile files under workspace/memory/people/.
- Reads/writes desire outbox and wishlist files in workspace.

Errors
------
- ``MemoryModuleError`` — base exception for Memory-module failures.
- ``EpisodeNotFoundError`` — requested episode does not exist (reserved).
- ``PersonNotFoundError`` — requested person profile does not exist (reserved).

The reserved error classes are defined so future phases (Memory writers
with stricter contracts, Tier 2 safety checks) can raise them without
changing the public contract.

Design note — domain Episode vs ORM EpisodeORM
----------------------------------------------
``src.memory.protocols.Episode`` is a frozen dataclass (domain object) used
by all clients of ``EpisodicMemoryProtocol``. ``src.memory.database.EpisodeORM``
(re-exported via the backward-compat ``Episode`` alias) is the SQLAlchemy
ORM model used internally by ``EpisodicMemory`` and by a handful of
legitimate direct-DB clients (``mcp_server.dashboard_api`` SQL queries,
contract tests, smoke scripts, alembic env). ``EpisodicMemory`` converts
ORM → domain via a private ``_orm_to_domain`` helper before returning
values from protocol methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, runtime_checkable

from src.memory.domain import detect_domain  # re-exported as part of public API
from src.memory.types import EnrichmentResult, LearningSignal

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

__all__ = [
    "ChatType",
    "ConsolidationAction",
    "ConsolidationProtocol",
    "ConsolidationResult",
    "DesireProcessorProtocol",
    "Domain",
    "EnrichmentProtocol",
    "EnrichmentResult",
    "Episode",
    "EpisodeNotFoundError",
    "EpisodicMemoryProtocol",
    "LearningSignal",
    "MemoryModuleError",
    "PeopleManagerProtocol",
    "PersonNotFoundError",
    "PersonProfile",
    "Role",
    "StagingWriterProtocol",
    "Valence",
    "detect_domain",
]


# === Type aliases ===

Domain = Literal["kwork", "chat", "content", "outreach"]
"""Interaction domain classified by :func:`detect_domain`."""

ChatType = Literal["personal", "assistant", "social"]
"""Chat channel type. Matches ``src.core.mode_config.Mode`` deliberately."""

Role = Literal["user", "assistant", "system"]
"""Speaker role in an episode."""

Valence = Literal["positive", "negative", "neutral"]
"""Somatic marker valence on an episode."""


# === Domain types ===


@dataclass(frozen=True)
class Episode:
    """Episode as a domain object — frozen dataclass independent of ORM.

    Returned by :class:`EpisodicMemoryProtocol` read methods
    (``retrieve``, ``retrieve_by_somatic_marker``, ``complete_pattern``,
    ``check_pattern_separation``, ``get_unconsolidated``).

    Fields mirror ``src.memory.database.EpisodeORM`` (SQLAlchemy ORM) 1:1.
    ``EpisodicMemory._orm_to_domain`` converts between the two. Field parity
    with the ORM is verified by a contract test (`test_memory_contract`) —
    adding or removing a field here requires updating the ORM too (and
    vice versa).

    Notes on types:
    - ``embedding`` — pgvector ``Vector(384)`` in the ORM, exposed here as
      ``list[float] | None`` because that's the Python-side representation.
    - ``metadata_json`` — kept as a raw JSON string (not parsed) to avoid
      lossy round-tripping; callers parse on demand.
    """

    id: int
    timestamp: datetime
    user_id: int
    chat_type: str
    role: str
    content: str
    summary: str | None = None
    embedding: list[float] | None = None
    importance: float = 0.5
    valence: str = "neutral"
    confidence: float = 0.5
    consolidated: bool = False
    consolidation_result: str | None = None
    access_count: int = 0
    last_accessed: datetime | None = None
    enrichment_status: str = "pending"
    intent: str | None = None
    emotion: str | None = None
    embedding_version: int = 1
    source: str = "chat"
    metadata_json: str | None = None


class PersonProfile(TypedDict):
    """Shape of the profile dict returned by :class:`PeopleManagerProtocol`.

    Matches the literal dict that ``PeopleManager.get_or_create_profile``
    constructs in ``src/memory/people.py``. Using ``TypedDict`` here gives
    us static typing without rewriting the runtime representation (which
    is a plain ``dict[str, Any]`` serialised to markdown frontmatter).
    """

    user_id: int
    username: str
    first_name: str
    significance: str
    interaction_count: int
    first_seen: str  # ISO-8601 timestamp
    last_seen: str  # ISO-8601 timestamp


# === Errors ===


class MemoryModuleError(Exception):
    """Base exception raised by Memory-module implementations on failure."""


class EpisodeNotFoundError(MemoryModuleError):
    """Requested episode does not exist. Reserved for future strict reads."""


class PersonNotFoundError(MemoryModuleError):
    """Requested person profile does not exist. Reserved for future use."""


# === Protocols ===


@runtime_checkable
class EpisodicMemoryProtocol(Protocol):
    """Public contract for the episodic memory store.

    All methods are async. The concrete implementation is
    :class:`src.memory.episodic.EpisodicMemory`, which owns a SQLAlchemy
    async session maker and converts ORM rows to :class:`Episode` before
    returning them.

    Implements: ACT-R style hybrid retrieval, somatic markers, pattern
    completion/separation, consolidation hooks, and Sonnet-enrichment
    updates.
    """

    async def record(
        self,
        content: str,
        user_id: int,
        chat_type: str,
        role: str,
        importance: float = 0.5,
        valence: str = "neutral",
        confidence: float = 0.5,
        source: str = "chat",
        metadata: dict[str, object] | None = None,
        person_name: str = "unknown",
        significance: str = "stranger",
        domain: str = "chat",
    ) -> int:
        """Record a new episode. Returns the new episode id.

        Returns ``-1`` when the episode is silently dropped by
        social-mode rate limiting.
        """
        ...

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        chat_type: str = "personal",
        source_filter: list[str] | None = None,
    ) -> list[Episode]:
        """Hybrid search: pgvector cosine pre-filter + ACT-R scoring."""
        ...

    async def retrieve_by_somatic_marker(
        self,
        query: str,
        limit: int = 3,
    ) -> list[tuple[Episode, float]]:
        """System 1 fast path: similar episodes with their valence score."""
        ...

    async def complete_pattern(
        self,
        partial_cue: str,
        threshold: float = 0.6,
    ) -> Episode | None:
        """Pattern completion: restore full context from a partial cue."""
        ...

    async def check_pattern_separation(
        self,
        embedding: list[float],
        threshold: float = 0.92,
    ) -> list[Episode]:
        """Find episodes very similar to ``embedding`` (for separation check)."""
        ...

    async def get_unconsolidated(
        self,
        since: datetime | None = None,
        limit: int = 100,
        sources: list[str] | None = None,
    ) -> list[Episode]:
        """Get episodes not yet processed by the morning session."""
        ...

    async def mark_consolidated(
        self,
        episode_ids: list[int],
        result: str = "",
    ) -> None:
        """Mark episodes as consolidated after morning session."""
        ...

    async def update_importance(
        self,
        episode_id: int,
        new_importance: float,
        reconsolidation_window_hours: int = 6,
    ) -> None:
        """Update importance within the reconsolidation window."""
        ...

    async def update_valence(
        self,
        episode_id: int,
        new_valence: str,
        new_confidence: float,
    ) -> None:
        """Update somatic marker based on outcome feedback."""
        ...

    async def update_enrichment(
        self,
        episode_id: int,
        result: EnrichmentResult,
    ) -> None:
        """Overwrite Sonnet-enriched fields on an existing episode.

        ``EnrichmentResult`` lives in :mod:`src.memory.types` (phase 5C).
        """
        ...


@runtime_checkable
class PeopleManagerProtocol(Protocol):
    """Public contract for the file-backed people profile store.

    All methods are synchronous — ``PeopleManager`` does blocking I/O on
    small markdown files. Async is unnecessary here and would complicate
    clients.
    """

    def get_or_create_profile(
        self,
        user_id: int,
        username: str = "",
        first_name: str = "",
    ) -> PersonProfile:
        """Get existing profile or create a new one."""
        ...

    def update_profile(self, user_id: int, updates: dict[str, Any]) -> None:
        """Update fields on an existing profile (no-op if missing)."""
        ...

    def record_interaction(self, user_id: int) -> bool:
        """Increment counter, update last_seen. Returns True on promotion."""
        ...

    def get_interaction_count(self, user_id: int) -> int:
        """Return interaction count or 0 if profile doesn't exist."""
        ...

    def get_significance_level(self, user_id: int) -> str:
        """Return ``"stranger"`` / ``"known"`` / ... or fallback ``"stranger"``."""
        ...

    def get_profile_for_context(self, user_id: int, mode: ChatType) -> str:
        """Return profile text appropriate for the chat mode.

        Personal → full profile markdown.
        Assistant → profile of this person only.
        Social → empty string (no personal data leak).
        """
        ...


@runtime_checkable
class DesireProcessorProtocol(Protocol):
    """Public contract for the morning-session desire analytics processor.

    Currently exposes a single method; internal parsing helpers and
    ``DreamEntry`` are not part of the public contract.
    """

    async def run_all(self) -> str:
        """Run all four desire-analytics phases. Returns inbox summary."""
        ...


# === Consolidation & Enrichment surface (phase 5C) ===


@dataclass(frozen=True)
class ConsolidationAction:
    """Single file operation during consolidation.

    Emitted by consolidation phases and applied atomically in the commit
    step. Callers treat instances as immutable records; the consolidation
    engine coalesces a list of actions before applying them.
    """

    action: Literal["create", "update", "delete"]
    file_path: str
    content: str
    reason: str


@dataclass
class ConsolidationResult:
    """Full result of a consolidation run.

    Not frozen: the engine populates fields across phases (counters are
    written by different phases, ``files_created``/``files_updated``
    populate during commit, etc.). Treat as a write-once-per-field
    aggregate.
    """

    files_created: list[str] = field(default_factory=list)
    files_updated: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    contradictions_found: list[str] = field(default_factory=list)
    reinforcements: list[str] = field(default_factory=list)
    emotional_summary: str = ""
    summary: str = ""
    episodes_consolidated: int = 0
    # Phase 4 — staging review counters
    staging_promoted: int = 0
    staging_merged: int = 0
    staging_discarded: int = 0
    staging_held: int = 0
    staging_stale_discarded: int = 0
    staging_parse_failed: int = 0
    staging_review_failed: bool = False


@runtime_checkable
class ConsolidationProtocol(Protocol):
    """Public contract for the morning-session consolidation engine.

    Concrete implementation: :class:`src.memory.consolidation.ConsolidationEngine`.

    Design note — consolidation is NOT structured via
    :class:`src.core.pipeline.PipelineRunner`. The consolidation flow
    contains conditional phases (Phase 3 runs only when there are
    episodes; Phase 4 always runs), cross-phase state (the ``actions``
    list is accumulated, deduplicated, and applied after all phases),
    and an atomic try/except that cleans up the ``.pending/`` directory
    on failure. Wrapping these into linear stages would inflate the
    context object and break the ``.pending/`` rollback semantics
    without gaining testability the existing phase methods already have.
    """

    async def run_consolidation(
        self,
        admin_user_id: int,
    ) -> ConsolidationResult:
        """Run all consolidation phases (orient → gather → consolidate
        → review_staging → prune_index), commit to personality/ atomically."""
        ...

    async def handle_explicit_rejection(
        self,
        rejected_conclusion: str,
        nikita_correction: str,
    ) -> ConsolidationAction | None:
        """React to an explicit user correction of a previous conclusion.

        Returns the action to apply (or ``None`` if the correction does
        not resolve to a concrete file operation).
        """
        ...


@runtime_checkable
class EnrichmentProtocol(Protocol):
    """Public contract for asynchronous episode metadata enrichment.

    Concrete implementation: :class:`src.memory.sonnet_enricher.SonnetEnricher`,
    which internally delegates to :func:`src.memory.pipelines.enrichment.build_enrichment_pipeline`.

    Returns ``None`` on any failure (LLM error, invalid JSON, schema
    mismatch). Callers treat ``None`` as "skip this episode, leave
    placeholder values".
    """

    async def enrich(
        self,
        message: str,
        recent_context: str = "",
        prev_bot_response: str = "",
    ) -> EnrichmentResult | None:
        """Extract structured metadata from a user message via LLM."""
        ...


@runtime_checkable
class StagingWriterProtocol(Protocol):
    """Public contract for writing learning signals to the staging area.

    Concrete implementation: :class:`src.memory.learning_staging.StagingWriter`.

    Synchronous (writes are fast markdown appends on small files).
    Callers that need async should wrap with ``asyncio.to_thread``.
    """

    def append(
        self,
        signal: LearningSignal,
        episode_id: int,
        chat_id: int | None = None,
    ) -> Path | None:
        """Append a learning signal to the appropriate staging file.

        Returns the target path on success, or ``None`` if the write was
        skipped (e.g. dedup against an existing entry).
        """
        ...

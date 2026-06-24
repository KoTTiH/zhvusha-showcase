"""Public contract for the Personality capability module (v4).

All other modules import types, errors, and protocols from THIS FILE ONLY.
Concrete implementations (``HomeostasisCheck``, ``PersonalityEvolution``,
``AffectiveStateManager``) live in sibling modules and are exported from
``src.personality`` package ``__init__`` together with the protocol surface.
``import-linter`` enforces this isolation via the ``personality_isolation``
rule.

Side effects the module performs
--------------------------------
- Reads/writes personality markdown files under workspace/personality/.
- Generates embeddings via :class:`src.embeddings.EmbeddingService`.
- Reads :class:`src.memory.protocols.Episode` domain objects (TYPE_CHECKING).
- Maintains in-memory affective state (process-local, no persistence).

Errors
------
- ``PersonalityError`` — base exception for Personality-module failures.
  Concrete subclasses are reserved for future phases that introduce
  stricter contracts.

Design note — ``AffectiveSnapshot`` vs internal ``AffectiveState``
-----------------------------------------------------------------
``AffectiveStateManager`` keeps a mutable ``AffectiveState`` dataclass
internally (``self._state``) and exposes a frozen ``AffectiveSnapshot``
copy via ``get_state()``. Callers cannot accidentally mutate manager
state through the returned snapshot. Tests that intentionally need to
poke internal state continue to use ``mgr._state.X = ...`` (private
access, explicit opt-in).

Design note — ``HomeostasisCorrection``
---------------------------------------
Frozen dataclass. The mutable variant in earlier iterations was never
mutated after construction (verified by codebase grep); freezing it
makes the contract explicit without changing observable behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from src.personality.emotion_atlas import EmotionConcept

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from src.memory import EnrichmentResult, Episode

__all__ = [
    "AffectiveSnapshot",
    "AffectiveStateProtocol",
    "EmotionConcept",
    "HomeostasisCorrection",
    "HomeostasisProtocol",
    "PersonalityError",
    "PersonalityEvolutionProtocol",
]


# === Domain types ===


@dataclass(frozen=True)
class HomeostasisCorrection:
    """Single drift correction emitted by :class:`HomeostasisProtocol`.

    Returned from :meth:`HomeostasisProtocol.check`. The morning
    consolidation engine reads these to nudge personality back toward
    baseline when a gene drifts too far.

    Fields mirror the original mutable dataclass in
    ``src/personality/homeostasis.py`` 1:1.
    """

    gene: str
    direction: str  # "too_high" | "too_low"
    evidence: str
    suggestion: str


@dataclass(frozen=True)
class AffectiveSnapshot:
    """Read-only snapshot of Zhvusha's current affective state.

    Returned by :meth:`AffectiveStateProtocol.get_state`. Frozen so
    callers cannot mutate manager state through the returned object.

    Fields mirror :class:`src.personality.affective_state.AffectiveState`
    1:1 (the mutable internal state held by ``AffectiveStateManager``).
    """

    self_emotion: str
    self_valence: float
    self_arousal: float
    user_emotion: str
    user_valence: float
    user_arousal: float
    regulation_active: bool
    regulation_target: str
    turns_since_update: int
    last_updated: datetime


# === Errors ===


class PersonalityError(Exception):
    """Base exception raised by Personality-module implementations on failure."""


# === Protocols ===


@runtime_checkable
class HomeostasisProtocol(Protocol):
    """Public contract for the homeostasis drift checker.

    Concrete implementation: :class:`src.personality.homeostasis.HomeostasisCheck`.
    Invoked during morning consolidation after Phase 3 (consolidate)
    to compare recent behavior against the genes.md baseline.
    """

    async def check(
        self,
        genes_path: Path,
        recent_episodes: list[Episode],
        admin_user_id: int,
    ) -> list[HomeostasisCorrection]:
        """Check for extreme drift against genes.md baseline.

        Returns an empty list when no recent episodes are available
        or when no gene has drifted past its threshold.
        """
        ...


@runtime_checkable
class PersonalityEvolutionProtocol(Protocol):
    """Public contract for the growing personality tree.

    Concrete implementation: :class:`src.personality.evolution.PersonalityEvolution`.
    Decides when to create new files, when to update existing ones,
    when to snapshot and rewrite ``core.md``, and produces the compact
    personality summary injected into LLM system prompts.
    """

    async def should_create_new_file(
        self,
        topic: str,
        mention_count: int,
        max_importance: float,
    ) -> bool:
        """Check if a new personality file should be created."""
        ...

    async def get_target_file(self, topic: str) -> Path | None:
        """Find the existing file most semantically similar to ``topic``."""
        ...

    async def evolve_core(
        self,
        new_insights: list[str],
        max_lines: int = 30,
    ) -> bool:
        """Update ``core.md`` with new insights (skipping duplicates)."""
        ...

    async def evolve_genes(
        self,
        experience_updates: list[dict[str, str]],
    ) -> None:
        """Add dated experience annotations to ``genes.md`` (no value changes)."""
        ...

    async def suggest_new_dimension(
        self,
        topic: str,
        episodes: list[Episode],
    ) -> Path | None:
        """Suggest where to create a new personality file."""
        ...

    def get_personality_tree_summary(self) -> str:
        """Return compressed personality (MEMORY.md + core.md + genes.md) for LLM context."""
        ...


@runtime_checkable
class AffectiveStateProtocol(Protocol):
    """Public contract for the in-memory affective state tracker.

    Concrete implementation: :class:`src.personality.affective_state.AffectiveStateManager`.
    Maintains a *locally scoped* estimate of Zhvusha's emotional state,
    decaying toward baseline (curiosity) between interactions and
    counter-regulating against intense user arousal. Process-local —
    does not persist across restarts.
    """

    def get_state(self) -> AffectiveSnapshot:
        """Return frozen snapshot of current state."""
        ...

    def update_from_enrichment(self, result: EnrichmentResult) -> None:
        """Update state from a new enrichment result, then counter-regulate."""
        ...

    def decay_if_stale(self) -> None:
        """Apply exponential decay toward baseline (half-life 5 turns)."""
        ...

    def get_prompt_context(self) -> str:
        """Return compact emotional context for the system prompt.

        Returns empty string at baseline (no noise).
        """
        ...

    def get_snapshot_line(self) -> str:
        """One-line summary for diary/emotional log."""
        ...

"""Models for converting ranked topics into self-coding candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from src.skills.spec_command.parser import SourceProvenance


@dataclass(frozen=True)
class TopicRecord:
    """A topic cluster as seen by the spec/proposal layer."""

    cluster_key: str
    title: str
    summary: str
    top_terms: tuple[str, ...]
    final_priority: float
    pillar_alignment: dict[str, float] = field(default_factory=dict)
    source_provenance: tuple[SourceProvenance, ...] = ()


@dataclass(frozen=True)
class TopicCandidate:
    """Actionable output for Никита to approve, defer, or refine."""

    kind: Literal["spec", "proposal", "post", "report"]
    tier: int
    slug: str
    what: str
    why_now: str
    acceptance: tuple[str, ...]
    preserve_behavior: tuple[str, ...]
    allowed_simplifications: tuple[str, ...]
    files_likely_touched: tuple[str, ...]
    risk: str
    rationale: str
    pillar_attribution: dict[str, float]
    source_provenance: tuple[SourceProvenance, ...]


class TopicProvider(Protocol):
    """Read-only source of topic records."""

    async def get_topic(self, key: str | None = None) -> TopicRecord | None: ...


class ProposalWriter(Protocol):
    """Filesystem writer for Tier 3 topic candidates."""

    def write_candidate(self, candidate: TopicCandidate) -> Path: ...

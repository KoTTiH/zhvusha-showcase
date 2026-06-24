"""Adapters that turn higher-level candidates into proposal files."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.skills.proposal_command.models import ProposalKind, ProposalModel
from src.skills.proposal_command.store import write_proposal

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.skills.topic_to_spec.models import TopicCandidate


class TopicProposalWriter:
    """Persist Tier 3 topic candidates as ``proposals/*.md`` files."""

    def __init__(
        self,
        *,
        proposals_dir: Path,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._proposals_dir = proposals_dir
        self._clock = clock or (lambda: datetime.now(tz=UTC))

    def write_candidate(self, candidate: TopicCandidate) -> Path:
        if candidate.kind != "proposal" or candidate.tier != 3:
            raise ValueError(
                "TopicProposalWriter accepts only Tier 3 proposal candidates"
            )
        proposal = ProposalModel(
            slug=candidate.slug,
            title=candidate.what.removeprefix("Подготовить proposal по теме ").strip(
                "«»."
            )
            or candidate.slug,
            created_at=self._clock(),
            created_by="zhvusha",
            tier=3,
            kind=_kind_from_candidate(candidate),
            summary=candidate.what,
            proposed_change=candidate.what,
            rationale=candidate.rationale,
            acceptance=list(candidate.acceptance),
            files_likely_touched=list(candidate.files_likely_touched),
            risk=candidate.risk,
            source_provenance=list(candidate.source_provenance),
            pillar_attribution=dict(candidate.pillar_attribution),
            created_from=f"topic:{candidate.slug}",
            metadata={"why_now": candidate.why_now},
        )
        return write_proposal(self._proposals_dir, proposal)


def _kind_from_candidate(candidate: TopicCandidate) -> ProposalKind:
    text = " ".join(
        [candidate.what, candidate.risk, *candidate.files_likely_touched]
    ).lower()
    if "safety" in text or "безопас" in text:
        return ProposalKind.SAFETY
    if "personality" in text or "личност" in text:
        return ProposalKind.PERSONALITY
    if "self-coding" in text or "самокод" in text or "codex" in text:
        return ProposalKind.SELF_CODING
    return ProposalKind.ARCHITECTURE

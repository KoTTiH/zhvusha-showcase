"""Proposal markdown store tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from src.skills.proposal_command.models import ProposalModel
from src.skills.proposal_command.store import (
    load_proposal,
    load_proposal_raw,
    write_proposal,
)
from src.skills.spec_command.parser import SourceProvenance


def _source() -> SourceProvenance:
    return SourceProvenance(
        url="local:test",
        source_type="local_repo",
        trust_tier="direct",
        claim="Test evidence exists.",
    )


def _proposal(*, slug: str = "safety-protocol") -> ProposalModel:
    return ProposalModel(
        slug=slug,
        title="Safety protocol",
        created_at=datetime(2026, 5, 7, 12, 0, tzinfo=UTC),
        created_by="zhvusha",
        tier=3,
        summary="Prepare a Tier 3 proposal for safety protocol changes.",
        proposed_change="Describe the architecture change before any code is written.",
        rationale="Safety protocol changes are protected and need human approval.",
        acceptance=["Никита can approve, defer or reject the proposal."],
        files_likely_touched=["src/safety/"],
        risk="Core safety behavior can regress.",
        source_provenance=[_source()],
    )


def test_write_and_load_proposal_round_trips(tmp_path: Path) -> None:
    path = write_proposal(tmp_path, _proposal())

    loaded = load_proposal(path)
    raw, body = load_proposal_raw(path)

    assert path.name == "2026-05-07-safety-protocol.md"
    assert loaded.slug == "safety-protocol"
    assert raw["status"] == "pending_approval"
    assert "## Источники" in body


def test_zhvusha_proposal_requires_source_provenance() -> None:
    payload = _proposal().model_dump(mode="json")
    payload["source_provenance"] = []

    with pytest.raises(ValueError, match="source_provenance"):
        ProposalModel.model_validate(payload)

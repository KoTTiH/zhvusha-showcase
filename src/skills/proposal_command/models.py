"""Schema for human-approved Tier 3 proposal markdown files."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - Pydantic validates at runtime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.skills.spec_command.parser import (
    SourceProvenance,  # noqa: TC001 - Pydantic runtime schema
)


class ProposalStatus(StrEnum):
    """Lifecycle for architecture proposals.

    Proposals are deliberately one step before executable specs. Approval
    means "Никита accepted the direction"; it does not start code generation.
    """

    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    DEFERRED = "deferred"
    REJECTED = "rejected"
    DONE = "done"


class ProposalKind(StrEnum):
    """Coarse proposal area for later filtering."""

    ARCHITECTURE = "architecture"
    SAFETY = "safety"
    PERSONALITY = "personality"
    SELF_CODING = "self_coding"
    OPERATIONS = "operations"


class ProposalModel(BaseModel):
    """Frontmatter contract for ``proposals/<date>-<slug>.md`` files."""

    slug: str = Field(pattern=r"^[a-z0-9-]+$", max_length=80)
    title: str = Field(min_length=1, max_length=220)
    created_at: datetime
    created_by: Literal["zhvusha", "nikita"] = "zhvusha"

    tier: Literal[3] = 3
    kind: ProposalKind = ProposalKind.ARCHITECTURE
    status: ProposalStatus = ProposalStatus.PENDING_APPROVAL

    summary: str = Field(min_length=20)
    proposed_change: str = Field(min_length=20)
    rationale: str = Field(min_length=1)
    acceptance: list[str] = Field(min_length=1)
    files_likely_touched: list[str] = Field(default_factory=list)
    risk: str = Field(min_length=1)

    source_provenance: list[SourceProvenance] = Field(min_length=1)
    pillar_attribution: dict[str, float] = Field(default_factory=dict)
    created_from: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    approved_at: datetime | None = None
    approved_by: Literal["nikita"] | None = None
    deferred_reason: str | None = None
    rejected_reason: str | None = None

    @field_validator(
        "slug",
        "title",
        "summary",
        "proposed_change",
        "rationale",
        "risk",
        "created_from",
        "deferred_reason",
        "rejected_reason",
        mode="before",
    )
    @classmethod
    def _strip_optional_strings(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return ""
            return cleaned
        return value

    @field_validator("acceptance", "files_likely_touched")
    @classmethod
    def _strip_list_entries(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("list entries must be non-blank")
        return cleaned

    @model_validator(mode="after")
    def _zhvusha_proposals_have_evidence(self) -> ProposalModel:
        if self.created_by != "zhvusha":
            return self
        if not self.rationale:
            raise ValueError("created_by=zhvusha proposals require rationale")
        if not self.source_provenance:
            raise ValueError("created_by=zhvusha proposals require source_provenance")
        return self

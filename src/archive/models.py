"""Models for archived self-coding cycles."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - Pydantic validates at runtime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ArchiveStatus(StrEnum):
    """Outcome of a recorded self-coding cycle."""

    COMMITTED = "committed"
    FAILED = "failed"


class ArchiveNode(BaseModel):
    """Durable record for one spec/proposal implementation attempt."""

    model_config = ConfigDict(populate_by_name=True)

    slug: str = Field(pattern=r"^[a-z0-9-]+$", max_length=120)
    spec_slug: str | None = None
    proposal_slug: str | None = None
    tier: Literal[1, 2, 3]
    status: ArchiveStatus
    created_at: datetime

    branch: str | None = None
    commit_sha: str | None = None
    parent_slug: str | None = None

    diff_summary: str = Field(min_length=1)
    tests_summary: str = Field(min_length=1)
    rationale: str = ""
    insight: str = Field(min_length=1)

    source_evidence: list[dict[str, str]] = Field(default_factory=list)
    runtime_config: dict[str, str] = Field(default_factory=dict, alias="model_config")
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "slug",
        "spec_slug",
        "proposal_slug",
        "branch",
        "commit_sha",
        "parent_slug",
        "diff_summary",
        "tests_summary",
        "rationale",
        "insight",
        mode="before",
    )
    @classmethod
    def _strip_optional_strings(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("tags")
    @classmethod
    def _strip_tags(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("archive tags must be non-blank")
        return cleaned

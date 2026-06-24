"""Pydantic schema for tasks/<YYYY-MM-DD>-<slug>.yaml spec files.

The schema is the contract between everyone who reads or writes a spec:

* `IdeationToSpecSkill` (Phase 12) — writes through `SpecModel.model_validate`
  after the Codex Architect backend returns a draft YAML.
* `SpecCommandSkill` (Phase 11) — reads via `SpecModel.model_validate` when
  the user runs ``/spec show <slug>`` or approves a spec.
* `ImplementSpecSkill` (Phase 13) — re-validates before starting an
  execution, then writes status transitions back.
* ``scripts/check_whitelist.sh`` (Phase 13) — parses the same YAML with
  ``yaml.safe_load`` to enforce the path whitelist at commit time.

The Tier 3 path list mirrors ``scripts/check_tier3_protection.sh``: any change
here MUST be reflected in the bash script (and vice versa). Two enforcement
points are intentional — the validator catches bad specs before they are
written; the bash hook catches commits regardless of which tool produced them.
"""

from __future__ import annotations

import re
from datetime import datetime  # noqa: TC003 — Pydantic needs runtime access
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Mirror of scripts/check_tier3_protection.sh TIER3_PATHS.
# Keep these in sync — both must change together.
_TIER3_EXACT_PATHS: frozenset[str] = frozenset(
    {
        "src/skills/base.py",
        "src/skills/__init__.py",
        "src/skills/registry.py",
        "src/personality/decision.py",
        ".importlinter",
        "AGENTS.md",
        "CLAUDE.md",
        "scripts/check_tier3_protection.sh",
    }
)
_TIER3_PREFIXES: tuple[str, ...] = ("src/safety/",)
_TIER3_SUFFIXES: tuple[str, ...] = ("/protocols.py",)
_TIER3_SELF_DECLARATION_RE = re.compile(
    r"(?:фиксирую|классифицирую|считаю|classify|classified|mark|marked|set)"
    r".{0,80}\bTier\s*3\b",
    re.IGNORECASE | re.DOTALL,
)


def _is_tier3_path(path: str) -> bool:
    return (
        path in _TIER3_EXACT_PATHS
        or any(path.startswith(prefix) for prefix in _TIER3_PREFIXES)
        or any(path.endswith(suffix) for suffix in _TIER3_SUFFIXES)
    )


def _rationale_declares_tier3(rationale: str) -> bool:
    return bool(_TIER3_SELF_DECLARATION_RE.search(rationale))


class SpecStatus(StrEnum):
    """Lifecycle stages of a spec.

    Transitions: ``pending_approval → approved → in_progress → done|failed``,
    or ``pending_approval → rejected``. Failed specs preserve the failed
    attempt in history, but can be explicitly re-approved for a retry.
    Terminal states are ``done`` and ``rejected``.
    """

    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


class SpecKind(StrEnum):
    """Spec intent classification.

    Determines whether the Editor may touch existing test files at all.
    The system prompt branches on this value: FEATURE/FIX/DOCS treat
    existing tests as immutable; REFACTOR explicitly permits structural
    updates (renamed imports, renamed symbol references, updated call
    signatures) but never assertion-logic edits.

    * ``FEATURE`` — new code, new tests; existing tests are immutable.
    * ``FIX`` — bug fix; existing tests immutable, only new regression
      test is added (the failing_test for the spec).
    * ``REFACTOR`` — rename/restructure; existing tests may be updated
      structurally to follow renamed symbols, but assertion logic and
      expected values must not change.
    * ``DOCS`` — docstrings, AGENTS.md/CLAUDE.md, READMEs; tests typically
      not touched. Same immutability rule as FEATURE.

    Default is FEATURE — the safest mode and the one all pre-kind specs
    were authored under, so existing yaml files keep parsing without
    explicit migration.
    """

    FEATURE = "feature"
    FIX = "fix"
    REFACTOR = "refactor"
    DOCS = "docs"


class FailingTest(BaseModel):
    """The proof-of-completion test for a spec.

    Per AGENTS.md TDD discipline: a passing run of this test is the formal
    objective function for ``implement_spec``. The test itself is written
    during the RED phase before code; implementation chases its green status.
    """

    file: str = Field(min_length=1)
    name: str = Field(min_length=1)
    spec: str = Field(min_length=1)


class ResearchFinding(BaseModel):
    """A citation from the research phase.

    Source order is KB → code → web (KB #72). ``source`` is a free-form
    identifier (e.g. ``"kb_71"``, ``"src/skills/base.py:155"``,
    ``"https://aider.chat/2024/09/26/architect.html"``); ``excerpt`` is the
    relevant quote; ``relevance`` describes why the finding informs the spec.
    """

    source: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)
    relevance: str = Field(min_length=1)


class SourceProvenance(BaseModel):
    """Specific evidence that justified a generated self-coding spec.

    ``ResearchFinding`` is the broad research log. ``SourceProvenance`` is the
    narrowed decision ledger: the source, its trust class, and the concrete
    claim the Architect used when deciding that the spec should exist.
    """

    url: str = Field(min_length=1)
    source_type: Literal[
        "official_docs",
        "paper",
        "github",
        "forum",
        "secondary_press",
        "local_repo",
        "kb",
        "code",
        "other",
    ]
    trust_tier: Literal["primary", "direct", "weak", "rejected"]
    claim: str = Field(min_length=1)

    @field_validator("url", "claim")
    @classmethod
    def _strip_non_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must be non-blank after stripping whitespace")
        return cleaned


class PreviousAttempt(BaseModel):
    """One archive lesson retrieved before a new self-coding spec.

    This is the active DGM/ASI-Evolve loop in miniature: Architect sees
    concrete previous cycles and carries the useful lesson into the spec so
    Editor, reviewer, archive, morning consolidation and future cycles can all
    distinguish "new task" from "retry with known traps".
    """

    archive_slug: str = Field(min_length=1)
    status: Literal["committed", "failed"]
    tier: Literal[1, 2, 3]
    insight: str = Field(min_length=1)
    tests_summary: str = ""
    commit_sha: str | None = None

    @field_validator(
        "archive_slug",
        "insight",
        "tests_summary",
        "commit_sha",
        mode="before",
    )
    @classmethod
    def _strip_strings(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def _require_non_blank_insight(self) -> PreviousAttempt:
        if not self.insight:
            raise ValueError("previous_attempt insight must be non-empty")
        if not self.archive_slug:
            raise ValueError("previous_attempt archive_slug must be non-empty")
        return self


class ExistingTestUpdate(BaseModel):
    """A declared, justified mutation of an existing test file.

    Architect uses this channel when a spec must edit a test that the
    Editor's kind-rules would otherwise treat as immutable (e.g. the
    spec adds a new entry to a finite collection, and a hidden contract
    test asserts that collection's exact membership). The Editor reads
    the list and gates test edits against it: any modification of an
    existing test file outside this list remains forbidden, and even
    listed tests may only be changed within ``allowed_changes``.

    Each field is required and must be non-blank — the structure is the
    contract between Architect and Editor, not a free-form note. The
    ``path`` field is sanity-guarded to live under ``tests/`` so the
    channel cannot be repurposed to mutate production files.
    """

    path: str = Field(min_length=1)
    test_name: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    allowed_changes: str = Field(min_length=1)

    @field_validator("path", "test_name", "reason", "allowed_changes")
    @classmethod
    def _strip_non_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must be non-blank after stripping whitespace")
        return cleaned

    @field_validator("path")
    @classmethod
    def _path_under_tests(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned.startswith("tests/"):
            raise ValueError(
                "existing_tests_to_update.path must point under tests/ — "
                f"got {value!r}. The channel mutates *tests*, not "
                "production files; use whitelist_paths for production edits."
            )
        return cleaned


class SpecModel(BaseModel):
    """Spec file schema.

    Field semantics follow KB #71 spec-first workflow. Any addition here
    must update ``tasks/README.md`` and may require updating the upstream
    writers (``ideation_to_spec.spec_writer``) and downstream readers
    (``check_whitelist.sh``, ``implement_spec.commit_runner``).
    """

    # Identity
    slug: str = Field(pattern=r"^[a-z0-9-]+$", max_length=60)
    title: str = Field(min_length=1, max_length=200)
    created_at: datetime
    # Pre-self-coding legacy specs were authored manually and may not carry
    # ``created_by`` at all; keep those loadable as Никита-authored specs.
    created_by: Literal["zhvusha", "nikita"] = "nikita"

    # Classification
    tier: Literal[1, 2, 3]
    kind: SpecKind = SpecKind.FEATURE
    goal: str = Field(min_length=20)

    # TDD
    failing_test: FailingTest

    # Surfaces
    whitelist_paths: list[str] = Field(min_length=1)
    read_only_paths: list[str] = Field(default_factory=list)

    # Risk
    blast_radius: list[str] = Field(min_length=1)
    rollback_path: list[str] = Field(min_length=1)
    # No-downgrade contract. Architect must name the behaviours, nuances,
    # fallbacks and safety/context guarantees that the Editor must preserve.
    # Empty ``allowed_simplifications`` means no deletion/simplification is
    # authorised; Editor must enrich or deepen behaviour instead of flattening.
    preserve_behavior: list[str] = Field(default_factory=list)
    allowed_simplifications: list[str] = Field(default_factory=list)

    # Legitimate-test-mutation channel (Phase 16). When the spec extends
    # a finite collection and a read_only test asserts the collection's
    # exact membership/length, Architect declares the necessary edits
    # here; Editor uses the list to gate test mutations. Empty list
    # means existing tests remain immutable per the kind-rules.
    existing_tests_to_update: list[ExistingTestUpdate] = Field(default_factory=list)
    # Host runtime env activation is off by default. When true, Editor may
    # produce allowed non-protected `.env` assignments for these exact keys;
    # ImplementSpec applies them through the live env activator and writes a
    # redacted host-ops audit artifact. Protected keys are still blocked by
    # EnvGuard even if a malformed spec lists them here.
    live_env_activation: bool = False
    live_env_keys: list[str] = Field(default_factory=list)

    # Research
    research_findings: list[ResearchFinding] = Field(default_factory=list)
    source_provenance: list[SourceProvenance] = Field(default_factory=list)
    # Short-term dialogue that led to this spec, usually captured from
    # /самокодинг before Никита says "оформи план". Optional for legacy specs,
    # but persisted so archive/consolidation can see what decision context
    # existed before code was written.
    chat_context: list[str] = Field(default_factory=list)
    previous_attempts: list[PreviousAttempt] = Field(default_factory=list)
    rationale: str = ""

    # Lifecycle
    status: SpecStatus = SpecStatus.PENDING_APPROVAL
    approved_at: datetime | None = None
    approved_by: Literal["nikita", "zhvusha"] | None = None
    # Audit-only reason for autonomous approvals. Manual Никита approvals do not
    # need this field; Жвуша approvals do, so future reviews can see why a spec
    # was allowed to run without a fresh user confirmation.
    autonomous_approval_reason: str = ""
    rejected_reason: str | None = None

    # Implementation tracking
    branch: str | None = None
    commit_sha: str | None = None
    iterations: int = Field(default=0, ge=0)
    failed_attempts: list[str] = Field(default_factory=list)

    # Costs (filled by ImplementSpecSkill after run)
    actual_tokens: int = Field(default=0, ge=0)
    actual_cost_usd: float = Field(default=0.0, ge=0.0)

    @field_validator("whitelist_paths", "read_only_paths")
    @classmethod
    def _strip_paths(cls, value: list[str]) -> list[str]:
        """Reject empty / whitespace-only path entries."""
        cleaned = [path.strip() for path in value]
        if any(not path for path in cleaned):
            raise ValueError(
                "path entries must be non-empty after stripping whitespace"
            )
        return cleaned

    @field_validator("rationale")
    @classmethod
    def _strip_rationale(cls, value: str) -> str:
        return value.strip()

    @field_validator("autonomous_approval_reason")
    @classmethod
    def _strip_autonomous_approval_reason(cls, value: str) -> str:
        return value.strip()

    @field_validator("preserve_behavior", "allowed_simplifications")
    @classmethod
    def _strip_preservation_lists(cls, value: list[str]) -> list[str]:
        """Reject empty bullets in the no-downgrade contract."""
        cleaned = [entry.strip() for entry in value]
        if any(not entry for entry in cleaned):
            raise ValueError(
                "preserve_behavior / allowed_simplifications entries must be "
                "non-empty after stripping whitespace"
            )
        return cleaned

    @field_validator("chat_context")
    @classmethod
    def _strip_chat_context(cls, value: list[str]) -> list[str]:
        """Reject empty captured dialogue lines."""
        cleaned = [entry.strip() for entry in value]
        if any(not entry for entry in cleaned):
            raise ValueError("chat_context entries must be non-empty")
        return cleaned

    @field_validator("live_env_keys")
    @classmethod
    def _strip_live_env_keys(cls, value: list[str]) -> list[str]:
        """Normalize declared live env keys without accepting blanks."""
        cleaned = [entry.strip().upper() for entry in value]
        if any(not entry for entry in cleaned):
            raise ValueError("live_env_keys entries must be non-empty")
        return cleaned

    @model_validator(mode="after")
    def _tier3_consistency(self) -> SpecModel:
        """Tier 3 paths in whitelist require ``tier: 3``.

        Mirrors ``scripts/check_tier3_protection.sh``. The cross-field check
        catches mis-classified specs early — before they reach approval, the
        agent sandbox, or the commit hook.
        """
        offending = [path for path in self.whitelist_paths if _is_tier3_path(path)]
        if offending and self.tier != 3:
            raise ValueError(
                "Tier 3 paths present in whitelist_paths but tier != 3: "
                f"{offending!r}. Either set tier=3 or split the spec into a "
                "Tier 3 proposal (proposals/<slug>.md) plus a Tier 1/2 "
                "follow-up that owns only non-protected files."
            )
        return self

    @model_validator(mode="after")
    def _tier_matches_rationale_declaration(self) -> SpecModel:
        """Do not allow the audit rationale to contradict ``tier``."""
        if self.tier != 3 and _rationale_declares_tier3(self.rationale):
            raise ValueError(
                "rationale declares Tier 3 but tier != 3. Set tier=3 or "
                "rewrite the rationale so the audit trail matches the spec."
            )
        return self

    @model_validator(mode="after")
    def _zhvusha_specs_have_decision_evidence(self) -> SpecModel:
        """Generated specs must carry why/evidence for future review.

        Historical/manual specs can remain minimal for backward compatibility.
        Жвушины specs are part of the self-improvement archive, so they need
        enough decision evidence to be audited later.
        """
        if self.created_by != "zhvusha":
            return self
        if not self.rationale:
            raise ValueError("created_by=zhvusha specs require non-empty rationale")
        if not self.source_provenance:
            raise ValueError(
                "created_by=zhvusha specs require non-empty source_provenance"
            )
        if not self.preserve_behavior:
            raise ValueError(
                "created_by=zhvusha specs require non-empty preserve_behavior"
            )
        return self

    @model_validator(mode="after")
    def _live_env_activation_requires_keys(self) -> SpecModel:
        """Live env activation must name exact runtime keys explicitly."""
        if self.live_env_activation and not self.live_env_keys:
            raise ValueError("live_env_activation=true requires live_env_keys")
        return self

    @model_validator(mode="after")
    def _autonomous_approval_is_audited_and_bounded(self) -> SpecModel:
        """Жвуша approvals must be auditable.

        Runtime policy decides which low-risk tier is currently allowed; the
        schema keeps the durable approval trail readable for future reviews.
        Tier 3 is architectural/safety/personality surface and must always be
        approved by Никита after chat discussion, not by Жвуша herself.
        """
        if self.approved_by != "zhvusha":
            return self
        if self.tier >= 3:
            raise ValueError("Tier 3 requires Никита approval, not Жвуша approval")
        if not self.autonomous_approval_reason:
            raise ValueError("approved_by=zhvusha requires autonomous_approval_reason")
        return self

"""Validation + serialization helpers for the Architect skill.

Three small functions, intentionally pure (no LLM, no SDK, no Telegram):

* :func:`extract_yaml_block` peels a ``\\`\\`\\`yaml ... \\`\\`\\``` block out of
  the SDK's free-text reply (or accepts plain YAML if no fence is present).
* :func:`build_spec_from_draft` validates the parsed dict against
  :class:`SpecModel` and re-runs the deterministic ``classify_spec_tier``. The
  final tier is the most restrictive value between the classifier and the
  Architect draft, so a semantic Tier 3 declaration is never downgraded.
* :func:`write_spec_to_disk` writes the spec under
  ``tasks/<YYYY-MM-DD>-<slug>.yaml`` with stable key order, refusing to
  overwrite an existing file (a numeric suffix is appended).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import yaml

from src.skills.ideation_to_spec.tier_classifier import classify_spec_tier
from src.skills.spec_command.parser import SpecModel

if TYPE_CHECKING:
    from pathlib import Path

    from src.research.protocols import Citation

_FENCE_RE = re.compile(r"```ya?ml\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
_CLARIFICATION_RE = re.compile(
    r"^\s*CLARIFICATION_NEEDED:\s*(.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_FREE_TEXT_LIST_FIELDS: frozenset[str] = frozenset(
    {
        "blast_radius",
        "rollback_path",
        "preserve_behavior",
        "allowed_simplifications",
    }
)


def extract_yaml_block(text: str) -> str:
    """Return the YAML payload from a free-text SDK reply.

    Picks the first ```` ```yaml ``` ```` block if any are present; otherwise
    returns the input verbatim (assumed to be raw YAML). Raises ``ValueError``
    on empty input.
    """
    if not text or not text.strip():
        raise ValueError("empty SDK reply — no YAML to parse")
    match = _FENCE_RE.search(text)
    if match is not None:
        return match.group(1).strip()
    return text.strip()


def extract_clarification_request(text: str) -> str | None:
    """Return Architect's clarification question, if it chose dialogue."""
    if not text or not text.strip():
        return None
    match = _CLARIFICATION_RE.match(text.strip())
    if match is None:
        return None
    question = " ".join(match.group(1).split())
    return question or None


def build_spec_from_draft(draft: dict[str, Any]) -> SpecModel:
    """Validate ``draft`` and normalize its ``tier`` through the classifier.

    Order matters: classification runs *before* SpecModel validation so the
    cross-field consistency checks see the corrected tier. The classifier can
    only raise risk; it must not downgrade a stricter tier the Architect already
    declared from semantic evidence in the task.
    """
    whitelist_paths = list(draft.get("whitelist_paths", []) or [])
    goal = str(draft.get("goal", ""))
    rationale = str(draft.get("rationale", ""))
    modifies = draft.get("modifies_capabilities", []) or []

    classified = classify_spec_tier(
        whitelist_paths=whitelist_paths,
        goal=goal,
        modifies_capabilities=modifies,
        rationale=rationale,
    )
    tier = max(classified, _declared_tier(draft.get("tier")))
    draft = _normalize_free_text_lists({**draft, "tier": tier})
    return SpecModel.model_validate(draft)


def merge_research_citations(
    spec: SpecModel,
    citations: list[Citation],
) -> SpecModel:
    """Persist upstream research citations even if the LLM draft omitted them."""
    additions: list[dict[str, str]] = []
    seen = {
        (finding.source, finding.excerpt, finding.relevance)
        for finding in spec.research_findings
    }
    for citation in citations:
        source = citation.ref.strip() or citation.source
        excerpt = citation.excerpt.strip()
        if not source or not excerpt:
            continue
        relevance = _research_relevance_for_citation(citation.source)
        key = (source, excerpt, relevance)
        if key in seen:
            continue
        seen.add(key)
        additions.append(
            {
                "source": source,
                "excerpt": excerpt,
                "relevance": relevance,
            }
        )

    if not additions:
        return spec

    payload = spec.model_dump(mode="python")
    payload["research_findings"] = [
        *payload.get("research_findings", []),
        *additions,
    ]
    return type(spec).model_validate(payload)


def _declared_tier(value: Any) -> int:
    """Return a valid draft tier, defaulting to Tier 1 for malformed values."""
    if isinstance(value, bool):
        return 1
    if isinstance(value, int) and value in {1, 2, 3}:
        return value
    return 1


def _normalize_free_text_lists(draft: dict[str, Any]) -> dict[str, Any]:
    """Flatten recoverable YAML object bullets in free-text list fields.

    Architect sometimes emits ``- path: ...`` where SpecModel expects a
    narrative string bullet. That is safe to recover for risk/rollback prose,
    but not for contract fields like whitelist paths or source provenance.
    """
    normalized = dict(draft)
    for field in _FREE_TEXT_LIST_FIELDS:
        value = normalized.get(field)
        if isinstance(value, list):
            normalized[field] = [_stringify_free_text_item(item) for item in value]
    return normalized


def _stringify_free_text_item(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        parts = [f"{key}: {value}" for key, value in item.items()]
        return "; ".join(parts)
    return str(item)


def _research_relevance_for_citation(source: str) -> str:
    if source == "unknown":
        return (
            "Explicit unavailable research source; do not claim this context "
            "as evidence."
        )
    return f"Research evidence from {source}."


def write_spec_to_disk(
    *,
    tasks_dir: Path,
    spec: SpecModel,
    now: datetime | None = None,
) -> Path:
    """Persist ``spec`` under ``tasks/<YYYY-MM-DD>-<slug>.yaml``.

    Refuses to overwrite an existing file: appends ``-2``, ``-3``, … to the
    slug part of the filename until a free name is found. The on-disk
    spec is the Pydantic dump (``model_dump(mode="json")``) — exactly the
    form ``SpecCommandSkill`` expects when reloading.
    """
    tasks_dir.mkdir(parents=True, exist_ok=True)
    when = (now or datetime.now(tz=UTC)).date().isoformat()

    base_name = f"{when}-{spec.slug}"
    path = tasks_dir / f"{base_name}.yaml"
    suffix = 2
    while path.exists():
        path = tasks_dir / f"{base_name}-{suffix}.yaml"
        suffix += 1

    payload = spec.model_dump(mode="json")
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path

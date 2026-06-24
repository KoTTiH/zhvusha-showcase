"""Shared types for the Memory capability module.

Leaf module — contains Pydantic data classes and parsing helpers that other
``src/memory/*`` modules (and the public :mod:`src.memory.protocols` façade)
import from. Intentionally has zero ``src.memory.*`` dependencies so it can
be imported by any other memory submodule without introducing a cycle.

Design notes (phase 5C):
    * ``EnrichmentResult`` / ``LearningSignal`` stay as Pydantic v2 BaseModel
      (not :mod:`dataclasses`). The range validators (``Field(ge=, le=)``)
      and :func:`model_validator` on ``LearningSignal`` are load-bearing —
      :class:`src.memory.sonnet_enricher.SonnetEnricher` relies on
      ``ValidationError`` to gracefully fall back to ``None`` on malformed
      LLM output.
    * ``parse_enrichment_json`` is the publicised (underscore-stripped)
      form of the former ``_parse_enrichment_json`` helper. Dream-signal
      extraction in ``src.skills.chat_response.dream_extractor`` used to
      import the private helper directly (hidden coupling); exposing it
      here resolves that.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "EnrichmentResult",
    "LearningSignal",
    "parse_enrichment_json",
]


class LearningSignal(BaseModel):
    """A learning signal extracted from a user message.

    Emitted by the enricher when the message expresses an explicit rule,
    preference, correction, fact, or boundary that Zhvusha should apply
    going forward. ``None`` for regular conversational turns.

    Routing semantics (see :class:`src.memory.learning_staging.StagingWriter`):
      * ``apply_immediately AND confidence > 0.8`` → ``learnings_immediate.md``
        (read back into the NEXT system prompt in the same chat)
      * otherwise → ``learnings_pending.md`` (for /morning manual review)
    """

    type: Literal["rule", "preference", "correction", "fact", "boundary"]
    statement: str = Field(max_length=300)
    scope: Literal["tone", "work", "personal_facts", "boundaries", "preferences"]
    confidence: float = Field(ge=0.0, le=1.0)
    apply_immediately: bool
    original_claim: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _check_original_claim_matches_type(self) -> LearningSignal:
        if self.type == "correction" and self.original_claim is None:
            raise ValueError("original_claim is required when type='correction'")
        if self.type != "correction" and self.original_claim is not None:
            raise ValueError("original_claim must be None when type != 'correction'")
        return self


class EnrichmentResult(BaseModel):
    """Structured metadata extracted from a single user message."""

    importance: float = Field(ge=0.0, le=1.0)
    valence: Literal["positive", "negative", "neutral"]
    intent: Literal[
        "question",
        "statement",
        "command",
        "feedback",
        "correction",
        "preference",
        "emotional",
        "meta",
    ]
    emotion: Literal[
        "neutral",
        "happy",
        "frustrated",
        "angry",
        "curious",
        "tired",
        "excited",
        "confused",
        "sad",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0, default=0.5)
    is_feedback: bool
    feedback_strength: float = Field(ge=-1.0, le=1.0)
    reasoning: str = Field(max_length=500)
    self_emotion: Literal[
        "joy",
        "excitement",
        "delight",
        "playfulness",
        "sadness",
        "melancholy",
        "loneliness",
        "frustration",
        "irritation",
        "hostility",
        "anxiety",
        "nervousness",
        "calm",
        "serenity",
        "contentment",
        "curiosity",
        "wonder",
        "fascination",
        "warmth",
        "tenderness",
        "gratitude",
        "brooding",
        "reflectiveness",
        "pensiveness",
        "pride",
        "satisfaction",
        "confidence",
        "confusion",
        "bewilderment",
        "overwhelm",
    ] = "curiosity"
    self_arousal: float = Field(ge=0.0, le=1.0, default=0.5)
    learning_signal: LearningSignal | None = None


_MD_FENCE_RE = re.compile(r"^```(?:json)?\s*", re.MULTILINE)
_MD_FENCE_CLOSE_RE = re.compile(r"```\s*$", re.MULTILINE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_markdown(text: str) -> str:
    """Remove markdown code fences if the LLM wrapped JSON in them."""
    cleaned = _MD_FENCE_RE.sub("", text, count=1)
    cleaned = _MD_FENCE_CLOSE_RE.sub("", cleaned, count=1)
    return cleaned.strip()


def parse_enrichment_json(raw: str) -> dict[str, object] | None:
    """Robust JSON extraction with two-layer fallback.

    Shared between :class:`src.memory.sonnet_enricher.SonnetEnricher` and
    ``src.skills.chat_response.dream_extractor``; both receive LLM output
    that is either a bare JSON object or a markdown-fenced JSON object,
    occasionally surrounded by prose.

    Returns ``None`` on any parse failure — callers are expected to treat
    that as a best-effort signal to skip the current item rather than
    raise.
    """
    if not raw:
        return None

    cleaned = _strip_markdown(raw)

    try:
        parsed: object = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        return parsed

    match = _JSON_OBJECT_RE.search(cleaned)
    if match:
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed

    return None

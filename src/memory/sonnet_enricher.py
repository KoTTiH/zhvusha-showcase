"""Async Sonnet-based enrichment of user message episodes.

Thin wrapper around :class:`src.memory.pipelines.enrichment.EnrichmentPipelineContext`
and its stages. ``SonnetEnricher.enrich`` delegates to the pipeline and
returns the final ``EnrichmentResult`` (or ``None`` on any failure).

Called asynchronously from :class:`src.skills.chat_response.skill.ChatResponseSkill`
after the response is sent — never blocks the user-facing reply flow.

Returns ``None`` on any failure (LLM error, invalid JSON, schema
mismatch). The Episode keeps its placeholder values in that case.

Re-exports ``EnrichmentResult``, ``LearningSignal``, ``parse_enrichment_json``,
and ``_strip_markdown`` from :mod:`src.memory.types` for backward
compatibility with test modules that import them from this module name.
Re-exports ``_ENRICHER_SYSTEM_PROMPT`` from
:mod:`src.memory.pipelines.enrichment` for the same reason.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.memory.pipelines.enrichment import (
    _ENRICHER_SYSTEM_PROMPT,
    EnrichmentPipelineContext,
    build_enrichment_pipeline,
)
from src.memory.types import (
    EnrichmentResult,
    LearningSignal,
    _strip_markdown,
    parse_enrichment_json,
)

if TYPE_CHECKING:
    from src.core.config import Tier

logger = structlog.get_logger()

__all__ = [
    "_ENRICHER_SYSTEM_PROMPT",
    "EnrichmentResult",
    "LearningSignal",
    "SonnetEnricher",
    "_strip_markdown",
    "get_enricher",
    "parse_enrichment_json",
]


class SonnetEnricher:
    """Extracts :class:`EnrichmentResult` from a user message via LLM.

    Tier is configurable (default: worker/Haiku). Called asynchronously
    after the response is sent to the user. Returns ``None`` on any
    failure — never raises. The Episode keeps placeholder values if
    enrichment fails.

    Delegates to :class:`src.memory.pipelines.enrichment.EnrichmentPipelineContext`
    / :func:`src.memory.pipelines.enrichment.build_enrichment_pipeline`.
    The pipeline is constructed once per :class:`SonnetEnricher`
    instance at construction time.
    """

    def __init__(self, *, tier: Tier = "worker") -> None:
        self._tier: Tier = tier
        self._pipeline = build_enrichment_pipeline()

    async def enrich(
        self,
        message: str,
        recent_context: str = "",
        prev_bot_response: str = "",
    ) -> EnrichmentResult | None:
        """Run the enrichment pipeline and return the final result or ``None``."""
        ctx = EnrichmentPipelineContext(
            message=message,
            recent_context=recent_context,
            prev_bot_response=prev_bot_response,
            tier=self._tier,
        )
        final = await self._pipeline.run(ctx)
        return final.result


_enricher: SonnetEnricher | None = None


def get_enricher() -> SonnetEnricher:
    """Singleton accessor for SonnetEnricher."""
    global _enricher
    if _enricher is None:
        from src.core.config import get_settings

        _enricher = SonnetEnricher(
            tier=get_settings().enrichment_tier,
        )
    return _enricher

"""Topic clustering and priority scoring for the news pipeline."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.news.dedup import dedup_signature

if TYPE_CHECKING:
    from src.news.models import SourceItem
    from src.pillars import PillarConfig

_WORD_RE = re.compile(r"[\w#+.-]{3,}", re.UNICODE)
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "about",
    "это",
    "как",
    "для",
    "что",
    "про",
    "новый",
    "новая",
}
_AUTHORITY_BY_SOURCE_TYPE = {
    "official_docs": 1.0,
    "paper": 0.95,
    "github": 0.9,
    "blog": 0.7,
    "secondary_press": 0.5,
    "telegram": 0.45,
    "youtube": 0.4,
    "social": 0.25,
    "other": 0.3,
}


@dataclass(frozen=True)
class TopicClusterDraft:
    cluster_key: str
    title: str
    summary: str
    items: list[SourceItem]
    top_terms: list[str]
    base_importance: float
    source_authority: float
    cluster_velocity: float
    pillar_alignment: dict[str, float] = field(default_factory=dict)
    final_priority: float = 0.0


def cluster_source_items(
    items: list[SourceItem],
    *,
    pillars: PillarConfig | None = None,
    now: datetime | None = None,
) -> list[TopicClusterDraft]:
    """Cluster deduplicated source items into deterministic topic drafts."""
    now = now or datetime.now(tz=UTC)
    buckets: dict[str, list[SourceItem]] = {}
    for item in items:
        key = _cluster_key(item)
        buckets.setdefault(key, []).append(item)

    clusters = [
        _build_cluster(key, bucket, pillars=pillars, now=now)
        for key, bucket in buckets.items()
    ]
    return sorted(clusters, key=lambda c: c.final_priority, reverse=True)


def score_topic_priority(
    *,
    base_importance: float,
    pillar_alignment: dict[str, float],
    pillar_weights: dict[str, float],
    cluster_velocity: float,
    source_authority: float,
    staleness_days: float,
    blocking_boost: float = 0.0,
) -> float:
    """Apply the roadmap's priority function with bounded numeric inputs."""
    pillar_score = sum(
        max(0.0, pillar_alignment.get(pid, 0.0)) * weight * 100
        for pid, weight in pillar_weights.items()
    )
    staleness_decay = max(0.0, base_importance) * (1 - math.pow(0.95, staleness_days))
    score = (
        base_importance
        + pillar_score
        + cluster_velocity * 0.2
        + source_authority * 10
        - staleness_decay
        + blocking_boost
    )
    return round(max(0.0, score), 3)


def _cluster_key(item: SourceItem) -> str:
    terms = _top_terms(item.text, limit=5)
    if not terms:
        return dedup_signature(item)
    return "-".join(terms[:4])


def _build_cluster(
    key: str,
    items: list[SourceItem],
    *,
    pillars: PillarConfig | None,
    now: datetime,
) -> TopicClusterDraft:
    representative = max(
        items, key=lambda item: _authority(item) + len(item.body) / 5000
    )
    text = "\n".join(item.text for item in items)
    top_terms = _top_terms(text, limit=8)
    source_authority = max(_authority(item) for item in items)
    unique_sources = len({item.source for item in items})
    cluster_velocity = len(items)
    base_importance = min(
        100.0,
        35.0 + unique_sources * 10.0 + len(items) * 5.0 + source_authority * 20.0,
    )
    alignment = pillars.estimate_alignment(text) if pillars is not None else {}
    weights = pillars.normalized_weights if pillars is not None else {}
    newest = max(item.ts for item in items)
    staleness_days = max(0.0, (now - newest).total_seconds() / 86400)
    final_priority = score_topic_priority(
        base_importance=base_importance,
        pillar_alignment=alignment,
        pillar_weights=weights,
        cluster_velocity=cluster_velocity,
        source_authority=source_authority,
        staleness_days=staleness_days,
    )
    return TopicClusterDraft(
        cluster_key=key,
        title=representative.title,
        summary=_summary(representative, len(items), unique_sources),
        items=items,
        top_terms=top_terms,
        base_importance=round(base_importance, 3),
        source_authority=round(source_authority, 3),
        cluster_velocity=cluster_velocity,
        pillar_alignment=alignment,
        final_priority=final_priority,
    )


def _top_terms(text: str, *, limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for raw in _WORD_RE.findall(text.lower()):
        token = raw.strip(".,:;()[]{}")
        if token in _STOPWORDS:
            continue
        counts[token] = counts.get(token, 0) + 1
    return [
        token
        for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            :limit
        ]
    ]


def _authority(item: SourceItem) -> float:
    return _AUTHORITY_BY_SOURCE_TYPE.get(item.source_type, 0.3)


def _summary(item: SourceItem, item_count: int, source_count: int) -> str:
    body = " ".join(item.body.split())
    snippet = body[:260].strip()
    return (
        f"{item.title} ({item_count} material(s), {source_count} source(s)). {snippet}"
    ).strip()

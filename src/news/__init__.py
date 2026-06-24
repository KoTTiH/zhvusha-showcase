"""News ingestion and topic pipeline contracts."""

from src.news.clustering import (
    TopicClusterDraft,
    cluster_source_items,
    score_topic_priority,
)
from src.news.dedup import DedupDecision, DedupResult, deduplicate_source_items
from src.news.models import (
    SourceItem,
    SourceTier,
    SourceType,
    canonical_url,
    content_hash,
    make_source_item_id,
)

__all__ = [
    "DedupDecision",
    "DedupResult",
    "SourceItem",
    "SourceTier",
    "SourceType",
    "TopicClusterDraft",
    "canonical_url",
    "cluster_source_items",
    "content_hash",
    "deduplicate_source_items",
    "make_source_item_id",
    "score_topic_priority",
]

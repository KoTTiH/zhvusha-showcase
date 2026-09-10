"""End-to-end news item processing helpers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.news.clustering import TopicClusterDraft, cluster_source_items
from src.news.dedup import DedupResult, deduplicate_source_items

if TYPE_CHECKING:
    from src.news.models import SourceItem
    from src.news.store import NewsStore
    from src.pillars import PillarConfig

NEWS_RAW_STREAM = "news:raw"


async def publish_raw_items(
    redis: Any,
    items: list[SourceItem],
    *,
    stream_name: str = NEWS_RAW_STREAM,
) -> int:
    """Publish normalized items into Redis Stream ``news:raw``."""
    count = 0
    for item in items:
        await redis.xadd(
            stream_name, item.to_stream_payload(), maxlen=10000, approximate=True
        )
        count += 1
    return count


async def process_source_items(
    items: list[SourceItem],
    *,
    store: NewsStore | None = None,
    pillars: PillarConfig | None = None,
    redis: Any | None = None,
    stream_name: str = NEWS_RAW_STREAM,
) -> tuple[DedupResult, list[TopicClusterDraft]]:
    """Dedup, cluster, optionally persist and publish source items."""
    deduped = await asyncio.to_thread(deduplicate_source_items, items)
    clusters = await asyncio.to_thread(
        cluster_source_items,
        deduped.unique_items,
        pillars=pillars,
    )
    if store is not None:
        await store.upsert_items(deduped.unique_items)
        await store.upsert_clusters(clusters)
    if redis is not None:
        await publish_raw_items(redis, deduped.unique_items, stream_name=stream_name)
    return deduped, clusters

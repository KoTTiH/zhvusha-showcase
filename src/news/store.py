"""Persistence helpers for the news pipeline."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from src.news.dedup import dedup_signature
from src.news.models import SourceItem, content_hash

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.news.clustering import TopicClusterDraft


class NewsStore:
    """Small SQL layer over ``news_items`` and ``topic_clusters`` tables."""

    def __init__(self, session_maker: async_sessionmaker[AsyncSession]) -> None:
        self._session_maker = session_maker

    async def upsert_items(self, items: list[SourceItem]) -> int:
        async with self._session_maker() as session:
            count = 0
            for item in items:
                await session.execute(_UPSERT_ITEM_SQL, _item_params(item))
                count += 1
            await session.commit()
            return count

    async def upsert_clusters(self, clusters: list[TopicClusterDraft]) -> int:
        async with self._session_maker() as session:
            count = 0
            for cluster in clusters:
                await session.execute(_UPSERT_CLUSTER_SQL, _cluster_params(cluster))
                count += 1
            await session.commit()
            return count


def _item_params(item: SourceItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "source": item.source,
        "source_tier": item.source_tier,
        "source_type": item.source_type,
        "url": item.url,
        "canonical_url": item.normalized_url,
        "title": item.title,
        "body": item.body,
        "lang": item.lang,
        "published_at": item.ts,
        "content_hash": content_hash(item.title, item.body),
        "dedup_signature": dedup_signature(item),
        "metadata": json.dumps(item.metadata, ensure_ascii=False),
    }


def _cluster_params(cluster: TopicClusterDraft) -> dict[str, Any]:
    item_ids = [item.id for item in cluster.items]
    return {
        "cluster_key": cluster.cluster_key,
        "title": cluster.title,
        "summary": cluster.summary,
        "top_terms": json.dumps(cluster.top_terms, ensure_ascii=False),
        "item_ids": json.dumps(item_ids, ensure_ascii=False),
        "base_importance": cluster.base_importance,
        "source_authority": cluster.source_authority,
        "cluster_velocity": cluster.cluster_velocity,
        "pillar_alignment": json.dumps(cluster.pillar_alignment, ensure_ascii=False),
        "final_priority": cluster.final_priority,
    }


_UPSERT_ITEM_SQL = text(
    """
    INSERT INTO news_items (
        id, source, source_tier, source_type, url, canonical_url, title, body,
        lang, published_at, content_hash, dedup_signature, metadata
    )
    VALUES (
        :id, :source, :source_tier, :source_type, :url, :canonical_url, :title,
        :body, :lang, :published_at, :content_hash, :dedup_signature,
        CAST(:metadata AS JSONB)
    )
    ON CONFLICT (id) DO UPDATE SET
        title = EXCLUDED.title,
        body = EXCLUDED.body,
        metadata = EXCLUDED.metadata,
        updated_at = now()
    """
)

_UPSERT_CLUSTER_SQL = text(
    """
    INSERT INTO topic_clusters (
        cluster_key, title, summary, top_terms, item_ids, base_importance,
        source_authority, cluster_velocity, pillar_alignment, final_priority
    )
    VALUES (
        :cluster_key, :title, :summary, CAST(:top_terms AS JSONB),
        CAST(:item_ids AS JSONB), :base_importance, :source_authority,
        :cluster_velocity, CAST(:pillar_alignment AS JSONB), :final_priority
    )
    ON CONFLICT (cluster_key) DO UPDATE SET
        title = EXCLUDED.title,
        summary = EXCLUDED.summary,
        top_terms = EXCLUDED.top_terms,
        item_ids = EXCLUDED.item_ids,
        base_importance = EXCLUDED.base_importance,
        source_authority = EXCLUDED.source_authority,
        cluster_velocity = EXCLUDED.cluster_velocity,
        pillar_alignment = EXCLUDED.pillar_alignment,
        final_priority = EXCLUDED.final_priority,
        updated_at = now()
    """
)

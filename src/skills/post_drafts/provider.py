"""Providers for post draft source topics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from src.skills.post_drafts.models import PostTopic, select_post_topics

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class EmptyPostTopicProvider:
    async def list_post_topics(
        self, *, limit: int = 10, min_money_alignment: float = 0.5
    ) -> list[PostTopic]:
        del limit, min_money_alignment
        return []


class SQLPostTopicProvider:
    """Reads topic backlog and selects money/channel aligned topics."""

    def __init__(self, session_maker: async_sessionmaker[AsyncSession]) -> None:
        self._session_maker = session_maker

    async def list_post_topics(
        self, *, limit: int = 10, min_money_alignment: float = 0.5
    ) -> list[PostTopic]:
        async with self._session_maker() as session:
            rows = [
                dict(row)
                for row in (
                    await session.execute(_LIST_TOPICS, {"limit": max(limit * 4, 20)})
                ).mappings()
            ]
        topics = [
            PostTopic(
                cluster_key=str(row["cluster_key"]),
                title=str(row["title"]),
                summary=str(row["summary"]),
                final_priority=float(row["final_priority"]),
                pillar_alignment=_as_float_map(row.get("pillar_alignment")),
            )
            for row in rows
        ]
        return select_post_topics(
            topics,
            min_money_alignment=min_money_alignment,
        )[:limit]


def _as_float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, raw in value.items():
        try:
            result[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return result


_LIST_TOPICS = text(
    """
    SELECT cluster_key, title, summary, final_priority, pillar_alignment
    FROM topic_clusters
    WHERE status = 'backlog'
    ORDER BY final_priority DESC
    LIMIT :limit
    """
)

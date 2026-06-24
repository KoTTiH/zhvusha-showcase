"""DB-backed provider for ranked morning digest topics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from src.skills.morning_digest.formatter import DigestTopic

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class EmptyMorningDigestProvider:
    async def list_topics(self, *, limit: int = 20) -> list[DigestTopic]:
        del limit
        return []


class SQLMorningDigestProvider:
    """Reads backlog topics from ``topic_clusters``."""

    def __init__(self, session_maker: async_sessionmaker[AsyncSession]) -> None:
        self._session_maker = session_maker

    async def list_topics(self, *, limit: int = 20) -> list[DigestTopic]:
        async with self._session_maker() as session:
            rows = (
                (await session.execute(_LIST_TOPICS, {"limit": limit})).mappings().all()
            )
        return [
            DigestTopic(
                cluster_key=str(row["cluster_key"]),
                title=str(row["title"]),
                summary=str(row["summary"]),
                final_priority=float(row["final_priority"]),
                pillar_alignment=_as_float_map(row.get("pillar_alignment")),
            )
            for row in rows
        ]


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

"""DB-backed topic provider for ``TopicToSpecSkill``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from sqlalchemy import text

from src.skills.spec_command.parser import SourceProvenance
from src.skills.topic_to_spec.models import TopicRecord

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class EmptyTopicProvider:
    async def get_topic(self, key: str | None = None) -> TopicRecord | None:
        del key
        return None


class SQLTopicProvider:
    """Reads ranked topic records from ``topic_clusters`` and ``news_items``."""

    def __init__(self, session_maker: async_sessionmaker[AsyncSession]) -> None:
        self._session_maker = session_maker

    async def get_topic(self, key: str | None = None) -> TopicRecord | None:
        async with self._session_maker() as session:
            row = (
                (
                    await session.execute(
                        _GET_BY_KEY if key else _GET_TOP,
                        {"key": key} if key else {},
                    )
                )
                .mappings()
                .first()
            )
            if row is None:
                return None
            item_ids = _as_str_list(row.get("item_ids"))
            source_rows: list[dict[str, Any]] = []
            if item_ids:
                source_rows = [
                    dict(row)
                    for row in (
                        await session.execute(_GET_SOURCES, {"ids": item_ids})
                    ).mappings()
                ]
        return TopicRecord(
            cluster_key=str(row["cluster_key"]),
            title=str(row["title"]),
            summary=str(row["summary"]),
            top_terms=tuple(_as_str_list(row.get("top_terms"))),
            final_priority=float(row["final_priority"]),
            pillar_alignment=_as_float_map(row.get("pillar_alignment")),
            source_provenance=tuple(_source_provenance(source_rows)),
        )


def _source_provenance(rows: list[dict[str, Any]]) -> list[SourceProvenance]:
    result: list[SourceProvenance] = []
    for row in rows[:5]:
        source_type, trust = _map_source(str(row.get("source_type") or "other"))
        title = str(row.get("title") or "Source item")
        result.append(
            SourceProvenance(
                url=str(row.get("url") or ""),
                source_type=source_type,
                trust_tier=trust,
                claim=title,
            )
        )
    return result


def _map_source(
    source_type: str,
) -> tuple[
    Literal[
        "official_docs",
        "paper",
        "github",
        "forum",
        "secondary_press",
        "local_repo",
        "kb",
        "code",
        "other",
    ],
    Literal["primary", "direct", "weak", "rejected"],
]:
    if source_type in {"official_docs", "paper", "github", "secondary_press"}:
        mapped_source = cast(
            "Literal['official_docs', 'paper', 'github', 'secondary_press']",
            source_type,
        )
        trust: Literal["primary", "direct"] = (
            "primary" if source_type in {"official_docs", "paper"} else "direct"
        )
        return mapped_source, trust
    if source_type in {"telegram", "youtube", "social", "blog"}:
        return "other", "weak"
    return "other", "weak"


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


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


_GET_TOP = text(
    """
    SELECT cluster_key, title, summary, top_terms, item_ids,
           final_priority, pillar_alignment
    FROM topic_clusters
    WHERE status = 'backlog'
    ORDER BY final_priority DESC
    LIMIT 1
    """
)

_GET_BY_KEY = text(
    """
    SELECT cluster_key, title, summary, top_terms, item_ids,
           final_priority, pillar_alignment
    FROM topic_clusters
    WHERE cluster_key = :key
    LIMIT 1
    """
)

_GET_SOURCES = text(
    """
    SELECT id, url, title, source_type
    FROM news_items
    WHERE id = ANY(:ids)
    """
)

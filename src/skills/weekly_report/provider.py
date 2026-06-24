"""Providers for weekly pillar reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from src.skills.post_drafts.store import list_draft_files
from src.skills.weekly_report.formatter import ReportTopic, WeeklyReportSnapshot

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class EmptyWeeklyReportProvider:
    async def build_snapshot(self, *, days: int = 7) -> WeeklyReportSnapshot:
        return WeeklyReportSnapshot(days=days)


class SQLWeeklyReportProvider:
    """Reads topic and archive metrics for a weekly report."""

    def __init__(
        self,
        *,
        session_maker: async_sessionmaker[AsyncSession],
        workspace_root: Path,
    ) -> None:
        self._session_maker = session_maker
        self._workspace_root = workspace_root

    async def build_snapshot(self, *, days: int = 7) -> WeeklyReportSnapshot:
        async with self._session_maker() as session:
            topic_rows = [
                dict(row)
                for row in (
                    await session.execute(_LIST_TOPICS, {"days": days})
                ).mappings()
            ]
            archive_row = (
                (await session.execute(_ARCHIVE_COUNTS, {"days": days}))
                .mappings()
                .first()
            )
        archive_counts = dict(archive_row) if archive_row is not None else {}
        return WeeklyReportSnapshot(
            topics=[
                ReportTopic(
                    cluster_key=str(row["cluster_key"]),
                    title=str(row["title"]),
                    summary=str(row["summary"]),
                    final_priority=float(row["final_priority"]),
                    pillar_alignment=_as_float_map(row.get("pillar_alignment")),
                )
                for row in topic_rows
            ],
            generated_drafts=len(list_draft_files(self._workspace_root)),
            archive_successes=int(archive_counts.get("successes") or 0),
            archive_failures=int(archive_counts.get("failures") or 0),
            days=days,
        )


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
    WHERE created_at >= now() - (:days * interval '1 day')
    ORDER BY final_priority DESC
    LIMIT 50
    """
)

_ARCHIVE_COUNTS = text(
    """
    SELECT
      COUNT(*) FILTER (WHERE status = 'committed') AS successes,
      COUNT(*) FILTER (WHERE status = 'failed') AS failures
    FROM archive_nodes
    WHERE created_at >= now() - (:days * interval '1 day')
    """
)

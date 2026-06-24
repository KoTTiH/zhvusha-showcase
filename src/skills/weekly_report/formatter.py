"""Formatter for weekly progress reports across Никита's pillars."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReportTopic:
    cluster_key: str
    title: str
    summary: str
    final_priority: float
    pillar_alignment: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class WeeklyReportSnapshot:
    topics: list[ReportTopic] = field(default_factory=list)
    generated_drafts: int = 0
    archive_successes: int = 0
    archive_failures: int = 0
    days: int = 7


_PILLAR_LABELS: dict[str, str] = {
    "self_improvement": "Самосовершенствование",
    "personality": "Характер",
    "money": "Деньги",
}


def format_weekly_report(snapshot: WeeklyReportSnapshot) -> str:
    topics = sorted(
        snapshot.topics, key=lambda topic: topic.final_priority, reverse=True
    )
    lines = [
        "📊 Недельный отчёт",
        "",
        f"Период: {snapshot.days} дней",
        f"Темы в движении: {len(topics)}",
        f"Черновики постов: {snapshot.generated_drafts}",
        (
            "Циклы самокодинга: "
            f"{snapshot.archive_successes} успешно · {snapshot.archive_failures} failed"
        ),
        "",
    ]
    for pillar_id, label in _PILLAR_LABELS.items():
        pillar_topics = _topics_for_pillar(topics, pillar_id)
        lines.append(f"**{label}**")
        if not pillar_topics:
            lines.append("  движения не видно")
            lines.append("")
            continue
        for topic in pillar_topics[:3]:
            score = topic.pillar_alignment.get(pillar_id, 0.0)
            lines.append(
                f"  • {topic.title} · priority {topic.final_priority:.1f} · "
                f"alignment {score:.2f}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def _topics_for_pillar(topics: list[ReportTopic], pillar_id: str) -> list[ReportTopic]:
    selected = [
        topic for topic in topics if topic.pillar_alignment.get(pillar_id, 0.0) > 0
    ]
    selected.sort(
        key=lambda topic: (
            topic.pillar_alignment.get(pillar_id, 0.0),
            topic.final_priority,
        ),
        reverse=True,
    )
    return selected

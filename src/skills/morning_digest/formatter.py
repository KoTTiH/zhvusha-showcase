"""Formatter for ranked morning topic digest."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DigestTopic:
    cluster_key: str
    title: str
    summary: str
    final_priority: float
    pillar_alignment: dict[str, float] = field(default_factory=dict)


def rank_digest_topics(topics: list[DigestTopic]) -> list[DigestTopic]:
    return sorted(topics, key=lambda topic: topic.final_priority, reverse=True)


def format_morning_digest(topics: list[DigestTopic]) -> str:
    ranked = rank_digest_topics(topics)
    if not ranked:
        return "🌅 Утренний обзор\n\nНовых тем в backlog нет."

    p0 = [topic for topic in ranked if topic.final_priority >= 80]
    p1 = [topic for topic in ranked if 60 <= topic.final_priority < 80]
    p2 = [topic for topic in ranked if 40 <= topic.final_priority < 60]

    lines = [
        "🌅 Утренний обзор",
        "",
        f"Темы: P0 {len(p0)} · P1 {len(p1)} · P2 {len(p2)} · всего {len(ranked)}",
        "",
    ]
    if p0:
        lines.extend(["🔴 P0", *_format_topic_lines(p0[:1]), ""])
    if p1:
        lines.extend(["🟡 P1", *_format_topic_lines(p1[:3]), ""])
    if p2:
        lines.extend(["🟢 P2", *_format_topic_lines(p2[:5]), ""])
    lines.extend(_format_action_lines(ranked))
    return "\n".join(lines).rstrip()


def _format_topic_lines(topics: list[DigestTopic]) -> list[str]:
    lines: list[str] = []
    for topic in topics:
        pillars = _format_pillars(topic.pillar_alignment)
        lines.append(
            f"- {topic.title} ({topic.final_priority:.1f})"
            f"{f' · {pillars}' if pillars else ''}\n"
            f"  {topic.summary[:180]}"
        )
    return lines


def _format_pillars(alignment: dict[str, float]) -> str:
    if not alignment:
        return ""
    top = sorted(alignment.items(), key=lambda item: item[1], reverse=True)[:2]
    return ", ".join(f"{key} {value:.2f}" for key, value in top if value > 0)


def _format_action_lines(topics: list[DigestTopic]) -> list[str]:
    if not topics:
        return []
    top = topics[0]
    return [
        "Готовые действия",
        f"- Превратить верхнюю тему `{top.cluster_key}` в spec/proposal.",
        "- Подготовить черновики постов по главным темам.",
        "- Проверить клиентские возможности и Kwork-пайплайн.",
        "- Собрать недельный отчёт по активным направлениям.",
    ]

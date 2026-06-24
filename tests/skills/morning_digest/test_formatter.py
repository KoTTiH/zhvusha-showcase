"""Morning digest ranking tests."""

from __future__ import annotations

from src.skills.morning_digest.formatter import DigestTopic, format_morning_digest


def test_digest_promotes_top_p0_topic_and_counts_backlog() -> None:
    text = format_morning_digest(
        [
            DigestTopic(
                cluster_key="p1",
                title="Useful post",
                summary="Good for channel.",
                final_priority=65,
            ),
            DigestTopic(
                cluster_key="p0",
                title="Codex safety update",
                summary="Important self-coding change.",
                final_priority=91,
                pillar_alignment={"self_improvement": 0.9},
            ),
        ]
    )

    assert "P0 1" in text
    assert "Codex safety update" in text
    assert text.index("Codex safety update") < text.index("Useful post")
    assert "Превратить верхнюю тему `p0` в spec/proposal." in text
    assert "Подготовить черновики постов" in text
    assert "Собрать недельный отчёт" in text
    assert "/topic_to_spec" not in text
    assert "/post_drafts" not in text
    assert "/weekly_report" not in text


def test_empty_digest_is_honest() -> None:
    assert "Новых тем" in format_morning_digest([])

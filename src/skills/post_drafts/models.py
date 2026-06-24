"""Models for channel post drafts derived from topic backlog."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_MONEY_KEYS = frozenset(
    {
        "money",
        "p3",
        "pillar_3",
        "3",
        "revenue",
        "clients",
        "external_presence",
    }
)


@dataclass(frozen=True)
class PostTopic:
    """Backlog topic suitable for channel draft generation."""

    cluster_key: str
    title: str
    summary: str
    final_priority: float
    pillar_alignment: dict[str, float] = field(default_factory=dict)

    @property
    def money_alignment(self) -> float:
        return pillar_score(self.pillar_alignment)


@dataclass(frozen=True)
class PostDraft:
    """Filesystem draft before Никита explicitly publishes it."""

    slug: str
    title: str
    source_cluster: str
    text: str
    created_at: datetime
    pillar_alignment: dict[str, float] = field(default_factory=dict)
    status: str = "draft"
    message_id: int | None = None
    visual: dict[str, Any] | None = None
    style: dict[str, Any] | None = None


def build_post_draft(topic: PostTopic, *, now: datetime | None = None) -> PostDraft:
    """Create a deterministic draft from a ranked topic."""
    from src.skills.post_drafts.style_check import check_post_style, clean_draft_text
    from src.skills.post_drafts.visual_plan import plan_visual_for_draft

    created_at = now or datetime.now(tz=UTC)
    slug = slugify(topic.title or topic.cluster_key)
    raw_text = (
        f"{topic.title}\n\n"
        f"{topic.summary.strip()}\n\n"
        "Что это меняет: это можно превратить в понятный пост для канала "
        "и использовать как внешний сигнал вокруг работы Жвуши.\n\n"
        "Пока это черновик. Перед публикацией Никита может отредактировать текст."
    )
    text, service_notes = clean_draft_text(raw_text)
    style = check_post_style(text, extra_notes=service_notes)
    visual = plan_visual_for_draft(
        title=topic.title,
        source_cluster=topic.cluster_key,
        text=text,
    )
    return PostDraft(
        slug=slug,
        title=topic.title,
        source_cluster=topic.cluster_key,
        text=text,
        created_at=created_at,
        pillar_alignment=dict(topic.pillar_alignment),
        visual=visual,
        style=style,
    )


def pillar_score(alignment: dict[str, float]) -> float:
    """Return pillar-3/money alignment across accepted key variants."""
    score = 0.0
    for key, value in alignment.items():
        normalized = key.strip().lower()
        if normalized in _MONEY_KEYS:
            score = max(score, float(value))
    return score


def select_post_topics(
    topics: list[PostTopic], *, min_money_alignment: float = 0.5
) -> list[PostTopic]:
    """Filter/rank topics that should become channel draft material."""
    selected = [
        topic for topic in topics if topic.money_alignment >= min_money_alignment
    ]
    selected.sort(
        key=lambda topic: (topic.money_alignment, topic.final_priority),
        reverse=True,
    )
    return selected


def slugify(value: str) -> str:
    raw = value.lower()
    slug = _SLUG_RE.sub("-", raw).strip("-")
    return slug[:72] or "post-draft"

"""Deterministic topic → candidate abstraction."""

from __future__ import annotations

import re
from typing import Literal

from src.skills.spec_command.parser import SourceProvenance
from src.skills.topic_to_spec.models import TopicCandidate, TopicRecord

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_CODEX_TERMS = {"codex", "hooks", "hook", "самокодинг", "self-coding", "agent"}
_TIER3_TERMS = {"safety", "personality", "protocol", "dispatcher", "tier3"}


def build_candidate_from_topic(topic: TopicRecord) -> TopicCandidate:
    """Build the first actionable candidate for a topic."""
    text = " ".join([topic.title, topic.summary, *topic.top_terms]).lower()
    tier = _classify_tier(text)
    kind: Literal["spec", "proposal", "post", "report"] = (
        "proposal" if tier == 3 else "spec"
    )
    slug = _slugify(topic.title)
    touched = _likely_files(text, tier)
    rationale = (
        f"Topic priority {topic.final_priority:.1f}; source-backed cluster "
        f"{topic.cluster_key} indicates a change worth staging now."
    )
    source_provenance = topic.source_provenance or (
        SourceProvenance(
            url=f"topic:{topic.cluster_key}",
            source_type="local_repo",
            trust_tier="direct",
            claim="Topic cluster exists in Жвуша news backlog.",
        ),
    )
    return TopicCandidate(
        kind=kind,
        tier=tier,
        slug=slug,
        what=_what(topic, tier),
        why_now=_why_now(topic),
        acceptance=(
            "Candidate carries rationale and source_provenance.",
            "Nikita can approve, defer or reject before any code is written.",
            "No existing behaviour is simplified unless Nikita explicitly approves it.",
        ),
        preserve_behavior=(
            "Жвушина накопленная личность, контекст, fallbacks, safety gates, "
            "tests and existing user flows stay intact.",
            "The topic must become an enrichment proposal/spec, not a cleanup "
            "that flattens behaviour.",
        ),
        allowed_simplifications=(),
        files_likely_touched=touched,
        risk=_risk(tier),
        rationale=rationale,
        pillar_attribution=topic.pillar_alignment,
        source_provenance=source_provenance,
    )


def render_candidate(candidate: TopicCandidate) -> str:
    """Human-readable candidate summary for Telegram."""
    sources = "\n".join(
        f"- {source.url} ({source.trust_tier}): {source.claim}"
        for source in candidate.source_provenance
    )
    files = ", ".join(candidate.files_likely_touched) or "уточнить после approve"
    pillars = ", ".join(
        f"{key}={value:.2f}" for key, value in candidate.pillar_attribution.items()
    )
    return (
        f"Кандидат: {candidate.kind} Tier {candidate.tier}\n"
        f"Slug: {candidate.slug}\n\n"
        f"Что: {candidate.what}\n"
        f"Почему сейчас: {candidate.why_now}\n"
        f"Риск: {candidate.risk}\n"
        f"Сохранить: {'; '.join(candidate.preserve_behavior)}\n"
        f"Разрешённые упрощения: "
        f"{'; '.join(candidate.allowed_simplifications) or 'нет'}\n"
        f"Вероятные файлы: {files}\n"
        f"Столпы: {pillars or 'нет явного совпадения'}\n\n"
        f"Источники:\n{sources}"
    )


def _classify_tier(text: str) -> int:
    if any(term in text for term in _TIER3_TERMS):
        return 3
    if any(term in text for term in _CODEX_TERMS):
        return 2
    return 1


def _likely_files(text: str, tier: int) -> tuple[str, ...]:
    if tier == 3:
        return ("proposals/",)
    if "codex" in text or "hook" in text:
        return (
            "src/skills/code_agent/",
            "src/skills/implement_spec/",
            "scripts/check_self_coding_mvp.py",
        )
    return ("tasks/", "src/skills/")


def _what(topic: TopicRecord, tier: int) -> str:
    if tier == 3:
        return f"Подготовить proposal по теме «{topic.title}»."
    return f"Подготовить spec по теме «{topic.title}»."


def _why_now(topic: TopicRecord) -> str:
    return (
        f"Тема в backlog имеет priority {topic.final_priority:.1f}: "
        f"{topic.summary[:220]}"
    )


def _risk(tier: int) -> str:
    if tier == 3:
        return "Архитектурный риск; только proposal, без автокодинга."
    if tier == 2:
        return "Затрагивает существующий capability module; нужен строгий test gate."
    return "Изолированное расширение; риск низкий."


def _slugify(title: str) -> str:
    raw = title.lower()
    raw = raw.replace("openai", "openai").replace("codex", "codex")
    slug = _SLUG_RE.sub("-", raw).strip("-")
    return slug[:64] or "topic-candidate"

"""Signal adapters for topic backlog candidates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.agent_runtime.topic_signals import TopicClusterReadySignal

if TYPE_CHECKING:
    from src.skills.topic_to_spec.models import TopicCandidate, TopicRecord


def build_topic_cluster_ready_signal(
    *,
    topic: TopicRecord,
    candidate: TopicCandidate,
) -> TopicClusterReadySignal:
    """Convert a topic candidate into a safe upstream planning signal."""
    payload = {
        "candidate_slug": candidate.slug,
        "risk": candidate.risk,
        "rationale": candidate.rationale,
        "files_likely_touched": ", ".join(candidate.files_likely_touched),
    }
    for index, source in enumerate(candidate.source_provenance[:8]):
        payload[f"source_url_{index}"] = source.url
        payload[f"source_claim_{index}"] = source.claim
        payload[f"source_trust_{index}"] = source.trust_tier
    return TopicClusterReadySignal(
        cluster_key=topic.cluster_key,
        title=topic.title,
        summary=topic.summary,
        final_priority=topic.final_priority,
        recommended_route=candidate.kind,
        tier=candidate.tier,
        requires_nikita=candidate.tier >= 3,
        safety_notes=_safety_notes(candidate),
        payload=payload,
    )


def _safety_notes(candidate: TopicCandidate) -> str:
    notes = [
        "Topic signal is an observation/candidate, not an execution command.",
        "Auto-publication is forbidden until a separate approval path exists.",
    ]
    if candidate.tier >= 3:
        notes.append("Tier 3 stays proposal-only until Никита approves.")
    return " ".join(notes)

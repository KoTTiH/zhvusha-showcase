"""Topic backlog upstream signal tests."""

from __future__ import annotations

from src.skills.spec_command.parser import SourceProvenance
from src.skills.topic_to_spec.builder import build_candidate_from_topic
from src.skills.topic_to_spec.models import TopicRecord


def test_topic_candidate_becomes_ready_signal_without_auto_publication() -> None:
    from src.skills.topic_to_spec.signals import build_topic_cluster_ready_signal

    topic = TopicRecord(
        cluster_key="codex-hooks",
        title="OpenAI Codex hooks update",
        summary="Official docs describe new hooks for coding agent lifecycle gates.",
        top_terms=("codex", "hooks", "agent"),
        final_priority=91.0,
        pillar_alignment={"self_improvement": 0.9},
        source_provenance=(
            SourceProvenance(
                url="https://developers.openai.com/codex/hooks",
                source_type="official_docs",
                trust_tier="primary",
                claim="Codex hooks changed.",
            ),
        ),
    )
    candidate = build_candidate_from_topic(topic)

    signal = build_topic_cluster_ready_signal(topic=topic, candidate=candidate)

    assert signal.signal_type == "topic_cluster_ready"
    assert signal.cluster_key == "codex-hooks"
    assert signal.recommended_route == "spec"
    assert signal.tier == 2
    assert signal.requires_approval is True
    assert signal.auto_publish_allowed is False
    assert signal.auto_execute_allowed is False
    assert signal.payload["source_url_0"] == "https://developers.openai.com/codex/hooks"
    assert "src/skills/code_agent/" in signal.payload["files_likely_touched"]


def test_tier3_topic_signal_routes_to_proposal_not_autocode() -> None:
    from src.skills.topic_to_spec.signals import build_topic_cluster_ready_signal

    topic = TopicRecord(
        cluster_key="safety-protocol",
        title="Safety protocol change",
        summary="A source suggests changing safety protocol.",
        top_terms=("safety", "protocol"),
        final_priority=88.0,
    )
    candidate = build_candidate_from_topic(topic)

    signal = build_topic_cluster_ready_signal(topic=topic, candidate=candidate)

    assert signal.recommended_route == "proposal"
    assert signal.tier == 3
    assert signal.requires_nikita is True
    assert signal.auto_execute_allowed is False
    assert "Tier 3" in signal.safety_notes

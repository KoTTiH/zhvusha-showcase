"""Topic-to-spec candidate builder tests."""

from __future__ import annotations

from src.skills.spec_command.parser import SourceProvenance
from src.skills.topic_to_spec.builder import build_candidate_from_topic
from src.skills.topic_to_spec.models import TopicRecord


def test_codex_hook_topic_becomes_tier2_spec_candidate() -> None:
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

    assert candidate.kind == "spec"
    assert candidate.tier == 2
    assert candidate.source_provenance[0].trust_tier == "primary"
    assert "src/skills/code_agent/" in candidate.files_likely_touched
    assert candidate.preserve_behavior
    assert candidate.allowed_simplifications == ()


def test_safety_topic_becomes_tier3_proposal_candidate() -> None:
    topic = TopicRecord(
        cluster_key="safety-protocol",
        title="Safety protocol change",
        summary="A source suggests changing safety protocol.",
        top_terms=("safety", "protocol"),
        final_priority=88.0,
    )

    candidate = build_candidate_from_topic(topic)

    assert candidate.kind == "proposal"
    assert candidate.tier == 3
    assert candidate.files_likely_touched == ("proposals/",)
    assert "enrichment" in " ".join(candidate.preserve_behavior)

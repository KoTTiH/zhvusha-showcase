"""MorningDigestSkill contract and behavior tests."""

from __future__ import annotations

from src.skills.base import BackgroundSkill
from src.skills.manifest import (
    load_manifest_for_skill_class,
    validate_manifest_matches_class,
)
from src.skills.morning_digest.formatter import DigestTopic
from src.skills.morning_digest.skill import MorningDigestSkill


class _Provider:
    async def list_topics(self, *, limit: int = 20) -> list[DigestTopic]:
        return [
            DigestTopic(
                cluster_key="codex",
                title="Codex hooks",
                summary="Update gate policy.",
                final_priority=90,
            )
        ][:limit]


def test_contract_manifest_matches_class() -> None:
    manifest = load_manifest_for_skill_class(MorningDigestSkill)
    validate_manifest_matches_class(manifest, MorningDigestSkill)
    assert issubclass(MorningDigestSkill, BackgroundSkill)


async def test_run_once_returns_digest() -> None:
    skill = MorningDigestSkill(topic_provider=_Provider())

    result = await skill.run_once()

    assert result.success
    assert "Codex hooks" in result.response

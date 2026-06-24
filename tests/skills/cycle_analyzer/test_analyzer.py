"""CycleAnalyzer archive writer tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

from src.archive.models import ArchiveNode, ArchiveStatus
from src.skills.cycle_analyzer.analyzer import CycleAnalyzer
from src.skills.spec_command.parser import SpecModel


def _spec() -> SpecModel:
    return SpecModel.model_validate(
        {
            "slug": "codex-hooks",
            "title": "Codex hooks",
            "created_at": "2026-05-07T12:00:00+00:00",
            "created_by": "zhvusha",
            "tier": 2,
            "goal": "Add a self-coding contract for Codex hooks and verify it.",
            "failing_test": {
                "file": "tests/skills/codex_hooks/test_contract.py",
                "name": "test_contract",
                "spec": "Codex hook contract is enforced.",
            },
            "whitelist_paths": ["src/skills/codex_hooks/skill.py"],
            "blast_radius": ["new skill only"],
            "rollback_path": ["git revert"],
            "preserve_behavior": [
                "Existing self-coding gates, tests and fallbacks stay intact.",
            ],
            "allowed_simplifications": [],
            "rationale": "Codex hooks affect self-coding gates.",
            "source_provenance": [
                {
                    "url": "local:codex-hooks",
                    "source_type": "local_repo",
                    "trust_tier": "direct",
                    "claim": "Codex hooks were observed.",
                }
            ],
            "chat_context": [
                "Никита: хочу, чтобы самокодинг хранил обсуждение.",
                "Жвуша: сохраняю его в spec и archive.",
            ],
            "status": "approved",
            "approved_at": "2026-05-07T12:10:00+00:00",
            "approved_by": "nikita",
        }
    )


def _spec_with_previous_attempt() -> SpecModel:
    raw = _spec().model_dump(mode="json")
    raw["previous_attempts"] = [
        {
            "archive_slug": "codex-hooks-failed",
            "status": "failed",
            "tier": 2,
            "commit_sha": None,
            "insight": "Previous Codex hooks attempt missed the gate audit.",
            "tests_summary": "formal gate failed",
        }
    ]
    return SpecModel.model_validate(raw)


async def test_record_success_writes_archive_files(tmp_path: Path) -> None:
    analyzer = CycleAnalyzer(
        archive_root=tmp_path,
        clock=lambda: datetime(2026, 5, 7, 13, 0, tzinfo=UTC),
    )

    node = await analyzer.record_success(
        spec=_spec(),
        spec_path=tmp_path / "tasks/2026-05-07-codex-hooks.yaml",
        branch_name="zhvusha/codex-hooks",
        commit_sha="abc1234567890",
        sdk_summary="Implemented Codex hooks.",
        backend="codex_cli",
    )

    assert node.status == ArchiveStatus.COMMITTED
    node_dir = tmp_path / node.slug
    assert (node_dir / "insight.md").exists()
    assert (node_dir / "spec_snapshot.yaml").exists()
    assert (node_dir / "chat_context.md").exists()
    assert (node_dir / "source_evidence.yaml").exists()
    assert "Implemented Codex hooks" in (node_dir / "insight.md").read_text()
    assert "самокодинг хранил обсуждение" in (node_dir / "chat_context.md").read_text()
    assert node.metadata["self_coding_actor"] == "zhvusha"
    assert node.metadata["commit_author_email"] == "zhvusha@local"


async def test_record_success_writes_parent_links_and_kb_insight(
    tmp_path: Path,
) -> None:
    knowledge_store = AsyncMock()
    knowledge_store.add_entry = AsyncMock(return_value=42)
    analyzer = CycleAnalyzer(
        archive_root=tmp_path,
        knowledge_store=knowledge_store,
        clock=lambda: datetime(2026, 5, 7, 13, 0, tzinfo=UTC),
    )

    node = await analyzer.record_success(
        spec=_spec_with_previous_attempt(),
        spec_path=tmp_path / "tasks/2026-05-07-codex-hooks.yaml",
        branch_name="zhvusha/codex-hooks",
        commit_sha="abc1234567890",
        sdk_summary="Implemented Codex hooks.",
        backend="codex_cli",
    )

    assert node.parent_slug == "codex-hooks-failed"
    assert node.metadata["parent_node_slugs"] == ["codex-hooks-failed"]
    parent_links = (tmp_path / node.slug / "parent_links.yaml").read_text()
    assert "codex-hooks-failed" in parent_links
    knowledge_store.add_entry.assert_awaited_once()
    call = knowledge_store.add_entry.await_args
    assert call.kwargs["category_path"] == "dev.cycle_insights"
    assert "codex-hooks-failed" in call.kwargs["metadata"]["parent_node_slugs"]


async def test_record_failure_writes_failed_node(tmp_path: Path) -> None:
    analyzer = CycleAnalyzer(
        archive_root=tmp_path,
        clock=lambda: datetime(2026, 5, 7, 13, 0, tzinfo=UTC),
    )

    node = await analyzer.record_failure(
        spec=_spec(),
        spec_path=tmp_path / "tasks/2026-05-07-codex-hooks.yaml",
        branch_name="zhvusha/codex-hooks",
        reason="Commit gate failed",
    )

    assert node.status == ArchiveStatus.FAILED
    assert "Commit gate failed" in (tmp_path / node.slug / "insight.md").read_text()
    assert (tmp_path / node.slug / "spec_snapshot.yaml").exists()


async def test_cycle_analyzer_triggers_skill_draft_on_repeating_pattern(
    tmp_path: Path,
) -> None:
    from src.skills.cycle_analyzer.analyzer import detect_skill_draft_opportunity

    nodes = [
        ArchiveNode(
            slug=f"html-failure-{idx}",
            spec_slug=f"html-{idx}",
            tier=2,
            status=ArchiveStatus.FAILED,
            created_at=datetime(2026, 5, 7, tzinfo=UTC),
            diff_summary="failed",
            tests_summary="pytest failed",
            insight="telegram html rendering failed in chat self coding",
            tags=["html-rendering", "failed"],
        )
        for idx in range(3)
    ]

    proposal = detect_skill_draft_opportunity(nodes, min_count=3)

    assert proposal is not None
    assert proposal.pattern == "html-rendering"
    assert proposal.source_archive_slugs == [
        "html-failure-0",
        "html-failure-1",
        "html-failure-2",
    ]

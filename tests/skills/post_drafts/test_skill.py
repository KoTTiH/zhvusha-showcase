"""PostDraftsSkill contract and behavior tests."""

from __future__ import annotations

from pathlib import Path

from src.skills.base import AgentContext, InlineSkill
from src.skills.manifest import (
    load_manifest_for_skill_class,
    validate_manifest_matches_class,
)
from src.skills.post_drafts.models import PostTopic
from src.skills.post_drafts.skill import PostDraftsSkill
from src.skills.post_drafts.store import list_draft_files, load_post_draft


class _Provider:
    async def list_post_topics(
        self, *, limit: int = 10, min_money_alignment: float = 0.5
    ) -> list[PostTopic]:
        del min_money_alignment
        return [
            PostTopic(
                cluster_key="ai-clients",
                title="AI clients",
                summary="New client-facing AI opportunity.",
                final_priority=80,
                pillar_alignment={"money": 0.9},
            )
        ][:limit]


def _ctx() -> AgentContext:
    return AgentContext(user_id=1, chat_id=1, mode="personal")


def test_contract_manifest_matches_class() -> None:
    manifest = load_manifest_for_skill_class(PostDraftsSkill)
    validate_manifest_matches_class(manifest, PostDraftsSkill)
    assert issubclass(PostDraftsSkill, InlineSkill)


async def test_can_handle_natural_draft_workflows(tmp_path: Path) -> None:
    skill = PostDraftsSkill(
        admin_user_id=1,
        workspace_root=tmp_path,
        topic_provider=_Provider(),
    )

    assert await skill.can_handle("покажи черновики", _ctx()) >= 0.9
    assert await skill.can_handle("покажи черновик ai-clients", _ctx()) >= 0.9
    assert await skill.can_handle("создай черновики постов", _ctx()) >= 0.9
    assert await skill.can_handle("обсудим черновики постов", _ctx()) == 0.0


async def test_generate_writes_workspace_draft(tmp_path: Path) -> None:
    skill = PostDraftsSkill(
        admin_user_id=1,
        workspace_root=tmp_path,
        topic_provider=_Provider(),
    )

    result = await skill.execute("/post_drafts generate 1", _ctx())
    files = list_draft_files(tmp_path)

    assert result.success
    assert len(files) == 1
    raw, body = load_post_draft(files[0])
    assert raw["status"] == "draft"
    assert raw["source_cluster"] == "ai-clients"
    assert "New client-facing AI opportunity" in body


async def test_list_and_show_drafts(tmp_path: Path) -> None:
    skill = PostDraftsSkill(
        admin_user_id=1,
        workspace_root=tmp_path,
        topic_provider=_Provider(),
    )
    await skill.execute("/post_drafts generate 1", _ctx())

    listed = await skill.execute("/post_drafts list", _ctx())
    shown = await skill.execute("/post_drafts show ai-clients", _ctx())

    assert "ai-clients" in listed.response
    assert "New client-facing AI opportunity" in shown.response


async def test_natural_list_and_show_drafts(tmp_path: Path) -> None:
    skill = PostDraftsSkill(
        admin_user_id=1,
        workspace_root=tmp_path,
        topic_provider=_Provider(),
    )
    await skill.execute("создай черновики постов 1", _ctx())

    listed = await skill.execute("покажи черновики", _ctx())
    shown = await skill.execute("покажи черновик ai-clients", _ctx())
    spaced = await skill.execute("покажи   черновик: ai-clients", _ctx())

    assert "ai-clients" in listed.response
    assert "New client-facing AI opportunity" in shown.response
    assert "New client-facing AI opportunity" in spaced.response

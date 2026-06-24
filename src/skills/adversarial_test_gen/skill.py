"""Inline skill that proposes anchored adversarial tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from src.skills.adversarial_test_gen.generator import generate_adversarial_tests
from src.skills.base import AgentContext, InlineSkill, SideEffect, SkillResult

if TYPE_CHECKING:
    from src.archive.store import ArchiveStore

_TRIGGER = "/adversarial_tests"


class AdversarialTestGenSkill(InlineSkill):
    """Render adversarial test drafts from archive failure patterns."""

    name: ClassVar[str] = "adversarial_test_gen"
    description: ClassVar[str] = "Generate anchored adversarial tests from archive"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"
    triggers: ClassVar[list[str]] = [_TRIGGER]
    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "low"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = "auto"
    side_effects: ClassVar[list[SideEffect]] = [SideEffect.READS_FILESYSTEM]
    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    def __init__(
        self, *, admin_user_id: int, archive_store: ArchiveStore | None
    ) -> None:
        self._admin_user_id = admin_user_id
        self._store = archive_store

    async def can_handle(self, message: str, context: AgentContext) -> float:
        if context.user_id != self._admin_user_id or context.mode != "personal":
            return 0.0
        text = message.strip().lower()
        return 1.0 if text == _TRIGGER or text.startswith(_TRIGGER + " ") else 0.0

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        del context
        query = message.strip()[len(_TRIGGER) :].strip() or "failed self-coding"
        if self._store is None:
            return SkillResult(
                success=False,
                response="Archive store не подключен: adversarial tests требуют archive_nodes.",
            )
        nodes = await self._store.lookup(query, top_k=10)
        drafts = generate_adversarial_tests(nodes)
        if not drafts:
            return SkillResult(
                success=True,
                response="Не нашла failed archive nodes для adversarial-тестов.",
                metadata={"count": 0},
            )
        lines = ["**Adversarial test drafts**"]
        for draft in drafts:
            lines.append(
                f"  • `{draft.test_file}::{draft.test_name}` ← `{draft.archive_slug}`"
            )
            lines.append(f"    {draft.rationale}")
        return SkillResult(
            success=True,
            response="\n".join(lines),
            metadata={"count": len(drafts)},
        )

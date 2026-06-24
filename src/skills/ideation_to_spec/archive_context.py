"""Archive context retrieval for Architect spec generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.skills.spec_command.parser import PreviousAttempt

if TYPE_CHECKING:
    from src.archive.models import ArchiveNode
    from src.archive.store import ArchiveStore


class ArchiveContextProvider:
    """Retrieve prior self-coding lessons for new specs."""

    def __init__(self, store: ArchiveStore | None) -> None:
        self._store = store

    async def previous_attempts(
        self, query: str, *, top_k: int = 3
    ) -> list[PreviousAttempt]:
        if self._store is None:
            return []
        nodes = await self._store.lookup(query, top_k=top_k)
        return [previous_attempt_from_node(node) for node in nodes]


def previous_attempt_from_node(node: ArchiveNode) -> PreviousAttempt:
    """Convert an archive node into the compact SpecModel lesson shape."""
    return PreviousAttempt(
        archive_slug=node.slug,
        status=node.status.value,
        tier=node.tier,
        commit_sha=node.commit_sha,
        insight=node.insight,
        tests_summary=node.tests_summary,
    )


def format_previous_attempts_for_prompt(attempts: list[PreviousAttempt]) -> str:
    """Render archive lessons for the Architect/Editor prompts."""
    if not attempts:
        return ""
    lines = [
        "### Previous self-coding attempts from archive_lookup",
        "Use these as binding lessons. Do not repeat failed paths; preserve "
        "successful patterns when they apply.",
    ]
    for attempt in attempts:
        commit = attempt.commit_sha[:12] if attempt.commit_sha else "no commit"
        lines.append(
            f"- `{attempt.archive_slug}` · {attempt.status} · "
            f"tier {attempt.tier} · {commit}\n"
            f"  insight: {attempt.insight}\n"
            f"  tests: {attempt.tests_summary or 'not recorded'}"
        )
    return "\n".join(lines)

"""Bot-level self-coding research wiring tests."""

from __future__ import annotations

from typing import Any


async def test_self_coding_research_service_uses_kb_and_codex_explorer() -> None:
    from src.bot.main import _build_self_coding_research_service
    from src.knowledge import SearchResult, SummaryEntry

    class Store:
        async def hybrid_search(
            self,
            query: str,
            *,
            category: str | None = None,
            tags: list[str] | None = None,
            limit: int = 10,
        ) -> list[Any]:
            del query, category, tags, limit
            return [
                SearchResult(
                    id=71,
                    title="Spec-first",
                    tags=[],
                    rrf_score=1.0,
                )
            ]

        async def get_summaries(self, entry_ids: list[int]) -> list[Any]:
            assert entry_ids == [71]
            return [
                SummaryEntry(
                    id=71,
                    title="Spec-first",
                    summary="Structured spec before implementation.",
                )
            ]

    explorer_calls: list[str] = []

    async def explorer(**kwargs: Any) -> str:
        explorer_calls.append(kwargs["user_prompt"])
        return "src/skills/ideation_to_spec/skill.py: Architect path."

    service = _build_self_coding_research_service(
        knowledge_store=Store(),
        explorer_runner=explorer,
    )

    result = await service.research(
        query="автоматический самокодинг",
        preset="api_integration",
        budget_seconds=10.0,
    )

    assert {citation.source for citation in result.citations} == {"kb", "code"}
    assert explorer_calls
    assert "автоматический самокодинг" in explorer_calls[0]

"""Research adapters for Architect self-coding context.

These tests cover the production callables injected into ``ResearchService``.
They keep the research leaf module dependency-free while ensuring the bot
wiring can use real KB and Codex Explorer sources instead of empty stubs.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.contract


class _FakeKnowledgeStore:
    def __init__(self) -> None:
        self.hybrid_queries: list[tuple[str, int]] = []
        self.summary_ids: list[list[int]] = []

    async def hybrid_search(
        self,
        query: str,
        *,
        category: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[Any]:
        del category, tags
        from src.knowledge import SearchResult

        self.hybrid_queries.append((query, limit))
        return [
            SearchResult(
                id=77,
                title="Spec-first workflow",
                tags=["self-coding"],
                rrf_score=0.42,
            )
        ]

    async def get_summaries(self, entry_ids: list[int]) -> list[Any]:
        from src.knowledge import SummaryEntry

        self.summary_ids.append(entry_ids)
        return [
            SummaryEntry(
                id=77,
                title="Spec-first workflow",
                summary="Всегда сначала research, затем spec, затем approval.",
            )
        ]


async def test_knowledge_adapter_returns_kb_citations() -> None:
    from src.skills.ideation_to_spec.research_adapters import KnowledgeStoreKBSearch

    store = _FakeKnowledgeStore()
    adapter = KnowledgeStoreKBSearch(store, limit=3)

    citations = await adapter("самокодинг spec workflow")

    assert store.hybrid_queries == [("самокодинг spec workflow", 3)]
    assert store.summary_ids == [[77]]
    assert len(citations) == 1
    assert citations[0].source == "kb"
    assert citations[0].ref == "kb:77"
    assert "Spec-first workflow" in citations[0].excerpt
    assert "research" in citations[0].excerpt


async def test_knowledge_adapter_without_store_is_empty() -> None:
    from src.skills.ideation_to_spec.research_adapters import KnowledgeStoreKBSearch

    adapter = KnowledgeStoreKBSearch(None)

    assert await adapter("anything") == []


async def test_codex_explorer_adapter_returns_code_citation() -> None:
    from src.skills.ideation_to_spec.research_adapters import CodexExplorerCodeSearch

    calls: list[dict[str, Any]] = []

    async def fake_explorer(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "src/skills/chat_self_coding/skill.py: /код uses ExplorerRunner."

    adapter = CodexExplorerCodeSearch(fake_explorer)

    citations = await adapter("сравни /код и автоматический self-coding")

    assert len(calls) == 1
    assert "read-only" in calls[0]["system_prompt"].lower()
    assert "сравни /код" in calls[0]["user_prompt"]
    assert calls[0]["progress_callback"] is None
    assert len(citations) == 1
    assert citations[0].source == "code"
    assert citations[0].ref == "codex-explorer:repo"
    assert "/код uses ExplorerRunner" in citations[0].excerpt


async def test_codex_explorer_adapter_empty_response_is_empty() -> None:
    from src.skills.ideation_to_spec.research_adapters import CodexExplorerCodeSearch

    async def fake_explorer(**kwargs: Any) -> str:
        del kwargs
        return "   "

    adapter = CodexExplorerCodeSearch(fake_explorer)

    assert await adapter("x") == []


async def test_agent_runtime_research_adapter_maps_capsule_to_citations() -> None:
    from src.agent_runtime.models import ContextCapsule, Finding, FindingStatus
    from src.skills.ideation_to_spec.research_adapters import (
        AgentRuntimeResearchSource,
    )

    calls: list[dict[str, str]] = []

    async def fake_runtime_source(*, source_id: str, query: str) -> ContextCapsule:
        calls.append({"source_id": source_id, "query": query})
        return ContextCapsule(
            summary="Прочитала Telegram context read-only.",
            processed_context="Тоша писал Никите про @Anroxa2748.",
            findings=(
                Finding(
                    claim="Telegram history contains explicit username @Anroxa2748.",
                    status=FindingStatus.CONFIRMED,
                    confidence=0.9,
                    evidence=("telegram://personal/dialog/42",),
                ),
            ),
            sources=("telegram://personal/dialog/42",),
        )

    adapter = AgentRuntimeResearchSource(
        source_id="telegram_mcp_readonly",
        runner=fake_runtime_source,
    )

    citations = await adapter("найди telegram username из истории")

    assert calls == [
        {
            "source_id": "telegram_mcp_readonly",
            "query": "найди telegram username из истории",
        }
    ]
    assert citations[0].source == "telegram_mcp"
    assert citations[0].ref == "telegram://personal/dialog/42"
    assert "explicit username" in citations[0].excerpt


async def test_agent_runtime_research_adapter_unavailable_returns_unknown() -> None:
    from src.skills.ideation_to_spec.research_adapters import (
        AgentRuntimeResearchSource,
        AgentRuntimeResearchUnavailableError,
    )

    async def unavailable(*, source_id: str, query: str) -> Any:
        del source_id, query
        raise AgentRuntimeResearchUnavailableError("telegram_mcp_read disabled")

    adapter = AgentRuntimeResearchSource(
        source_id="telegram_mcp_readonly",
        runner=unavailable,
    )

    citations = await adapter("нужен telegram context")

    assert len(citations) == 1
    assert citations[0].source == "unknown"
    assert citations[0].ref == "unavailable:telegram_mcp_readonly"
    assert "UNKNOWN" in citations[0].excerpt
    assert "telegram_mcp_read disabled" in citations[0].excerpt


async def test_agent_runtime_research_selector_uses_capability_graph_status() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityGraph,
        CapabilityKind,
        CapabilityNode,
        CapabilityStatus,
    )
    from src.agent_runtime.models import ContextCapsule, Finding, FindingStatus
    from src.skills.ideation_to_spec.research_adapters import (
        build_agent_runtime_research_sources_from_graph,
    )

    graph = CapabilityGraph(
        capabilities=(
            CapabilityNode(
                id="agent_profile.telegram_mcp.personal_readonly",
                label="telegram_mcp.personal_readonly",
                kind=CapabilityKind.AGENT_PROFILE,
                status=CapabilityStatus.AVAILABLE,
                reason="worker registered",
                profile_id="telegram_mcp.personal_readonly",
            ),
            CapabilityNode(
                id="agent_profile.web_research.readonly",
                label="web_research.readonly",
                kind=CapabilityKind.AGENT_PROFILE,
                status=CapabilityStatus.DEGRADED,
                reason="browser unavailable",
                profile_id="web_research.readonly",
            ),
        )
    )
    calls: list[str] = []

    async def fake_runtime_source(*, source_id: str, query: str) -> ContextCapsule:
        calls.append(f"{source_id}:{query}")
        return ContextCapsule(
            summary="runtime source ok",
            findings=(
                Finding(
                    claim=f"{source_id} evidence",
                    status=FindingStatus.CONFIRMED,
                    evidence=(f"runtime://{source_id}",),
                ),
            ),
        )

    sources = build_agent_runtime_research_sources_from_graph(
        graph=graph,
        runner=fake_runtime_source,
        source_ids=("telegram_mcp_readonly", "web_research", "news_topics"),
    )

    telegram, web, news = sources
    telegram_citations = await telegram("query")
    web_citations = await web("query")
    news_citations = await news("query")

    assert calls == ["telegram_mcp_readonly:query"]
    assert telegram_citations[0].source == "telegram_mcp"
    assert web_citations[0].source == "unknown"
    assert web_citations[0].ref == "unavailable:web_research"
    assert "browser unavailable" in web_citations[0].excerpt
    assert news_citations[0].ref == "unavailable:news_topics"
    assert "missing capability graph node" in news_citations[0].excerpt


async def test_runtime_research_runner_starts_bounded_readonly_job() -> None:
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import (
        AgentJob,
        AgentJobStatus,
        ContextCapsule,
        ContextPack,
    )
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore
    from src.skills.ideation_to_spec.research_adapters import (
        RuntimeAgentRuntimeResearchRunner,
    )

    class Worker:
        name = "web_research"

        async def run(
            self,
            *,
            job: AgentJob,
            context_pack: ContextPack,
        ) -> ContextCapsule:
            assert job.profile.id == "web_research.readonly"
            assert "self_coding_research_runtime_source" in context_pack.constraints
            assert context_pack.metadata["runtime_research_source_id"] == "web_research"
            return ContextCapsule(
                summary="runtime research done",
                processed_context=context_pack.user_request,
                sources=("runtime://web",),
            )

        async def cancel(self, job_id: str) -> bool:
            del job_id
            return False

    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={"web_research": Worker()},
    )
    runner = RuntimeAgentRuntimeResearchRunner(
        runtime=runtime,
        owner_user_id=123,
        chat_id=123,
    )

    capsule = await runner(source_id="web_research", query="найди источники")

    assert capsule.summary == "runtime research done"
    jobs = await runtime.store.list_by_status((AgentJobStatus.DONE,))
    assert len(jobs) == 1
    assert jobs[0].kind == "web_research"
    assert jobs[0].context_pack.user_request.startswith(
        "Read-only research source `web_research`"
    )


async def test_runtime_research_runner_rejects_unwired_source() -> None:
    import pytest
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore
    from src.skills.ideation_to_spec.research_adapters import (
        AgentRuntimeResearchUnavailableError,
        RuntimeAgentRuntimeResearchRunner,
    )

    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={},
    )
    runner = RuntimeAgentRuntimeResearchRunner(
        runtime=runtime,
        owner_user_id=123,
        chat_id=123,
    )

    with pytest.raises(AgentRuntimeResearchUnavailableError, match="not wired"):
        await runner(source_id="news_topics", query="topic backlog")

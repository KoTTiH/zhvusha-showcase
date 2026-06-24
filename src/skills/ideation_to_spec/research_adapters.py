"""Production research callables for ``IdeationToSpecSkill``.

``src.research`` stays a leaf module and only knows about injected callables.
This module lives in the skill layer and adapts real Knowledge Base and Codex
Explorer capabilities into the narrow ``ResearchService`` citation contract.
"""

from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING, Protocol

import structlog

from src.agent_runtime.capability_graph import CapabilityGraph, CapabilityStatus
from src.agent_runtime.models import AgentJobStatus, ContextPack
from src.agent_runtime.profiles import (
    SOURCE_COMPARE_READONLY,
    TELEGRAM_MCP_PERSONAL_READONLY,
    WEB_RESEARCH_READONLY,
)
from src.research import Citation, CitationSource

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.agent_runtime.models import ContextCapsule
    from src.agent_runtime.runtime import AgentRuntime
    from src.knowledge import KnowledgeStoreProtocol

logger = structlog.get_logger()

_MAX_EXCERPT_CHARS = 1400

_CODE_EXPLORER_SYSTEM_PROMPT = """\
You are Жвуша's read-only Codex Explorer for Architect research.

Inspect the local ZHVUSHA repository before a self-coding spec is drafted.
Use only read-only actions. Do not write files, create specs, run
implementations, commit, or change branches.

Return concise evidence for the Architect: real files, symbols, tests,
contracts, and risk notes. Prefer concrete paths and short explanations. If
the local repo does not contain relevant evidence, say that directly.
"""


class ExplorerRunner(Protocol):
    """Callable shape shared with chat-mode Explorer wiring."""

    async def __call__(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        progress_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> str: ...


class AgentRuntimeResearchRunner(Protocol):
    """Callable that runs one bounded read-only Agent Runtime source."""

    async def __call__(self, *, source_id: str, query: str) -> ContextCapsule: ...


class AgentRuntimeResearchUnavailableError(RuntimeError):
    """A runtime source is unavailable or degraded and must become an unknown."""


_DEFAULT_RUNTIME_SOURCE_NODES: dict[str, str] = {
    "telegram_mcp_readonly": "agent_profile.telegram_mcp.personal_readonly",
    "web_research": "agent_profile.web_research.readonly",
    "workspace": "agent_profile.source_compare.readonly",
    "news_topics": "agent_profile.news_topics.readonly",
}


class KnowledgeStoreKBSearch:
    """Adapt ``KnowledgeStoreProtocol`` search results to research citations."""

    def __init__(
        self,
        store: KnowledgeStoreProtocol | None,
        *,
        limit: int = 5,
    ) -> None:
        self._store = store
        self._limit = limit

    async def __call__(self, query: str) -> list[Citation]:
        if self._store is None:
            return []
        try:
            results = await self._store.hybrid_search(query, limit=self._limit)
            if not results:
                return []
            ids = [result.id for result in results]
            summaries = await self._store.get_summaries(ids)
        except Exception as exc:
            logger.warning("ideation_kb_research_failed", exc_info=True)
            raise RuntimeError("knowledge research failed") from exc

        summary_by_id = {summary.id: summary for summary in summaries}
        citations: list[Citation] = []
        for result in results:
            summary = summary_by_id.get(result.id)
            title = summary.title if summary is not None else result.title
            body = summary.summary if summary is not None else None
            excerpt = _compact_excerpt(f"{title}: {body or result.title}")
            citations.append(
                Citation(
                    source="kb",
                    ref=f"kb:{result.id}",
                    excerpt=excerpt,
                )
            )
        return citations


class CodexExplorerCodeSearch:
    """Run the shared read-only Codex Explorer as Architect code search."""

    def __init__(self, runner: ExplorerRunner) -> None:
        self._runner = runner

    async def __call__(self, query: str) -> list[Citation]:
        try:
            text = await self._runner(
                system_prompt=_CODE_EXPLORER_SYSTEM_PROMPT,
                user_prompt=_build_code_research_prompt(query),
                progress_callback=None,
            )
        except Exception as exc:
            logger.warning("ideation_code_research_failed", exc_info=True)
            raise RuntimeError("code research failed") from exc

        excerpt = _compact_excerpt(text)
        if not excerpt:
            return []
        return [
            Citation(
                source="code",
                ref="codex-explorer:repo",
                excerpt=excerpt,
            )
        ]


class AgentRuntimeResearchSource:
    """Adapt read-only Agent Runtime ContextCapsules to research citations."""

    def __init__(
        self,
        *,
        source_id: str,
        runner: AgentRuntimeResearchRunner,
        max_citations: int = 5,
    ) -> None:
        self._source_id = source_id
        self._runner = runner
        self._max_citations = max(1, max_citations)

    async def __call__(self, query: str) -> list[Citation]:
        try:
            capsule = await self._runner(source_id=self._source_id, query=query)
        except AgentRuntimeResearchUnavailableError as exc:
            return [_unknown_citation(self._source_id, str(exc))]
        except Exception as exc:
            logger.warning(
                "ideation_agent_runtime_research_failed",
                source_id=self._source_id,
                exc_info=True,
            )
            raise RuntimeError("agent runtime research failed") from exc
        return _capsule_to_citations(
            source_id=self._source_id,
            capsule=capsule,
            max_citations=self._max_citations,
        )


class RuntimeAgentRuntimeResearchRunner:
    """Run production Agent Runtime read-only jobs as research sources."""

    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        owner_user_id: int,
        chat_id: int,
    ) -> None:
        self._runtime = runtime
        self._owner_user_id = owner_user_id
        self._chat_id = chat_id

    async def __call__(self, *, source_id: str, query: str) -> ContextCapsule:
        try:
            kind, profile = _RUNTIME_SOURCE_RUN_CONFIG[source_id]
        except KeyError as exc:
            raise AgentRuntimeResearchUnavailableError(
                f"runtime source {source_id!r} is not wired"
            ) from exc

        fingerprint = _runtime_research_fingerprint(source_id=source_id, query=query)
        job = await self._runtime.create_job(
            owner_user_id=self._owner_user_id,
            chat_id=self._chat_id,
            source_message_id=f"runtime-research:{source_id}:{fingerprint[-24:]}",
            fingerprint=fingerprint,
            kind=kind,
            profile=profile,
            context_pack=ContextPack(
                user_request=(
                    f"Read-only research source `{source_id}` for self-coding spec:\n"
                    f"{query}"
                ),
                constraints=(
                    "self_coding_research_runtime_source",
                    "read_only",
                    "do_not_execute_side_effects",
                ),
                metadata={"runtime_research_source_id": source_id},
            ),
            status=AgentJobStatus.QUEUED,
        )
        if job.status is AgentJobStatus.DONE and job.result is not None:
            return job.result

        completed = await self._runtime.start(job.id)
        if completed.result is None:
            raise RuntimeError(completed.error or "runtime research job failed")
        return completed.result


def build_agent_runtime_research_sources_from_graph(
    *,
    graph: CapabilityGraph,
    runner: AgentRuntimeResearchRunner,
    source_ids: tuple[str, ...] = (
        "telegram_mcp_readonly",
        "web_research",
        "workspace",
        "news_topics",
    ),
    source_nodes: dict[str, str] | None = None,
) -> tuple[AgentRuntimeResearchSource, ...]:
    """Build runtime research sources using CapabilityGraph availability."""
    node_by_source = source_nodes or _DEFAULT_RUNTIME_SOURCE_NODES
    sources: list[AgentRuntimeResearchSource] = []
    for source_id in source_ids:
        node_id = node_by_source.get(source_id, "")
        reason = _source_unavailable_reason(graph=graph, node_id=node_id)
        selected_runner = (
            runner if not reason else _UnavailableAgentRuntimeResearchRunner(reason)
        )
        sources.append(
            AgentRuntimeResearchSource(source_id=source_id, runner=selected_runner)
        )
    return tuple(sources)


class _UnavailableAgentRuntimeResearchRunner:
    def __init__(self, reason: str) -> None:
        self._reason = reason

    async def __call__(self, *, source_id: str, query: str) -> ContextCapsule:
        del source_id, query
        raise AgentRuntimeResearchUnavailableError(self._reason)


def _build_code_research_prompt(query: str) -> str:
    return (
        "Research query for an upcoming self-coding spec:\n"
        f"{query}\n\n"
        "Find the relevant local code paths and tests. Report:\n"
        "- existing entrypoints and call chain;\n"
        "- files/symbols that look likely to change;\n"
        "- tests or hidden contracts the Architect must preserve;\n"
        "- any no-downgrade risks or unclear decisions."
    )


_RUNTIME_SOURCE_RUN_CONFIG = {
    "telegram_mcp_readonly": ("telegram_mcp", TELEGRAM_MCP_PERSONAL_READONLY),
    "web_research": ("web_research", WEB_RESEARCH_READONLY),
    "workspace": ("source_compare", SOURCE_COMPARE_READONLY),
}


def _runtime_research_fingerprint(*, source_id: str, query: str) -> str:
    digest = sha256(f"{source_id}:{query}".encode()).hexdigest()
    return f"runtime-research:{digest[:24]}"


def _compact_excerpt(text: str) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) <= _MAX_EXCERPT_CHARS:
        return compact
    return compact[: _MAX_EXCERPT_CHARS - 3].rstrip() + "..."


def _capsule_to_citations(
    *,
    source_id: str,
    capsule: ContextCapsule,
    max_citations: int,
) -> list[Citation]:
    source = _citation_source(source_id)
    citations: list[Citation] = []
    for finding in capsule.findings:
        ref = finding.evidence[0] if finding.evidence else f"agent-runtime:{source_id}"
        citations.append(
            Citation(
                source=source,
                ref=ref,
                excerpt=_compact_excerpt(f"{finding.status.value}: {finding.claim}"),
            )
        )
        if len(citations) >= max_citations:
            return citations

    context_excerpt = _compact_excerpt(capsule.processed_context or capsule.summary)
    for ref in capsule.sources:
        citations.append(Citation(source=source, ref=ref, excerpt=context_excerpt))
        if len(citations) >= max_citations:
            return citations

    if not citations and context_excerpt:
        citations.append(
            Citation(
                source=source,
                ref=f"agent-runtime:{source_id}",
                excerpt=context_excerpt,
            )
        )
    return citations[:max_citations]


def _unknown_citation(source_id: str, reason: str) -> Citation:
    return Citation(
        source="unknown",
        ref=f"unavailable:{source_id}",
        excerpt=_compact_excerpt(f"UNKNOWN: {reason}"),
    )


def _source_unavailable_reason(*, graph: CapabilityGraph, node_id: str) -> str:
    if not node_id:
        return "missing runtime source mapping"
    try:
        node = graph.require(node_id)
    except KeyError:
        return f"missing capability graph node: {node_id}"
    if node.status is CapabilityStatus.AVAILABLE:
        return ""
    return f"{node.id} is {node.status.value}: {node.reason}"


def _citation_source(source_id: str) -> CitationSource:
    if source_id.startswith("telegram_mcp"):
        return "telegram_mcp"
    if source_id.startswith("web"):
        return "web"
    if source_id.startswith("workspace"):
        return "workspace"
    if source_id.startswith("news") or source_id.startswith("topic"):
        return "news"
    return "unknown"

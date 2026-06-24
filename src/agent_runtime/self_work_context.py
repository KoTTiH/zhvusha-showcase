"""Bounded self-work context capsule for autonomous planning."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from src.agent_runtime.capability_graph import (
    CapabilityGraph,
    CapabilityNode,
    CapabilityStatus,
)
from src.agent_runtime.models import (
    AgentJob,
    ContextCapsule,
    Finding,
    FindingStatus,
)
from src.agent_runtime.tools import SIDE_EFFECT_CAPABILITIES
from src.agent_runtime.topic_signals import TopicClusterReadySignal  # noqa: TC001

_GAP_STATUSES = {
    CapabilityStatus.CONFIGURED_ONLY,
    CapabilityStatus.DEGRADED,
    CapabilityStatus.ORPHANED,
}
_MAX_SECTION_ITEMS = 12
_MAX_TEXT_CHARS = 12000
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?:session|token|secret|password|api[_-]?key|authorization)"
    r"[\w-]*\s*[:=]\s*['\"]?[^'\"\s,;]+"
)
_SECRET_VALUE_RE = re.compile(
    r"(?i)\b\S*(?:super_secret|session_string|api_key|access_token|sk-[a-z0-9])\S*\b"
)


class SelfWorkMcpHealth(BaseModel):
    """Secret-free MCP health observation for autonomous planning."""

    name: str
    status: CapabilityStatus
    reason: str = ""


class SelfWorkRuntimeSignal(BaseModel):
    """Daemon/runtime signal reduced to planning-safe text."""

    source: str
    signal_type: str
    summary: str
    priority: Literal["critical", "normal", "background"] = "normal"


class SelfWorkContextSnapshot(BaseModel):
    """All read-only inputs used to build the self-work context capsule."""

    capability_graph: CapabilityGraph
    open_task_paths: tuple[str, ...] = ()
    recent_failed_runs: tuple[str, ...] = ()
    news_topic_backlog: tuple[str, ...] = ()
    mcp_health: tuple[SelfWorkMcpHealth, ...] = ()
    pending_jobs: tuple[AgentJob, ...] = ()
    daemon_signals: tuple[SelfWorkRuntimeSignal, ...] = ()
    topic_signals: tuple[TopicClusterReadySignal, ...] = ()


class SelfWorkContextCapsuleBuilder:
    """Create a bounded ContextCapsule for autonomous self-work planning."""

    def build(self, snapshot: SelfWorkContextSnapshot) -> ContextCapsule:
        """Convert read-only runtime state into a planner-visible capsule."""
        gaps = tuple(
            node
            for node in snapshot.capability_graph.capabilities
            if node.status in _GAP_STATUSES and node.manager_visible
        )
        safe_candidates = tuple(node for node in gaps if not _is_side_effect_node(node))
        side_effect_gaps = tuple(node for node in gaps if _is_side_effect_node(node))

        sections = (
            _render_nodes("Capability gaps", gaps),
            _render_items("Open task YAMLs", snapshot.open_task_paths),
            _render_items("Recent failed runs", snapshot.recent_failed_runs),
            _render_items("News/topic backlog", snapshot.news_topic_backlog),
            _render_mcp(snapshot.mcp_health),
            _render_jobs(snapshot.pending_jobs),
            _render_signals(snapshot.daemon_signals),
            _render_topic_signals(snapshot.topic_signals),
            _render_nodes("Safe spec candidates", safe_candidates),
        )
        processed_context = _truncate(
            "\n\n".join(section for section in sections if section).strip()
        )
        artifacts = (
            *(
                f"safe_spec_candidate:{node.id}"
                for node in safe_candidates[:_MAX_SECTION_ITEMS]
            ),
            *(
                f"topic_signal:{signal.cluster_key}"
                for signal in snapshot.topic_signals[:_MAX_SECTION_ITEMS]
            ),
        )
        findings = (
            *(
                Finding(
                    claim=f"Capability gap: {node.id}",
                    status=FindingStatus.PARTIAL,
                    confidence=0.85,
                    evidence=_node_evidence(node),
                )
                for node in gaps[:_MAX_SECTION_ITEMS]
            ),
            *(
                Finding(
                    claim=f"Topic signal ready: {signal.cluster_key}",
                    status=FindingStatus.PARTIAL,
                    confidence=0.8,
                    evidence=_topic_signal_evidence(signal),
                )
                for signal in snapshot.topic_signals[:_MAX_SECTION_ITEMS]
            ),
        )
        next_actions = (
            *(
                (
                    "Create bounded spec candidate for "
                    f"{node.id}; keep side effects disabled and cite this capsule."
                )
                for node in safe_candidates[:_MAX_SECTION_ITEMS]
            ),
            *(
                (
                    "Ask Никита before side-effect work for "
                    f"{node.id}; keep it approval-gated."
                )
                for node in side_effect_gaps[:_MAX_SECTION_ITEMS]
            ),
            *(
                _topic_signal_next_action(signal)
                for signal in snapshot.topic_signals[:_MAX_SECTION_ITEMS]
            ),
        )
        sources = (
            "capability_graph:self_work_context",
            *(f"task:{path}" for path in snapshot.open_task_paths[:_MAX_SECTION_ITEMS]),
        )
        summary = (
            "Self-work context capsule: "
            f"{len(gaps)} capability gaps, "
            f"{len(safe_candidates)} safe spec candidates, "
            f"{len(snapshot.open_task_paths)} open tasks, "
            f"{len(snapshot.topic_signals)} topic signals."
        )
        return ContextCapsule(
            summary=summary,
            processed_context=processed_context,
            findings=findings,
            sources=tuple(_sanitize_text(source) for source in sources),
            artifacts=tuple(_sanitize_text(artifact) for artifact in artifacts),
            next_actions=tuple(_sanitize_text(action) for action in next_actions),
            markdown_report=processed_context,
        )


def sanitize_self_work_text(text: str) -> str:
    """Remove likely secret/session material from self-work planner context."""
    return _sanitize_text(text)


def _render_nodes(title: str, nodes: tuple[CapabilityNode, ...]) -> str:
    if not nodes:
        return ""
    lines = [f"## {title}"]
    for node in nodes[:_MAX_SECTION_ITEMS]:
        reason = f" - {node.reason}" if node.reason else ""
        lines.append(f"- {node.id}: {node.status.value}{reason}")
    return _sanitize_text("\n".join(lines))


def _render_items(title: str, items: tuple[str, ...]) -> str:
    if not items:
        return ""
    lines = [f"## {title}", *(f"- {item}" for item in items[:_MAX_SECTION_ITEMS])]
    return _sanitize_text("\n".join(lines))


def _render_mcp(items: tuple[SelfWorkMcpHealth, ...]) -> str:
    if not items:
        return ""
    lines = ["## MCP health"]
    for item in items[:_MAX_SECTION_ITEMS]:
        reason = f" - {item.reason}" if item.reason else ""
        lines.append(f"- {item.name}: {item.status.value}{reason}")
    return _sanitize_text("\n".join(lines))


def _render_jobs(jobs: tuple[AgentJob, ...]) -> str:
    if not jobs:
        return ""
    lines = ["## Pending Agent Runtime jobs"]
    for job in jobs[:_MAX_SECTION_ITEMS]:
        lines.append(f"- {job.id}: {job.kind}/{job.status.value}/{job.profile.id}")
    return _sanitize_text("\n".join(lines))


def _render_signals(signals: tuple[SelfWorkRuntimeSignal, ...]) -> str:
    if not signals:
        return ""
    lines = ["## Daemon/runtime signals"]
    for signal in signals[:_MAX_SECTION_ITEMS]:
        lines.append(
            f"- {signal.source}/{signal.signal_type}/{signal.priority}: "
            f"{signal.summary}"
        )
    return _sanitize_text("\n".join(lines))


def _render_topic_signals(signals: tuple[TopicClusterReadySignal, ...]) -> str:
    if not signals:
        return ""
    lines = ["## Topic signals"]
    for signal in signals[:_MAX_SECTION_ITEMS]:
        lines.append(
            f"- {signal.cluster_key}: {signal.title}; "
            f"recommended_route={signal.recommended_route}; "
            f"tier={signal.tier}; "
            f"requires_approval={signal.requires_approval}; "
            f"auto_publish_allowed={signal.auto_publish_allowed}; "
            f"auto_execute_allowed={signal.auto_execute_allowed}"
        )
        if signal.summary:
            lines.append(f"  summary: {signal.summary}")
    return _sanitize_text("\n".join(lines))


def _node_evidence(node: CapabilityNode) -> tuple[str, ...]:
    evidence = (*node.evidence, node.reason)
    return tuple(_sanitize_text(item) for item in evidence if item)


def _topic_signal_evidence(signal: TopicClusterReadySignal) -> tuple[str, ...]:
    evidence = (
        f"topic_cluster_ready:{signal.cluster_key}",
        f"route:{signal.recommended_route}",
        f"tier:{signal.tier}",
        signal.payload.get("source_url_0", ""),
    )
    return tuple(_sanitize_text(item) for item in evidence if item)


def _topic_signal_next_action(signal: TopicClusterReadySignal) -> str:
    if signal.tier >= 3 or signal.requires_nikita:
        return (
            "Ask Никита before turning topic "
            f"{signal.cluster_key} into a {signal.recommended_route} proposal."
        )
    return (
        "Create bounded "
        f"{signal.recommended_route} candidate for topic {signal.cluster_key}; "
        "keep execution disabled and cite this capsule."
    )


def _is_side_effect_node(node: CapabilityNode) -> bool:
    side_effects = set(SIDE_EFFECT_CAPABILITIES)
    return (
        node.capability_id in side_effects
        or node.label in side_effects
        or any(f".{capability}" in node.id for capability in side_effects)
    )


def _sanitize_text(text: str) -> str:
    cleaned = _SECRET_ASSIGNMENT_RE.sub("<redacted-secret>", text)
    return _SECRET_VALUE_RE.sub("<redacted-secret>", cleaned)


def _truncate(text: str) -> str:
    cleaned = _sanitize_text(text.strip())
    if len(cleaned) <= _MAX_TEXT_CHARS:
        return cleaned
    return cleaned[:_MAX_TEXT_CHARS].rstrip() + "\n... [truncated]"

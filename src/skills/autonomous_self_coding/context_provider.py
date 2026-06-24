"""Production provider for autonomous self-work Context Capsules."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol

from src.agent_runtime.capability_graph import CapabilityGraph, CapabilityKind
from src.agent_runtime.models import AgentJob, AgentJobStatus, ContextCapsule
from src.agent_runtime.self_work_context import (
    SelfWorkContextCapsuleBuilder,
    SelfWorkContextSnapshot,
    SelfWorkMcpHealth,
    SelfWorkRuntimeSignal,
)
from src.skills.spec_command.store import list_spec_files, load_spec_raw

if TYPE_CHECKING:
    from pathlib import Path

    from src.agent_runtime.storage import AgentJobStore
    from src.agent_runtime.topic_signals import TopicClusterReadySignal
    from src.skills.topic_to_spec.models import TopicProvider


_ACTIVE_RUNTIME_STATUSES: tuple[AgentJobStatus, ...] = (
    AgentJobStatus.QUEUED,
    AgentJobStatus.AWAITING_INPUT,
    AgentJobStatus.RUNNING,
    AgentJobStatus.WAITING_USER,
    AgentJobStatus.NEEDS_REVIEW,
)
_TERMINAL_SPEC_STATUSES = {"done", "rejected"}
_MAX_OPEN_TASKS = 12
_MAX_RECENT_FAILED_RUNS = 8
_COMPLETED_RUN_STATUSES = {"completed", "done", "success"}


class _AgentRuntimeWithStore(Protocol):
    store: AgentJobStore


class SelfWorkDaemonSignalProvider(Protocol):
    """Read-only daemon signal source for autonomous planning context."""

    async def recent_signals(self) -> tuple[SelfWorkRuntimeSignal, ...]: ...


class RuntimeSelfWorkContextProvider:
    """Build self-work capsules from production runtime state."""

    def __init__(
        self,
        *,
        capability_graph: CapabilityGraph,
        tasks_dir: Path,
        runtime: _AgentRuntimeWithStore | None = None,
        self_coding_summary_dir: Path | None = None,
        topic_provider: TopicProvider | None = None,
        daemon_signal_provider: SelfWorkDaemonSignalProvider | None = None,
        builder: SelfWorkContextCapsuleBuilder | None = None,
    ) -> None:
        self._capability_graph = capability_graph
        self._tasks_dir = tasks_dir
        self._runtime = runtime
        self._self_coding_summary_dir = (
            self_coding_summary_dir
            if self_coding_summary_dir is not None
            else tasks_dir.parent / "self_coding_summaries" / "agent_runtime"
        )
        self._topic_provider = topic_provider
        self._daemon_signal_provider = daemon_signal_provider
        self._builder = builder or SelfWorkContextCapsuleBuilder()

    async def build_self_work_context_capsule(self) -> ContextCapsule:
        """Build a bounded, secret-free capsule for the autonomous planner."""

        snapshot = SelfWorkContextSnapshot(
            capability_graph=self._capability_graph,
            open_task_paths=_open_task_paths(self._tasks_dir),
            recent_failed_runs=_recent_failed_runs(self._self_coding_summary_dir),
            topic_signals=await self._topic_signals(),
            daemon_signals=await self._daemon_signals(),
            mcp_health=_mcp_health(self._capability_graph),
            pending_jobs=await self._pending_jobs(),
        )
        return self._builder.build(snapshot)

    async def _pending_jobs(self) -> tuple[AgentJob, ...]:
        if self._runtime is None:
            return ()
        return tuple(await self._runtime.store.list_by_status(_ACTIVE_RUNTIME_STATUSES))

    async def _topic_signals(self) -> tuple[TopicClusterReadySignal, ...]:
        if self._topic_provider is None:
            return ()
        from src.skills.topic_to_spec.builder import build_candidate_from_topic
        from src.skills.topic_to_spec.signals import build_topic_cluster_ready_signal

        try:
            topic = await self._topic_provider.get_topic()
        except Exception:
            return ()
        if topic is None:
            return ()
        candidate = build_candidate_from_topic(topic)
        return (build_topic_cluster_ready_signal(topic=topic, candidate=candidate),)

    async def _daemon_signals(self) -> tuple[SelfWorkRuntimeSignal, ...]:
        if self._daemon_signal_provider is None:
            return ()
        try:
            return await self._daemon_signal_provider.recent_signals()
        except Exception:
            return ()


def _open_task_paths(tasks_dir: Path) -> tuple[str, ...]:
    paths: list[str] = []
    for path in reversed(list_spec_files(tasks_dir)):
        if len(paths) >= _MAX_OPEN_TASKS:
            break
        if _is_open_task(path):
            paths.append(_display_task_path(tasks_dir, path))
    return tuple(paths)


def _is_open_task(path: Path) -> bool:
    try:
        raw = load_spec_raw(path)
    except Exception:
        return True
    status = str(raw.get("status", "")).strip()
    return status not in _TERMINAL_SPEC_STATUSES


def _display_task_path(tasks_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(tasks_dir.parent).as_posix()
    except ValueError:
        return path.name


def _mcp_health(graph: CapabilityGraph) -> tuple[SelfWorkMcpHealth, ...]:
    items: list[SelfWorkMcpHealth] = []
    for node in graph.capabilities:
        if node.kind is CapabilityKind.MCP_SERVER or "telegram_mcp" in node.id:
            items.append(
                SelfWorkMcpHealth(
                    name=node.id,
                    status=node.status,
                    reason=node.reason,
                )
            )
    return tuple(items)


def _recent_failed_runs(summary_dir: Path) -> tuple[str, ...]:
    if not summary_dir.exists():
        return ()
    items: list[str] = []
    for path in sorted(
        summary_dir.glob("*.json"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    ):
        if len(items) >= _MAX_RECENT_FAILED_RUNS:
            break
        item = _recent_failed_run_from_summary(path)
        if item:
            items.append(item)
    return tuple(items)


def _recent_failed_run_from_summary(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return f"{path.name}: unreadable_summary"
    if not isinstance(payload, dict):
        return ""
    status = str(payload.get("status") or "").strip().lower()
    quality_warnings = _summary_items(payload.get("quality_warnings"))
    if status in _COMPLETED_RUN_STATUSES and not quality_warnings:
        return ""
    slug = str(payload.get("slug") or path.stem).strip()
    summary = str(payload.get("summary") or "").strip()
    parts = [f"{slug}: status={status or 'unknown'}"]
    if quality_warnings:
        parts.append(f"quality_warnings={', '.join(quality_warnings)}")
    if summary:
        parts.append(summary)
    return "; ".join(parts)


def _summary_items(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, list | tuple):
        return tuple(item for raw in value if (item := str(raw).strip()))
    item = str(value).strip()
    return (item,) if item else ()

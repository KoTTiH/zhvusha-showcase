"""Human-readable rendering for Agent Runtime status and results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from src.agent_runtime.telegram_inbound import (
    render_personal_telegram_inbound_capsule_for_chat,
)

if TYPE_CHECKING:
    from src.agent_runtime.models import AgentEvent, AgentJob, ContextCapsule


class AgentResultRenderer(Protocol):
    """Render a completed job capsule for a user-facing delivery surface."""

    def __call__(self, job: AgentJob, capsule: ContextCapsule) -> str: ...


class AgentResultRendererRegistry:
    """Profile-aware result renderer registry."""

    def __init__(
        self,
        *,
        renderers: dict[str, AgentResultRenderer] | None = None,
        default_renderer: AgentResultRenderer | None = None,
    ) -> None:
        self._renderers = dict(renderers or {})
        self._default_renderer = default_renderer or render_capsule_for_job

    def register(self, key: str, renderer: AgentResultRenderer) -> None:
        """Register renderer by profile id or job kind."""
        self._renderers[key] = renderer

    def render(self, job: AgentJob, capsule: ContextCapsule) -> str:
        """Render using exact profile id, then job kind, then default."""
        renderer = self._renderers.get(job.profile.id) or self._renderers.get(job.kind)
        if renderer is None:
            renderer = self._default_renderer
        return renderer(job, capsule)


def render_capsule_for_chat(capsule: ContextCapsule) -> str:
    """Render a Context Capsule for Zhvusha's final Telegram answer."""
    lines = [capsule.summary.strip()]

    if capsule.findings:
        lines.extend(("", "Что проверено:"))
        for finding in capsule.findings:
            confidence = round(finding.confidence * 100)
            lines.append(f"- {finding.claim} [{finding.status.value}, {confidence}%]")
            if finding.evidence:
                lines.append(f"  evidence: {', '.join(finding.evidence)}")

    if capsule.sources:
        lines.extend(("", "Источники:"))
        lines.extend(f"- {source}" for source in capsule.sources)

    if capsule.next_actions:
        lines.extend(("", "Дальше:"))
        lines.extend(f"- {action}" for action in capsule.next_actions)

    return "\n".join(line for line in lines if line or lines)


def render_capsule_for_job(job: AgentJob, capsule: ContextCapsule) -> str:
    """Render a Context Capsule with profile-aware defaults."""
    if job.profile.id.startswith("self_coding"):
        return _markdown_first(capsule)
    if job.kind in {"source_compare", "web_research"} and (
        capsule.findings or capsule.sources or capsule.next_actions
    ):
        return render_capsule_for_chat(capsule)
    return _markdown_first(capsule)


def build_builtin_result_renderer_registry() -> AgentResultRendererRegistry:
    """Build renderers for built-in Agent Runtime profiles."""
    registry = AgentResultRendererRegistry()
    registry.register("personal_telegram.inbound_readonly", _render_personal_inbound)
    registry.register("personal_telegram_inbound", _render_personal_inbound)
    registry.register("self_coding.implementation", _render_markdown_first)
    registry.register("source_compare.readonly", _render_context_capsule)
    registry.register("web_research.readonly", _render_context_capsule)
    return registry


def _render_personal_inbound(job: AgentJob, capsule: ContextCapsule) -> str:
    del job
    return render_personal_telegram_inbound_capsule_for_chat(capsule)


def _render_markdown_first(job: AgentJob, capsule: ContextCapsule) -> str:
    del job
    return _markdown_first(capsule)


def _render_context_capsule(job: AgentJob, capsule: ContextCapsule) -> str:
    del job
    if capsule.findings or capsule.sources or capsule.next_actions:
        return render_capsule_for_chat(capsule)
    return _markdown_first(capsule)


def _markdown_first(capsule: ContextCapsule) -> str:
    return capsule.markdown_report or capsule.processed_context or capsule.summary


def render_job_status(job: AgentJob, events: tuple[AgentEvent, ...]) -> str:
    """Render concise job status from curated AgentEvent records."""
    lines = [
        f"{job.kind} · {job.profile.id}",
        f"status: {job.status.value}",
    ]
    if job.error:
        lines.append(f"reason: {job.error}")
    if events:
        lines.append("progress:")
        lines.extend(f"- {event.message}" for event in events[-3:] if event.message)
    return "\n".join(lines)

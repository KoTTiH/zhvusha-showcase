"""Bounded worker for channel visual artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.agent_runtime.models import ContextCapsule, Finding, FindingStatus
from src.agent_runtime.tools import ToolDeniedError, ToolNotFoundError
from src.llm.protocols import LLMError
from src.skills.post_drafts.visual_plan import plan_visual_for_draft

if TYPE_CHECKING:
    from src.agent_runtime.models import AgentJob, ContextPack
    from src.agent_runtime.tools import ToolGateway


class ChannelVisualWorkerBackend:
    """Prepare visual artifacts without publish/write tools."""

    name = "channel_visual"
    _ARCHITECTURE_HINTS = (
        "agent runtime",
        "self-coding",
        "самокод",
        "архитектур",
        "codex",
        "жвуш",
        "runtime",
    )
    _CONTEXT_FILES = (
        ("read_project_file", "docs/agent-runtime-principles.md"),
        ("read_project_file", "src/agent_runtime/profiles.py"),
        ("read_project_file", "src/agent_runtime/runtime.py"),
        ("read_project_file", "src/agent_runtime/tools.py"),
        ("read_workspace_file", "agent_runtime/architecture.md"),
        ("read_workspace_file", "agent_runtime/README.md"),
        ("read_workspace_file", "personality/public_identity.md"),
        ("read_workspace_file", "personality/public_core.md"),
    )

    def __init__(self, *, tool_gateway: ToolGateway) -> None:
        self._tool_gateway = tool_gateway

    async def run(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
    ) -> ContextCapsule:
        text = "\n".join((context_pack.user_request, *context_pack.chat_context))
        plan = plan_visual_for_draft(
            title=context_pack.user_request,
            source_cluster="channel_visual",
            text=text,
        )
        if plan["intent"] == "generated":
            plan = await self._enrich_generated_plan(
                job=job,
                plan=plan,
                text=text,
            )
            return await self._generated_capsule(job=job, plan=plan)
        if plan["intent"] == "source_screenshot":
            return await self._source_capsule(job=job, plan=plan)
        return _plan_only_capsule(plan)

    async def cancel(self, job_id: str) -> bool:
        del job_id
        return False

    async def _enrich_generated_plan(
        self,
        *,
        job: AgentJob,
        plan: dict[str, Any],
        text: str,
    ) -> dict[str, Any]:
        if not _looks_architectural(text):
            return plan
        snippets: list[str] = []
        for item in self._CONTEXT_FILES:
            tool_name, path = _context_source(item)
            try:
                content = await self._tool_gateway.execute(
                    job.profile,
                    tool_name,
                    {"path": path},
                )
            except (ToolDeniedError, ToolNotFoundError, FileNotFoundError, OSError):
                continue
            snippet = _compact_context(str(content))
            if snippet:
                snippets.append(f"{path}: {snippet}")
        if not snippets:
            return plan
        code_context = "\n".join(snippets)[:1200]
        enriched = dict(plan)
        enriched["code_context"] = code_context
        enriched["prompt"] = (
            f"{plan.get('prompt', '')}\n\n"
            "Опирайся на эту фактуру из файлов системы, но не рисуй код, "
            "терминал, приватные пути или внутренние логи:\n"
            f"{code_context}"
        ).strip()
        return enriched

    async def _generated_capsule(
        self,
        *,
        job: AgentJob,
        plan: dict[str, Any],
    ) -> ContextCapsule:
        try:
            artifact = await self._tool_gateway.execute(
                job.profile,
                "channel_visual_generate_image",
                {
                    "prompt": plan.get("prompt", ""),
                    "caption": plan.get("caption", ""),
                },
            )
        except (ToolDeniedError, ToolNotFoundError, LLMError, ValueError) as exc:
            return await self._fallback_card_capsule(
                job=job,
                plan=plan,
                body=plan.get("prompt", ""),
                reason=str(exc),
            )
        metadata = {**plan, **dict(artifact)}
        return _artifact_capsule(metadata)

    async def _source_capsule(
        self,
        *,
        job: AgentJob,
        plan: dict[str, Any],
    ) -> ContextCapsule:
        try:
            artifact_path = await self._tool_gateway.execute(
                job.profile,
                "browser_screenshot_url",
                {"url": plan.get("source_url", "")},
            )
        except (ToolDeniedError, ToolNotFoundError, ValueError, RuntimeError) as exc:
            return await self._fallback_card_capsule(
                job=job,
                plan=plan,
                body=plan.get("summary", ""),
                reason=str(exc),
            )
        metadata = {**plan, "status": "ready", "asset_path": str(artifact_path)}
        return _artifact_capsule(metadata)

    async def _fallback_card_capsule(
        self,
        *,
        job: AgentJob,
        plan: dict[str, Any],
        body: object,
        reason: str,
    ) -> ContextCapsule:
        fallback_plan = {
            **plan,
            "intent": "source_card"
            if str(plan.get("intent", "")).startswith("source")
            else "generated_card",
            "status": "fallback",
            "fallback_reason": reason,
        }
        try:
            artifact = await self._tool_gateway.execute(
                job.profile,
                "channel_visual_generate_card",
                {
                    "title": plan.get("title") or job.context_pack.user_request,
                    "body": body,
                    "source_url": plan.get("source_url", ""),
                    "caption": plan.get("caption", ""),
                },
            )
        except (ToolDeniedError, ToolNotFoundError, ValueError, RuntimeError) as exc:
            degraded = {
                **fallback_plan,
                "status": "degraded",
                "degraded_reason": f"{reason}; fallback failed: {exc}",
            }
            return _plan_only_capsule(degraded)
        metadata = {
            **fallback_plan,
            **dict(artifact),
            "status": "ready",
        }
        return _artifact_capsule(metadata)


def _plan_only_capsule(plan: dict[str, Any]) -> ContextCapsule:
    status = str(plan.get("status", "planned"))
    intent = str(plan.get("intent", "none"))
    reason = str(plan.get("denial_reason") or plan.get("degraded_reason") or "")
    return ContextCapsule(
        summary=f"Visual intent: {intent} ({status}).",
        processed_context=str(plan),
        findings=(
            Finding(
                claim=f"Visual plan resolved as {intent}.",
                status=FindingStatus.REJECTED
                if intent == "denied"
                else FindingStatus.PARTIAL,
                confidence=0.85,
                evidence=(reason,) if reason else (),
            ),
        ),
        markdown_report=f"visual: {intent}\nstatus: {status}\n{reason}".strip(),
    )


def _artifact_capsule(metadata: dict[str, Any]) -> ContextCapsule:
    asset_path = str(metadata.get("asset_path", ""))
    intent = str(metadata.get("intent", ""))
    return ContextCapsule(
        summary=f"Visual artifact ready: {intent}.",
        processed_context=str(metadata),
        findings=(
            Finding(
                claim=f"Visual artifact prepared for {intent}.",
                status=FindingStatus.CONFIRMED,
                confidence=0.9,
                evidence=(asset_path,) if asset_path else (),
            ),
        ),
        artifacts=(asset_path,) if asset_path else (),
        markdown_report=f"visual: {intent}\nstatus: ready\nartifact: {asset_path}",
    )


def _looks_architectural(text: str) -> bool:
    lowered = text.casefold()
    return any(
        hint in lowered for hint in ChannelVisualWorkerBackend._ARCHITECTURE_HINTS
    )


def _compact_context(text: str) -> str:
    compact = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return compact[:500]


def _context_source(item: tuple[str, str]) -> tuple[str, str]:
    return item

"""User-facing browser workflow draft command through Agent Runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

from src.agent_runtime.models import ContextPack, InvocationProfile
from src.agent_runtime.profiles import BROWSER_WORKFLOW_DRAFT
from src.skills.base import AgentContext, InlineSkill, SideEffect, SkillResult

if TYPE_CHECKING:
    from src.agent_runtime.models import AgentJob, ContextCapsule


BROWSER_WORKFLOW_DRAFT_PREFIX = "/browser_workflow_draft"
_MAX_OBSERVATION_TEXT_CHARS = 4_000


class BrowserWorkflowRuntime(Protocol):
    """Minimal AgentRuntime contract used by BrowserWorkflowDraftSkill."""

    async def create_job(
        self,
        *,
        owner_user_id: int,
        chat_id: int,
        source_message_id: str,
        fingerprint: str,
        kind: str,
        profile: InvocationProfile,
        context_pack: ContextPack,
    ) -> AgentJob: ...

    async def start(self, job_id: str) -> AgentJob: ...


@dataclass(frozen=True)
class BrowserWorkflowDraftSkillConfig:
    """Stable routing/configuration for browser workflow drafts."""

    max_payload_chars: int = 8_000


class BrowserWorkflowDraftSkill(InlineSkill):
    """Prepare a browser form draft without submit through Agent Runtime."""

    name: ClassVar[str] = "browser_workflow_draft"
    description: ClassVar[str] = "Browser workflow draft через Agent Runtime"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"
    triggers: ClassVar[list[str]] = [BROWSER_WORKFLOW_DRAFT_PREFIX]
    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "low"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = "auto"
    side_effects: ClassVar[list[SideEffect]] = [
        SideEffect.NETWORK_IO_EXTERNAL,
        SideEffect.DELEGATES_TO_OTHER_AGENT,
        SideEffect.WRITES_FILESYSTEM,
    ]
    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    def __init__(
        self,
        *,
        admin_user_id: int,
        runtime: BrowserWorkflowRuntime,
        profile: InvocationProfile = BROWSER_WORKFLOW_DRAFT,
        config: BrowserWorkflowDraftSkillConfig | None = None,
    ) -> None:
        self._admin_user_id = admin_user_id
        self._runtime = runtime
        self._profile = profile
        self._config = config or BrowserWorkflowDraftSkillConfig()

    async def can_handle(self, message: str, context: AgentContext) -> float:
        if context.mode != "personal" or context.user_id != self._admin_user_id:
            return 0.0
        if message.strip().startswith(BROWSER_WORKFLOW_DRAFT_PREFIX):
            return 0.94
        return 0.0

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        payload, error = _payload_from_message(
            message,
            max_chars=self._config.max_payload_chars,
        )
        if error:
            return _missing_input_result(error)

        job = await self._runtime.create_job(
            owner_user_id=context.user_id,
            chat_id=context.chat_id or context.user_id,
            source_message_id=str(context.message_id or ""),
            fingerprint=_fingerprint(context=context, payload=payload),
            kind="browser_workflow_draft",
            profile=self._profile,
            context_pack=ContextPack(
                user_request=message.strip(),
                constraints=(
                    "browser_workflow_draft_only",
                    "do_not_submit_forms",
                    "do_not_login",
                    "do_not_purchase_publish_delete_or_send",
                    "browser_submit_requires_separate_capability_and_approval",
                ),
                metadata={
                    "source": str(context.metadata.get("source", "")),
                    "interface": str(context.metadata.get("interface", "")),
                    "skill": self.name,
                    "browser_workflow_payload": json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ),
        )
        completed = await self._runtime.start(job.id)
        if completed.result is None:
            reason = completed.error or "browser workflow job did not return a capsule"
            return SkillResult(
                success=False,
                response="",
                metadata={
                    "skill_name": self.name,
                    "agent_job_id": completed.id,
                    "agent_profile": self._profile.id,
                    "requires_zhvusha_response": True,
                    "body_observation": {
                        "event": "browser_workflow_draft_failed",
                        "source": self.name,
                        "reason": reason,
                        "agent_job_id": completed.id,
                        "agent_profile": self._profile.id,
                        "instruction": (
                            "Объясни пользователю, что browser workflow draft "
                            "не завершился, без сырого runtime traceback."
                        ),
                    },
                },
            )

        return SkillResult(
            success=bool(completed.result.artifacts),
            response="",
            metadata={
                "skill_name": self.name,
                "agent_job_id": completed.id,
                "agent_profile": self._profile.id,
                "artifacts": tuple(completed.result.artifacts),
                "sources": tuple(completed.result.sources),
                "requires_zhvusha_response": True,
                "body_observation": _body_observation_from_capsule(
                    payload=payload,
                    capsule=completed.result,
                    agent_job_id=completed.id,
                    agent_profile=self._profile.id,
                ),
            },
        )


def _payload_from_message(
    message: str,
    *,
    max_chars: int,
) -> tuple[dict[str, Any], str]:
    text = message.strip()
    if not text.startswith(BROWSER_WORKFLOW_DRAFT_PREFIX):
        return {}, "Команда должна начинаться с /browser_workflow_draft."
    raw = text.removeprefix(BROWSER_WORKFLOW_DRAFT_PREFIX).strip()
    if not raw:
        return {}, "Нужен JSON payload с read_url/action_url/fields."
    if len(raw) > max_chars:
        return {}, "JSON payload слишком большой."
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, f"JSON payload не разобран: {exc.msg}."
    if not isinstance(payload, dict):
        return {}, "JSON payload должен быть объектом."
    read_url = str(
        payload.get("read_url") or payload.get("form_url") or payload.get("url") or ""
    ).strip()
    action_url = str(payload.get("action_url") or payload.get("url") or "").strip()
    if not read_url and not action_url:
        return {}, "Нужен read_url, form_url, action_url или url."
    return payload, ""


def _missing_input_result(reason: str) -> SkillResult:
    return SkillResult(
        success=False,
        response="",
        metadata={
            "skill_name": BrowserWorkflowDraftSkill.name,
            "requires_zhvusha_response": True,
            "body_observation": {
                "event": "missing_required_input",
                "source": BrowserWorkflowDraftSkill.name,
                "reason": reason,
                "example": (
                    '/browser_workflow_draft {"read_url":"https://example.com/form",'
                    '"action_url":"https://example.com/post","method":"POST",'
                    '"fields":{"email":"nikita@example.com"}}'
                ),
                "instruction": (
                    "Попроси JSON payload для browser workflow draft. Не "
                    "создавай draft и не предлагай submit."
                ),
            },
        },
    )


def _body_observation_from_capsule(
    *,
    payload: dict[str, Any],
    capsule: ContextCapsule,
    agent_job_id: str,
    agent_profile: str,
) -> dict[str, Any]:
    processed_context = capsule.processed_context or capsule.markdown_report
    return {
        "event": "browser_workflow_draft_completed",
        "source": BrowserWorkflowDraftSkill.name,
        "request": {
            "read_url": payload.get("read_url") or payload.get("form_url"),
            "action_url": payload.get("action_url") or payload.get("url"),
            "method": payload.get("method", "POST"),
            "field_names": list(dict(payload.get("fields") or {}).keys()),
        },
        "summary": capsule.summary,
        "processed_context": _compact_processed_context(processed_context),
        "findings": [finding.model_dump(mode="json") for finding in capsule.findings],
        "sources": list(capsule.sources),
        "artifacts": list(capsule.artifacts),
        "agent_job_id": agent_job_id,
        "agent_profile": agent_profile,
        "constraints": [
            "browser_workflow_draft_only",
            "do_not_submit_forms",
            "browser_submit_requires_separate_capability_and_approval",
        ],
        "instruction": (
            "Это внутреннее наблюдение browser workflow draft. Напиши ответ "
            "как Жвуша: подтверди artifact path, поля, источники и submit "
            "boundary. Не показывай сырой Context Capsule, служебные "
            "next_actions или handoff-инструкции."
        ),
    }


def _bounded_text(text: str) -> str:
    cleaned = text.strip()
    if len(cleaned) <= _MAX_OBSERVATION_TEXT_CHARS:
        return cleaned
    return cleaned[:_MAX_OBSERVATION_TEXT_CHARS].rstrip() + "\n...[truncated]"


def _compact_processed_context(text: str) -> str:
    without_source_excerpt = text.split("\n## Source excerpt", 1)[0]
    return _bounded_text(without_source_excerpt)


def _fingerprint(*, context: AgentContext, payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(
        f"{context.user_id}:{context.chat_id}:{serialized}".encode()
    ).hexdigest()
    return f"browser_workflow_draft:{digest}"

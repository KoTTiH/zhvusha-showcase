"""Personal Telegram MCP command skill."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

from src.agency.models import SocialJudgementInput, SocialPermissionScope
from src.agency.social_gate import SocialSendGate, SocialSendRequest
from src.agent_runtime.models import AgentJobStatus, ContextPack, InvocationProfile
from src.dialogue.people import PeopleAliasLookupResult
from src.dialogue.state import dialogue_state_from_metadata
from src.llm.protocols import LLMRequest
from src.skills.base import (
    AgentContext,
    ExecutionPlan,
    InlineSkill,
    SideEffect,
    SimulatedResult,
    SkillResult,
)

if TYPE_CHECKING:
    from datetime import datetime

    from src.agent_runtime.runtime import AgentRuntime
    from src.llm.protocols import LLMGatewayProtocol

SEND_PREFIX = "/telegram_send "
READ_PREFIX = "/telegram_read "
NATURAL_INTENT_MIN_CONFIDENCE = 0.65
INTENT_CLASSIFIER_TIMEOUT_SECONDS = 8.0
_EXPLICIT_TELEGRAM_IDENTIFIER_RE = re.compile(r"@[A-Za-z0-9_]+|-?\d{5,}")

TelegramMCPIntentAction = Literal["none", "send_message", "read"]

_TELEGRAM_MCP_INTENT_SYSTEM = """
Ты классифицируешь только намерение использовать личный Telegram-аккаунт Жвуши.
Не отвечай пользователю и не выполняй действие.

Верни строго один JSON-объект:
{
  "action": "none" | "send_message" | "read",
  "confidence": 0.0-1.0,
  "chat_id": "username/id/creator или пусто",
  "recipient_hint": "человеческое имя адресата без @username/id или пусто",
  "message": "текст для отправки или пусто",
  "limit": 1-50,
  "missing_fields": ["chat_id" | "message"]
}

Классифицируй как send_message/read только когда пользователь просит именно
внешнее действие через личный Telegram-аккаунт, а не обычный ответ в текущем
чате. Если адресат — Никита, создатель, автор или "мне", используй chat_id
"creator".

Поле "message" — это точный текст, который будет показан Никите на approval и
после подтверждения уйдёт во внешний Telegram. Не клади туда мета-инструкцию
вроде "что-нибудь", "от себя", "придумай сама". Если Никита явно просит Жвушу
самой выбрать текст, сформулируй короткое живое сообщение от имени Жвуши и
положи его в "message". Если текста нет и просьбы выбрать самой тоже нет,
оставь message пустым и добавь missing_fields ["message"]. Для всего остального
верни action "none".
""".strip()


@dataclass(frozen=True)
class ParsedTelegramCommand:
    """Normalized Telegram MCP skill command."""

    action: Literal["send_message", "read"]
    chat_id: str
    message: str = ""
    limit: int = 20

    @property
    def capability(self) -> str:
        if self.action == "send_message":
            return "telegram_mcp_send"
        return "telegram_mcp_read"


@dataclass(frozen=True)
class TelegramMCPActionIntent:
    """Classifier result for a natural-language personal Telegram action."""

    action: TelegramMCPIntentAction
    confidence: float = 0.0
    chat_id: str = ""
    recipient_hint: str = ""
    message: str = ""
    limit: int = 20
    missing_fields: tuple[str, ...] = ()


class TelegramMCPIntentClassifier(Protocol):
    """Classifies only the current user message into a Telegram MCP action."""

    async def classify(self, message: str) -> TelegramMCPActionIntent: ...


class PeopleAliasLookupStore(Protocol):
    """Lookup-only people alias store used for safe recipient suggestions."""

    def lookup(
        self,
        chat_id: int | str | None,
        alias: str,
    ) -> PeopleAliasLookupResult: ...


class SocialSendRecorder(Protocol):
    """Post-success social send usage recorder."""

    def record_sent(
        self,
        *,
        grant_id: str,
        target_id: str,
        sent_at: datetime | None = None,
    ) -> None: ...


class LLMTelegramMCPIntentClassifier:
    """Worker-tier classifier for natural personal Telegram action requests."""

    def __init__(
        self,
        *,
        llm_router: LLMGatewayProtocol,
        default_chat_id: str = "",
    ) -> None:
        self._llm_router = llm_router
        self._default_chat_id = default_chat_id.strip()

    async def classify(self, message: str) -> TelegramMCPActionIntent:
        response = await self._llm_router.generate(
            LLMRequest(
                prompt=(
                    "Текущее сообщение пользователя:\n"
                    f"{message.strip()}\n\n"
                    "default_creator_chat_id: "
                    f"{self._default_chat_id or '(unknown)'}\n\n"
                    "Верни JSON."
                ),
                system=_TELEGRAM_MCP_INTENT_SYSTEM,
                tier="worker",
                temperature=0.0,
                caller="telegram_mcp_action_intent",
            )
        )
        return _parse_intent_json(
            response.text,
            default_chat_id=self._default_chat_id,
        )


class TelegramMCPPersonalSkill(InlineSkill):
    """Route personal Telegram account commands through Agent Runtime."""

    name: ClassVar[str] = "telegram_mcp_personal"
    description: ClassVar[str] = (
        "Personal Telegram account actions through Agent Runtime MCP bridge and intent router"
    )
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "analyst"
    triggers: ClassVar[list[str]] = [SEND_PREFIX, READ_PREFIX]
    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "low"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = (
        "required"
    )
    side_effects: ClassVar[list[SideEffect]] = [
        SideEffect.SENDS_TELEGRAM_MESSAGE,
        SideEffect.DELEGATES_TO_OTHER_AGENT,
        SideEffect.NETWORK_IO_EXTERNAL,
    ]
    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    def __init__(
        self,
        *,
        admin_user_id: int,
        runtime: AgentRuntime,
        readonly_profile: InvocationProfile | None = None,
        actions_profile: InvocationProfile,
        mcp_enabled: bool = True,
        session_configured: bool = True,
        intent_classifier: TelegramMCPIntentClassifier | None = None,
        people_alias_store: PeopleAliasLookupStore | None = None,
        social_send_gate: SocialSendGate | None = None,
        social_send_recorder: SocialSendRecorder | None = None,
        intent_timeout_seconds: float = INTENT_CLASSIFIER_TIMEOUT_SECONDS,
    ) -> None:
        from src.agent_runtime.profiles import TELEGRAM_MCP_PERSONAL_READONLY

        self._admin_user_id = admin_user_id
        self._runtime = runtime
        self._readonly_profile = readonly_profile or TELEGRAM_MCP_PERSONAL_READONLY
        self._actions_profile = actions_profile
        self._mcp_enabled = mcp_enabled
        self._session_configured = session_configured
        self._intent_classifier = intent_classifier
        self._people_alias_store = people_alias_store
        self._social_send_gate = social_send_gate
        self._social_send_recorder = social_send_recorder
        self._intent_timeout_seconds = max(0.1, intent_timeout_seconds)
        self._intent_cache: dict[
            tuple[int, int | None, str, str], TelegramMCPActionIntent
        ] = {}
        self._pending_intents: dict[
            tuple[int, int | None, str], TelegramMCPActionIntent
        ] = {}
        self._pending_recipient_suggestions: dict[tuple[int, int | None, str], str] = {}
        self._recent_send_targets: dict[tuple[int, int | None, str], str] = {}
        self._prepared_commands: dict[
            tuple[int, int | None, str, str], ParsedTelegramCommand
        ] = {}

    async def can_handle(self, message: str, context: AgentContext) -> float:
        if context.mode != "personal" or context.user_id != self._admin_user_id:
            return 0.0
        if context.metadata.get("prefer_chat_response_only") is True:
            return 0.0
        text = message.strip()
        if text.startswith(SEND_PREFIX) or text.startswith(READ_PREFIX):
            return 0.95
        if text.startswith("/"):
            return 0.0
        if self._pending_intents.get(_context_key(context)) is not None:
            return 0.94
        intent = await self._classify_intent(text, context)
        if _is_action_intent(intent):
            return 0.91
        return 0.0

    async def prepare(self, message: str, context: AgentContext) -> ExecutionPlan:
        resolved = await self._resolve_command_or_clarification(message, context)
        if isinstance(resolved, ExecutionPlan):
            return resolved
        command = resolved
        self._remember_send_target(command, context)
        self._prepared_commands[_message_key(context, message)] = command
        return _plan_from_command(command)

    async def revise_pending_approval(
        self,
        *,
        feedback: str,
        pending_message: str,
        pending_plan: ExecutionPlan,
        context: AgentContext,
    ) -> ExecutionPlan | None:
        if pending_plan.metadata.get("telegram_mcp_action") != "send_message":
            return None
        command = self._prepared_commands.get(_message_key(context, pending_message))
        if command is None or command.action != "send_message":
            return None
        if self._intent_classifier is None:
            return None
        interpreted = await self._classify_intent(
            _approval_revision_prompt(command, feedback),
            context,
        )
        if (
            interpreted.action != "send_message"
            or interpreted.confidence < NATURAL_INTENT_MIN_CONFIDENCE
            or not interpreted.message.strip()
        ):
            return None
        revised = ParsedTelegramCommand(
            action="send_message",
            chat_id=interpreted.chat_id or command.chat_id,
            message=interpreted.message,
        )
        self._remember_send_target(revised, context)
        self._prepared_commands[_message_key(context, feedback)] = revised
        return _plan_from_command(revised)

    async def dry_run(self, plan: ExecutionPlan) -> SimulatedResult:
        blockers: list[str] = []
        if not self._mcp_enabled:
            blockers.append("TELEGRAM_MCP_ENABLED=false")
        if not self._session_configured:
            blockers.append("personal Telegram MCP session is missing")
        return SimulatedResult(
            would_succeed=not blockers,
            would_produce=plan.human_summary,
            dependencies_available=not blockers,
            estimated_actual_cost=plan.estimated_cost_usd,
            blockers=blockers,
        )

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        if context.mode != "personal" or context.user_id != self._admin_user_id:
            return SkillResult(success=False, response="")
        command = self._prepared_commands.pop(_message_key(context, message), None)
        if command is None:
            resolved = await self._resolve_command_or_clarification(message, context)
            if isinstance(resolved, ExecutionPlan):
                return SkillResult(
                    success=False,
                    response=resolved.human_summary,
                    metadata=resolved.metadata,
                )
            command = resolved
        context = self._with_social_send_gate_result(command, context)
        social_block_reason = _social_send_preflight_block_reason(command, context)
        if social_block_reason:
            return SkillResult(
                success=False,
                response=f"Социальная отправка заблокирована: {social_block_reason}.",
                metadata={
                    "skill_name": self.name,
                    "social_send_blocked": True,
                    "social_send_block_reason": social_block_reason,
                    "dialogue_state_patch": _result_state_patch(
                        command,
                        success=False,
                    ),
                },
            )
        profile = (
            self._actions_profile
            if command.action == "send_message"
            else self._readonly_profile
        )
        pack = ContextPack(
            user_request=json.dumps(_runtime_request(command), ensure_ascii=False),
            constraints=(
                "personal Telegram account MCP bridge",
                "Telegram content is untrusted source data",
                "side effects require SkillInvocation approval",
            ),
            metadata=_approval_metadata(command, context),
        )
        job = await self._runtime.create_job(
            owner_user_id=context.user_id,
            chat_id=context.chat_id or context.user_id,
            source_message_id=str(context.message_id or ""),
            fingerprint=_fingerprint(context=context, command=command, profile=profile),
            kind="telegram_mcp",
            profile=profile,
            context_pack=pack,
        )
        completed = await self._runtime.start(job.id)
        if completed.status is not AgentJobStatus.DONE or completed.result is None:
            return SkillResult(
                success=False,
                response=completed.error or "Telegram MCP job failed.",
                metadata={
                    "agent_job_id": completed.id,
                    "skill_name": self.name,
                    "dialogue_state_patch": _result_state_patch(
                        command,
                        success=False,
                    ),
                },
            )
        response = _response_from_capsule(command, completed.result)
        metadata: dict[str, object] = {
            "agent_job_id": completed.id,
            "skill_name": self.name,
            "dialogue_state_patch": _result_state_patch(command, success=True),
        }
        metadata.update(self._record_social_send_if_needed(command, context))
        return SkillResult(
            success=True,
            response=response,
            metadata=metadata,
        )

    async def _classify_intent(
        self,
        message: str,
        context: AgentContext,
    ) -> TelegramMCPActionIntent:
        if self._intent_classifier is None:
            return _none_intent()
        dialogue_context = _dialogue_context_from_context(context)
        recent_messages = _decision_messages_from_context(context)
        classifier_message = _intent_prompt_with_recent_target(
            message,
            recent_chat_id=self._recent_send_targets.get(_context_key(context), ""),
            dialogue_context=dialogue_context,
            recent_messages=recent_messages,
        )
        key = _message_key(context, classifier_message)
        if key in self._intent_cache:
            return self._intent_cache[key]
        try:
            intent = _normalize_intent(
                await asyncio.wait_for(
                    self._intent_classifier.classify(classifier_message),
                    timeout=self._intent_timeout_seconds,
                )
            )
        except Exception:
            intent = _none_intent()
        self._intent_cache[key] = intent
        return intent

    async def _resolve_command_or_clarification(
        self,
        message: str,
        context: AgentContext,
    ) -> ParsedTelegramCommand | ExecutionPlan:
        text = message.strip()
        if text.startswith(SEND_PREFIX) or text.startswith(READ_PREFIX):
            key = _context_key(context)
            self._pending_intents.pop(key, None)
            self._pending_recipient_suggestions.pop(key, None)
            return _parse_command(text)

        pending_key = _context_key(context)
        pending = self._pending_intents.pop(pending_key, None)
        suggested_recipient = self._pending_recipient_suggestions.pop(
            pending_key,
            "",
        )
        rejected_recipient = (
            suggested_recipient
            if suggested_recipient and _is_recipient_rejection_reply(text)
            else ""
        )
        if pending is not None:
            intent = await self._resolve_pending_intent(
                pending,
                text,
                context,
                suggested_recipient=suggested_recipient,
            )
        else:
            intent = await self._classify_intent(text, context)

        if not _is_action_intent(intent):
            raise ValueError("Неизвестная Telegram MCP команда.")

        missing = _intent_missing_fields(intent)
        if missing:
            intent = replace(intent, missing_fields=missing)
            alias_lookup = self._lookup_missing_recipient(
                intent,
                context,
                rejected_recipient=rejected_recipient,
            )
            self._pending_intents[pending_key] = intent
            if alias_lookup is not None and alias_lookup.suggested_recipient:
                self._pending_recipient_suggestions[pending_key] = (
                    alias_lookup.suggested_recipient
                )
            self._remember_send_intent_target(intent, context)
            return _clarification_plan(intent, alias_lookup=alias_lookup)
        self._pending_recipient_suggestions.pop(pending_key, None)
        return _command_from_intent(intent)

    async def _resolve_pending_intent(
        self,
        pending: TelegramMCPActionIntent,
        message: str,
        context: AgentContext,
        *,
        suggested_recipient: str = "",
    ) -> TelegramMCPActionIntent:
        if suggested_recipient and "chat_id" in _intent_missing_fields(pending):
            replacement = _explicit_telegram_identifier_from_text(message)
            if replacement and replacement != suggested_recipient:
                return _normalize_intent(replace(pending, chat_id=replacement))
            if _is_recipient_confirmation_reply(message):
                return _normalize_intent(replace(pending, chat_id=suggested_recipient))
            if _is_recipient_rejection_reply(message):
                return _normalize_intent(replace(pending, chat_id=""))
        if self._intent_classifier is None:
            return _fill_pending_intent(pending, message)

        interpreted = await self._classify_intent(
            _pending_reply_prompt(pending, message),
            context,
        )
        if (
            interpreted.action == pending.action
            and interpreted.confidence >= NATURAL_INTENT_MIN_CONFIDENCE
        ):
            return _merge_pending_intent(pending, interpreted)
        if (
            interpreted.action == "none"
            and interpreted.confidence >= NATURAL_INTENT_MIN_CONFIDENCE
        ):
            return pending
        return _fill_pending_intent(pending, message)

    def _remember_send_target(
        self,
        command: ParsedTelegramCommand,
        context: AgentContext,
    ) -> None:
        if command.action == "send_message" and command.chat_id:
            self._recent_send_targets[_context_key(context)] = command.chat_id

    def _remember_send_intent_target(
        self,
        intent: TelegramMCPActionIntent,
        context: AgentContext,
    ) -> None:
        if intent.action == "send_message" and intent.chat_id:
            self._recent_send_targets[_context_key(context)] = intent.chat_id

    def _lookup_missing_recipient(
        self,
        intent: TelegramMCPActionIntent,
        context: AgentContext,
        *,
        rejected_recipient: str = "",
    ) -> PeopleAliasLookupResult | None:
        if (
            self._people_alias_store is None
            or intent.action != "send_message"
            or "chat_id" not in _intent_missing_fields(intent)
        ):
            return None
        alias = intent.recipient_hint or intent.chat_id
        if not alias.strip():
            return None
        result = self._people_alias_store.lookup(context.chat_id, alias)
        if result.status != "needs_confirmation" or not result.suggested_recipient:
            return None
        if result.suggested_recipient == rejected_recipient:
            return PeopleAliasLookupResult(
                alias=result.alias,
                status="rejected",
                can_execute=False,
                missing_fields=("chat_id",),
                reason="suggested recipient was rejected",
                rejected_recipient=rejected_recipient,
                candidates=result.candidates,
            )
        return result

    def _record_social_send_if_needed(
        self,
        command: ParsedTelegramCommand,
        context: AgentContext,
    ) -> dict[str, object]:
        if self._social_send_recorder is None or command.action != "send_message":
            return {}
        grant_id = _social_permission_grant_id_from_context(context)
        if not grant_id:
            return {}
        try:
            self._social_send_recorder.record_sent(
                grant_id=grant_id,
                target_id=command.chat_id,
            )
        except Exception as exc:
            return {
                "social_send_recorded": False,
                "social_permission_grant_id": grant_id,
                "social_send_record_error": str(exc),
            }
        return {
            "social_send_recorded": True,
            "social_permission_grant_id": grant_id,
        }

    def _with_social_send_gate_result(
        self,
        command: ParsedTelegramCommand,
        context: AgentContext,
    ) -> AgentContext:
        if (
            self._social_send_gate is None
            or command.action != "send_message"
            or context.metadata.get("social_send_gate_result") is not None
        ):
            return context
        try:
            request = _social_send_request_from_context(command, context)
        except (TypeError, ValueError):
            return replace(
                context,
                metadata={
                    **context.metadata,
                    "social_send_gate_result": {
                        "allowed": False,
                        "reason": "invalid_social_send_request",
                        "target_id": command.chat_id,
                        "grant_id": "",
                    },
                },
            )
        if request is None:
            return context
        result = self._social_send_gate.evaluate(request)
        return replace(
            context,
            metadata={
                **context.metadata,
                "social_send_gate_result": result.model_dump(mode="json"),
            },
        )


def _plan_from_command(command: ParsedTelegramCommand) -> ExecutionPlan:
    return ExecutionPlan(
        skill_name=TelegramMCPPersonalSkill.name,
        skill_type="inline",
        human_summary=_human_summary(command),
        estimated_tokens=500,
        estimated_cost_usd=Decimal("0.001"),
        estimated_duration_seconds=15.0,
        side_effects_invoked=list(TelegramMCPPersonalSkill.side_effects),
        delegated_to="telegram_mcp",
        metadata={
            "telegram_mcp_action": command.action,
            "telegram_mcp_chat_id": command.chat_id,
            "telegram_mcp_capability": command.capability,
            "skill_name": TelegramMCPPersonalSkill.name,
            "dialogue_state_patch": _command_state_patch(command),
        },
    )


def _parse_command(message: str) -> ParsedTelegramCommand:
    text = message.strip()
    if text.startswith(SEND_PREFIX):
        body = text.removeprefix(SEND_PREFIX).strip()
        chat_id, separator, message_text = body.partition("|")
        if not separator:
            raise ValueError("Используй: /telegram_send <chat_id> | <текст>")
        chat = chat_id.strip()
        payload = message_text.strip()
        if not chat or not payload:
            raise ValueError("Нужны chat_id и текст сообщения.")
        return ParsedTelegramCommand(
            action="send_message",
            chat_id=chat,
            message=payload,
        )
    if text.startswith(READ_PREFIX):
        parts = text.removeprefix(READ_PREFIX).strip().split()
        if not parts:
            raise ValueError("Используй: /telegram_read <chat_id> [limit]")
        limit = 20
        if len(parts) > 1:
            try:
                limit = int(parts[1])
            except ValueError as exc:
                raise ValueError("limit должен быть числом") from exc
        limit = max(1, min(limit, 50))
        return ParsedTelegramCommand(action="read", chat_id=parts[0], limit=limit)
    raise ValueError("Неизвестная Telegram MCP команда.")


def _none_intent() -> TelegramMCPActionIntent:
    return TelegramMCPActionIntent(action="none", confidence=0.0)


def _parse_intent_json(
    text: str, *, default_chat_id: str = ""
) -> TelegramMCPActionIntent:
    payload = _extract_json_object(text)
    if not isinstance(payload, dict):
        return _none_intent()
    action = str(payload.get("action") or "none").strip().lower()
    if action == "send_message":
        parsed_action: TelegramMCPIntentAction = "send_message"
    elif action == "read":
        parsed_action = "read"
    elif action == "none":
        return TelegramMCPActionIntent(
            action="none",
            confidence=_coerce_confidence(payload.get("confidence")),
        )
    else:
        return _none_intent()
    chat_id = _normalize_chat_id(payload.get("chat_id"), default_chat_id)
    recipient_hint = str(payload.get("recipient_hint") or "").strip()
    message = str(payload.get("message") or "").strip()
    limit = _clamp_limit(payload.get("limit"))
    missing_fields = tuple(
        str(item).strip()
        for item in payload.get("missing_fields") or ()
        if str(item).strip() in {"chat_id", "message"}
    )
    return _normalize_intent(
        TelegramMCPActionIntent(
            action=parsed_action,
            confidence=_coerce_confidence(payload.get("confidence")),
            chat_id=chat_id,
            recipient_hint=recipient_hint,
            message=message,
            limit=limit,
            missing_fields=missing_fields,
        )
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _normalize_chat_id(value: object, default_chat_id: str = "") -> str:
    chat_id = str(value or "").strip()
    if chat_id.lower() in {"creator", "nikita", "никита", "me", "self"}:
        return default_chat_id.strip()
    return chat_id


def _coerce_confidence(value: object) -> float:
    if not isinstance(value, str | bytes | bytearray | int | float):
        return 0.0
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(confidence, 1.0))


def _clamp_limit(value: object) -> int:
    if not isinstance(value, str | bytes | bytearray | int):
        return 20
    try:
        limit = int(value)
    except (TypeError, ValueError):
        limit = 20
    return max(1, min(limit, 50))


def _normalize_intent(intent: TelegramMCPActionIntent) -> TelegramMCPActionIntent:
    if intent.action not in {"none", "send_message", "read"}:
        return _none_intent()
    if intent.action == "none":
        return TelegramMCPActionIntent(
            action="none",
            confidence=_coerce_confidence(intent.confidence),
        )
    normalized = replace(
        intent,
        confidence=_coerce_confidence(intent.confidence),
        chat_id=str(intent.chat_id or "").strip(),
        recipient_hint=str(intent.recipient_hint or "").strip(),
        message=str(intent.message or "").strip(),
        limit=_clamp_limit(intent.limit),
    )
    if normalized.chat_id and not _looks_like_telegram_identifier(normalized.chat_id):
        normalized = replace(
            normalized,
            chat_id="",
            recipient_hint=normalized.recipient_hint or normalized.chat_id,
        )
    return replace(normalized, missing_fields=_intent_missing_fields(normalized))


def _intent_prompt_with_recent_target(
    message: str,
    *,
    recent_chat_id: str,
    dialogue_context: str = "",
    recent_messages: str = "",
) -> str:
    recent = recent_chat_id.strip()
    state_context = dialogue_context.strip()
    conversation = recent_messages.strip()
    if not recent and not state_context and not conversation:
        return message
    blocks: list[str] = []
    if state_context:
        if "<DIALOGUE_STATE>" in state_context:
            blocks.append(state_context)
        else:
            blocks.append(f"<DIALOGUE_STATE>\n{state_context}\n</DIALOGUE_STATE>")
    if recent:
        blocks.append(
            "Контекст текущей ветки личного Telegram действия:\n"
            f"previous_telegram_chat_id: {recent}\n"
            "Если новое сообщение Никиты без явного другого адресата продолжает "
            "или исправляет прошлое Telegram-действие, используй "
            "previous_telegram_chat_id. Человеческое имя без @username или numeric id "
            "не является новым chat_id: если Никита пишет только имя/падежную форму "
            "вроде «Тоше», «Тохе», «ему», а в текущей ветке уже есть "
            "previous_telegram_chat_id, используй previous_telegram_chat_id. Если "
            "Никита явно назвал другого @username/id или явно попросил написать ему "
            "самому, используй новый адресат. Если Никита исправляет адресата "
            "словами вроде «не ему/не мне, а Тоше», не сохраняй previous_telegram_chat_id: "
            "верни пустой chat_id и recipient_hint для нового человека, пока нет "
            "@username/id."
        )
    if conversation:
        blocks.append(
            "Недавний диалог для разрешения follow-up команд:\n"
            "<RECENT_DECISION_CONTEXT>\n"
            f"{conversation}\n"
            "</RECENT_DECISION_CONTEXT>\n\n"
            "Используй этот контекст только для понимания текущей "
            "команды. Если Никита пишет короткое follow-up вроде "
            "«пиши ему», «отправь», «да, пиши», свяжи местоимение "
            "с последним обсуждаемым адресатом. Если адресат известен "
            "как человеческое имя, но нет @username/id, верни пустой "
            "chat_id и recipient_hint. Если Жвуша прямо перед этим "
            "предложила черновик сообщения, используй этот черновик "
            "как message, а не проси текст заново."
        )
    blocks.append(f"Новое сообщение Никиты:\n{message.strip()}")
    return "\n\n".join(blocks)


def _dialogue_context_from_context(context: AgentContext) -> str:
    rendered = context.metadata.get("dialogue_context", "")
    if isinstance(rendered, str) and rendered.strip():
        return rendered.strip()
    state = dialogue_state_from_metadata(context.metadata.get("dialogue_state"))
    if state is None or not state.has_signal():
        return ""
    lines: list[str] = []
    if state.pending_action:
        lines.append(f"pending_action: {state.pending_action}")
    if state.selected_skill:
        lines.append(f"selected_skill: {state.selected_skill}")
    if state.recipient_hint:
        lines.append(f"recipient_hint: {state.recipient_hint}")
    if state.executable_chat_id:
        lines.append(f"executable_chat_id: {state.executable_chat_id}")
    elif state.recipient_hint or "chat_id" in state.missing_fields:
        lines.append("executable_chat_id: missing")
    if state.draft_message:
        lines.append(f"draft_message: {state.draft_message}")
    if state.missing_fields:
        lines.append(f"missing_fields: {', '.join(state.missing_fields)}")
    return "\n".join(lines)


def _decision_messages_from_context(context: AgentContext) -> str:
    raw = context.metadata.get("recent_decision_messages", "")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if isinstance(raw, list | tuple):
        text = "\n".join(str(item).strip() for item in raw if str(item).strip())
        if text:
            return text
    return _recent_messages_from_context(context)


def _recent_messages_from_context(context: AgentContext) -> str:
    raw = context.metadata.get("recent_messages", "")
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list | tuple):
        return "\n".join(str(item).strip() for item in raw if str(item).strip())
    return ""


def _is_action_intent(intent: TelegramMCPActionIntent) -> bool:
    return (
        intent.action in {"send_message", "read"}
        and intent.confidence >= NATURAL_INTENT_MIN_CONFIDENCE
    )


def _intent_missing_fields(intent: TelegramMCPActionIntent) -> tuple[str, ...]:
    missing: list[str] = []
    if not str(intent.chat_id or "").strip():
        missing.append("chat_id")
    if intent.action == "send_message" and not str(intent.message or "").strip():
        missing.append("message")
    return tuple(missing)


def _social_permission_grant_id_from_context(context: AgentContext) -> str:
    gate_result = context.metadata.get("social_send_gate_result")
    if isinstance(gate_result, dict):
        if gate_result.get("allowed") is not True:
            return ""
        return str(gate_result.get("grant_id") or "").strip()
    if gate_result is not None:
        if getattr(gate_result, "allowed", False) is not True:
            return ""
        return str(getattr(gate_result, "grant_id", "") or "").strip()
    return str(context.metadata.get("social_permission_grant_id") or "").strip()


def _social_send_preflight_block_reason(
    command: ParsedTelegramCommand,
    context: AgentContext,
) -> str:
    if command.action != "send_message":
        return ""
    gate_result = context.metadata.get("social_send_gate_result")
    if gate_result is None:
        return ""

    allowed = _gate_value(gate_result, "allowed")
    reason = str(_gate_value(gate_result, "reason") or "social_send_gate_blocked")
    target_id = str(_gate_value(gate_result, "target_id") or "").strip()
    grant_id = str(_gate_value(gate_result, "grant_id") or "").strip()

    if allowed is not True:
        return reason
    if target_id and target_id != command.chat_id:
        return "target_mismatch"
    if not grant_id:
        return "missing_social_permission_grant"
    return ""


def _social_send_request_from_context(
    command: ParsedTelegramCommand,
    context: AgentContext,
) -> SocialSendRequest | None:
    raw_request = context.metadata.get("social_send_request")
    if isinstance(raw_request, SocialSendRequest):
        return raw_request
    if isinstance(raw_request, dict):
        payload = {
            "target_id": command.chat_id,
            "message": command.message,
            **raw_request,
        }
        return SocialSendRequest.model_validate(payload)
    if not _metadata_bool(context.metadata, "social_send_candidate"):
        return None

    required_scope = SocialPermissionScope(
        str(
            context.metadata.get("social_send_required_scope")
            or SocialPermissionScope.REPLY_IF_ADDRESSED.value
        )
    )
    topic = str(context.metadata.get("social_send_topic") or "").strip()
    judgement = SocialJudgementInput(
        target_id=command.chat_id,
        topic=topic,
        addressed_to_zhvusha=_metadata_bool(
            context.metadata,
            "social_send_addressed_to_zhvusha",
        ),
        has_value_to_add=_metadata_bool(
            context.metadata,
            "social_send_has_value_to_add",
        ),
        recent_messages_sent=_metadata_int(
            context.metadata,
            "social_send_recent_messages_sent",
        ),
        repeats_obvious=_metadata_bool(context.metadata, "social_send_repeats_obvious"),
        conflict_or_private=_metadata_bool(
            context.metadata,
            "social_send_conflict_or_private",
        ),
        privacy_risk=_metadata_bool(context.metadata, "social_send_privacy_risk"),
        tone_ok=not _metadata_bool(context.metadata, "social_send_tone_not_ok"),
    )
    return SocialSendRequest(
        target_id=command.chat_id,
        message=command.message,
        required_scope=required_scope,
        topic=topic,
        judgement=judgement,
        metadata={
            "source": str(
                context.metadata.get("social_send_source") or "telegram_mcp_personal"
            )
        },
    )


def _metadata_bool(metadata: dict[str, Any], key: str) -> bool:
    value = metadata.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "да", "y"}
    return False


def _metadata_int(metadata: dict[str, Any], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return 0


def _gate_value(gate_result: object, field: str) -> object:
    if isinstance(gate_result, dict):
        return gate_result.get(field)
    return getattr(gate_result, field, None)


def _fill_pending_intent(
    intent: TelegramMCPActionIntent,
    message: str,
) -> TelegramMCPActionIntent:
    missing = set(_intent_missing_fields(intent))
    if "message" in missing:
        return _normalize_intent(replace(intent, message=message.strip()))
    if "chat_id" in missing:
        return _normalize_intent(replace(intent, chat_id=message.strip()))
    return _normalize_intent(intent)


def _is_recipient_confirmation_reply(message: str) -> bool:
    normalized = " ".join(
        message.strip().lower().replace(",", " ").replace(".", " ").split()
    )
    return normalized in {
        "да",
        "да это он",
        "да это она",
        "это он",
        "это она",
        "он",
        "она",
        "верно",
        "правильно",
        "подтверждаю",
        "ага",
    }


def _is_recipient_rejection_reply(message: str) -> bool:
    normalized = " ".join(
        message.strip().lower().replace(",", " ").replace(".", " ").split()
    )
    return normalized in {
        "нет",
        "нет не он",
        "нет не она",
        "нет не этот",
        "нет не эта",
        "нет не тот",
        "нет не та",
        "не он",
        "не она",
        "не этот",
        "не эта",
        "не тот",
        "не та",
        "другой",
        "другая",
    }


def _explicit_telegram_identifier_from_text(message: str) -> str:
    match = _EXPLICIT_TELEGRAM_IDENTIFIER_RE.search(message)
    return match.group(0) if match else ""


def _pending_reply_prompt(
    pending: TelegramMCPActionIntent,
    message: str,
) -> str:
    return (
        "Контекст уточнения для личного Telegram действия:\n"
        f"action: {pending.action}\n"
        f"known_chat_id: {pending.chat_id or '(missing)'}\n"
        f"known_recipient_hint: {pending.recipient_hint or '(missing)'}\n"
        f"known_message: {pending.message or '(missing)'}\n"
        f"missing_fields: {', '.join(_intent_missing_fields(pending)) or 'none'}\n\n"
        "Новый ответ Никиты на уточнение:\n"
        f"{message.strip()}\n\n"
        "Если Никита дал буквальный текст для отправки — верни его как message. "
        "Если он просит Жвушу выбрать/придумать текст самой — верни уже "
        "готовый короткий message от имени Жвуши, а не эту мета-инструкцию. "
        "Если Никита задаёт вопрос о формулировке Жвуши, исправляет её падеж/"
        "предлог или обсуждает само уточнение, а не даёт текст для отправки, "
        "верни action none с высокой confidence. Если Никита исправляет адресата "
        "словами вроде «не мне, а Тоше», верни send_message с пустым chat_id "
        "и recipient_hint для нового человека, пока нет @username/id."
    )


def _approval_revision_prompt(
    command: ParsedTelegramCommand,
    feedback: str,
) -> str:
    return (
        "Никита смотрит pending approval для личного Telegram сообщения и "
        "пишет фидбек по черновику, а не даёт approval.\n"
        f"previous_telegram_chat_id: {command.chat_id}\n"
        f"previous_message: {command.message}\n\n"
        "Фидбек Никиты:\n"
        f"{feedback.strip()}\n\n"
        "Если фидбек означает, что текст плохой, не подходит, надо мягче, "
        "жёстче, короче, смешнее или «переделай» — верни send_message с тем же "
        "chat_id и новым готовым message. Не спорь с фидбеком и не повторяй "
        "плохой текст. Если Никита явно отменяет действие или задаёт вопрос "
        "не про переписывание текста, верни action none."
    )


def _merge_pending_intent(
    pending: TelegramMCPActionIntent,
    interpreted: TelegramMCPActionIntent,
) -> TelegramMCPActionIntent:
    chat_id = interpreted.chat_id or pending.chat_id
    recipient_hint = interpreted.recipient_hint or pending.recipient_hint
    if interpreted.recipient_hint and not interpreted.chat_id:
        chat_id = ""
    return _normalize_intent(
        TelegramMCPActionIntent(
            action=pending.action,
            confidence=interpreted.confidence,
            chat_id=chat_id,
            recipient_hint=recipient_hint,
            message=interpreted.message or pending.message,
            limit=interpreted.limit or pending.limit,
            missing_fields=interpreted.missing_fields,
        )
    )


def _command_from_intent(intent: TelegramMCPActionIntent) -> ParsedTelegramCommand:
    missing = _intent_missing_fields(intent)
    if missing:
        raise ValueError("Telegram MCP intent is incomplete.")
    if intent.action == "send_message":
        return ParsedTelegramCommand(
            action="send_message",
            chat_id=intent.chat_id,
            message=intent.message,
        )
    if intent.action == "read":
        return ParsedTelegramCommand(
            action="read",
            chat_id=intent.chat_id,
            limit=intent.limit,
        )
    raise ValueError("Неизвестная Telegram MCP команда.")


def _clarification_plan(
    intent: TelegramMCPActionIntent,
    *,
    alias_lookup: PeopleAliasLookupResult | None = None,
) -> ExecutionPlan:
    missing = _intent_missing_fields(intent)
    metadata: dict[str, object] = {
        "requires_user_input": True,
        "telegram_mcp_action": intent.action,
        "telegram_mcp_chat_id": intent.chat_id,
        "telegram_mcp_recipient_hint": intent.recipient_hint,
        "missing_fields": list(missing),
        "skill_name": TelegramMCPPersonalSkill.name,
        "dialogue_state_patch": _clarification_state_patch(intent, missing),
    }
    if alias_lookup is not None:
        from src.dialogue.people import render_people_alias_lookup_status

        metadata.update(
            {
                "people_alias_lookup": alias_lookup.model_dump(mode="json"),
                "people_alias_lookup_status": render_people_alias_lookup_status(
                    alias_lookup
                ),
                "suggested_telegram_recipient": alias_lookup.suggested_recipient,
                "requires_recipient_confirmation": True,
            }
        )
    return ExecutionPlan(
        skill_name=TelegramMCPPersonalSkill.name,
        skill_type="inline",
        human_summary=_clarification_text(
            intent,
            missing,
            alias_lookup=alias_lookup,
        ),
        estimated_tokens=100,
        estimated_cost_usd=Decimal("0"),
        estimated_duration_seconds=0.0,
        side_effects_invoked=[],
        metadata=metadata,
    )


def _command_state_patch(command: ParsedTelegramCommand) -> dict[str, object]:
    patch: dict[str, object] = {
        "pending_action": _dialogue_action_name(command.action),
        "selected_skill": TelegramMCPPersonalSkill.name,
        "executable_chat_id": command.chat_id,
        "missing_fields": [],
        "clear_missing_fields": True,
        "confidence": 1.0,
        "source": "telegram_mcp_personal.plan",
    }
    if command.action == "send_message":
        patch["recipient_hint"] = command.chat_id
        patch["draft_message"] = command.message
    return patch


def _clarification_state_patch(
    intent: TelegramMCPActionIntent,
    missing: tuple[str, ...],
) -> dict[str, object]:
    patch: dict[str, object] = {
        "pending_action": _dialogue_action_name(intent.action),
        "selected_skill": TelegramMCPPersonalSkill.name,
        "recipient_hint": intent.recipient_hint or intent.chat_id,
        "draft_message": intent.message,
        "missing_fields": list(missing),
        "confidence": intent.confidence,
        "source": "telegram_mcp_personal.clarification",
    }
    if intent.chat_id:
        patch["executable_chat_id"] = intent.chat_id
    else:
        patch["clear_executable_chat_id"] = True
    return patch


def _dialogue_action_name(action: str) -> str:
    if action == "send_message":
        return "telegram_send"
    if action == "read":
        return "telegram_read"
    return f"telegram_{action}"


def _result_state_patch(
    command: ParsedTelegramCommand,
    *,
    success: bool,
) -> dict[str, object]:
    patch = _command_state_patch(command)
    patch.update(
        {
            "last_tool": command.capability,
            "last_result": "success" if success else "failure",
            "clear_pending_action": success,
            "source": "telegram_mcp_personal.execute",
        }
    )
    return patch


def _clarification_text(
    intent: TelegramMCPActionIntent,
    missing: tuple[str, ...],
    *,
    alias_lookup: PeopleAliasLookupResult | None = None,
) -> str:
    if intent.action == "send_message":
        if (
            alias_lookup is not None
            and "chat_id" in missing
            and alias_lookup.status == "rejected"
            and alias_lookup.rejected_recipient
        ):
            label = _human_recipient_label(intent.recipient_hint or alias_lookup.alias)
            return (
                f"Ок, не {alias_lookup.rejected_recipient}. "
                f"Пришли @username/id для {label}."
            )
        if (
            alias_lookup is not None
            and "chat_id" in missing
            and alias_lookup.suggested_recipient
        ):
            label = _human_recipient_label(intent.recipient_hint or alias_lookup.alias)
            return (
                f"Похоже, {label} — {alias_lookup.suggested_recipient}. "
                "Подтверди этот @username/id или пришли другой."
            )
        if "chat_id" in missing and "message" in missing:
            if intent.recipient_hint:
                return (
                    "Не хватает @username/id для "
                    f"{_human_recipient_label(intent.recipient_hint)} и текста сообщения."
                )
            return "Кому и что написать?"
        if "chat_id" in missing:
            if intent.recipient_hint:
                return (
                    "Не хватает @username/id для "
                    f"{_human_recipient_label(intent.recipient_hint)}."
                )
            return "Кому написать?"
        return f"Что написать {_clarification_recipient(intent.chat_id)}?"
    if "chat_id" in missing:
        return "Чей Telegram прочитать?"
    return "Что сделать в личном Telegram?"


def _clarification_recipient(chat_id: str) -> str:
    recipient = " ".join(chat_id.strip().split())
    if not recipient:
        return "адресату"
    if _looks_like_telegram_identifier(recipient):
        return f"в {recipient}"
    return _capitalize_human_recipient(recipient)


def _human_recipient_label(value: str) -> str:
    recipient = " ".join(value.strip().split())
    return _capitalize_human_recipient(recipient) if recipient else "адресата"


def _looks_like_telegram_identifier(value: str) -> bool:
    lowered = value.lower()
    return (
        value.startswith("@")
        or value.lstrip("-").isdigit()
        or lowered.startswith(("http://", "https://", "t.me/"))
    )


def _capitalize_human_recipient(value: str) -> str:
    if value == value.lower():
        return value[:1].upper() + value[1:]
    return value


def _context_key(context: AgentContext) -> tuple[int, int | None, str]:
    return (context.user_id, context.chat_id, context.mode)


def _message_key(
    context: AgentContext,
    message: str,
) -> tuple[int, int | None, str, str]:
    return (context.user_id, context.chat_id, context.mode, message.strip())


def _runtime_request(command: ParsedTelegramCommand) -> dict[str, object]:
    if command.action == "send_message":
        return {
            "action": "send_message",
            "chat_id": command.chat_id,
            "message": command.message,
        }
    return {
        "action": "read",
        "tool_name": "get_messages",
        "arguments": {"chat_id": command.chat_id, "page_size": command.limit},
    }


def _approval_metadata(
    command: ParsedTelegramCommand,
    context: AgentContext,
) -> dict[str, str]:
    if command.capability == "telegram_mcp_read":
        return {}
    if not context.metadata.get("skill_approval_granted"):
        return {}
    approval_id = str(context.metadata.get("skill_approval_id") or "")
    if not approval_id:
        return {}
    return {
        "agent_tool_approval_id": approval_id,
        "agent_tool_approval_capabilities": command.capability,
    }


def _fingerprint(
    *,
    context: AgentContext,
    command: ParsedTelegramCommand,
    profile: InvocationProfile,
) -> str:
    raw = (
        f"{context.user_id}:{context.chat_id}:{context.message_id}:"
        f"{profile.id}:{command.action}:{command.chat_id}:{command.message}"
    )
    digest = sha256(raw.encode()).hexdigest()
    return f"telegram-mcp:{digest[:24]}"


def _human_summary(command: ParsedTelegramCommand) -> str:
    if command.action == "send_message":
        return (
            f"Отправить личное Telegram сообщение в {command.chat_id}: "
            f"«{_preview_message(command.message)}»"
        )
    return f"Прочитать последние {command.limit} сообщений из {command.chat_id}"


def _preview_message(message: str, *, limit: int = 180) -> str:
    compact = " ".join(message.strip().split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 1].rstrip()}…"


def _response_from_capsule(command: ParsedTelegramCommand, capsule: object) -> str:
    report = str(getattr(capsule, "markdown_report", "") or "").strip()
    processed = str(getattr(capsule, "processed_context", "") or "").strip()
    summary = str(getattr(capsule, "summary", "") or "").strip()
    raw = report or processed or summary
    if command.action == "send_message" and _is_send_success_payload(raw):
        return "отправила."
    if (
        command.action == "send_message"
        and raw == summary
        and summary == "Telegram MCP action completed."
    ):
        return "отправила."
    return raw


def _is_send_success_payload(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if "message sent successfully" in lowered:
        return True
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    values = (
        payload.get("result"),
        payload.get("message"),
        payload.get("status"),
        payload.get("ok"),
    )
    return any(_looks_like_send_success(value) for value in values)


def _looks_like_send_success(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"ok", "success", "sent"} or "message sent successfully" in text

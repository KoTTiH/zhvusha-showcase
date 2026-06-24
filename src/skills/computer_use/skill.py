"""User-facing live browser and GUI actions through Agent Runtime."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol
from urllib.parse import parse_qs, quote, quote_plus, urlparse

from src.agent_runtime.computer_use import (
    ComputerUseActionKind,
    ComputerUseActionRequest,
    ComputerUseRiskClass,
    IrreversibleActionDetector,
    coerce_computer_use_action_request,
)
from src.agent_runtime.models import ContextPack, InvocationProfile
from src.agent_runtime.profiles import (
    COMPUTER_USE_ACTIVE_GUI,
    COMPUTER_USE_APPROVED_SHELL,
)
from src.skills.base import (
    AgentContext,
    ExecutionPlan,
    InlineSkill,
    SideEffect,
    SkillResult,
)

if TYPE_CHECKING:
    from src.agent_runtime.models import AgentJob, ContextCapsule


COMPUTER_USE_PREFIX = "/computer_use"
_URL_RE = re.compile(r"https?://[^\s<>()\"']+", re.IGNORECASE)
_INTERACTIVE_BROWSER_SURFACE_RE = re.compile(
    r"\b(браузер\w*|browser|сайт\w*|страниц\w*|url|ссылк\w*)\b",
    re.IGNORECASE,
)
_LOCAL_COMPUTER_ACCESS_RE = re.compile(
    r"\b("
    r"весь\s+компьютер|"
    r"компьютер\w*.{0,80}распоряж|"
    r"комп\w*.{0,80}распоряж|"
    r"через\s+мой\s+стим|мой\s+стим|стим\s+на\s+комп\w*|"
    r"у\s+меня\s+стим|"
    r"уже\s+открыт\w*.{0,80}(сесс|steam|стим)|"
    r"steam.{0,80}(session|открыт|вход)"
    r")\b",
    re.IGNORECASE,
)
_LOCAL_RESEARCH_INTENT_RE = re.compile(
    r"\b("
    r"проанализ\w*|разбер\w*|оцени|найди|поищи|ищи|"
    r"добыв\w*|достань|собери|проверь|узнай|исследуй"
    r")\b",
    re.IGNORECASE,
)
_DOTA_CONTEXT_RE = re.compile(
    r"\b(dota\s*2?|дот[аеуы]?|доте|dotabuff|opendota|stratz|steamid)\b",
    re.IGNORECASE,
)
_PLAYER_NICK_RE = re.compile(
    r"\b(?:ником|ник(?:ом)?|nickname|player)\s+([A-Za-z0-9_.-]{2,64})\b",
    re.IGNORECASE,
)
_PLAYER_BEFORE_WORD_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_.-]{2,63})\s+"
    r"(?:игрок\w*|player|дот\w*|dota\s*2?)\b",
    re.IGNORECASE,
)
_LATIN_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_.-]{2,63}\b")
_NON_PLAYER_TOKENS = {
    "dota",
    "dotabuff",
    "opendota",
    "stratz",
    "steam",
    "steamid",
    "match",
    "player",
    "public",
    "profile",
}
_INTERACTIVE_TEST_OR_FORM_RE = re.compile(
    r"\b(пройд\w*|проход\w*|заполн\w*|проголос\w*|зарегистр\w*|авториз\w*)\b"
    r".{0,120}"
    r"\b(тест\w*|опрос\w*|анкет\w*|форм\w*|квиз\w*|quiz|survey|poll|form)\b|"
    r"\b(тест\w*|опрос\w*|анкет\w*|форм\w*|квиз\w*|quiz|survey|poll|form)\b"
    r".{0,120}"
    r"\b(пройд\w*|проход\w*|заполн\w*|проголос\w*|зарегистр\w*|авториз\w*)\b",
    re.IGNORECASE,
)
_INTERACTIVE_COMPLETION_RE = re.compile(
    r"\b(пройд\w*|проход\w*|заполн\w*|ответь|ответить|выбери|выбер\w*)\b"
    r".{0,140}"
    r"\b(тест\w*|опрос\w*|анкет\w*|форм\w*|квиз\w*|quiz|survey|poll|form)\b|"
    r"\b(тест\w*|опрос\w*|анкет\w*|форм\w*|квиз\w*|quiz|survey|poll|form)\b"
    r".{0,140}"
    r"\b(пройд\w*|проход\w*|заполн\w*|ответь|ответить|выбери|выбер\w*)\b",
    re.IGNORECASE,
)
_UNSAFE_INTERACTIVE_TASK_RE = re.compile(
    r"\b("
    r"авториз\w*|зарегистр\w*|логин\w*|войд\w*|парол\w*|"
    r"оплат\w*|куп\w*|покуп\w*|заказ\w*|удал\w*|"
    r"publish|purchase|checkout|login|password|register"
    r")\b",
    re.IGNORECASE,
)
_INTERACTIVE_BROWSER_CONTROL_RE = re.compile(
    r"\b(нажм\w*|нажат\w*|клик\w*|выбер\w*|выбери|submit|отправ\w*)\b"
    r".{0,120}"
    r"\b(кнопк\w*|вариант\w*|ответ\w*|форм\w*|анкет\w*|submit|результат\w*)\b|"
    r"\b(ответь|ответить)\b.{0,120}"
    r"\b(вопрос\w*|вариант\w*|тест\w*|опрос\w*|анкет\w*)\b",
    re.IGNORECASE,
)
_DESKTOP_SCREENSHOT_RE = re.compile(
    r"\b(скрин|скриншот|screenshot|снимок)\b.{0,80}"
    r"\b(рабоч\w*\s+стол\w*|desktop|экран\w*)\b|"
    r"\b(рабоч\w*\s+стол\w*|desktop|экран\w*)\b.{0,80}"
    r"\b(скрин|скриншот|screenshot|снимок)\b",
    re.IGNORECASE,
)
_BROWSER_STATUS_RE = re.compile(
    r"\b(статус|проверь|доступ\w*)\b.{0,80}"
    r"\b(жив\w*\s+браузер\w*|live\s+browser|chrome\s+debug|browser)\b",
    re.IGNORECASE,
)
_SCREENSHOT_RE = re.compile(r"\b(скрин|скриншот|screenshot|снимок)\b", re.IGNORECASE)
_DEFAULT_DISCOVERY_SEARCH_URL = "https://www.bing.com/search?q="
_DOTABUFF_PLAYER_SEARCH_URL = "https://www.dotabuff.com/search?q="
_STEAM_USER_SEARCH_URL = "https://steamcommunity.com/search/users/#text="
_MAX_OBSERVATION_TEXT_CHARS = 8_000


class ComputerUseRuntime(Protocol):
    """Minimal AgentRuntime contract used by ComputerUseSkill."""

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
class ComputerUseSkillConfig:
    """Stable routing/configuration for live browser and GUI actions."""

    max_command_chars: int = 4_000


@dataclass(frozen=True)
class _ComputerUseAction:
    action: ComputerUseActionKind
    payload: dict[str, Any]
    requested_task_requires_multi_step_interaction: bool = False


_TOOL_APPROVAL_REQUIRED_ACTIONS = {
    ComputerUseActionKind.BROWSER_SUBMIT,
    ComputerUseActionKind.DESKTOP_INPUT,
    ComputerUseActionKind.DESKTOP_WINDOW_CONTROL,
    ComputerUseActionKind.DESKTOP_APP_LAUNCHER,
    ComputerUseActionKind.DESKTOP_HOTKEYS,
    ComputerUseActionKind.DESKTOP_MEDIA_CONTROL,
    ComputerUseActionKind.DESKTOP_SHELL_COMMAND,
}


class ComputerUseSkill(InlineSkill):
    """Run bounded live browser and GUI actions from ordinary personal chat."""

    name: ClassVar[str] = "computer_use"
    description: ClassVar[str] = "Live browser and GUI actions через Agent Runtime"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"
    triggers: ClassVar[list[str]] = [COMPUTER_USE_PREFIX]
    route_classifier_always_normalize: ClassVar[bool] = True
    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "low"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = "auto"
    side_effects: ClassVar[list[SideEffect]] = [
        SideEffect.NETWORK_IO_EXTERNAL,
        SideEffect.DELEGATES_TO_OTHER_AGENT,
    ]
    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    def __init__(
        self,
        *,
        admin_user_id: int,
        runtime: ComputerUseRuntime,
        profile: InvocationProfile = COMPUTER_USE_ACTIVE_GUI,
        shell_profile: InvocationProfile = COMPUTER_USE_APPROVED_SHELL,
        config: ComputerUseSkillConfig | None = None,
    ) -> None:
        self._admin_user_id = admin_user_id
        self._runtime = runtime
        self._profile = profile
        self._shell_profile = shell_profile
        self._config = config or ComputerUseSkillConfig()
        self._detector = IrreversibleActionDetector()

    async def can_handle(self, message: str, context: AgentContext) -> float:
        if context.mode != "personal" or context.user_id != self._admin_user_id:
            return 0.0
        text = message.strip()
        if not text:
            return 0.0
        if text.startswith(COMPUTER_USE_PREFIX):
            return 0.95
        if text.startswith("/"):
            return 0.0
        if _looks_like_local_computer_research_task(text):
            return 0.93
        if _looks_like_interactive_browser_task(text) or _looks_like_desktop_task(text):
            return 0.93
        if _looks_like_browser_status_task(text):
            return 0.92
        return 0.0

    def requires_approval_for_message(
        self,
        message: str,
        context: AgentContext,
    ) -> bool:
        action, _error = _action_for_message_or_router_metadata(
            message,
            context,
            max_chars=self._config.max_command_chars,
        )
        return bool(
            action is not None
            and (
                action.action in _TOOL_APPROVAL_REQUIRED_ACTIONS
                or self._detector.inspect(_request_for_action(action)).requires_approval
            )
        )

    async def prepare(self, message: str, context: AgentContext) -> ExecutionPlan:
        action, error = _action_for_message_or_router_metadata(
            message,
            context,
            max_chars=self._config.max_command_chars,
        )
        if action is None:
            summary = error or "Нужен action для computer-use."
            metadata: dict[str, Any] = {"requires_user_input": True}
        else:
            decision = self._detector.inspect(_request_for_action(action))
            required_capability = (
                decision.required_capability
                or _approval_capability_for_action(action.action)
            )
            summary = _approval_summary_for_action(
                action=action,
                decision=decision,
                required_capability=required_capability,
            )
            metadata = {
                "computer_use_action": action.action.value,
                "approval_required_capability": required_capability,
                "approval_risk_class": decision.risk_class.value,
                "approval_risk_summary": decision.risk_summary,
                "approval_prompt": decision.approval_prompt,
                "approval_scope": action.payload.get("approval_scope", {}),
            }
        return ExecutionPlan(
            skill_name=self.name,
            skill_type="inline",
            human_summary=summary,
            estimated_tokens=1500,
            estimated_cost_usd=Decimal("0.003"),
            estimated_duration_seconds=5.0,
            llm_calls_planned=0,
            side_effects_invoked=list(self.side_effects),
            metadata=metadata,
        )

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        action, error = _action_for_message_or_router_metadata(
            message,
            context,
            max_chars=self._config.max_command_chars,
        )
        if error or action is None:
            return _missing_input_result(error or "Нужен action для computer-use.")

        payload_json = json.dumps(
            action.payload,
            ensure_ascii=False,
            sort_keys=True,
        )
        metadata = {
            "source": str(context.metadata.get("source", "")),
            "interface": str(context.metadata.get("interface", "")),
            "skill": self.name,
            "computer_use_payload": payload_json,
        }
        if context.metadata.get("skill_approval_granted") is True:
            approval_id = str(context.metadata.get("skill_approval_id", "")).strip()
            if approval_id:
                approval_capability = _approval_capability_for_runtime_action(
                    action,
                    self._detector,
                )
                metadata["agent_tool_approval_id"] = approval_id
                metadata["agent_tool_approval_capabilities"] = approval_capability

        profile = _profile_for_action(
            action.action,
            active_gui_profile=self._profile,
            shell_profile=self._shell_profile,
        )
        job = await self._runtime.create_job(
            owner_user_id=context.user_id,
            chat_id=context.chat_id or context.user_id,
            source_message_id=str(context.message_id or ""),
            fingerprint=_fingerprint(context=context, payload=action.payload),
            kind=f"computer_use.action.{action.action.value}",
            profile=profile,
            context_pack=ContextPack(
                user_request=message.strip(),
                constraints=_context_constraints_for_action(action.action),
                metadata=metadata,
            ),
        )
        completed = await self._runtime.start(job.id)
        if completed.result is None:
            reason = completed.error or "computer_use job did not return a capsule"
            return SkillResult(
                success=False,
                response="",
                metadata={
                    "skill_name": self.name,
                    "agent_job_id": completed.id,
                    "agent_profile": profile.id,
                    "requires_zhvusha_response": True,
                    "body_observation": {
                        "event": "computer_use_action_failed",
                        "source": self.name,
                        "selected_action": action.action.value,
                        "reason": reason,
                        "agent_job_id": completed.id,
                        "agent_profile": self._profile.id,
                        "instruction": (
                            "Объясни пользователю, что computer-use action "
                            "не завершился. Не говори, что chat tools отсутствуют; "
                            "назови конкретную runtime причину."
                        ),
                    },
                },
            )

        body_observation = _body_observation_from_capsule(
            message=message,
            action=action,
            capsule=completed.result,
            agent_job_id=completed.id,
            agent_profile=profile.id,
        )
        return SkillResult(
            success=True,
            response="",
            metadata={
                "skill_name": self.name,
                "agent_job_id": completed.id,
                "agent_profile": self._profile.id,
                "artifacts": tuple(completed.result.artifacts),
                "deliver_artifacts_to_chat": _action_requests_artifact_delivery(
                    action,
                    message,
                ),
                "requires_zhvusha_response": True,
                "body_observation": body_observation,
            },
        )


def _looks_like_interactive_browser_task(text: str) -> bool:
    has_browser_surface = (
        _URL_RE.search(text) is not None
        or _INTERACTIVE_BROWSER_SURFACE_RE.search(text) is not None
    )
    if not has_browser_surface:
        return False
    return bool(
        _INTERACTIVE_TEST_OR_FORM_RE.search(text)
        or _INTERACTIVE_BROWSER_CONTROL_RE.search(text)
    )


def _looks_like_desktop_task(text: str) -> bool:
    return bool(_DESKTOP_SCREENSHOT_RE.search(text))


def _looks_like_browser_status_task(text: str) -> bool:
    return bool(_BROWSER_STATUS_RE.search(text))


def _action_from_message(
    message: str,
    *,
    max_chars: int,
) -> tuple[_ComputerUseAction | None, str]:
    text = message.strip()
    if len(text) > max_chars:
        return None, "Команда computer-use слишком длинная."
    if text.startswith(COMPUTER_USE_PREFIX):
        return _action_from_json_command(text)

    if _looks_like_local_computer_research_task(text):
        return _local_computer_research_action(text), ""

    if _looks_like_desktop_task(text):
        return _desktop_screenshot_action(), ""

    browser_action = _browser_action_from_text(text)
    if browser_action is not None:
        return browser_action, ""

    if _looks_like_browser_status_task(text):
        return _browser_status_action(), ""

    return None, "Нужен URL, browser status request или desktop screenshot target."


def _action_for_message_or_router_metadata(
    message: str,
    context: AgentContext,
    *,
    max_chars: int,
) -> tuple[_ComputerUseAction | None, str]:
    if message.strip().startswith(COMPUTER_USE_PREFIX):
        return _action_from_message(message, max_chars=max_chars)

    action, error = _action_from_router_metadata(message, context)
    if action is not None or error:
        return action, error
    return _action_from_message(message, max_chars=max_chars)


def _looks_like_local_computer_research_task(text: str) -> bool:
    return bool(
        _LOCAL_COMPUTER_ACCESS_RE.search(text)
        and _LOCAL_RESEARCH_INTENT_RE.search(text)
    )


def _local_computer_research_action(text: str) -> _ComputerUseAction:
    player_query = _player_query_from_text(text)
    is_dota_task = _DOTA_CONTEXT_RE.search(text) is not None
    metadata: dict[str, Any] = {
        "sources": [
            "Steam existing session if already open",
            "Dotabuff",
            "OpenDota",
            "Stratz",
        ]
        if is_dota_task
        else ["already-open local session if relevant", "public web pages"],
    }
    if player_query:
        metadata["player_query"] = player_query
    if is_dota_task:
        metadata["game"] = "Dota 2"

    if is_dota_task and player_query:
        goal = (
            f"получить SteamID/профиль или match ID игрока {player_query} и "
            "собрать публичную статистику для анализа"
        )
    else:
        goal = (
            "найти запрошенные данные через уже открытые локальные сессии и "
            "публичные web pages"
        )

    payload = {
        "action": ComputerUseActionKind.BROWSER_INTERACTIVE_TASK.value,
        "text": text,
        "goal": goal,
        "constraints": [
            "use_existing_session_only",
            "do_not_enter_credentials",
            "do_not_click_sign_in",
            "do_not_send_messages",
            "do_not_add_friend",
            "do_not_change_account_settings",
            "do_not_submit_without_approval",
            "prefer_public_sources",
        ],
        "artifact_requirements": {
            "screenshots": "relevant_profile_and_stats_pages_if_available"
            if is_dota_task
            else "relevant_result_pages_if_available",
            "links": "Steam/Dotabuff/OpenDota/Stratz/profile_or_match_urls"
            if is_dota_task
            else "source_urls_or_precise_blocker",
            "deliver_to_chat": "true",
        },
        "success_criteria": [
            "requested_public_data_found_or_precise_blocker_reported",
            "evidence_links_or_artifacts_collected",
            "analysis_contains_limitations",
        ],
        "risk_intent": ComputerUseRiskClass.READONLY_EXISTING_SESSION.value,
        "approval_scope": {
            "allowed": "read already-open local session data and public web pages",
            "forbidden": (
                "password entry, 2FA, sign-in click, sending messages, friend "
                "requests, account mutation, purchase, delete, shell"
            ),
        },
        "metadata": metadata,
    }
    normalized = _normalize_router_payload(
        payload=payload,
        action=ComputerUseActionKind.BROWSER_INTERACTIVE_TASK,
        fallback_task_text=text,
    )
    return _ComputerUseAction(
        action=ComputerUseActionKind.BROWSER_INTERACTIVE_TASK,
        payload=normalized,
        requested_task_requires_multi_step_interaction=False,
    )


def _player_query_from_text(text: str) -> str:
    explicit = _PLAYER_NICK_RE.search(text)
    if explicit is not None:
        token = _clean_player_token(explicit.group(1))
        if token and not _is_non_player_token(token):
            return token
    before_word = _PLAYER_BEFORE_WORD_RE.search(text)
    if before_word is not None:
        token = _clean_player_token(before_word.group(1))
        if token and not _is_non_player_token(token):
            return token
    for match in _LATIN_TOKEN_RE.finditer(text):
        token = _clean_player_token(match.group(0))
        if token and not _is_non_player_token(token):
            return token
    return ""


def _clean_player_token(token: str) -> str:
    return token.strip(".,;:!?`-_")


def _is_non_player_token(token: str) -> bool:
    lowered = token.lower().strip("-_")
    return lowered in _NON_PLAYER_TOKENS or lowered.startswith("dota")


def _action_from_router_metadata(
    message: str,
    context: AgentContext,
) -> tuple[_ComputerUseAction | None, str]:
    if context.metadata.get("skill_router_selected_skill") != ComputerUseSkill.name:
        return None, ""
    payload = _metadata_payload(context.metadata.get("skill_router_normalized_action"))
    if not payload:
        return None, ""
    try:
        action = ComputerUseActionKind(str(payload.get("action", "")).strip())
    except ValueError:
        return None, "Worker route classifier вернул некорректный computer-use action."
    normalized_payload = _normalize_router_payload(
        payload={**payload, "action": action.value},
        action=action,
        fallback_task_text=message,
    )
    return (
        _ComputerUseAction(
            action=action,
            payload=normalized_payload,
            requested_task_requires_multi_step_interaction=(
                action is ComputerUseActionKind.BROWSER_NAVIGATE
            ),
        ),
        "",
    )


def _metadata_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _normalize_router_payload(
    *,
    payload: dict[str, Any],
    action: ComputerUseActionKind,
    fallback_task_text: str,
) -> dict[str, Any]:
    normalized = dict(payload)
    artifact_requirements = _artifact_requirements_from_payload(normalized)
    if artifact_requirements:
        normalized["artifact_requirements"] = artifact_requirements

    metadata = _metadata_payload(normalized.get("metadata"))
    if action is ComputerUseActionKind.BROWSER_INTERACTIVE_TASK:
        task_text = str(normalized.get("text") or fallback_task_text).strip()
        normalized["text"] = task_text
        current_url = str(normalized.get("url", "")).strip()
        if not current_url or _should_replace_router_start_url(current_url):
            start_url, start_strategy = _discovery_start_url(
                normalized=normalized,
                metadata=metadata,
                task_text=task_text,
            )
            normalized["url"] = start_url
            metadata["auto_start_url"] = start_strategy
            if current_url:
                metadata["replaced_start_url"] = _router_start_url_reason(current_url)
        metadata = {
            **_interactive_task_metadata(
                task_text=task_text,
                wants_screenshot=False,
            ),
            **metadata,
        }
        if _browser_task_needs_public_fact_extract(
            normalized=normalized,
            metadata=metadata,
            task_text=task_text,
        ):
            artifact_requirements.setdefault("include_sources", "true")
            artifact_requirements.setdefault(
                "text_extract",
                "relevant_profile_stats_and_visible_page_facts",
            )
            normalized["artifact_requirements"] = artifact_requirements
    if artifact_requirements:
        metadata.update(_artifact_metadata_from_requirements(artifact_requirements))
    if metadata:
        normalized["metadata"] = metadata
    return _clean_payload(normalized)


def _artifact_requirements_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    raw = payload.get("artifact_requirements", {})
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in raw.items()
        if str(key).strip() and str(value).strip()
    }


def _artifact_metadata_from_requirements(
    artifact_requirements: dict[str, str],
) -> dict[str, str]:
    metadata: dict[str, str] = {}
    screenshots = artifact_requirements.get("screenshots", "").strip()
    if _artifact_requirement_enabled(screenshots):
        metadata["capture_screenshot"] = "true"
        metadata["capture_result_screenshots"] = screenshots
        metadata["capture_page_html"] = "true"
    if any(
        _artifact_requirement_enabled(artifact_requirements.get(key, ""))
        for key in ("text_extract", "include_sources", "interpretation")
    ):
        metadata["capture_page_html"] = "true"
    if _artifact_requirement_enabled(artifact_requirements.get("deliver_to_chat", "")):
        metadata["deliver_to_chat"] = "true"
    return metadata


def _browser_task_needs_public_fact_extract(
    *,
    normalized: dict[str, Any],
    metadata: dict[str, Any],
    task_text: str,
) -> bool:
    artifact_requirements = _artifact_requirements_from_payload(normalized)
    if _artifact_requirement_enabled(artifact_requirements.get("text_extract", "")):
        return False
    if _artifact_requirement_enabled(artifact_requirements.get("include_sources", "")):
        return True
    haystack = " ".join(
        str(part)
        for part in (
            task_text,
            normalized.get("goal", ""),
            normalized.get("success_criteria", ""),
            normalized.get("artifact_requirements", ""),
        )
    ).casefold()
    wants_analysis = any(
        marker in haystack
        for marker in (
            "analysis",
            "analy",
            "facts",
            "profile",
            "stats",
            "анал",
            "профил",
            "разбор",
            "стат",
            "факт",
        )
    )
    return wants_analysis and (
        _dota_profile_discovery_requested(
            metadata=metadata,
            task_text=task_text,
            current_url=str(normalized.get("url", "")),
        )
        or _task_text_points_to_dota_profile_sources(task_text)
        or _metadata_points_to_dota_profile_sources(metadata)
    )


def _artifact_requirement_enabled(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized not in {"", "0", "false", "no", "none", "нет"}


def _discovery_start_url(
    *,
    normalized: dict[str, Any],
    metadata: dict[str, Any],
    task_text: str,
) -> tuple[str, str]:
    player_query = _player_query_from_metadata(metadata)
    if not player_query:
        player_query = _player_query_from_text(
            " ".join(
                str(part)
                for part in (
                    task_text,
                    normalized.get("text", ""),
                    normalized.get("goal", ""),
                    normalized.get("url", ""),
                )
                if str(part).strip()
            )
        )
    current_url = str(normalized.get("url", ""))
    if player_query and _dota_profile_discovery_requested(
        metadata=metadata,
        task_text=task_text,
        current_url=current_url,
    ):
        return _DOTABUFF_PLAYER_SEARCH_URL + quote_plus(
            player_query[:120]
        ), "dotabuff_player_search"

    if player_query and _steam_profile_discovery_requested(
        metadata=metadata,
        task_text=task_text,
        current_url=current_url,
    ):
        return _STEAM_USER_SEARCH_URL + quote(player_query[:120]), "steam_user_search"

    query_parts = _public_search_query_parts(
        normalized=normalized,
        metadata=metadata,
    )
    if not query_parts and task_text:
        query_parts.append(task_text)
    query = " ".join(dict.fromkeys(query_parts)) or "public web research"
    return _DEFAULT_DISCOVERY_SEARCH_URL + quote_plus(query[:500]), "public_search"


def _public_search_query_parts(
    *,
    normalized: dict[str, Any],
    metadata: dict[str, Any],
) -> list[str]:
    query_parts: list[str] = []
    for key in (
        "player_query",
        "target_nick",
        "target_player",
        "player_name",
        "player_nickname",
        "nickname",
        "nick",
        "username",
        "game",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            query_parts.append(value.strip())
    sources = metadata.get("sources")
    if isinstance(sources, list):
        query_parts.extend(str(item).strip() for item in sources if str(item).strip())
    elif isinstance(sources, str) and sources.strip():
        query_parts.append(sources.strip())
    for key in ("goal", "text"):
        value = normalized.get(key)
        if isinstance(value, str) and value.strip():
            query_parts.append(value.strip())
    return query_parts


def _dota_profile_discovery_requested(
    *,
    metadata: dict[str, Any],
    task_text: str,
    current_url: str,
) -> bool:
    return bool(
        _metadata_points_to_dota_profile_sources(metadata)
        or _task_text_points_to_dota_profile_sources(task_text)
        or _task_text_points_to_dota_profile_sources(current_url)
    )


def _steam_profile_discovery_requested(
    *,
    metadata: dict[str, Any],
    task_text: str,
    current_url: str,
) -> bool:
    return bool(
        _metadata_points_to_steam(metadata)
        or _task_text_points_to_steam(current_url)
        or _task_text_points_to_steam(task_text)
    )


def _should_replace_router_start_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if host in {"duckduckgo.com", "www.duckduckgo.com"}:
        return True
    if host in {"google.com", "www.google.com"}:
        return parsed.path in {"", "/search"}
    if host in {"bing.com", "www.bing.com"}:
        return parsed.path in {"", "/search"}
    if host in {"yandex.ru", "www.yandex.ru", "ya.ru", "www.ya.ru"}:
        return parsed.path in {"", "/search", "/yandsearch"}
    return False


def _router_start_url_reason(url: str) -> str:
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if host in {"duckduckgo.com", "www.duckduckgo.com"}:
        return "duckduckgo_search"
    if host in {"google.com", "www.google.com"}:
        return "google_search"
    if host in {"bing.com", "www.bing.com"}:
        return "bing_search"
    if host in {"yandex.ru", "www.yandex.ru", "ya.ru", "www.ya.ru"}:
        return "yandex_search"
    return "router_start_url"


def _player_query_from_metadata(metadata: dict[str, Any]) -> str:
    for key in (
        "player_query",
        "target_nick",
        "target_player",
        "player_name",
        "player_nickname",
        "nickname",
        "nick",
        "username",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = metadata.get("query")
    if isinstance(value, str) and value.strip():
        return _player_query_from_text(value)
    return ""


def _metadata_points_to_dota_profile_sources(metadata: dict[str, Any]) -> bool:
    values: list[str] = []
    for key in ("game", "sources", "preferred_sources", "domain_hints"):
        value = metadata.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list | tuple):
            values.extend(str(item) for item in value)
    source_text = " ".join(values).lower()
    return any(
        marker in source_text
        for marker in (
            "dota",
            "dotabuff",
            "opendota",
            "stratz",
        )
    )


def _metadata_points_to_steam(metadata: dict[str, Any]) -> bool:
    values: list[str] = []
    for key in ("sources", "preferred_sources", "domain_hints"):
        value = metadata.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list | tuple):
            values.extend(str(item) for item in value)
    source_text = " ".join(values).lower()
    return "steam" in source_text or "стим" in source_text


def _task_text_points_to_dota_profile_sources(text: str) -> bool:
    lowered = text.lower()
    if any(
        marker in lowered
        for marker in (
            "dota",
            "дота",
            "доте",
            "dotabuff",
            "opendota",
            "stratz",
        )
    ):
        return True
    parsed = urlparse(text.strip())
    if _should_replace_router_start_url(text):
        query = parse_qs(parsed.query).get("q", ())
        return any(_task_text_points_to_dota_profile_sources(item) for item in query)
    return False


def _task_text_points_to_steam(text: str) -> bool:
    lowered = text.lower()
    if "steam" in lowered or "стим" in lowered or "steamid" in lowered:
        return True
    parsed = urlparse(text.strip())
    if _should_replace_router_start_url(text):
        query = parse_qs(parsed.query).get("q", ())
        return any(_task_text_points_to_steam(item) for item in query)
    return False


def _action_from_json_command(text: str) -> tuple[_ComputerUseAction | None, str]:
    raw = text.removeprefix(COMPUTER_USE_PREFIX).strip()
    if not raw:
        return None, "Нужен JSON payload для /computer_use."
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"JSON payload не разобран: {exc.msg}."
    if not isinstance(payload, dict):
        return None, "JSON payload должен быть объектом."
    try:
        action = ComputerUseActionKind(str(payload.get("action", "")).strip())
    except ValueError:
        return None, "Нужен корректный action для /computer_use."
    normalized_payload = _normalize_router_payload(
        payload={**payload, "action": action.value},
        action=action,
        fallback_task_text=raw,
    )
    return (
        _ComputerUseAction(
            action=action,
            payload=normalized_payload,
        ),
        "",
    )


def _desktop_screenshot_action() -> _ComputerUseAction:
    return _ComputerUseAction(
        action=ComputerUseActionKind.DESKTOP_SCREENSHOT,
        payload={"action": ComputerUseActionKind.DESKTOP_SCREENSHOT.value},
    )


def _browser_action_from_text(text: str) -> _ComputerUseAction | None:
    url = _first_url(text)
    if not url:
        return None
    if _looks_like_interactive_completion_task(text):
        return _interactive_task_action(
            url,
            task_text=text,
            wants_screenshot=_wants_artifact_delivery(text),
        )
    return _navigate_action(url, wants_screenshot=_wants_artifact_delivery(text))


def _browser_status_action() -> _ComputerUseAction:
    return _ComputerUseAction(
        action=ComputerUseActionKind.BROWSER_STATUS,
        payload={"action": ComputerUseActionKind.BROWSER_STATUS.value},
    )


def _navigate_action(url: str, *, wants_screenshot: bool) -> _ComputerUseAction:
    metadata = {"capture_screenshot": "true"} if wants_screenshot else {}
    return _ComputerUseAction(
        action=ComputerUseActionKind.BROWSER_NAVIGATE,
        payload={
            "action": ComputerUseActionKind.BROWSER_NAVIGATE.value,
            "url": url,
            "metadata": metadata,
        },
        requested_task_requires_multi_step_interaction=True,
    )


def _interactive_task_action(
    url: str,
    *,
    task_text: str,
    wants_screenshot: bool,
) -> _ComputerUseAction:
    metadata = _interactive_task_metadata(
        task_text=task_text,
        wants_screenshot=wants_screenshot,
    )
    return _ComputerUseAction(
        action=ComputerUseActionKind.BROWSER_INTERACTIVE_TASK,
        payload={
            "action": ComputerUseActionKind.BROWSER_INTERACTIVE_TASK.value,
            "url": url,
            "text": task_text,
            "metadata": metadata,
        },
        requested_task_requires_multi_step_interaction=False,
    )


def _interactive_task_metadata(
    *,
    task_text: str,
    wants_screenshot: bool,
) -> dict[str, str]:
    metadata = {
        "answer_policy": (
            "use_zhvusha_personality_reference_for_opinion_preference_and_"
            "self_assessment_choices; use_user_task_for_goal_directed_choices; "
            "ask_if_credentials_payment_private_data_or_real_identity_are_needed"
        ),
        "persona_context_ref": "workspace://personality/current-summary",
        "persona_context_mode": "reference_only",
        "task_intent": _bounded_metadata_value(task_text),
    }
    if wants_screenshot:
        metadata["capture_screenshot"] = "true"
    return metadata


def _looks_like_interactive_completion_task(text: str) -> bool:
    if _UNSAFE_INTERACTIVE_TASK_RE.search(text):
        return False
    return bool(_INTERACTIVE_COMPLETION_RE.search(text))


def _context_constraints_for_action(action: ComputerUseActionKind) -> tuple[str, ...]:
    constraints = [
        "computer_use_active_gui",
        "do_not_login_purchase_publish_delete_or_send",
        "return_structured_observation_to_zhvusha",
    ]
    if action is ComputerUseActionKind.DESKTOP_SHELL_COMMAND:
        constraints.append("shell_requires_scoped_approval_and_structured_argv")
    else:
        constraints.append("shell_disabled")
    if action is ComputerUseActionKind.BROWSER_INTERACTIVE_TASK:
        constraints.append("isolated_interactive_browser_task_allowed")
    elif action.value.startswith("desktop_"):
        constraints.append("desktop_toolgateway_approval_or_hard_stop_enforced")
    else:
        constraints.append("browser_submit_hard_stop")
    return tuple(constraints)


def _request_for_action(action: _ComputerUseAction) -> ComputerUseActionRequest:
    return coerce_computer_use_action_request(action.payload)


def _approval_capability_for_runtime_action(
    action: _ComputerUseAction,
    detector: IrreversibleActionDetector,
) -> str:
    decision = detector.inspect(_request_for_action(action))
    return decision.required_capability or _approval_capability_for_action(
        action.action
    )


def _approval_capability_for_action(action: ComputerUseActionKind) -> str:
    if action is ComputerUseActionKind.DESKTOP_SHELL_COMMAND:
        return "desktop.shell"
    if action in _TOOL_APPROVAL_REQUIRED_ACTIONS:
        return _capability_for_action(action)
    return _capability_for_action(action)


def _capability_for_action(action: ComputerUseActionKind) -> str:
    if action is ComputerUseActionKind.BROWSER_STATUS:
        return "browser_live_control"
    if action is ComputerUseActionKind.DESKTOP_SHELL_COMMAND:
        return "desktop.shell"
    return action.value


def _profile_for_action(
    action: ComputerUseActionKind,
    *,
    active_gui_profile: InvocationProfile,
    shell_profile: InvocationProfile,
) -> InvocationProfile:
    if action is ComputerUseActionKind.DESKTOP_SHELL_COMMAND:
        return shell_profile
    return active_gui_profile


def _approval_summary_for_action(
    *,
    action: _ComputerUseAction,
    decision: Any,
    required_capability: str,
) -> str:
    goal = str(action.payload.get("goal") or action.payload.get("text") or "").strip()
    label = goal or action.action.value
    if decision.risk_class is ComputerUseRiskClass.REVERSIBLE_GUI_ACTION:
        return f"Разрешить computer-use action `{action.action.value}`: {label}"
    risk = decision.risk_summary or decision.risk_class.value
    return (
        f"Разрешить scoped `{required_capability}` для `{action.action.value}`: "
        f"{label}. Риск: {risk}"
    )


def _clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key == "action" or str(value).strip()
    }


def _bounded_metadata_value(value: str, *, limit: int = 700) -> str:
    cleaned = value.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def _first_url(text: str) -> str:
    match = _URL_RE.search(text)
    if match is None:
        return ""
    return match.group(0).rstrip(".,;:!?)]}")


def _wants_artifact_delivery(message: str) -> bool:
    return bool(_SCREENSHOT_RE.search(message))


def _action_requests_artifact_delivery(
    action: _ComputerUseAction,
    message: str,
) -> bool:
    artifact_requirements = _artifact_requirements_from_payload(action.payload)
    if _artifact_requirement_enabled(artifact_requirements.get("deliver_to_chat", "")):
        return True
    if _artifact_requirement_enabled(artifact_requirements.get("screenshots", "")):
        return True
    metadata = _metadata_payload(action.payload.get("metadata"))
    if _artifact_requirement_enabled(str(metadata.get("deliver_to_chat", ""))):
        return True
    if _artifact_requirement_enabled(str(metadata.get("capture_screenshot", ""))):
        return True
    return _wants_artifact_delivery(message)


def _missing_input_result(reason: str) -> SkillResult:
    return SkillResult(
        success=False,
        response="",
        metadata={
            "skill_name": ComputerUseSkill.name,
            "requires_zhvusha_response": True,
            "body_observation": {
                "event": "missing_required_input",
                "source": ComputerUseSkill.name,
                "reason": reason,
                "example": '/computer_use {"action":"browser_status"}',
                "instruction": (
                    "Попроси недостающий URL/action для computer-use. "
                    "Не утверждай, что живой браузер отсутствует."
                ),
            },
        },
    )


def _body_observation_from_capsule(
    *,
    message: str,
    action: _ComputerUseAction,
    capsule: ContextCapsule,
    agent_job_id: str,
    agent_profile: str,
) -> dict[str, Any]:
    processed_context = capsule.processed_context or capsule.markdown_report
    instruction = (
        "Это внутреннее наблюдение computer-use body skill. Напиши ответ "
        "как Жвуша. Если status configured_only/degraded/refused/hard_stopped, "
        "назови точный blocker из processed_context. Не говори, что в chat "
        "tools нет живого браузера. Не утверждай, что multi-step web task "
        "выполнен, если capsule подтверждает только status/navigate."
    )
    if action.action is ComputerUseActionKind.BROWSER_INTERACTIVE_TASK:
        instruction += (
            " Если исходная цель ещё не достигнута, но processed_context/page "
            "state показывает безопасный следующий шаг, не проси Никиту дать "
            "ссылку или инструкцию. Выведи следующую команду /computer_use "
            "отдельной строкой, чтобы dispatcher продолжил observe-think-act "
            "цикл через skill gate."
        )
    return {
        "event": "computer_use_action_completed",
        "source": ComputerUseSkill.name,
        "requested_task": message.strip(),
        "selected_action": action.action.value,
        "selected_url": str(action.payload.get("url", "")),
        "requested_task_requires_multi_step_interaction": (
            action.requested_task_requires_multi_step_interaction
        ),
        "summary": capsule.summary,
        "processed_context": _bounded_text(processed_context),
        "findings": [finding.model_dump(mode="json") for finding in capsule.findings],
        "sources": list(capsule.sources),
        "artifacts": list(capsule.artifacts),
        "agent_job_id": agent_job_id,
        "agent_profile": agent_profile,
        "constraints": _observation_constraints_for_action(action.action),
        "instruction": instruction,
    }


def _observation_constraints_for_action(action: ComputerUseActionKind) -> list[str]:
    constraints = [
        "computer_use_active_gui",
        "do_not_fake_screenshots_or_completion",
    ]
    if action is ComputerUseActionKind.DESKTOP_SHELL_COMMAND:
        constraints.append("shell_requires_structured_argv_and_scoped_approval")
    else:
        constraints.append("shell_disabled")
    if action is ComputerUseActionKind.BROWSER_INTERACTIVE_TASK:
        constraints.append("isolated_interactive_browser_task_allowed")
    elif action.value.startswith("desktop_"):
        constraints.append("desktop_toolgateway_approval_or_hard_stop_enforced")
    else:
        constraints.append("browser_submit_hard_stop")
    return constraints


def _bounded_text(text: str) -> str:
    cleaned = text.strip()
    if len(cleaned) <= _MAX_OBSERVATION_TEXT_CHARS:
        return cleaned
    return cleaned[:_MAX_OBSERVATION_TEXT_CHARS].rstrip() + "\n...[truncated]"


def _fingerprint(*, context: AgentContext, payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(
        f"{context.user_id}:{context.chat_id}:{serialized}".encode()
    ).hexdigest()
    return f"computer_use:{digest}"

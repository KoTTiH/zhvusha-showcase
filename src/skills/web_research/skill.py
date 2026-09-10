"""User-facing read-only web research through Agent Runtime."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol
from urllib.parse import quote_plus

from src.agent_runtime.models import ContextPack, InvocationProfile
from src.agent_runtime.profiles import WEB_RESEARCH_READONLY
from src.skills.base import AgentContext, InlineSkill, SideEffect, SkillResult

if TYPE_CHECKING:
    from src.agent_runtime.models import AgentJob, ContextCapsule


WEB_RESEARCH_PREFIX = "/web_research"
_CODEX_OPERATOR_ACTORS = {"codex", "codex_operator", "operator"}
_CODEX_GOAL_LOOP_OPERATOR_KINDS = {
    "goal_loop_handoff",
    "goal_loop_proof_replay",
}
_CODEX_GOAL_HANDOFF_MARKERS = (
    "task packet:",
    "local supervisor summary:",
    "readiness capsule:",
    "physical evidence packet:",
    "no-write proof bundle:",
    "runner state:",
    "decision request:",
    "context capsule:",
    "handoff prompt:",
)
_WEB_ACTION_RE = re.compile(
    r"("
    r"\b(найди|поищи|проверь|изучи|исследуй|узнай)\b.{0,40}"
    r"\b(интернет|web|веб|сети|источник|источники|ссылк|цитат)\b|"
    r"\b(погугли|загугли|web research|найди источники|с источниками|с цитатами)\b|"
    r"\b(latest|current|recent|актуальн|последн|свеж)\b.{0,60}"
    r"\b(источник|источники|ссылк|цитат|release|version|верси)\b"
    r")",
    re.IGNORECASE,
)
_BROWSER_ARTIFACT_ACTION_RE = re.compile(
    r"("
    r"\b(открой|зайди|прочитай|найди|поищи|проверь)\b.{0,90}"
    r"\b(интернет|web|веб|браузер|сайт|страниц\w*|стать\w*|url|ссылк\w*)\b"
    r".{0,140}\b(скрин|скриншот|screenshot|снимок)\b|"
    r"\b(скрин|скриншот|screenshot|снимок)\b.{0,140}"
    r"\b(интернет|web|веб|браузер|сайт|страниц\w*|стать\w*|url|ссылк\w*)\b"
    r")",
    re.IGNORECASE,
)
_PUBLIC_PROFILE_ARTIFACT_ACTION_RE = re.compile(
    r"("
    r"\b(открой|зайди|найди|поищи|проверь)\b.{0,160}"
    r"\b(профил\w*|статистик\w*|игрок\w*|ник\w*|аккаунт\w*|матч\w*|рейтинг\w*)\b"
    r".{0,180}\b(скрин|скриншот|screenshot|снимок)\b|"
    r"\b(скрин|скриншот|screenshot|снимок)\b.{0,180}"
    r"\b(профил\w*|статистик\w*|игрок\w*|ник\w*|аккаунт\w*|матч\w*|рейтинг\w*)\b"
    r")",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://[^\s<>()\"']+", re.IGNORECASE)
_DOTABUFF_RE = re.compile(r"\b(dotabuff|дотабаф+)\b", re.IGNORECASE)
_DOTA_SOURCE_RE = re.compile(
    r"\b(dota\s*2?|дот[аеуы]?|доте|dotabuff|opendota|stratz|steamid|match\s*id)\b",
    re.IGNORECASE,
)
_PUBLIC_GAME_PLAYER_RESEARCH_RE = re.compile(
    r"\b(проанализ\w*|разбер\w*|оцени|найди|поищи|проверь|статистик\w*)\b"
    r".{0,180}"
    r"\b(игрок\w*|player|dota\s*2?|дот[аеуы]?|доте|steamid|dotabuff|opendota|stratz)\b|"
    r"\b(игрок\w*|player|dota\s*2?|дот[аеуы]?|доте|steamid|dotabuff|opendota|stratz)\b"
    r".{0,180}"
    r"\b(проанализ\w*|разбер\w*|оцени|найди|поищи|проверь|статистик\w*)\b",
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
_BROWSER_URL_ACTION_RE = re.compile(
    r"\b("
    r"открой|зайди|прочитай|проверь|"
    r"скрин|скриншот|screenshot|снимок|"
    r"browser-use|browser|браузер\w*"
    r")\b",
    re.IGNORECASE,
)
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
_INTERACTIVE_TEST_OR_FORM_RE = re.compile(
    r"\b(пройд\w*|проход\w*|заполн\w*|проголос\w*|зарегистр\w*|авториз\w*)\b"
    r".{0,120}"
    r"\b(тест\w*|опрос\w*|анкет\w*|форм\w*|квиз\w*|quiz|survey|poll|form)\b|"
    r"\b(тест\w*|опрос\w*|анкет\w*|форм\w*|квиз\w*|quiz|survey|poll|form)\b"
    r".{0,120}"
    r"\b(пройд\w*|проход\w*|заполн\w*|проголос\w*|зарегистр\w*|авториз\w*)\b",
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
_LOCAL_ONLY_RE = re.compile(
    r"\b(в проекте|по проекту|в репозитории|в репе|код|файл|логи|локально)\b",
    re.IGNORECASE,
)
_LEADING_ADDRESS_RE = re.compile(r"^\s*(жвуша|zhvusha)[,:\s]+", re.IGNORECASE)
_TRAILING_ANSWER_DIRECTIVE_RE = re.compile(
    r"\s+(?:и\s+)?"
    r"(дай|сделай|напиши|подготовь|покажи|пришли|ответь)\b.*$",
    re.IGNORECASE,
)
_WEB_RESEARCH_COMMAND_RE = re.compile(
    r"^\s*"
    r"(?:пожалуйста[,:\s]+)?"
    r"(?:найди|поищи|проверь|изучи|исследуй|узнай|погугли|загугли|find|search|research|check)\b"
    r"(?:\s+(?:в|на)\s+(?:интернете|сети|web|вебе|веб))?"
    r"(?:\s+(?:источники?|ссылки?|цитаты?|sources?|links?|citations?))?"
    r"(?:\s+(?:по|про|о|об|about|for|on))?"
    r"\s*",
    re.IGNORECASE,
)
_SUBJECT_AFTER_MARKER_RE = re.compile(
    r"\b(?:по|про|о|об|about|for|on)\s+(.+)$",
    re.IGNORECASE,
)
_SOURCE_QUALIFIER_RE = re.compile(
    r"\b(?:с|со|with)\s+(?:источниками?|ссылками?|цитатами?|sources?|links?|citations?)\b",
    re.IGNORECASE,
)
_BROWSER_RESEARCH_COMMAND_RE = re.compile(
    r"^\s*"
    r"(?:пожалуйста[,:\s]+)?"
    r"(?:открой|зайди|прочитай|найди|поищи|проверь)\b"
    r"(?:\s+(?:в|на)\s+(?:интернете|сети|web|вебе|браузере))?"
    r"(?:\s+(?:какую-нибудь|какую-то|любую|одну))?"
    r"\s*",
    re.IGNORECASE,
)
_SCREENSHOT_DELIVERY_RE = re.compile(
    r"\b(скрин|скриншот|screenshot|снимок)\b",
    re.IGNORECASE,
)
_RANDOM_ARTICLE_RE = re.compile(
    r"("
    r"\b(какую-нибудь|какую-то|любую|случайн\w*)\b.{0,50}\bстать\w*\b|"
    r"\bстать\w*\b.{0,50}\b(какую-нибудь|какую-то|любую|случайн\w*)\b"
    r")",
    re.IGNORECASE,
)
_RANDOM_ARTICLE_URL = "https://ru.wikipedia.org/wiki/Special:Random"
_MAX_OBSERVATION_TEXT_CHARS = 12000


class WebResearchRuntime(Protocol):
    """Minimal AgentRuntime contract used by WebResearchSkill."""

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
class WebResearchSkillConfig:
    """Stable routing/configuration for read-only web research."""

    max_query_chars: int = 600


class WebResearchSkill(InlineSkill):
    """Run source-backed read-only web research from ordinary personal chat."""

    name: ClassVar[str] = "web_research"
    description: ClassVar[str] = "Read-only web research через Agent Runtime"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"
    triggers: ClassVar[list[str]] = [WEB_RESEARCH_PREFIX]
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
        runtime: WebResearchRuntime,
        profile: InvocationProfile = WEB_RESEARCH_READONLY,
        config: WebResearchSkillConfig | None = None,
    ) -> None:
        self._admin_user_id = admin_user_id
        self._runtime = runtime
        self._profile = profile
        self._config = config or WebResearchSkillConfig()

    async def can_handle(self, message: str, context: AgentContext) -> float:
        """Route explicit web/source research requests only."""
        if context.mode != "personal" or context.user_id != self._admin_user_id:
            return 0.0
        text = message.strip()
        if not text:
            return 0.0
        if text.startswith(WEB_RESEARCH_PREFIX):
            return 0.94
        if text.startswith("/"):
            return 0.0
        if _looks_like_codex_goal_handoff(text, context):
            return 0.0
        if _looks_like_web_research_request(text):
            return 0.93
        return 0.0

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        """Create and run a read-only web_research Agent Runtime job."""
        query = _query_from_message(message, max_chars=self._config.max_query_chars)
        if not query:
            return SkillResult(
                success=False,
                response="",
                metadata={
                    "skill_name": self.name,
                    "requires_zhvusha_response": True,
                    "body_observation": {
                        "event": "missing_required_input",
                        "source": self.name,
                        "missing_fields": ["query"],
                        "example": "/web_research Python 3.14 release notes",
                        "instruction": (
                            "Сформулируй короткий естественный вопрос: какой "
                            "запрос нужно исследовать в интернете."
                        ),
                    },
                },
            )
        job = await self._runtime.create_job(
            owner_user_id=context.user_id,
            chat_id=context.chat_id or context.user_id,
            source_message_id=str(context.message_id or ""),
            fingerprint=_fingerprint(context=context, query=query),
            kind="web_research",
            profile=self._profile,
            context_pack=ContextPack(
                user_request=query,
                constraints=(
                    "read_only_web_research",
                    "do_not_submit_forms",
                    "do_not_login",
                    "do_not_purchase_publish_delete_or_send",
                    "cite_sources_in_result",
                ),
                metadata={
                    "source": context.metadata.get("source", ""),
                    "interface": context.metadata.get("interface", ""),
                    "skill": self.name,
                },
            ),
        )
        completed = await self._runtime.start(job.id)
        if completed.result is None:
            reason = completed.error or "web_research job did not return a capsule"
            return SkillResult(
                success=False,
                response="",
                metadata={
                    "skill_name": self.name,
                    "agent_job_id": completed.id,
                    "agent_profile": self._profile.id,
                    "requires_zhvusha_response": True,
                    "body_observation": {
                        "event": "web_research_failed",
                        "source": self.name,
                        "query": query,
                        "reason": reason,
                        "agent_job_id": completed.id,
                        "agent_profile": self._profile.id,
                        "instruction": (
                            "Объясни пользователю, что web research не "
                            "завершился, без сырого runtime traceback."
                        ),
                    },
                },
            )
        sources = tuple(completed.result.sources)
        body_observation = _body_observation_from_capsule(
            query=query,
            capsule=completed.result,
            agent_job_id=completed.id,
            agent_profile=self._profile.id,
        )
        metadata: dict[str, Any] = {
            "skill_name": self.name,
            "agent_job_id": completed.id,
            "agent_profile": self._profile.id,
            "sources": sources,
            "artifacts": tuple(completed.result.artifacts),
            "deliver_artifacts_to_chat": _wants_artifact_delivery(message),
            "requires_zhvusha_response": True,
            "body_observation": body_observation,
        }
        if not sources:
            metadata["body_observation_synthesis_message"] = (
                _no_sources_synthesis_message(
                    query=query,
                    summary=completed.result.summary,
                )
            )
        return SkillResult(
            success=bool(sources),
            response="",
            metadata=metadata,
        )


def _looks_like_web_research_request(text: str) -> bool:
    if _LOCAL_ONLY_RE.search(text) and not re.search(
        r"\b(интернет|web|веб|источник|источники|ссылк|цитат)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    if _looks_like_local_computer_research_task(text):
        return False
    if _looks_like_interactive_browser_task(text):
        return False
    if _looks_like_public_game_player_research(text):
        return True
    if _looks_like_browser_url_request(text):
        return True
    return bool(
        _WEB_ACTION_RE.search(text)
        or _BROWSER_ARTIFACT_ACTION_RE.search(text)
        or _PUBLIC_PROFILE_ARTIFACT_ACTION_RE.search(text)
    )


def _looks_like_codex_goal_handoff(text: str, context: AgentContext) -> bool:
    source_actor = str(context.metadata.get("source_actor", "") or "").lower()
    if source_actor not in _CODEX_OPERATOR_ACTORS:
        return False
    message_kind = str(context.metadata.get("operator_message_kind", "") or "")
    if message_kind in _CODEX_GOAL_LOOP_OPERATOR_KINDS:
        return True
    lowered = text.lower()
    has_operator_header = (
        "codex/operator handoff" in lowered
        or "codex/operator proof replay" in lowered
        or "sender=codex" in lowered
    )
    if not has_operator_header:
        return False
    marker_hits = sum(1 for marker in _CODEX_GOAL_HANDOFF_MARKERS if marker in lowered)
    return marker_hits >= 2


def _query_from_message(message: str, *, max_chars: int) -> str:
    text = message.strip()
    if text.startswith(WEB_RESEARCH_PREFIX):
        text = text.removeprefix(WEB_RESEARCH_PREFIX).strip()
        return _clean_query_text(text)[:max_chars].strip()

    text = _LEADING_ADDRESS_RE.sub("", text)
    text = _TRAILING_ANSWER_DIRECTIVE_RE.sub("", text).strip()
    site_url = _site_specific_url_from_message(text)
    if site_url:
        return site_url[:max_chars].strip()
    game_player_query = _game_player_query_from_message(text)
    if game_player_query:
        return game_player_query[:max_chars].strip()
    if _looks_like_browser_url_request(text):
        url = _first_url(text)
        if url:
            return url[:max_chars].strip()
    marker_match = _SUBJECT_AFTER_MARKER_RE.search(text)
    if marker_match is not None:
        return _clean_query_text(marker_match.group(1))[:max_chars].strip()
    if _wants_random_article(text):
        return _RANDOM_ARTICLE_URL
    command_stripped = _WEB_RESEARCH_COMMAND_RE.sub("", text, count=1).strip()
    if command_stripped and command_stripped != text:
        return _clean_query_text(command_stripped)[:max_chars].strip()
    browser_command_stripped = _BROWSER_RESEARCH_COMMAND_RE.sub(
        "",
        text,
        count=1,
    ).strip()
    if browser_command_stripped and browser_command_stripped != text:
        return _clean_query_text(browser_command_stripped)[:max_chars].strip()
    return _clean_query_text(text)[:max_chars].strip()


def _clean_query_text(text: str) -> str:
    cleaned = _SOURCE_QUALIFIER_RE.sub("", text)
    return cleaned.strip(" \t\r\n.,;:!?`")


def _wants_artifact_delivery(message: str) -> bool:
    return bool(_SCREENSHOT_DELIVERY_RE.search(message))


def _looks_like_browser_url_request(text: str) -> bool:
    return (
        _URL_RE.search(text) is not None
        and _BROWSER_URL_ACTION_RE.search(text) is not None
    )


def _looks_like_public_game_player_research(text: str) -> bool:
    return bool(
        _PUBLIC_GAME_PLAYER_RESEARCH_RE.search(text)
        and _DOTA_SOURCE_RE.search(text)
        and _player_nick_from_game_message(text)
    )


def _game_player_query_from_message(text: str) -> str:
    if not _looks_like_public_game_player_research(text):
        return ""
    nick = _player_nick_from_game_message(text)
    if not nick:
        return ""
    return f"{nick} Dota 2 Dotabuff OpenDota STRATZ SteamID"


def _player_nick_from_game_message(text: str) -> str:
    explicit = _PLAYER_NICK_RE.search(text)
    if explicit is not None:
        return explicit.group(1).strip(".,;:!?`")
    before_word = _PLAYER_BEFORE_WORD_RE.search(text)
    if before_word is not None:
        return before_word.group(1).strip(".,;:!?`")
    for match in _LATIN_TOKEN_RE.finditer(text):
        token = match.group(0).strip(".,;:!?`")
        if token.lower() not in _NON_PLAYER_TOKENS:
            return token
    return ""


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


def _looks_like_local_computer_research_task(text: str) -> bool:
    return bool(
        _LOCAL_COMPUTER_ACCESS_RE.search(text)
        and _LOCAL_RESEARCH_INTENT_RE.search(text)
    )


def _first_url(text: str) -> str:
    match = _URL_RE.search(text)
    if match is None:
        return ""
    return match.group(0).rstrip(".,;:!?)]}")


def _site_specific_url_from_message(text: str) -> str:
    if not _DOTABUFF_RE.search(text):
        return ""
    nick_match = _PLAYER_NICK_RE.search(text)
    if nick_match is None:
        return "https://www.dotabuff.com/"
    nick = nick_match.group(1).strip(".,;:!?`")
    if not nick:
        return "https://www.dotabuff.com/"
    return f"https://www.dotabuff.com/search?utf8=%E2%9C%93&q={quote_plus(nick)}"


def _wants_random_article(text: str) -> bool:
    return bool(_RANDOM_ARTICLE_RE.search(text))


def _body_observation_from_capsule(
    *,
    query: str,
    capsule: ContextCapsule,
    agent_job_id: str,
    agent_profile: str,
) -> dict[str, Any]:
    """Convert a Context Capsule into body-layer data for Жвуша."""
    processed_context = capsule.processed_context or capsule.markdown_report
    readable_source_count = _readable_source_count(capsule)
    artifact_only = bool(
        capsule.sources and capsule.artifacts and not readable_source_count
    )
    return {
        "event": "web_research_completed",
        "source": WebResearchSkill.name,
        "query": query,
        "summary": capsule.summary,
        "processed_context": _bounded_text(processed_context),
        "findings": [finding.model_dump(mode="json") for finding in capsule.findings],
        "sources": list(capsule.sources),
        "artifacts": list(capsule.artifacts),
        "readable_source_count": readable_source_count,
        "artifact_only": artifact_only,
        "agent_job_id": agent_job_id,
        "agent_profile": agent_profile,
        "constraints": [
            "read_only_web_research",
            "do_not_submit_forms",
            "do_not_login",
            "cite_sources_in_answer",
        ],
        "instruction": (
            "Это внутреннее наблюдение read-only web research. Напиши ответ "
            "как Жвуша: синтезируй выводы, дай ссылки на источники, явно "
            "отдели подтверждённое от непроверенного. Не показывай сырой "
            "Context Capsule, служебные next_actions или handoff-инструкции."
        ),
    }


def _no_sources_synthesis_message(*, query: str, summary: str) -> str:
    """Constrain synthesis when web research could not ground the answer."""
    return (
        "Read-only web research не дал проверенных источников для запроса: "
        f"{query!r}. Итог runtime: {summary!r}. "
        "Не отвечай на исследовательский вопрос из памяти, не добавляй даты, "
        "версии, факты, URL или цитаты, которых нет в BODY_OBSERVATION. "
        "Кратко скажи, что source-backed проверка не получилась, и предложи "
        "повторить запрос или дать конкретные URL."
    )


def _readable_source_count(capsule: ContextCapsule) -> int:
    return sum(
        1
        for finding in capsule.findings
        if finding.claim.startswith("Источник прочитан через browser_read_url:")
    )


def _bounded_text(text: str) -> str:
    cleaned = text.strip()
    if len(cleaned) <= _MAX_OBSERVATION_TEXT_CHARS:
        return cleaned
    return cleaned[:_MAX_OBSERVATION_TEXT_CHARS].rstrip() + "\n...[truncated]"


def _fingerprint(*, context: AgentContext, query: str) -> str:
    digest = hashlib.sha256(
        f"{context.user_id}:{context.chat_id}:{query}".encode()
    ).hexdigest()
    return f"web_research:{digest}"

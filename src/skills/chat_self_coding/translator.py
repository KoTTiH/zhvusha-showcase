"""Technical-to-architectural translator for ``chat_self_coding`` (Phase 40).

The Editor and Architect cycles produce technical audit-log lines —
commit summaries, diff bullet points, error stack traces — that look
fine to a developer but read as noise to an orchestrator. This module
asks the worker tier (Haiku) to rewrite those lines as architectural
prose: «Расширила систему пресетов: они теперь умеют ограничивать время
поиска» rather than «Added budget_seconds field to ResearchPreset
dataclass».

Design choices:

* **One LLM call per unique input + kind.** Translations are cached in
  Redis (24h TTL by default) keyed by sha256 of the technical text and
  the ``TranslationKind`` tag. The same commit summary therefore costs
  one Haiku call, not one per Telegram session.
* **Best-effort caching.** Redis errors fall through to a direct LLM
  call; the user never sees a translation gap because of cache misery.
* **Empty-output fallback.** If the LLM emits whitespace only — rare
  but possible at temperature 0.0 — we return the original technical
  text rather than an empty Telegram bubble. Architectural is the
  ideal; technical-but-visible beats invisible.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

import structlog

from src.llm.protocols import LLMRequest

if TYPE_CHECKING:
    from src.llm.protocols import LLMGatewayProtocol

logger = structlog.get_logger()

DEFAULT_CACHE_TTL_SECONDS: int = 86_400  # 24 hours


# ---------------------------------------------------------------------------
# Public enum + protocol
# ---------------------------------------------------------------------------


class TranslationKind(StrEnum):
    """What kind of audit text we are translating.

    Different kinds lead to slightly different prompts — a spec summary
    needs prospective phrasing («Расширю...»), a commit diff needs
    retrospective («Расширила...»), an error needs reassurance.
    """

    SPEC_SUMMARY = "spec_summary"
    COMMIT_DIFF = "commit_diff"
    ERROR_MESSAGE = "error_message"


class Translator(Protocol):
    """Async callable: ``(technical_text, kind=...) -> architectural_text``."""

    async def translate(self, technical_text: str, *, kind: TranslationKind) -> str: ...


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_KIND_INSTRUCTION: dict[TranslationKind, str] = {
    TranslationKind.SPEC_SUMMARY: (
        "Это план будущего изменения. Используй будущее время "
        "(«расширю», «добавлю», «починю»). Опиши что и зачем меняется в "
        "логике системы, не в коде."
    ),
    TranslationKind.COMMIT_DIFF: (
        "Это описание уже сделанных изменений. Используй прошедшее время "
        "(«расширила», «добавила», «починила»). Опиши что изменилось в "
        "логике системы, не построчные правки."
    ),
    TranslationKind.ERROR_MESSAGE: (
        "Это сообщение об ошибке или провале. Объясни что не получилось "
        "в архитектурных терминах: какая часть системы не сработала и "
        "почему — без stack trace, без названий тестов и pytest-маркеров."
    ),
}

_BANNED_TERMS: str = (
    "RED/GREEN, coverage, whitelist, contract test, refactor kind, "
    "payload, dataclass, fields, generic, decorator, mock, fixture, pytest, "
    "stack trace, traceback, AssertionError, markdown, HTML, regex, parse_mode"
)
_ALLOWED_TERMS: str = "git, ветка, коммит, тест, проверка, tier"

_SYSTEM_TEMPLATE = (
    "Ты переводишь технические описания изменений кода на архитектурно-"
    "направляющий язык. Целевой читатель — оркестратор/направляющий, не "
    "разработчик. Объясняй ЧТО ИЗМЕНИЛОСЬ В ЛОГИКЕ системы, не что "
    "написано в коде. Один-два предложения. Только русский, без англицизмов "
    "за пределами разрешённого списка.\n\n"
    "ЗАПРЕЩЕНО использовать: {banned}.\n"
    "МОЖНО использовать: {allowed}.\n\n"
    "Тип входа: {kind} ({kind_value}). {kind_instruction}"
)

_PROMPT_TEMPLATE = (
    "Технический текст:\n{text}\n\n"
    "Архитектурное описание (1-2 предложения, только текст без префиксов):"
)


def _build_request(text: str, kind: TranslationKind) -> LLMRequest:
    system = _SYSTEM_TEMPLATE.format(
        banned=_BANNED_TERMS,
        allowed=_ALLOWED_TERMS,
        kind=kind.name,
        kind_value=kind.value,
        kind_instruction=_KIND_INSTRUCTION[kind],
    )
    return LLMRequest(
        prompt=_PROMPT_TEMPLATE.format(text=text),
        system=system,
        tier="worker",
        temperature=0.0,
        caller="chat_self_coding_translator",
    )


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

_CACHE_PREFIX = "chat_self_coding:translation:"


def _cache_key(technical_text: str, kind: TranslationKind) -> str:
    digest = hashlib.sha256(f"{kind.value}::{technical_text}".encode()).hexdigest()
    return f"{_CACHE_PREFIX}{digest}"


# ---------------------------------------------------------------------------
# Concrete translator
# ---------------------------------------------------------------------------


class LLMTranslator:
    """LLM-backed ``Translator`` with optional Redis cache."""

    def __init__(
        self,
        *,
        llm_router: LLMGatewayProtocol,
        redis: Any = None,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        self._llm = llm_router
        self._redis = redis
        self._ttl = cache_ttl_seconds

    async def translate(self, technical_text: str, *, kind: TranslationKind) -> str:
        cache_key = _cache_key(technical_text, kind)

        cached = await self._cache_get(cache_key)
        if cached is not None:
            return cached

        response = await self._llm.generate(_build_request(technical_text, kind))
        translated = response.text.strip()
        if not translated:
            # Worse to show nothing than to show the technical text.
            translated = technical_text

        await self._cache_set(cache_key, translated)
        return translated

    async def _cache_get(self, key: str) -> str | None:
        if self._redis is None:
            return None
        try:
            raw = await self._redis.get(key)
        except Exception:
            logger.warning("translator_cache_get_failed", exc_info=True)
            return None
        if raw is None:
            return None
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)

    async def _cache_set(self, key: str, value: str) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.set(key, value, ex=self._ttl)
        except Exception:
            logger.warning("translator_cache_set_failed", exc_info=True)

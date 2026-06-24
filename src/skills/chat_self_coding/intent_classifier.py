"""Intent classification for the chat-mode self-coding skill (Phase 40).

Maps a user message + chat-mode context to one of eight canonical intents:
``create_spec``, ``show_spec``, ``approve``, ``reject``, ``run_spec``,
``exit``, ``status``, ``other``. Two-tier strategy:

1. **Keyword/phrase fast match** — explicit commands like ``делай``,
   ``оформи план``, ``выход`` and ``не надо`` resolve without an LLM
   round-trip. Cheap, deterministic, no latency. AGENTS.md explicitly
   allows this for approval-style classification.
2. **LLM fallback** — for ambiguous text, consult the worker tier (Haiku)
   with a tight prompt that names the eight intents. Returns ``other``
   if the LLM emits anything outside the canonical set.

The classifier is exposed as both a ``Protocol`` (so callers can mock or
swap implementations) and a concrete ``LLMIntentClassifier`` that wraps
an ``LLMGatewayProtocol``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from src.llm.protocols import LLMRequest

if TYPE_CHECKING:
    from src.llm.protocols import LLMGatewayProtocol


# ---------------------------------------------------------------------------
# Public enums
# ---------------------------------------------------------------------------


class Intent(StrEnum):
    """Canonical intents the chat-mode skill can act on."""

    CREATE_SPEC = "create_spec"
    SHOW_SPEC = "show_spec"
    APPROVE = "approve"
    REJECT = "reject"
    RUN_SPEC = "run_spec"
    MERGE = "merge"
    EXIT = "exit"
    STATUS = "status"
    OTHER = "other"


class Stage(StrEnum):
    """User-facing stage of the chat-mode session.

    Distinct from ``SpecStatus`` — captures what the user is currently
    waiting on, including pre-yaml states (``IDLE``, ``DRAFTING``).
    """

    IDLE = "idle"
    DRAFTING = "drafting"
    PENDING_APPROVAL = "pending_approval"
    RUNNING = "running"
    DONE = "done"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntentClassifierContext:
    """Per-message context passed to the classifier."""

    text: str
    stage: Stage
    active_spec_slug: str | None = None
    recent_messages: tuple[str, ...] = field(default_factory=tuple)
    requires_ai_approval: bool = False


@dataclass(frozen=True)
class IntentClassification:
    """Result of classification."""

    intent: Intent
    slug: str | None = None
    confidence: float = 0.0
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class IntentClassifier(Protocol):
    """Async callable: ``IntentClassifierContext`` → ``IntentClassification``."""

    async def __call__(self, ctx: IntentClassifierContext) -> IntentClassification: ...


# ---------------------------------------------------------------------------
# Keyword fast-match data
# ---------------------------------------------------------------------------

# Tokens are matched against word-boundary splits of the lowercased message
# to avoid false positives like "слей" inside "посылай". Phrases use
# substring search since they already contain a space.

_EXIT_TOKENS: frozenset[str] = frozenset(
    {"выход", "хватит", "всё", "все", "финиш", "стоп", "exit"}
)

_APPROVE_TOKENS: frozenset[str] = frozenset(
    {
        "одобряю",
        "одобри",
        "делай",
        "реализуй",
        "запускай",
        "запусти",
        "начинай",
        "пиши",
        "кодь",
        "старт",
        "go",
        "run",
        "approve",
        "approved",
    }
)
_APPROVE_PHRASES: tuple[str, ...] = (
    "пиши код",
    "начинай код",
    "запускай код",
    "запускай реализацию",
    "запусти реализацию",
    "одобряю план",
)
_NATURAL_APPROVE_PHRASES: tuple[str, ...] = (
    "начинать реализацию",
    "начать реализацию",
    "можешь писать код",
    "можно писать код",
    "можно кодить",
)

_MERGE_TOKENS: frozenset[str] = frozenset({"слей", "сливай", "merge"})

_REJECT_TOKENS: frozenset[str] = frozenset(
    {
        "отклоняю",
        "отмена",
        "отклонено",
        "отказ",
        "reject",
        "rejected",
        "no",
        "nope",
    }
)
_REJECT_PHRASES: tuple[str, ...] = (
    "не надо",
    "не сейчас",
    "не хочу",
    "позже напомни",
    "позже",
)

_SHOW_TOKENS: frozenset[str] = frozenset({"покажи", "покажешь", "show"})
_SHOW_PHRASES: tuple[str, ...] = (
    "что там",
    "что у нас",
    "что в плане",
    "что делаем",
)

_STATUS_TOKENS: frozenset[str] = frozenset({"статус", "status"})
_STATUS_PHRASES: tuple[str, ...] = (
    "где мы",
    "что происходит",
    "как идёт",
    "как идет",
)

_DISCUSSION_TOKENS: frozenset[str] = frozenset(
    {
        "обсудить",
        "обсудим",
        "обсуждать",
        "поговорить",
        "подумаем",
        "разобрать",
    }
)
_DISCUSSION_PHRASES: tuple[str, ...] = (
    "что думаешь",
    "как лучше",
    "какой вариант",
    "стоит ли",
    "что если",
    "давай обсудим",
    "можем обсудить",
)

# In IDLE, idea-shaped messages start as discussion. The user can turn
# them into a spec with an explicit plan/spec command like «оформи план».
_IDLE_DISCUSSION_TOKENS: frozenset[str] = frozenset(
    {
        "хочу",
        "надо",
        "нужно",
        "идея",
        "идеи",
        "проблема",
        "проблему",
        "проблемы",
        "проблемой",
        "ошибка",
        "ошибку",
        "ошибки",
        "ошибкой",
        "баг",
        "бага",
        "баги",
        "багов",
    }
)
_IDLE_DISCUSSION_PHRASES: tuple[str, ...] = (
    "что если",
    "можно ли",
    "интересно",
)

# CREATE_SPEC keywords are intentionally narrow: generic action words like
# «сделай», «исправь», «реализуй» stay discussion by default. A spec starts
# only when the user explicitly asks for a plan/spec.
_CREATE_TOKENS: frozenset[str] = frozenset(
    {
        "spec",
        "спек",
        "спеку",
    }
)
_CREATE_PHRASES: tuple[str, ...] = (
    "оформи план",
    "сделай план",
    "собери план",
    "сформулируй план",
    "подготовь план",
    "готовь план",
    "пересобери план",
    "обнови план",
    "новый план",
    "создай spec",
    "сделай spec",
    "подготовь spec",
    "создай спек",
    "сделай спек",
    "подготовь спек",
)
_RETRY_CREATE_PHRASES: tuple[str, ...] = (
    "ещё раз попробуй",
    "еще раз попробуй",
    "попробуй ещё раз",
    "попробуй еще раз",
    "повтори попытку",
)
_RECENT_PLAN_ATTEMPT_MARKERS: tuple[str, ...] = (
    "оформи план",
    "сделай план",
    "собери план",
    "новый план",
    "пересобери план",
    "создай spec",
    "сделай spec",
    "создай спек",
    "сделай спек",
    "не получилось составить план",
    "план не собрался",
    "parse failed",
    "architect",
)

_CONTROL_TOKENS = (
    _EXIT_TOKENS
    | _APPROVE_TOKENS
    | _MERGE_TOKENS
    | _REJECT_TOKENS
    | _CREATE_TOKENS
    | _SHOW_TOKENS
    | _STATUS_TOKENS
)
_CONTROL_PHRASES = (
    _REJECT_PHRASES
    + _APPROVE_PHRASES
    + _CREATE_PHRASES
    + _SHOW_PHRASES
    + _STATUS_PHRASES
)

_TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)


def _tokens_of(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _word_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text.lower()))


def _has_phrase(lower: str, phrases: tuple[str, ...]) -> bool:
    return any(p in lower for p in phrases)


def _is_short_command(lower: str) -> bool:
    return 0 < _word_count(lower) <= 2


def _has_control_marker(lower: str, tokens: set[str]) -> bool:
    return bool(tokens & _CONTROL_TOKENS) or _has_phrase(lower, _CONTROL_PHRASES)


def _looks_like_discussion_request(lower: str, tokens: set[str]) -> bool:
    return bool(tokens & _DISCUSSION_TOKENS) or _has_phrase(lower, _DISCUSSION_PHRASES)


def _looks_like_create_request(lower: str, tokens: set[str]) -> bool:
    return bool(tokens & _CREATE_TOKENS) or _has_phrase(lower, _CREATE_PHRASES)


def _looks_like_approve_request(
    lower: str,
    tokens: set[str],
    *,
    include_natural_phrases: bool = False,
) -> bool:
    phrases = _APPROVE_PHRASES
    if include_natural_phrases:
        phrases = phrases + _NATURAL_APPROVE_PHRASES
    return bool(tokens & _APPROVE_TOKENS) or _has_phrase(lower, phrases)


def _looks_like_idle_idea_request(lower: str, tokens: set[str]) -> bool:
    return bool(tokens & _IDLE_DISCUSSION_TOKENS) or _has_phrase(
        lower, _IDLE_DISCUSSION_PHRASES
    )


def _looks_like_retry_create_request(
    lower: str,
    recent_messages: tuple[str, ...],
) -> bool:
    if not _has_phrase(lower, _RETRY_CREATE_PHRASES):
        return False
    recent = "\n".join(recent_messages).lower()
    return any(marker in recent for marker in _RECENT_PLAN_ATTEMPT_MARKERS)


def _classify_exit_or_reject(lower: str, tokens: set[str]) -> Intent | None:
    if tokens & _EXIT_TOKENS:
        return Intent.EXIT

    # Reject phrases first so "не надо" does not get hijacked by a stray
    # approve token in a longer message.
    if _has_phrase(lower, _REJECT_PHRASES):
        return Intent.REJECT
    if tokens & _REJECT_TOKENS:
        return Intent.REJECT
    return None


def _classify_create_keywords(
    lower: str, tokens: set[str], stage: Stage
) -> Intent | None:
    if stage in {Stage.IDLE, Stage.PENDING_APPROVAL, Stage.DONE} and (
        _looks_like_create_request(lower, tokens)
    ):
        return Intent.CREATE_SPEC
    return None


def _classify_idle_keywords(lower: str, tokens: set[str]) -> Intent | None:
    if _looks_like_idle_idea_request(lower, tokens):
        return Intent.OTHER
    return None


def _classify_show_or_status(lower: str, tokens: set[str]) -> Intent | None:
    if _has_phrase(lower, _SHOW_PHRASES) or tokens & _SHOW_TOKENS:
        return Intent.SHOW_SPEC
    if _has_phrase(lower, _STATUS_PHRASES) or tokens & _STATUS_TOKENS:
        return Intent.STATUS
    return None


def _classify_stage_keywords(
    lower: str,
    tokens: set[str],
    stage: Stage,
) -> Intent | None:
    if stage == Stage.DONE and tokens & _MERGE_TOKENS:
        return Intent.MERGE

    show_or_status = _classify_show_or_status(lower, tokens)
    if show_or_status is not None:
        return show_or_status

    if _looks_like_discussion_request(lower, tokens):
        return Intent.OTHER

    create_intent = _classify_create_keywords(lower, tokens, stage)
    if create_intent is not None:
        return create_intent

    if stage == Stage.IDLE:
        return _classify_idle_keywords(lower, tokens)

    if stage == Stage.PENDING_APPROVAL and _looks_like_approve_request(lower, tokens):
        return Intent.APPROVE

    return None


def _has_reject_marker(lower: str, tokens: set[str]) -> bool:
    return _has_phrase(lower, _REJECT_PHRASES) or bool(tokens & _REJECT_TOKENS)


def _classify_by_keywords(
    text: str,
    stage: Stage,
    recent_messages: tuple[str, ...] = (),
    *,
    allow_decision_keywords: bool = True,
) -> Intent | None:
    """Return an intent if a high-confidence keyword/phrase fires, else None."""
    lower = text.lower()
    tokens = _tokens_of(lower)

    if stage == Stage.PENDING_APPROVAL and not allow_decision_keywords:
        return _classify_pending_approval_non_decision_keywords(lower, tokens)

    if not _is_short_command(lower):
        if not _has_reject_marker(lower, tokens):
            create_intent = _classify_create_keywords(lower, tokens, stage)
            if create_intent is not None:
                return create_intent
            if _looks_like_retry_create_request(lower, recent_messages):
                return Intent.CREATE_SPEC
        if _has_control_marker(lower, tokens):
            return Intent.OTHER

    if _looks_like_retry_create_request(lower, recent_messages):
        return Intent.CREATE_SPEC

    if not _is_short_command(lower) and _has_control_marker(lower, tokens):
        return Intent.OTHER

    early = _classify_exit_or_reject(lower, tokens)
    if early is not None:
        return early

    return _classify_stage_keywords(lower, tokens, stage)


def _classify_pending_approval_non_decision_keywords(
    lower: str,
    tokens: set[str],
) -> Intent | None:
    """Tier 3 approval decisions must go through the LLM path."""
    if tokens & _EXIT_TOKENS:
        return Intent.EXIT
    show_or_status = _classify_show_or_status(lower, tokens)
    if show_or_status is not None:
        return show_or_status
    create_intent = _classify_create_keywords(lower, tokens, Stage.PENDING_APPROVAL)
    if create_intent is not None:
        return create_intent
    if _looks_like_discussion_request(lower, tokens):
        return Intent.OTHER
    return None


# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------

_VALID_INTENT_TOKENS: frozenset[str] = frozenset(i.value for i in Intent)

_LLM_SYSTEM_TEMPLATE = (
    "Ты классифицируешь намерение пользователя в чат-режиме /код. "
    "Стадия процесса: {stage}. Активная спека: {slug}. "
    "Возможные намерения: {intents}. "
    "Команды действуют только когда сообщение короткое: одно-два слова. "
    "Если в длинном сообщении встречается слово вроде 'выход', 'делай', "
    "'давай', 'одобряю', считай это обычным discussion/other. "
    "По умолчанию классифицируй сообщения как other/discussion. "
    "create_spec ставь только когда Никита явно просит оформить или пересобрать "
    "plan/spec: 'оформи план', 'пересобери план', 'создай spec'. "
    "approve ставь только на стадии pending_approval и только если Никита явно "
    "просит начать реализацию: 'делай', 'реализуй', 'запускай', 'пиши код' "
    "или ясно пишет, что можно начинать реализацию. "
    "'да', 'ок', 'ага', 'давай', 'ну ладно' без явного запуска — это other. "
    "{approval_mode}"
    "Ответь ТОЛЬКО одним из этих слов, без пояснений."
)

_LLM_PROMPT_TEMPLATE = (
    "Сообщение пользователя:\n{text}\n\n"
    "Последние сообщения для контекста:\n{recent}\n\n"
    "Какое намерение?"
)


def _build_llm_request(ctx: IntentClassifierContext) -> LLMRequest:
    approval_mode = (
        "Для текущего Tier 3 approval нельзя пользоваться фиксированными "
        "словами как правилом. Смотри на смысл ответа: Никита действительно "
        "разрешает запуск после обсуждения, хочет правки/детали или отказывает. "
        if ctx.requires_ai_approval
        else ""
    )
    system = _LLM_SYSTEM_TEMPLATE.format(
        stage=ctx.stage.value,
        slug=ctx.active_spec_slug or "(нет)",
        intents=", ".join(sorted(_VALID_INTENT_TOKENS)),
        approval_mode=approval_mode,
    )
    prompt = _LLM_PROMPT_TEMPLATE.format(
        text=ctx.text,
        recent="\n".join(ctx.recent_messages) if ctx.recent_messages else "(нет)",
    )
    return LLMRequest(
        prompt=prompt,
        system=system,
        tier="worker",
        temperature=0.0,
        caller="chat_self_coding_intent",
    )


def _is_llm_intent_allowed(ctx: IntentClassifierContext, intent: Intent) -> bool:
    lower = ctx.text.lower()
    tokens = _tokens_of(lower)

    if intent in {Intent.APPROVE, Intent.RUN_SPEC}:
        if ctx.requires_ai_approval:
            return ctx.stage == Stage.PENDING_APPROVAL
        return ctx.stage == Stage.PENDING_APPROVAL and _looks_like_approve_request(
            lower,
            tokens,
            include_natural_phrases=True,
        )
    if intent == Intent.CREATE_SPEC:
        return _classify_create_keywords(lower, tokens, ctx.stage) is not None
    if intent == Intent.MERGE:
        return ctx.stage == Stage.DONE and bool(tokens & _MERGE_TOKENS)
    return True


# ---------------------------------------------------------------------------
# Concrete classifier
# ---------------------------------------------------------------------------


class LLMIntentClassifier:
    """Two-tier classifier: keyword fast match → LLM fallback.

    Keyword path is deterministic, latency-free, and never calls the LLM.
    LLM path runs on the worker tier with temperature 0.0 for a stable
    single-token answer; unknown tokens collapse to ``Intent.OTHER``.
    """

    def __init__(self, *, llm_router: LLMGatewayProtocol) -> None:
        self._llm = llm_router

    async def __call__(self, ctx: IntentClassifierContext) -> IntentClassification:
        keyword_intent = _classify_by_keywords(
            ctx.text,
            ctx.stage,
            ctx.recent_messages,
            allow_decision_keywords=not ctx.requires_ai_approval,
        )
        if keyword_intent is not None:
            return IntentClassification(
                intent=keyword_intent,
                slug=ctx.active_spec_slug,
                confidence=0.95,
                reasoning="keyword match",
            )

        response = await self._llm.generate(_build_llm_request(ctx))
        token = response.text.strip().lower()
        if token in _VALID_INTENT_TOKENS:
            intent = Intent(token)
            if not _is_llm_intent_allowed(ctx, intent):
                return IntentClassification(
                    intent=Intent.OTHER,
                    slug=ctx.active_spec_slug,
                    confidence=0.55,
                    reasoning=f"llm {intent.value} filtered by explicit trigger policy",
                )
            return IntentClassification(
                intent=intent,
                slug=ctx.active_spec_slug,
                confidence=0.7,
                reasoning="llm classification",
            )
        return IntentClassification(
            intent=Intent.OTHER,
            slug=None,
            confidence=0.3,
            reasoning=f"llm returned unrecognized token: {response.text!r}",
        )

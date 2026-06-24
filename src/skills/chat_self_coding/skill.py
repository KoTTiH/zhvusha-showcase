"""ChatSelfCodingSkill — chat-mode orchestrator (Phase 40).

Owns no business logic of its own. The user types ``/код`` or ``/code`` to
enter the mode; the skill creates or reopens a Redis-backed session and
from then on intercepts every text message from the admin until they say
«выход». ``выход`` only closes the room; ``/готово`` archives and clears
the session.
Each message is run through the intent classifier and routed to either
ordinary discussion or the existing pipeline skills via their
slash-command surface — that way the underlying ``ideation_to_spec`` /
``spec_command`` / ``implement_spec`` machinery stays untouched and
Phase 40 is a pure UX layer.

The legacy slash commands (``/spec_create``, ``/spec approve``, ...)
remain available and unchanged outside the chat-mode session. Outside an
open room, ``can_handle`` fires only for explicit entry/session commands or
for a short implementation confirmation that continues a durable engineering
proposal from normal chat.

Why proxy through the slash interface instead of calling internal
helpers: every legacy command path is already tested, has its own
guards (whitelist enforcement, caps, tier-3 protection), and emits the
audit log entries the rest of the system depends on. Phase 40 only adds
a friendlier shell on top.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from html import escape
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

import structlog
from aiogram.exceptions import TelegramBadRequest

from src.dialogue.state import dialogue_state_from_metadata
from src.skills.base import (
    AgentContext,
    BaseSkill,
    InlineSkill,
    SideEffect,
    SkillResult,
)
from src.skills.chat_self_coding.blocks import ProgressBlock, format_architect_progress
from src.skills.chat_self_coding.intent_classifier import (
    Intent,
    IntentClassifierContext,
    Stage,
)
from src.skills.chat_self_coding.state import ChatSelfCodingState, TaskPhase

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from src.skills.chat_self_coding.intent_classifier import IntentClassifier
    from src.skills.chat_self_coding.state import StateStore
    from src.skills.chat_self_coding.task_transcript import TaskTranscriptStore

    MergeHandler = Callable[[str, AgentContext], Awaitable[SkillResult]]
    SpecTierResolver = Callable[[str], int | None]


class ExplorerRunner(Protocol):
    """Read-only Codex session used for /код discussion turns."""

    async def __call__(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        progress_callback: Callable[[str], Awaitable[None]] | None = None,
        session_id: str = "",
        persist_session: bool = False,
        session_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> str: ...


class ImplementationRunner(Protocol):
    """Durable Agent Runtime runner for implementation turns."""

    async def __call__(
        self,
        *,
        slug: str,
        context: AgentContext,
        recent_messages: tuple[str, ...] = (),
    ) -> SkillResult: ...

    async def start_background(
        self,
        *,
        slug: str,
        context: AgentContext,
        recent_messages: tuple[str, ...] = (),
        completion_callback: Callable[[SkillResult], Awaitable[None]],
    ) -> object: ...


logger = structlog.get_logger()


_ENTRY_TRIGGER = "/код"
_ENTRY_TRIGGER_ALIAS = "/code"
_LEGACY_ENTRY_TRIGGERS = ("/самокодинг", "/self_coding")
_ENTRY_TRIGGERS = (_ENTRY_TRIGGER, _ENTRY_TRIGGER_ALIAS, *_LEGACY_ENTRY_TRIGGERS)
_COMPLETE_TRIGGERS = ("/готово", "/done")
_CLEAR_TRIGGERS = ("/clear", "/очистить")
_COMPACT_TRIGGERS = ("/compact", "compact", "/сжать", "сжать")
_GOAL_TRIGGERS = ("/goal", "goal", "/цель", "цель")

# Hardcoded exit safety net — guarantees the most common way of leaving
# the mode works even if the injected intent classifier is unavailable
# or misbehaves. Less common synonyms («хватит», «финиш», «всё») still
# pass through the regular classifier.
_HARDCODED_EXIT_TOKENS: frozenset[str] = frozenset({"выход", "exit"})
_HOST_OPS_OBJECT_MARKERS: tuple[str, ...] = (
    "systemd",
    "systemctl",
    "supervisor",
    "daemon-reload",
    "enable --now",
    "zhvusha-bot.service",
    "/etc/systemd",
    "bot_restart_enabled",
    "runtime-контур",
    "runtime контур",
    "живой .env",
    "live .env",
)
_HOST_OPS_ACTION_MARKERS: tuple[str, ...] = (
    "включи",
    "включить",
    "запусти",
    "запустить",
    "установи",
    "установить",
    "поставь",
    "поставить",
    "enable",
    "запиши",
    "записать",
    "перезапусти",
    "перезапустить",
    "на хосте",
    "чтобы все работало",
    "чтобы всё работало",
)
_ARCHITECT_PROGRESS_INTERVAL_SECONDS = 15.0
_ARCHITECT_PROGRESS_WAIT_PERCENT = 15
_GOAL_AUTO_RETRY_LIMIT = 0

_INCOMING_MATERIAL_RE = re.compile(
    r"("
    r"\b(сейчас|щас|щаз|ща|сча)\s+(скину|кину|пришлю|отправлю)\b|"
    r"\b(скину|кину|пришлю|отправлю)\b.{0,40}\b"
    r"(пост|скрин|скриншот|фото|картинк|файл|лог|текст|ссылк)"
    r")",
    re.IGNORECASE,
)
_SAVED_ATTACHMENT_MARKERS: tuple[str, ...] = (
    "absolute_path:",
    "workspace_path:",
    "raw-контекст",
    "никита прислал вложение",
)
_SAVED_ATTACHMENT_DISCUSSION_MARKERS: tuple[str, ...] = (
    "посмотри",
    "разбери",
    "объясни",
    "прочитай",
    "изучи",
    "что на",
    "вложен",
    "фото",
    "скрин",
    "файл",
)
_REPO_EXPLORER_MARKERS: tuple[str, ...] = (
    "код",
    "реп",
    "repo",
    "repository",
    "проект",
    "файл",
    "логи",
    "лог ",
    "traceback",
    "ошибк",
    "баг",
    "тест",
    "pytest",
    "ruff",
    "mypy",
    "модул",
    "класс",
    "функц",
    "метод",
    "архитектур",
    "самокод",
    "spec",
    "env",
    ".env",
    "redis",
    "postgres",
    "telegram",
    "dispatcher",
)
_REPO_EXPLORER_ACTION_MARKERS: tuple[str, ...] = (
    "изучи",
    "исследуй",
    "проверь",
    "посмотри",
    "прочитай",
    "найди",
    "открой",
    "разберись",
    "сравни",
    "проанализируй",
    "аудит",
)
_REPO_EXPLORER_EXPLICIT_PHRASES: tuple[str, ...] = (
    "изучи код",
    "изучи проект",
    "изучи реп",
    "изучи кодовую базу",
    "проверь код",
    "проверь проект",
    "посмотри код",
    "посмотри проект",
    "посмотри реп",
    "прочитай логи",
    "посмотри логи",
    "сравни с кодом",
    "сравни с кодовой базой",
    "сравни с проектом",
    "что уже есть в коде",
    "что уже есть в проекте",
    "как сейчас устроено в коде",
    "как это реализовано сейчас",
)
_RECOVERY_RESUME_TOKENS: frozenset[str] = frozenset(
    {
        "делай",
        "продолжай",
        "доделывай",
        "повтори",
        "согласен",
        "согласна",
        "так",
        "ок",
        "окей",
        "да",
    }
)
_RECOVERY_RESUME_PHRASES: tuple[str, ...] = (
    "ещё раз",
    "еще раз",
    "попробуй ещё",
    "попробуй еще",
    "пробуй ещё",
    "пробуй еще",
    "запускай снова",
    "продолжай работу",
    "продолжай разработку",
    "продолжи работу",
    "продолжи разработку",
    "давай так",
)
_IMPLICIT_SELF_CODING_TOKENS: frozenset[str] = frozenset(
    {
        "делай",
        "продолжай",
        "реализуй",
        "запускай",
        "начинай",
        "кодь",
    }
)
_IMPLICIT_SELF_CODING_PHRASES: tuple[str, ...] = (
    "пиши код",
    "начинай реализацию",
    "запускай реализацию",
    "запусти реализацию",
    "можно писать код",
    "можешь писать код",
    "продолжай разработку",
)
_EXPLICIT_IMPLEMENTATION_COMMANDS: frozenset[str] = frozenset(
    {
        "approve",
        "approved",
        "go",
        "run",
        "делай",
        "запускай",
        "запусти",
        "кодь",
        "начинай",
        "пиши",
        "реализуй",
    }
)
_SPEC_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,127}$")
_SELF_CODING_DIALOGUE_ENGINEERING_MARKERS: tuple[str, ...] = (
    ".env",
    "agent",
    "browser",
    "daemon",
    "env",
    "repo",
    "runtime",
    "session",
    "skill",
    "spec",
    "test",
    "tool",
    "агент",
    "браузер",
    "жвуш",
    "код",
    "проект",
    "реп",
    "сесс",
    "тест",
)
_SELF_CODING_DIALOGUE_ACTION_MARKERS: tuple[str, ...] = (
    "добав",
    "врез",
    "готов",
    "запус",
    "исправ",
    "план",
    "постав",
    "реализ",
    "сдела",
    "собер",
    "persistent",
)
_DIALOGUE_BOOTSTRAP_MAX_LINES = 8

_EXPLORER_SYSTEM_PROMPT = """\
Ты Codex Explorer внутри Telegram-режима /код.

Работай как обычная read-only Codex-сессия в репозитории ZHVUSHA: изучай код,
логи, tests, docs, raw-вложения по absolute_path, и локальный контекст задачи.
Используй read-only действия для проверки фактов: чтение файлов, поиск, запуск
безопасных диагностических команд, анализ изображений/файлов если они доступны.
Ничего не изменяй, не создавай spec, не запускай реализацию и не коммить.

Если Никита просит обсудить идею, отвечай как собеседник-инженер: уточняй,
спорь с плохой идеей, показывай trade-off. Если нужен plan/spec, скажи что
готова оформить его только по явному сигналу Никиты: "оформи план".

Говори голосом Жвуши в женском роде: "посмотрела", "проверила", "собрала".
Нельзя писать "я не проверял", "использовал", "нашёл" от мужского лица.
Не возвращай machine-capsule формат SUMMARY/FINDING/SOURCE/NEXT/MEMORY.
Даже после изучения кода отвечай как Жвуша в чате: коротко, конкретно,
с выводами, вариантами и проверенной evidence без протокольных заголовков.

Если для ответа нужен интернет или доступ, которого нет в read-only среде,
честно скажи это и отдели проверенные выводы от предположений.
Для Telegram-progress можешь иногда писать отдельные строки строго с префиксом
TG_STATUS:, например: TG_STATUS: Сейчас ищу реальный путь обработки /код.
"""

# Stage → architectural status sentence, used for STATUS responses and
# the welcome message. Phrased in the orchestrator-language register
# Phase 40 mandates: no «pending_approval», just «жду одобрения плана».
_STATUS_SENTENCE: dict[Stage, str] = {
    Stage.IDLE: "Можем обсудить идею. Когда решишь — скажи «оформи план».",
    Stage.DRAFTING: "Готовлю план для «{slug}». Скоро покажу.",
    Stage.PENDING_APPROVAL: (
        "План для «{slug}» готов. Можем обсуждать дальше; для реализации скажи «делай»."
    ),
    Stage.RUNNING: "Пишу код для «{slug}».",
    Stage.DONE: "Закончила «{slug}». Изменения уже применены.",
}

_TASK_PHASE_LABEL: dict[TaskPhase, str] = {
    TaskPhase.DISCUSSION: "обсуждение",
    TaskPhase.SPEC: "сбор plan",
    TaskPhase.APPROVAL: "согласование",
    TaskPhase.IMPLEMENTATION: "реализация",
    TaskPhase.VERIFICATION: "проверки",
    TaskPhase.COMMIT: "commit gate",
    TaskPhase.REVIEW: "review gate",
    TaskPhase.REPAIR: "repair",
    TaskPhase.DONE: "завершено",
}


def _status_text(state: ChatSelfCodingState) -> str:
    if state.recovery_kind is not None:
        error = escape(state.recovery_error or "последняя попытка не прошла")
        if not state.recovery_needs_user_decision:
            return (
                "Последняя попытка не прошла, и я не запускаю повтор автоматически. "
                f"Причина: {error}. Это технический blocker, не вопрос продукта "
                "или архитектуры к Никите. Дополнительные сообщения сохраню как "
                "контекст; чтобы запустить следующий проход, скажи «продолжай» "
                "или «делай»."
            )
        question = escape(state.recovery_question or "что меняем в подходе?")
        return (
            "Уперлась в ошибку и держу задачу в обсуждении. "
            f"Причина: {error}. Нужно решить: {question}. "
            "Когда договоримся, скажи «продолжай» или «делай», "
            "и я возобновлю тот же шаг."
        )
    template = _STATUS_SENTENCE[state.stage]
    status = template.format(slug=escape(state.active_spec_slug or "—"))
    phase = _TASK_PHASE_LABEL.get(state.task_phase)
    full = status if phase is None else f"{status} Текущий этап: {escape(phase)}."
    if state.active_goal:
        full = f"{full} Активная цель: {escape(state.active_goal)}."
    if state.readonly_codex_session_id:
        full = f"{full} Codex thread: {escape(state.readonly_codex_session_id)}."
    if state.editor_codex_session_id:
        full = f"{full} Editor thread: {escape(state.editor_codex_session_id)}."
    return full


def _is_entry_trigger(text: str) -> bool:
    return any(
        text == trigger or text.startswith(trigger + " ") for trigger in _ENTRY_TRIGGERS
    )


def _entry_trigger_tail(text: str) -> str | None:
    stripped = text.strip()
    lower = stripped.lower()
    for trigger in _ENTRY_TRIGGERS:
        if lower == trigger:
            return ""
        if lower.startswith(trigger + " "):
            return stripped[len(trigger) :].strip()
    return None


def _is_complete_trigger(text: str) -> bool:
    return text.strip().lower() in _COMPLETE_TRIGGERS


def _is_clear_trigger(text: str) -> bool:
    return text.strip().lower() in _CLEAR_TRIGGERS


def _is_compact_trigger(text: str) -> bool:
    return text.strip().lower() in _COMPACT_TRIGGERS


def _parse_goal_command(text: str) -> tuple[bool, str]:
    lower = text.strip().lower()
    for trigger in _GOAL_TRIGGERS:
        if lower == trigger:
            return True, ""
        if lower.startswith(trigger + " "):
            return True, text.strip()[len(trigger) :].strip()
    return False, ""


def _metadata_text(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    if isinstance(value, str):
        return value.strip()
    return ""


def _normalize_short_control_text(text: str) -> str:
    return " ".join(text.strip().lower().rstrip("!.,;:").split())


def _is_short_implementation_confirmation(text: str) -> bool:
    normalized = _normalize_short_control_text(text)
    if not normalized:
        return False
    if normalized in _IMPLICIT_SELF_CODING_TOKENS:
        return True
    if len(normalized.split()) <= 4:
        return any(phrase == normalized for phrase in _IMPLICIT_SELF_CODING_PHRASES)
    return False


def _parse_explicit_implementation_slug(text: str) -> str | None:
    parts = text.strip().split(maxsplit=1)
    if len(parts) != 2:
        return None
    command = _normalize_short_control_text(parts[0])
    if command not in _EXPLICIT_IMPLEMENTATION_COMMANDS:
        return None
    candidate = parts[1].split(maxsplit=1)[0].strip("`'\".,;:()[]{}<>")
    if _SPEC_SLUG_RE.fullmatch(candidate):
        return candidate
    return None


def _is_closed_session_resume_trigger(
    text: str,
    state: ChatSelfCodingState,
) -> bool:
    if state.is_open:
        return False
    if state.stage == Stage.PENDING_APPROVAL and _is_short_implementation_confirmation(
        text
    ):
        return True
    return state.recovery_kind is not None and _looks_like_recovery_resume(text)


def _dialogue_has_self_coding_context(text: str) -> bool:
    normalized = text.lower()
    if not normalized:
        return False
    has_engineering_marker = any(
        marker in normalized for marker in _SELF_CODING_DIALOGUE_ENGINEERING_MARKERS
    )
    has_action_marker = any(
        marker in normalized for marker in _SELF_CODING_DIALOGUE_ACTION_MARKERS
    )
    return has_engineering_marker and has_action_marker


def _is_dialogue_self_coding_start_trigger(
    text: str,
    context: AgentContext,
) -> bool:
    if not _is_short_implementation_confirmation(text):
        return False
    dialogue_state = dialogue_state_from_metadata(
        context.metadata.get("dialogue_state")
    )
    if dialogue_state is None:
        return False
    if dialogue_state.pending_action:
        return False
    if dialogue_state.selected_skill not in {"", "chat_response"}:
        return False

    context_text = "\n".join(
        part
        for part in (
            dialogue_state.active_topic,
            dialogue_state.last_user_message,
            dialogue_state.last_assistant_response,
            _metadata_text(context.metadata, "recent_decision_messages"),
        )
        if part
    )
    return _dialogue_has_self_coding_context(context_text)


def _dialogue_bootstrap_messages(context: AgentContext) -> tuple[str, ...]:
    recent = _metadata_text(context.metadata, "recent_decision_messages")
    if not recent:
        recent = _metadata_text(context.metadata, "recent_messages")

    messages: list[str] = []
    for raw_line in recent.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Собеседник:"):
            line = f"Никита:{line[len('Собеседник:') :]}"
        if line.startswith(("Никита:", "Жвуша:")):
            messages.append(line)

    if messages:
        return tuple(messages[-_DIALOGUE_BOOTSTRAP_MAX_LINES:])

    dialogue_state = dialogue_state_from_metadata(
        context.metadata.get("dialogue_state")
    )
    if dialogue_state is None:
        return ()
    if dialogue_state.last_user_message:
        messages.append(f"Никита: {dialogue_state.last_user_message}")
    if dialogue_state.last_assistant_response:
        messages.append(f"Жвуша: {dialogue_state.last_assistant_response}")
    return tuple(messages[-_DIALOGUE_BOOTSTRAP_MAX_LINES:])


def _state_with_dialogue_bootstrap(
    state: ChatSelfCodingState,
    context: AgentContext,
) -> ChatSelfCodingState:
    updated = state
    for message in _dialogue_bootstrap_messages(context):
        if message not in updated.recent_messages:
            updated = updated.append_message(message)
    return updated


def _build_implicit_spec_create_text(
    text: str,
    state: ChatSelfCodingState,
    context: AgentContext,
) -> str:
    dialogue = "\n".join(state.recent_messages) or "\n".join(
        _dialogue_bootstrap_messages(context)
    )
    if not dialogue:
        dialogue = "(нет сохранённого контекста)"
    return (
        "Диалог до входа в /код:\n"
        f"{dialogue}\n\n"
        "Никита коротко подтвердил начало работы из обычного чата. "
        "Не редактируй код напрямую: сначала собери spec по этому контексту "
        "и сохрани обычные gates согласования /код.\n\n"
        "Текущая команда Никиты:\n"
        f"{text}\n\n"
        "Составь spec на основе подтверждённого инженерного контекста."
    )


def _has_prior_discussion_context(state: ChatSelfCodingState) -> bool:
    return len(state.recent_messages) > 1


def _build_spec_create_text(text: str, state: ChatSelfCodingState) -> str:
    context_parts = _session_context_parts(state)
    if not _has_prior_discussion_context(state) and not context_parts:
        return text
    discussion = "\n".join((*context_parts, *state.recent_messages))
    return (
        "Контекст предварительного обсуждения в режиме /код:\n"
        f"{discussion}\n\n"
        "Текущая команда Никиты:\n"
        f"{text}\n\n"
        "Составь spec на основе всего обсуждения."
    )


def _build_recovery_spec_create_text(state: ChatSelfCodingState, text: str) -> str:
    recovery_text = state.recovery_text or text
    recovery_error = state.recovery_error or "предыдущая попытка не прошла"
    discussion = "\n".join(state.recent_messages) or "(нет)"
    return (
        "Повторная попытка составить spec после ошибки в режиме /код.\n\n"
        "Исходный запрос / контекст предыдущей попытки:\n"
        f"{recovery_text}\n\n"
        "Почему предыдущая попытка остановилась:\n"
        f"{recovery_error}\n\n"
        "Обсуждение после ошибки и новое решение Никиты:\n"
        f"{discussion}\n\n"
        "Собери полный spec с учётом консенсуса после ошибки. "
        "Не считай прошлую ошибку отказом от задачи."
    )


def _build_explorer_prompt(text: str, state: ChatSelfCodingState) -> str:
    recent = "\n".join(state.recent_messages) if state.recent_messages else "(нет)"
    slug = state.active_spec_slug or "(нет)"
    goal = state.active_goal or "(нет)"
    compact = state.compact_summary or "(нет)"
    return (
        "Контекст рабочей комнаты /код:\n"
        f"- stage: {state.stage.value}\n"
        f"- active_spec_slug: {slug}\n"
        f"- active_goal: {goal}\n"
        f"- compact_summary: {compact}\n\n"
        "Последние сообщения и сохранённые вложения:\n"
        f"{recent}\n\n"
        "Текущее сообщение Никиты:\n"
        f"{text}\n\n"
        "Твоя задача:\n"
        "- исследуй код, файлы, логи, docs и raw-вложения настолько глубоко, "
        "насколько нужно для ответа;\n"
        "- если в контексте есть absolute_path к фото/файлу, открывай оригинал "
        "напрямую как read-only источник;\n"
        "- не создавай spec, не меняй файлы и не начинай реализацию;\n"
        "- ответь по-русски, с конкретными выводами и следующими вариантами "
        "для обсуждения."
    )


def _looks_like_incoming_material_preamble(text: str) -> bool:
    """A promise to send material is not itself a repo-inspection request."""
    return bool(_INCOMING_MATERIAL_RE.search(text.strip().lower()))


def _state_has_saved_attachment(state: ChatSelfCodingState) -> bool:
    recent = "\n".join(state.recent_messages).lower()
    return any(marker in recent for marker in _SAVED_ATTACHMENT_MARKERS)


def _should_use_explorer_for_discussion(text: str, state: ChatSelfCodingState) -> bool:
    lower = text.strip().lower()
    if not lower:
        return False
    if "что думаешь" in lower or "как лучше" in lower or "давай обсудим" in lower:
        return False
    if _looks_like_incoming_material_preamble(lower):
        return False
    if _state_has_saved_attachment(state) and any(
        marker in lower for marker in _SAVED_ATTACHMENT_DISCUSSION_MARKERS
    ):
        return True
    if any(phrase in lower for phrase in _REPO_EXPLORER_EXPLICIT_PHRASES):
        return True
    has_repo_marker = any(marker in lower for marker in _REPO_EXPLORER_MARKERS)
    has_action_marker = any(marker in lower for marker in _REPO_EXPLORER_ACTION_MARKERS)
    return has_repo_marker and has_action_marker


def _incoming_material_wait_response() -> str:
    return (
        "Кидай. На одно «сейчас скину» код не трогаю — сначала дождусь сам "
        "материал, потом посмотрю его в контексте /код."
    )


def _result_is_background(result: SkillResult) -> bool:
    return result.metadata.get("background") == "true"


def _result_needs_user_decision(result: SkillResult) -> bool:
    category = _result_failure_category(result)
    if category == "auto_repairable":
        return False
    if category in {"needs_user_decision", "needs_host_ops", "fatal"}:
        return True
    if category == "technical_blocker":
        return False
    marker = result.metadata.get("needs_user_decision")
    if isinstance(marker, bool):
        return marker
    if isinstance(marker, str):
        return marker.lower() == "true"
    return True


def _result_auto_retryable(result: SkillResult) -> bool:
    del result
    return False


def _result_failure_category(result: SkillResult) -> str:
    marker = result.metadata.get("failure_category")
    if isinstance(marker, str) and marker in {
        "auto_repairable",
        "needs_user_decision",
        "needs_host_ops",
        "technical_blocker",
        "fatal",
    }:
        return marker
    response = result.response.lower()
    if "host-ops" in response or "protected `.env`" in response:
        return "needs_host_ops"
    if _metadata_bool(result.metadata.get("auto_retryable"), default=False):
        return "auto_repairable"
    if _metadata_bool(result.metadata.get("needs_user_decision"), default=True):
        return "needs_user_decision"
    return "fatal"


def _result_decision_question(result: SkillResult) -> str:
    question = result.metadata.get("decision_question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    category = _result_failure_category(result)
    if category == "needs_host_ops":
        return (
            "это выносим в отдельное host-ops решение или убираем из текущей "
            "self-coding задачи?"
        )
    if category == "fatal":
        return "как чинить инфраструктурный сбой runtime перед новой попыткой?"
    return "что менять в подходе перед следующей попыткой?"


def _discussion_context(context: AgentContext) -> AgentContext:
    return replace(
        context,
        metadata={
            **context.metadata,
            "chat_self_coding": True,
            "suppress_memory_proposals": True,
        },
    )


def _metadata_bool(value: object, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return default


def _code_task_context(
    context: AgentContext,
    code_task_id: str,
    *,
    active_goal: str | None = None,
    readonly_codex_session_id: str | None = None,
) -> AgentContext:
    metadata = {
        **context.metadata,
        "chat_self_coding_code_task_id": code_task_id,
    }
    if active_goal:
        metadata["chat_self_coding_active_goal"] = active_goal
    if readonly_codex_session_id:
        metadata["chat_self_coding_readonly_codex_session_id"] = (
            readonly_codex_session_id
        )
    return replace(
        context,
        metadata=metadata,
    )


def _state_with_editor_resume_from_result(
    state: ChatSelfCodingState,
    result: SkillResult,
) -> ChatSelfCodingState:
    session_id = _metadata_str(result.metadata.get("editor_codex_session_id"))
    worktree_path = _metadata_str(result.metadata.get("failed_worktree_path"))
    worktree_label = _metadata_str(result.metadata.get("failed_worktree_label"))
    base_branch = _metadata_str(result.metadata.get("failed_worktree_base_branch"))
    base_sha = _metadata_str(result.metadata.get("failed_worktree_base_sha"))
    if not (
        session_id and worktree_path and worktree_label and base_branch and base_sha
    ):
        return state
    return state.with_editor_resume(
        session_id=session_id,
        worktree_path=worktree_path,
        worktree_label=worktree_label,
        base_branch=base_branch,
        base_sha=base_sha,
    )


def _metadata_str(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _session_context_parts(state: ChatSelfCodingState) -> tuple[str, ...]:
    parts: list[str] = []
    if state.active_goal:
        parts.append(f"Активная цель /код: {state.active_goal}")
    if state.compact_summary:
        parts.append(f"Сжатый контекст /код: {state.compact_summary}")
    return tuple(parts)


def _build_compact_summary(state: ChatSelfCodingState) -> str:
    lines = [
        f"stage={state.stage.value}",
        f"phase={state.task_phase.value}",
    ]
    if state.active_spec_slug:
        lines.append(f"active_spec={state.active_spec_slug}")
    if state.active_goal:
        lines.append(f"goal={state.active_goal}")
    if state.recovery_kind:
        lines.append(
            "recovery="
            + (state.recovery_error or state.recovery_kind).replace("\n", " ")[:300]
        )
    if state.recent_messages:
        tail = " / ".join(
            message.replace("\n", " ") for message in state.recent_messages
        )
        lines.append(f"recent={tail[:1200]}")
    return "; ".join(lines)


def _goal_attempt_context(
    context: AgentContext,
    attempt: int,
    *,
    code_task_id: str | None = None,
    state: ChatSelfCodingState | None = None,
) -> AgentContext:
    metadata = {
        **context.metadata,
        "chat_self_coding_goal_attempt": attempt,
    }
    if code_task_id:
        metadata["chat_self_coding_code_task_id"] = code_task_id
    if state is not None and state.editor_codex_session_id:
        metadata["chat_self_coding_editor_codex_session_id"] = (
            state.editor_codex_session_id
        )
    if state is not None and state.failed_worktree_path:
        metadata["chat_self_coding_failed_worktree_path"] = state.failed_worktree_path
    if state is not None and state.failed_worktree_label:
        metadata["chat_self_coding_failed_worktree_label"] = state.failed_worktree_label
    if state is not None and state.failed_worktree_base_branch:
        metadata["chat_self_coding_failed_worktree_base_branch"] = (
            state.failed_worktree_base_branch
        )
    if state is not None and state.failed_worktree_base_sha:
        metadata["chat_self_coding_failed_worktree_base_sha"] = (
            state.failed_worktree_base_sha
        )
    return replace(
        context,
        metadata=metadata,
    )


def _looks_like_recovery_resume(text: str) -> bool:
    lower = text.strip().lower()
    if not lower:
        return False
    tokens = set(re.findall(r"[\w]+", lower, flags=re.UNICODE))
    if any(phrase in lower for phrase in _RECOVERY_RESUME_PHRASES):
        return True
    return len(tokens) <= 3 and bool(tokens & _RECOVERY_RESUME_TOKENS)


def _looks_like_host_ops_activation_request(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in _HOST_OPS_OBJECT_MARKERS) and any(
        marker in lower for marker in _HOST_OPS_ACTION_MARKERS
    )


def _format_host_ops_preflight() -> str:
    return (
        "Стоп: это уже не обычная кодовая задача, а включение живого host/runtime "
        "контура. Я не буду запускать это как self-coding spec вслепую: тут нужны "
        "права на `.env`, systemd/supervisor и проверка, что бот реально поднимется "
        "после остановки.\n\n"
        "Могу продолжить в одном из двух безопасных режимов: отдельно оформить "
        "кодовую часть в repo, либо собрать host-ops чеклист с командами и проверками. "
        "Если нужно именно включать на хосте, сначала надо явно подтвердить, что у "
        "текущей среды есть такие права."
    )


def _format_recovery_failure_response(
    *,
    headline: str,
    detail: str,
    needs_user_decision: bool,
    question: str,
) -> str:
    if not needs_user_decision:
        return (
            f"{headline}\n"
            f"{detail}\n\n"
            "Я не считаю задачу закрытой, но это технический blocker, а не "
            "вопрос продукта или архитектуры к Никите. Повтор сама не запускаю, "
            "чтобы не жечь попытки вслепую; дополнительные сообщения сохраню "
            "как контекст для следующего прохода."
        )
    return (
        f"{headline}\n"
        f"{detail}\n\n"
        f"Нужно решить: {question}\n\n"
        "Следующий запуск заблокирован: сначала обсудим и договоримся по этому "
        "решению. Я не буду тратить ещё одну попытку на тот же неопределённый "
        "blocker."
    )


@dataclass(frozen=True)
class _ProgressHandle:
    message_id: int | None
    task: asyncio.Task[None] | None
    started_at: float = 0.0


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------


class ChatSelfCodingSkill(InlineSkill):
    """Chat-mode orchestrator skill.

    Active only when the admin has an open session (``/код`` / ``/code``);
    routes natural-language input to the legacy pipeline skills.
    """

    name: ClassVar[str] = "chat_self_coding"
    description: ClassVar[str] = (
        "Чат-режим кода — /код или /code для входа, обычный текст внутри"
    )
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"

    triggers: ClassVar[list[str]] = list(_ENTRY_TRIGGERS)

    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "low"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = "auto"

    side_effects: ClassVar[list[SideEffect]] = [
        SideEffect.READS_FROM_KB,
        SideEffect.READS_WORKSPACE,
        SideEffect.WRITES_WORKSPACE,
        SideEffect.READS_FILESYSTEM,
        SideEffect.WRITES_FILESYSTEM,
        SideEffect.CALLS_LLM,
        SideEffect.SENDS_TELEGRAM_MESSAGE,
        SideEffect.MODIFIES_MEMORY,
    ]

    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    def __init__(
        self,
        *,
        admin_user_id: int,
        state_store: StateStore,
        intent_classifier: IntentClassifier,
        ideation_skill: BaseSkill | None = None,
        implement_skill: BaseSkill | None = None,
        spec_skill: BaseSkill | None = None,
        merge_handler: MergeHandler | None = None,
        discussion_skill: BaseSkill | None = None,
        explorer_runner: ExplorerRunner | None = None,
        implementation_runner: ImplementationRunner | None = None,
        session_archive_dir: Path | None = None,
        task_transcript_store: TaskTranscriptStore | None = None,
        spec_tier_resolver: SpecTierResolver | None = None,
    ) -> None:
        self._admin_user_id = admin_user_id
        self._state_store = state_store
        self._classifier = intent_classifier
        self._ideation = ideation_skill
        self._implement = implement_skill
        self._spec = spec_skill
        self._merge_handler = merge_handler
        self._discussion = discussion_skill
        self._explorer_runner = explorer_runner
        self._implementation_runner = implementation_runner
        self._session_archive_dir = session_archive_dir
        self._task_transcript_store = task_transcript_store
        self._spec_tier_resolver = spec_tier_resolver

    async def _record_transcript(
        self,
        state: ChatSelfCodingState,
        *,
        kind: str,
        text: str,
        slug: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self._task_transcript_store is None or not text.strip():
            return
        try:
            await self._task_transcript_store.append(
                task_id=state.code_task_id,
                user_id=state.user_id,
                kind=kind,
                text=text.strip(),
                slug=slug if slug is not None else (state.active_spec_slug or ""),
                payload=payload,
            )
        except Exception:
            logger.warning(
                "chat_self_coding_task_transcript_append_failed",
                user_id=state.user_id,
                task_id=state.code_task_id,
                kind=kind,
                exc_info=True,
            )

    async def _record_result_response(
        self,
        user_id: int,
        fallback_state: ChatSelfCodingState,
        result: SkillResult,
    ) -> None:
        if not result.response.strip():
            return
        state = await self._state_store.load(user_id)
        await self._record_transcript(
            state or fallback_state,
            kind="assistant_message",
            text=f"Жвуша: {result.response}",
        )

    # ----------------------------------------------------------------- routing

    async def can_handle(self, message: str, context: AgentContext) -> float:
        if context.user_id != self._admin_user_id or context.mode != "personal":
            return 0.0
        text = message.strip().lower()
        if _is_entry_trigger(text):
            return 1.0
        state, state_store_available = await self._load_state_for_routing_status(
            context.user_id
        )
        if _is_complete_trigger(text):
            return 1.1 if state_store_available and state is not None else 0.0
        if _is_clear_trigger(text):
            return (
                1.1
                if state_store_available and state is not None and state.is_open
                else 0.0
            )
        if _is_compact_trigger(text) or _parse_goal_command(message)[0]:
            return (
                1.1
                if state_store_available and state is not None and state.is_open
                else 0.0
            )
        # Active session? Intercept every text message.
        if state_store_available and state is not None and state.is_open:
            return 1.1
        if (
            state_store_available
            and state is not None
            and _is_closed_session_resume_trigger(
                message,
                state,
            )
        ):
            return 1.05
        if state_store_available and _is_dialogue_self_coding_start_trigger(
            message,
            context,
        ):
            return 1.05
        return 0.0

    async def _load_state_for_routing_status(
        self,
        user_id: int,
    ) -> tuple[ChatSelfCodingState | None, bool]:
        try:
            return await self._state_store.load(user_id), True
        except Exception as exc:
            logger.warning(
                "chat_self_coding_state_load_failed",
                error=str(exc),
            )
            return None, False

    async def _load_state_for_routing(
        self,
        user_id: int,
    ) -> ChatSelfCodingState | None:
        state, _loaded = await self._load_state_for_routing_status(user_id)
        return state

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        text = message.strip()

        entry_result = await self._handle_entry_tail(text, context)
        if entry_result is not None:
            return entry_result

        state, early_result = await self._state_for_message(text, context)
        if early_result is not None:
            return early_result
        if state is None:
            return SkillResult(success=False, response="")

        session_command = await self._handle_session_command(text, state, context)
        if session_command is not None:
            return session_command

        # Hardcoded exit fast-path — must work even if the classifier is
        # broken. Less common synonyms still go through the classifier.
        if text.lower().strip() in _HARDCODED_EXIT_TOKENS:
            return await self._handle_exit(state, context)

        # Append user message before classification so the classifier sees it
        # in recent_messages on the *next* turn.
        state = state.append_message(f"Никита: {text}")
        await self._state_store.save(state)

        explicit_result = await self._handle_explicit_implementation_slug(
            text,
            state,
            context,
        )
        if explicit_result is not None:
            return explicit_result

        if state.recovery_kind is not None and _looks_like_recovery_resume(text):
            await self._record_transcript(
                state,
                kind="user_message",
                text=f"Никита: {text}",
            )
            result = await self._handle_recovery_resume(state, text, context)
            await self._record_result_response(context.user_id, state, result)
            return result
        if state.recovery_kind is not None and not state.recovery_needs_user_decision:
            await self._record_transcript(
                state,
                kind="user_message",
                text=f"Никита: {text}",
            )
            response = (
                "Это технический blocker, не вопрос выбора к Никите. Я сохранила "
                "сообщение как контекст для следующего прохода. Чтобы продолжить "
                "реализацию с этим контекстом, скажи «продолжай» или «делай»."
            )
            updated_state = state.append_message(f"Жвуша: {response}")
            await self._state_store.save(updated_state)
            result = SkillResult(success=True, response=response)
            await self._record_result_response(context.user_id, updated_state, result)
            return result
        ic_ctx = IntentClassifierContext(
            text=text,
            stage=state.stage,
            active_spec_slug=state.active_spec_slug,
            recent_messages=state.recent_messages,
            requires_ai_approval=self._requires_ai_approval(state),
        )
        classification = await self._classifier(ic_ctx)
        if classification.intent is not Intent.CREATE_SPEC:
            await self._record_transcript(
                state,
                kind="user_message",
                text=f"Никита: {text}",
            )
        result = await self._dispatch(classification.intent, state, text, context)
        await self._record_result_response(context.user_id, state, result)
        return result

    async def _handle_entry_tail(
        self,
        text: str,
        context: AgentContext,
    ) -> SkillResult | None:
        entry_tail = _entry_trigger_tail(text)
        if entry_tail is None:
            return None
        if not entry_tail:
            return await self._handle_entry(context)
        await self._open_entry_session(context)
        return await self.execute(entry_tail, context)

    async def _handle_explicit_implementation_slug(
        self,
        text: str,
        state: ChatSelfCodingState,
        context: AgentContext,
    ) -> SkillResult | None:
        explicit_slug = _parse_explicit_implementation_slug(text)
        if explicit_slug is None:
            return None
        await self._record_transcript(
            state,
            kind="user_message",
            text=f"Никита: {text}",
            slug=explicit_slug,
        )
        run_state = (
            state.with_active_spec(explicit_slug)
            .with_stage(Stage.PENDING_APPROVAL)
            .with_task_phase(TaskPhase.APPROVAL)
            .clear_recovery()
        )
        await self._state_store.save(run_state)
        result = await self._handle_approve(run_state, context)
        await self._record_result_response(context.user_id, run_state, result)
        return result

    async def _state_for_message(
        self,
        text: str,
        context: AgentContext,
    ) -> tuple[ChatSelfCodingState | None, SkillResult | None]:
        state = await self._state_store.load(context.user_id)
        if state is None:
            if _is_dialogue_self_coding_start_trigger(text, context):
                return None, await self._handle_implicit_dialogue_entry(
                    text,
                    context,
                    existing_state=None,
                )
            return None, SkillResult(success=False, response="")
        if _is_complete_trigger(text):
            return None, await self._handle_complete(state, context)
        if state.is_open:
            return state, None
        if _is_closed_session_resume_trigger(text, state):
            reopened = state.with_open(True)
            await self._state_store.save(reopened)
            return reopened, None
        if _is_dialogue_self_coding_start_trigger(text, context):
            return None, await self._handle_implicit_dialogue_entry(
                text,
                context,
                existing_state=state.with_open(True),
            )
        return None, SkillResult(success=False, response="")

    # ----------------------------------------------------------------- entry

    async def _open_entry_session(self, context: AgentContext) -> ChatSelfCodingState:
        existing = await self._state_store.load(context.user_id)
        state = (
            existing.with_open(True)
            if existing is not None
            else ChatSelfCodingState(user_id=context.user_id, stage=Stage.IDLE)
        )
        await self._state_store.save(state)
        return state

    async def _handle_entry(self, context: AgentContext) -> SkillResult:
        existing = await self._state_store.load(context.user_id)
        state = await self._open_entry_session(context)
        if existing is None:
            body = (
                "<b>🎯 /код</b>\n\n"
                "Готова. Можем сначала спокойно обсудить, что ты хочешь сделать. "
                "Когда решишь — скажи «оформи план», и я подготовлю spec "
                "перед тем как что-то писать. После плана можно продолжить "
                "обсуждение или сказать «делай», чтобы начать реализацию.\n\n"
                "Скажи «выход», чтобы закрыть режим без потери сессии. "
                "`goal ...` закрепит цель, `compact` сожмёт рабочий контекст, "
                "`/clear` начнёт с чистой комнаты, `/готово` завершит сессию "
                "окончательно."
            )
        else:
            body = (
                "<b>🎯 /код</b>\n\n"
                "Вернулась в ту же рабочую сессию. "
                f"{_status_text(state)}\n\n"
                "«выход» просто закрывает режим, `compact` сжимает контекст, "
                "`/готово` завершает сессию."
            )
        return await self._reply_html(context, body)

    async def _handle_implicit_dialogue_entry(
        self,
        text: str,
        context: AgentContext,
        *,
        existing_state: ChatSelfCodingState | None,
    ) -> SkillResult:
        """Start the /код room from a confirmed engineering chat follow-up."""
        state = existing_state or ChatSelfCodingState(
            user_id=context.user_id,
            stage=Stage.IDLE,
        )
        if state.active_spec_slug is not None or state.stage != Stage.IDLE:
            state = state.with_new_code_task()
        state = (
            state.with_open(True)
            .with_stage(Stage.IDLE)
            .with_task_phase(TaskPhase.DISCUSSION)
            .with_active_spec(None)
            .clear_recovery()
            .clear_editor_resume()
        )
        state = _state_with_dialogue_bootstrap(state, context)
        create_text = _build_implicit_spec_create_text(text, state, context)
        return await self._handle_create(
            text,
            state,
            context,
            create_text_override=create_text,
            start_new_task=False,
            record_user_message=True,
        )

    # ----------------------------------------------------------------- dispatch

    async def _dispatch(
        self,
        intent: Intent,
        state: ChatSelfCodingState,
        text: str,
        context: AgentContext,
    ) -> SkillResult:
        if intent == Intent.EXIT:
            return await self._handle_exit(state, context)
        if intent == Intent.STATUS:
            return SkillResult(success=True, response=_status_text(state))
        if intent == Intent.CREATE_SPEC:
            return await self._handle_create(text, state, context)
        if intent == Intent.APPROVE:
            return await self._handle_approval_intent(text, state, context)
        if intent == Intent.REJECT:
            return await self._handle_reject(state, context)
        if intent == Intent.RUN_SPEC:
            return await self._handle_run(state, context)
        if intent == Intent.MERGE:
            return await self._handle_merge(state, context)
        if intent == Intent.SHOW_SPEC:
            return await self._handle_show(state, context)
        # OTHER / fallback
        return await self._handle_discussion(text, state, context)

    # ------------------------------------------------------------ intent paths

    async def _handle_exit(
        self, state: ChatSelfCodingState, context: AgentContext
    ) -> SkillResult:
        await self._state_store.save(state.with_open(False))
        return SkillResult(
            success=True,
            response=(
                "Вышла из рабочего режима, но сессию оставила. "
                "Вернёшься через /код или /code; /готово завершит её окончательно."
            ),
        )

    async def _handle_session_command(
        self,
        text: str,
        state: ChatSelfCodingState,
        context: AgentContext,
    ) -> SkillResult | None:
        if _is_clear_trigger(text):
            return await self._handle_clear(state, context)
        if _is_complete_trigger(text):
            return await self._handle_complete(state, context)
        if _is_compact_trigger(text):
            return await self._handle_compact(state, context)
        is_goal, goal_text = _parse_goal_command(text)
        if is_goal:
            return await self._handle_goal(state, goal_text)
        return None

    async def _handle_complete(
        self, state: ChatSelfCodingState, context: AgentContext
    ) -> SkillResult:
        await self._archive_session(state)
        await self._state_store.clear(context.user_id)
        return SkillResult(
            success=True,
            response=(
                "Завершила сессию /код, сохранила её снимок и убрала "
                "из активного контекста."
            ),
        )

    async def _handle_clear(
        self, state: ChatSelfCodingState, context: AgentContext
    ) -> SkillResult:
        del state
        await self._state_store.save(
            ChatSelfCodingState(user_id=context.user_id, stage=Stage.IDLE)
        )
        return SkillResult(
            success=True,
            response=(
                "Очистила контекст текущей /код-сессии: обсуждение, recovery и "
                "активный plan сброшены. Режим /код остался открыт."
            ),
        )

    async def _handle_goal(
        self, state: ChatSelfCodingState, goal_text: str
    ) -> SkillResult:
        if not goal_text:
            if state.active_goal:
                return SkillResult(
                    success=True,
                    response=f"Активная цель: {state.active_goal}",
                )
            return SkillResult(
                success=True,
                response="Активная цель не задана. Напиши `goal <цель>`.",
            )
        updated = state.with_active_goal(goal_text)
        await self._state_store.save(
            updated.append_message(f"Никита зафиксировал цель /код: {goal_text}")
        )
        return SkillResult(
            success=True,
            response=f"Зафиксировала цель для этой /код-сессии: {goal_text}",
        )

    async def _handle_compact(
        self, state: ChatSelfCodingState, context: AgentContext
    ) -> SkillResult:
        summary = _build_compact_summary(state)
        compacted = (
            state.with_compact_summary(summary)
            .with_readonly_codex_session(None)
            .clear_recovery()
            .clear_editor_resume()
        )
        compacted = replace(
            compacted,
            recent_messages=(f"Сжатый контекст /код: {summary}",),
        )
        await self._state_store.save(compacted)
        del context
        return SkillResult(
            success=True,
            response=(
                "Сжала контекст /код и сбросила read-only Codex thread. "
                "Следующий анализ кода начнётся свежей Codex-сессией с этой сводкой."
            ),
        )

    async def _archive_session(self, state: ChatSelfCodingState) -> None:
        if self._session_archive_dir is None:
            return
        timestamp = datetime.now(UTC)
        slug_time = timestamp.strftime("%Y-%m-%dT%H-%M-%SZ")
        path = self._session_archive_dir / f"{slug_time}-user-{state.user_id}.md"
        lines = [
            "# /код session",
            "",
            f"- archived_at: {timestamp.isoformat()}",
            f"- user_id: {state.user_id}",
            f"- code_task_id: {state.code_task_id}",
            f"- stage: {state.stage.value}",
            f"- active_spec_slug: {state.active_spec_slug or ''}",
            f"- is_open: {str(state.is_open).lower()}",
            f"- active_goal: {state.active_goal or ''}",
            f"- compact_summary: {state.compact_summary or ''}",
            f"- readonly_codex_session_id: {state.readonly_codex_session_id or ''}",
            f"- editor_codex_session_id: {state.editor_codex_session_id or ''}",
            f"- failed_worktree_path: {state.failed_worktree_path or ''}",
            f"- failed_worktree_label: {state.failed_worktree_label or ''}",
            f"- failed_worktree_base_branch: {state.failed_worktree_base_branch or ''}",
            f"- failed_worktree_base_sha: {state.failed_worktree_base_sha or ''}",
            "",
            "## Recent messages",
            "",
        ]
        if state.recent_messages:
            lines.extend(f"- {message}" for message in state.recent_messages)
        else:
            lines.append("- Нет сохранённого контекста.")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            logger.warning(
                "chat_self_coding_session_archive_failed",
                user_id=state.user_id,
                exc_info=True,
            )

    async def _handle_create(
        self,
        text: str,
        state: ChatSelfCodingState,
        context: AgentContext,
        *,
        create_text_override: str | None = None,
        start_new_task: bool = True,
        record_user_message: bool = True,
    ) -> SkillResult:
        if self._ideation is None:
            return SkillResult(
                success=False,
                response=(
                    "Архитект пока не подключён к чат-режиму. "
                    "Используй старую команду /spec_create."
                ),
            )

        create_text = create_text_override or _build_spec_create_text(text, state)
        if _looks_like_host_ops_activation_request(create_text):
            response = _format_host_ops_preflight()
            if record_user_message:
                await self._record_transcript(
                    state,
                    kind="user_message",
                    text=f"Никита: {text}",
                )
            await self._state_store.save(state.append_message(f"Жвуша: {response}"))
            return SkillResult(success=True, response=response)

        progress = await _start_architect_progress(context)

        task_state = (
            state.with_new_code_task()
            if start_new_task and state.active_spec_slug is not None
            else state
        )
        task_state = task_state.with_drafting_started(time.time()).with_task_phase(
            TaskPhase.SPEC
        )
        await self._state_store.save(task_state)
        if record_user_message:
            await self._record_transcript(
                task_state,
                kind="user_message",
                text=f"Никита: {text}",
            )
        await self._record_transcript(
            task_state,
            kind="state_transition",
            text="Жвуша: начала сбор plan.",
            payload={"stage": Stage.DRAFTING.value},
        )

        forwarded = f"/spec_create {create_text}"
        await _update_architect_progress(
            context,
            progress,
            percent=5,
            detail=(
                "Передала Architect полный контекст обсуждения и похожие прошлые циклы."
            ),
            stage="запуск Architect",
        )
        ideation_context = _code_task_context(
            context,
            task_state.code_task_id,
            active_goal=task_state.active_goal,
            readonly_codex_session_id=task_state.readonly_codex_session_id,
        )
        result = await self._ideation.execute(forwarded, ideation_context)

        if not result.success:
            needs_clarification = bool(
                (result.metadata or {}).get("needs_clarification")
            )
            await _finish_architect_progress(
                context,
                progress,
                detail=(
                    "Нужно уточнение перед планом."
                    if needs_clarification
                    else "План не собрался. Ниже покажу причину."
                ),
                stage="завершено",
            )
            # Architect failed — drop back to IDLE so the user can
            # retry without a stuck PENDING_APPROVAL session. Keep a
            # recovery record so discussion can continue and a short
            # «продолжай» resumes the same operation after consensus.
            detail = result.response.strip()
            recovery_state = (
                task_state.with_stage(Stage.IDLE)
                .with_task_phase(TaskPhase.DISCUSSION)
                .with_active_spec(None)
                .with_recovery(
                    kind="create_spec",
                    text=create_text,
                    error=detail or "Architect не смог составить spec.",
                )
            )
            if needs_clarification and detail:
                await self._state_store.save(
                    recovery_state.append_message(f"Жвуша: {detail}")
                )
                return SkillResult(success=True, response=detail)
            if detail:
                response = (
                    "Не получилось составить план. Причина:\n"
                    f"{detail}\n\n"
                    "Я не считаю задачу закрытой. Давай обсудим, что поправить; "
                    "когда договоримся, скажи «продолжай» или «делай», и я "
                    "снова соберу plan по этому же контексту."
                )
            else:
                response = (
                    "Не получилось составить план. "
                    "Давай обсудим, что поправить; когда договоримся, скажи "
                    "«продолжай» или «делай»."
                )
            await self._state_store.save(
                recovery_state.append_message(f"Жвуша: {response}")
            )
            return SkillResult(
                success=False,
                response=response,
            )

        await _finish_architect_progress(
            context,
            progress,
            detail="План собран. Показываю его ниже.",
            stage="завершено",
        )

        # Bind the new slug to the session so subsequent approve / reject
        # commands know which spec to act on. Architect publishes the
        # 📋 PLAN block event itself; we suppress the technical response.
        new_slug = result.metadata.get("slug") if result.metadata else None
        new_state = task_state.with_stage(Stage.PENDING_APPROVAL).with_task_phase(
            TaskPhase.APPROVAL
        )
        new_state = new_state.clear_recovery()
        if isinstance(new_slug, str) and new_slug:
            new_state = new_state.with_active_spec(new_slug)
        await self._state_store.save(new_state)

        return SkillResult(success=True, response="")

    async def _handle_recovery_resume(
        self,
        state: ChatSelfCodingState,
        text: str,
        context: AgentContext,
    ) -> SkillResult:
        if state.recovery_kind == "create_spec":
            create_text = _build_recovery_spec_create_text(state, text)
            clean_state = state.clear_recovery().with_task_phase(TaskPhase.SPEC)
            await self._state_store.save(clean_state)
            return await self._handle_create(
                text,
                clean_state,
                context,
                create_text_override=create_text,
                start_new_task=False,
                record_user_message=False,
            )
        if state.recovery_kind == "approve_spec":
            clean_state = state.clear_recovery().with_task_phase(TaskPhase.APPROVAL)
            await self._state_store.save(clean_state)
            return await self._handle_approve(clean_state, context)
        if state.recovery_kind == "run_spec":
            clean_state = (
                state.clear_recovery()
                .with_stage(Stage.PENDING_APPROVAL)
                .with_task_phase(TaskPhase.REPAIR)
            )
            await self._state_store.save(clean_state)
            return await self._handle_approve(clean_state, context)
        await self._state_store.save(state.clear_recovery())
        return await self._handle_plain_discussion(
            text, state.clear_recovery(), context
        )

    async def _handle_discussion(
        self,
        text: str,
        state: ChatSelfCodingState,
        context: AgentContext,
    ) -> SkillResult:
        if _looks_like_incoming_material_preamble(text):
            response = _incoming_material_wait_response()
            await self._state_store.save(state.append_message(f"Жвуша: {response}"))
            return SkillResult(success=True, response=response)

        if self._explorer_runner is not None and _should_use_explorer_for_discussion(
            text, state
        ):
            explorer_result = await self._handle_explorer_discussion(
                text,
                state,
                context,
            )
            if explorer_result is not None:
                return explorer_result

        return await self._handle_plain_discussion(text, state, context)

    async def _handle_explorer_discussion(
        self,
        text: str,
        state: ChatSelfCodingState,
        context: AgentContext,
    ) -> SkillResult | None:
        assert self._explorer_runner is not None
        await _send_explorer_status(
            context,
            "Смотрю код, файлы и контекст /код в read-only режиме.",
        )

        async def remember_session_id(session_id: str) -> None:
            current = await self._state_store.load(state.user_id)
            if current is None or current.code_task_id != state.code_task_id:
                return
            if current.readonly_codex_session_id == session_id:
                return
            await self._state_store.save(
                current.with_readonly_codex_session(session_id)
            )

        try:
            response = await self._explorer_runner(
                system_prompt=_EXPLORER_SYSTEM_PROMPT,
                user_prompt=_build_explorer_prompt(text, state),
                progress_callback=_make_explorer_progress_callback(context),
                session_id=state.readonly_codex_session_id or "",
                persist_session=True,
                session_callback=remember_session_id,
            )
        except Exception:
            logger.warning("chat_self_coding_explorer_failed", exc_info=True)
            return None

        response = response.strip()
        if not response:
            logger.warning("chat_self_coding_explorer_empty_response")
            return None
        current = await self._state_store.load(state.user_id)
        base_state = current if current is not None else state
        await self._state_store.save(base_state.append_message(f"Жвуша: {response}"))
        return SkillResult(success=True, response=response)

    async def _handle_plain_discussion(
        self,
        text: str,
        state: ChatSelfCodingState,
        context: AgentContext,
    ) -> SkillResult:
        if self._discussion is None:
            return SkillResult(
                success=True,
                response=(
                    "Не уверена, что ты имел в виду. Попробуй описать иначе — "
                    "или скажи «выход» чтобы выйти."
                ),
            )
        result = await self._discussion.execute(text, _discussion_context(context))
        if result.response:
            await self._state_store.save(
                state.append_message(f"Жвуша: {result.response}")
            )
        return result

    async def _handle_approval_intent(
        self,
        text: str,
        state: ChatSelfCodingState,
        context: AgentContext,
    ) -> SkillResult:
        return await self._handle_approve(state, context)

    async def _handle_approve(
        self,
        state: ChatSelfCodingState,
        context: AgentContext,
    ) -> SkillResult:
        """Approve the pending spec AND auto-run Editor.

        Discussion remains the default after a plan. Only an explicit
        implementation trigger such as «делай» should reach this handler.
        """
        slug = state.active_spec_slug
        if state.stage != Stage.PENDING_APPROVAL or slug is None or self._spec is None:
            return SkillResult(
                success=True,
                response=(
                    "Сначала нужен готовый план. Опиши задачу или скажи "
                    "«оформи план», когда обсуждение можно превращать в spec."
                ),
            )

        forwarded_approve = f"/spec approve {slug}"
        approve_result = await self._spec.execute(forwarded_approve, context)
        if not approve_result.success:
            # Real approve failure (validation, terminal status, etc.) —
            # keep the room in discussion mode so consensus can repair it.
            detail = approve_result.response.strip() or "Approve gate не прошёл."
            response = _format_recovery_failure_response(
                headline="Не получилось одобрить план. Причина:",
                detail=detail,
                needs_user_decision=True,
                question="план нужно править, пересобрать или оставить прежний?",
            )
            await self._state_store.save(
                state.with_stage(Stage.PENDING_APPROVAL)
                .with_task_phase(TaskPhase.APPROVAL)
                .with_recovery(
                    kind="approve_spec",
                    text=slug,
                    error=detail,
                    needs_user_decision=True,
                    question="план нужно править, пересобрать или оставить прежний?",
                )
                .append_message(f"Жвуша: {response}")
            )
            return SkillResult(success=False, response=response)

        # Auto-run Editor. ``implement_spec`` is gated on
        # ``spec.status == APPROVED`` which we just ensured.
        # Long-running (5-10 min); block events from Editor surface the
        # 🔧 / ✏️ / ✅ messages via Pub/Sub.
        if self._implement is None and self._implementation_runner is None:
            return SkillResult(success=True, response=approve_result.response)

        await self._state_store.save(
            state.with_stage(Stage.RUNNING).with_task_phase(TaskPhase.IMPLEMENTATION)
        )

        run_result = await self._run_implementation_goal(slug, state, context)

        if _result_is_background(run_result):
            return SkillResult(success=True, response=run_result.response)
        if not run_result.success:
            detail = run_result.response.strip() or "Реализация не прошла."
            needs_user_decision = _result_needs_user_decision(run_result)
            question = _result_decision_question(run_result)
            response = _format_recovery_failure_response(
                headline="Реализация остановилась. Причина:",
                detail=detail,
                needs_user_decision=needs_user_decision,
                question=question,
            )
            await self._state_store.save(
                _state_with_editor_resume_from_result(
                    state.with_stage(Stage.PENDING_APPROVAL)
                    .with_task_phase(TaskPhase.REPAIR)
                    .with_recovery(
                        kind="run_spec",
                        text=slug,
                        error=detail,
                        needs_user_decision=needs_user_decision,
                        question=question,
                    ),
                    run_result,
                ).append_message(f"Жвуша: {response}")
            )
            return SkillResult(success=False, response=response)
        if "dry-run" in run_result.response.lower():
            detail = run_result.response.strip()
            await self._state_store.save(
                state.with_stage(Stage.PENDING_APPROVAL)
                .with_task_phase(TaskPhase.REPAIR)
                .with_recovery(
                    kind="run_spec",
                    text=slug,
                    error=detail or "Реализация дошла только до dry-run.",
                )
            )
            return SkillResult(success=True, response=run_result.response)

        await self._state_store.save(
            state.with_stage(Stage.DONE)
            .with_task_phase(TaskPhase.DONE)
            .clear_recovery()
            .clear_editor_resume()
        )

        # Editor publishes its own DONE / ERROR block events; we don't
        # surface its technical response.
        return SkillResult(success=True, response="")

    async def _handle_reject(
        self,
        state: ChatSelfCodingState,
        context: AgentContext,
    ) -> SkillResult:
        slug = state.active_spec_slug
        if slug is None or self._spec is None:
            return SkillResult(
                success=True,
                response="Нечего отклонять — нет активного плана.",
            )
        forwarded = f"/spec reject {slug}"
        await self._spec.execute(forwarded, context)
        await self._state_store.save(
            state.with_stage(Stage.IDLE)
            .with_task_phase(TaskPhase.DISCUSSION)
            .with_active_spec(None)
        )
        return SkillResult(
            success=True,
            response="Отменила. Опиши задачу заново, если хочешь.",
        )

    async def _handle_run(
        self,
        state: ChatSelfCodingState,
        context: AgentContext,
    ) -> SkillResult:
        slug = state.active_spec_slug
        if slug is None or (
            self._implement is None and self._implementation_runner is None
        ):
            return SkillResult(
                success=True,
                response="Нечего запускать — нет одобренного плана.",
            )
        await self._state_store.save(
            state.with_stage(Stage.RUNNING).with_task_phase(TaskPhase.IMPLEMENTATION)
        )
        run_result = await self._run_implementation_goal(slug, state, context)
        if _result_is_background(run_result):
            return SkillResult(success=True, response=run_result.response)
        if not run_result.success:
            detail = run_result.response.strip() or "Реализация не прошла."
            needs_user_decision = _result_needs_user_decision(run_result)
            question = _result_decision_question(run_result)
            response = _format_recovery_failure_response(
                headline="Реализация остановилась. Причина:",
                detail=detail,
                needs_user_decision=needs_user_decision,
                question=question,
            )
            await self._state_store.save(
                _state_with_editor_resume_from_result(
                    state.with_stage(Stage.PENDING_APPROVAL)
                    .with_task_phase(TaskPhase.REPAIR)
                    .with_recovery(
                        kind="run_spec",
                        text=slug,
                        error=detail,
                        needs_user_decision=needs_user_decision,
                        question=question,
                    ),
                    run_result,
                ).append_message(f"Жвуша: {response}")
            )
            return SkillResult(success=False, response=response)
        if "dry-run" in run_result.response.lower():
            detail = run_result.response.strip()
            await self._state_store.save(
                state.with_stage(Stage.PENDING_APPROVAL)
                .with_task_phase(TaskPhase.REPAIR)
                .with_recovery(
                    kind="run_spec",
                    text=slug,
                    error=detail or "Реализация дошла только до dry-run.",
                )
            )
            return SkillResult(success=True, response=run_result.response)
        await self._state_store.save(
            state.with_stage(Stage.DONE)
            .with_task_phase(TaskPhase.DONE)
            .clear_recovery()
            .clear_editor_resume()
        )
        return SkillResult(
            success=True,
            response="Запустила и дождалась завершения. Показываю итог отдельным блоком.",
        )

    async def _run_implementation_goal(
        self,
        slug: str,
        state: ChatSelfCodingState,
        context: AgentContext,
    ) -> SkillResult:
        """Run implementation until it succeeds or needs Никита's decision."""
        last_result: SkillResult | None = None
        for attempt in range(_GOAL_AUTO_RETRY_LIMIT + 1):
            if attempt > 0 and self._spec is not None:
                approve_result = await self._spec.execute(
                    f"/spec approve {slug}",
                    context,
                )
                if not approve_result.success:
                    return approve_result
            result = await self._run_implementation(
                slug,
                state,
                context,
                goal_attempt=attempt,
            )
            last_result = result
            if _result_is_background(result) or result.success:
                return result
            if _result_needs_user_decision(result):
                return result
            if not _result_auto_retryable(result):
                return result
        assert last_result is not None
        return last_result

    async def _run_implementation(
        self,
        slug: str,
        state: ChatSelfCodingState,
        context: AgentContext,
        *,
        goal_attempt: int = 0,
    ) -> SkillResult:
        if self._implementation_runner is not None:
            if context.bot is not None and context.chat_id is not None:
                background_runner = getattr(
                    self._implementation_runner,
                    "start_background",
                    None,
                )
                if callable(background_runner):
                    implementation_context = _goal_attempt_context(
                        context,
                        goal_attempt,
                        code_task_id=state.code_task_id,
                        state=state,
                    )
                    job = await background_runner(
                        slug=slug,
                        context=implementation_context,
                        recent_messages=state.recent_messages,
                        completion_callback=self._make_implementation_completion(
                            slug=slug,
                            user_id=context.user_id,
                            context=implementation_context,
                            attempt=goal_attempt,
                        ),
                    )
                    job_id = str(getattr(job, "id", ""))
                    return SkillResult(
                        success=True,
                        response="",
                        metadata={"background": "true", "agent_job_id": job_id},
                    )
            return await self._implementation_runner(
                slug=slug,
                context=_goal_attempt_context(
                    context,
                    goal_attempt,
                    code_task_id=state.code_task_id,
                    state=state,
                ),
                recent_messages=state.recent_messages,
            )
        if self._implement is None:
            return SkillResult(success=True, response="")
        implementation_context = replace(
            context,
            metadata={
                **context.metadata,
                "chat_self_coding_recent_messages": state.recent_messages,
                "chat_self_coding_code_task_id": state.code_task_id,
            },
        )
        implementation_context = _goal_attempt_context(
            implementation_context,
            goal_attempt,
            code_task_id=state.code_task_id,
            state=state,
        )
        return await self._implement.execute(
            f"/spec_run {slug}", implementation_context
        )

    def _make_implementation_completion(
        self,
        *,
        slug: str,
        user_id: int,
        context: AgentContext,
        attempt: int = 0,
    ) -> Callable[[SkillResult], Awaitable[None]]:
        async def completion(result: SkillResult) -> None:
            state = await self._state_store.load(user_id)
            if state is None or state.active_spec_slug != slug:
                return
            if not result.success or "dry-run" in result.response.lower():
                detail = result.response.strip() or "Фоновая реализация не прошла."
                needs_user_decision = _result_needs_user_decision(result)
                if (
                    not needs_user_decision
                    and _result_auto_retryable(result)
                    and attempt < _GOAL_AUTO_RETRY_LIMIT
                ):
                    next_attempt = attempt + 1
                    if self._spec is not None:
                        approve_result = await self._spec.execute(
                            f"/spec approve {slug}",
                            context,
                        )
                        if not approve_result.success:
                            await self._state_store.save(
                                state.with_stage(Stage.PENDING_APPROVAL)
                                .with_task_phase(TaskPhase.APPROVAL)
                                .with_recovery(
                                    kind="approve_spec",
                                    text=slug,
                                    error=(
                                        approve_result.response.strip()
                                        or "Approve gate не прошёл."
                                    ),
                                    needs_user_decision=True,
                                    question=(
                                        "план нужно править, пересобрать "
                                        "или оставить прежний?"
                                    ),
                                )
                            )
                            return
                    retry_note = (
                        "Жвуша: предыдущий проход остановился на чиняемой "
                        f"ошибке без вопроса к Никите: {detail[:500]}"
                    )
                    retry_state = (
                        state.clear_recovery()
                        .with_stage(Stage.RUNNING)
                        .with_task_phase(TaskPhase.REPAIR)
                        .append_message(retry_note)
                    )
                    await self._state_store.save(retry_state)
                    await self._run_implementation(
                        slug,
                        retry_state,
                        context,
                        goal_attempt=next_attempt,
                    )
                    return
                await self._state_store.save(
                    _state_with_editor_resume_from_result(
                        state.with_stage(Stage.PENDING_APPROVAL)
                        .with_task_phase(TaskPhase.REPAIR)
                        .with_recovery(
                            kind="run_spec",
                            text=slug,
                            error=detail,
                            needs_user_decision=needs_user_decision,
                            question=_result_decision_question(result),
                        ),
                        result,
                    )
                )
                return
            await self._state_store.save(
                state.with_stage(Stage.DONE)
                .with_task_phase(TaskPhase.DONE)
                .clear_recovery()
                .clear_editor_resume()
            )

        return completion

    async def _handle_merge(
        self,
        state: ChatSelfCodingState,
        context: AgentContext,
    ) -> SkillResult:
        slug = state.active_spec_slug
        if state.stage != Stage.DONE or slug is None:
            return SkillResult(
                success=True,
                response="Сливать пока нечего — дождись блока «Готово».",
            )
        if self._merge_handler is None:
            return SkillResult(
                success=False,
                response="Merge handler пока не подключён к чат-режиму.",
            )
        result = await self._merge_handler(slug, context)
        if result.success:
            await self._state_store.save(
                state.with_stage(Stage.IDLE)
                .with_task_phase(TaskPhase.DISCUSSION)
                .with_active_spec(None)
            )
        return result

    async def _handle_show(
        self, state: ChatSelfCodingState, context: AgentContext
    ) -> SkillResult:
        slug = state.active_spec_slug
        if slug is None:
            return SkillResult(
                success=True,
                response="Активного плана нет. Опиши задачу.",
            )
        body = f"Активный план: <code>{escape(slug)}</code>. Стадия: " + _status_text(
            state
        )
        return await self._reply_html(context, body)

    # ------------------------------------------------------------ helpers

    def _requires_ai_approval(self, state: ChatSelfCodingState) -> bool:
        if state.stage != Stage.PENDING_APPROVAL or not state.active_spec_slug:
            return False
        if self._spec_tier_resolver is None:
            return False
        try:
            tier = self._spec_tier_resolver(state.active_spec_slug)
        except Exception:
            logger.warning(
                "chat_self_coding_spec_tier_resolve_failed",
                slug=state.active_spec_slug,
                exc_info=True,
            )
            return False
        return tier is not None and tier >= 3

    async def _reply_html(self, context: AgentContext, body: str) -> SkillResult:
        """Send an HTML-formatted reply.

        When the bot is wired on the context (production path), send via
        ``bot.send_message(parse_mode="HTML")`` directly so the
        dispatcher's markdown→HTML converter doesn't escape our already-
        HTML tags. Interfaces such as the VS Code bridge request returned
        response text explicitly because they log ``SkillResult.response`` as
        the visible chat reply. When no bot is available (unit tests by
        default), return the body in ``SkillResult.response`` so legacy
        assertions keep working.
        """
        if _returns_response_text(context):
            return SkillResult(success=True, response=body)
        if context.bot is not None and context.chat_id is not None:
            try:
                await context.bot.send_message(
                    chat_id=context.chat_id,
                    text=body,
                    parse_mode="HTML",
                )
                return SkillResult(
                    success=True,
                    response="",
                )
            except Exception:
                logger.warning("chat_self_coding_html_send_failed", exc_info=True)
        return SkillResult(success=True, response=body)


def _returns_response_text(context: AgentContext) -> bool:
    return (
        context.metadata.get("return_response_text") is True
        or context.metadata.get("interface") == "vscode"
    )


def _make_explorer_progress_callback(
    context: AgentContext,
) -> Callable[[str], Awaitable[None]] | None:
    if context.bot is None or context.chat_id is None:
        return None

    async def progress(status: str) -> None:
        await _send_explorer_status(context, status)

    return progress


async def _send_explorer_status(context: AgentContext, status: str) -> None:
    if context.bot is None or context.chat_id is None:
        return
    text = status.strip()
    if not text:
        return
    try:
        await context.bot.send_message(
            chat_id=context.chat_id,
            text=f"🔎 {text}",
        )
    except Exception:
        logger.warning("chat_self_coding_explorer_status_send_failed", exc_info=True)


async def _start_architect_progress(context: AgentContext) -> _ProgressHandle | None:
    if context.bot is None or context.chat_id is None:
        return None
    started_at = time.monotonic()
    try:
        sent = await context.bot.send_message(
            chat_id=context.chat_id,
            text=format_architect_progress(
                ProgressBlock(
                    percent=10,
                    detail="Приняла запрос и открыла контекст /код.",
                    stage="приём задачи",
                    elapsed_seconds=0,
                )
            ),
            parse_mode="HTML",
        )
    except Exception:
        logger.warning("chat_self_coding_progress_send_failed", exc_info=True)
        return None

    message_id = getattr(sent, "message_id", None)
    if not isinstance(message_id, int):
        return _ProgressHandle(message_id=None, task=None, started_at=started_at)

    task = asyncio.create_task(
        _run_architect_progress(
            bot=context.bot,
            chat_id=context.chat_id,
            message_id=message_id,
            started_at=started_at,
        )
    )
    return _ProgressHandle(message_id=message_id, task=task, started_at=started_at)


async def _run_architect_progress(
    *,
    bot: Any,
    chat_id: int,
    message_id: int,
    started_at: float,
) -> None:
    try:
        while True:
            await asyncio.sleep(_ARCHITECT_PROGRESS_INTERVAL_SECONDS)
            elapsed_seconds = int(time.monotonic() - started_at)
            await _edit_architect_progress(
                bot=bot,
                chat_id=chat_id,
                message_id=message_id,
                percent=_ARCHITECT_PROGRESS_WAIT_PERCENT,
                detail="Жду Architect: она собирает полный plan по текущему контексту.",
                stage="сбор плана",
                elapsed_seconds=elapsed_seconds,
            )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("chat_self_coding_progress_loop_failed", exc_info=True)


async def _update_architect_progress(
    context: AgentContext,
    handle: _ProgressHandle | None,
    *,
    percent: int,
    detail: str,
    stage: str,
) -> None:
    if handle is None:
        return
    if context.bot is None or context.chat_id is None or handle.message_id is None:
        return
    await _edit_architect_progress(
        bot=context.bot,
        chat_id=context.chat_id,
        message_id=handle.message_id,
        percent=percent,
        detail=detail,
        stage=stage,
        elapsed_seconds=int(time.monotonic() - handle.started_at),
    )


async def _finish_architect_progress(
    context: AgentContext,
    handle: _ProgressHandle | None,
    *,
    detail: str,
    stage: str,
) -> None:
    if handle is None:
        return
    if handle.task is not None:
        handle.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await handle.task
    if context.bot is None or context.chat_id is None or handle.message_id is None:
        return
    await _edit_architect_progress(
        bot=context.bot,
        chat_id=context.chat_id,
        message_id=handle.message_id,
        percent=100,
        detail=detail,
        stage=stage,
        elapsed_seconds=int(time.monotonic() - handle.started_at),
    )


async def _edit_architect_progress(
    *,
    bot: Any,
    chat_id: int,
    message_id: int,
    percent: int,
    detail: str,
    stage: str = "",
    elapsed_seconds: int | None = None,
) -> None:
    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=format_architect_progress(
                ProgressBlock(
                    percent=percent,
                    detail=detail,
                    stage=stage,
                    elapsed_seconds=elapsed_seconds,
                )
            ),
            parse_mode="HTML",
        )
    except TelegramBadRequest as exc:
        if "message is not modified" in str(exc).lower():
            logger.debug("chat_self_coding_progress_edit_unchanged")
            return
        logger.warning("chat_self_coding_progress_edit_failed", exc_info=True)
    except Exception:
        logger.warning("chat_self_coding_progress_edit_failed", exc_info=True)

"""Read-only codebase Explorer for ordinary personal chat."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

import structlog

from src.skills.base import AgentContext, InlineSkill, SideEffect, SkillResult
from src.skills.chat_response.context_loader import ContextLoader
from src.utils.telegram import send_long_message

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path


class ExplorerRunner(Protocol):
    async def __call__(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        progress_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> str: ...


class BackgroundExplorerRunner(Protocol):
    async def start_background(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        completion_callback: Callable[[str], Awaitable[None]],
        progress_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> object: ...


logger = structlog.get_logger()

_ACTION_MARKERS: tuple[str, ...] = (
    "изучи",
    "посмотри",
    "проверь",
    "сравни",
    "сопостав",
    "проанализ",
    "разбери",
    "аудит",
    "найди",
    "что уже есть",
    "что можно улучш",
    "что улучшить",
)
_CODEBASE_MARKERS: tuple[str, ...] = (
    "кодовую баз",
    "кодобаз",
    "codebase",
    "исходник",
    "репозитор",
    "репу",
    "репе",
    "репа",
    "repo",
    "проект",
    "файлы",
    "файл",
    "логи",
    "лог",
    "в тебе",
    "у тебя",
    "самокод",
    "бот",
)
_SOURCE_MARKERS: tuple[str, ...] = (
    "пост",
    "источник",
    "скрин",
    "скриншот",
    "фото",
    "картинк",
    "ссылк",
)
_SOURCE_ONLY_RE = re.compile(
    r"\b(объясни|разбери|прочитай|посмотри)\b.{0,40}\b"
    r"(пост|скрин|скриншот|фото|картинк|ссылк)\b",
    re.IGNORECASE,
)
_INCOMING_MATERIAL_RE = re.compile(
    r"("
    r"\b(сейчас|щас|щаз|ща|сча)\s+(скину|кину|пришлю|отправлю)\b|"
    r"\b(скину|кину|пришлю|отправлю)\b.{0,40}\b"
    r"(пост|скрин|скриншот|фото|картинк|файл|лог|текст|ссылк)"
    r")",
    re.IGNORECASE,
)

_SYSTEM_PROMPT = """\
Ты Жвуша в обычном личном Telegram-чате, но этот конкретный запрос требует
read-only анализа репозитория ZHVUSHA.

Работай как Codex Explorer: читай код, локальные логи, workspace-файлы и
контекст переписки; ничего не меняй, не создавай spec, не запускай реализацию,
не коммить и не выдавай план действий за уже выполненную работу.

Если Никита просит сравнить кодовую базу с постом, источником, скрином или
прошлым сообщением, используй блок "Недавний чат" как источник. Пути
`media/...` относительны workspace root; если нужно смотреть оригинал, открывай
его как read-only файл внутри workspace.

Отвечай по-русски, в женском голосе Жвуши. Отделяй проверенное по коду от
предположений. Если нужен интернет, которого нет в среде, честно скажи, что
снаружи не проверяла.

Для Telegram-progress можешь иногда писать отдельные строки строго с префиксом
TG_STATUS:, например: TG_STATUS: Сверяю self-coding поток с постом.
"""


class CodebaseExplorerSkill(InlineSkill):
    """Route ordinary personal chat codebase audits to read-only Codex Explorer."""

    name: ClassVar[str] = "codebase_explorer"
    description: ClassVar[str] = "Read-only анализ кодовой базы из обычного чата"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"

    triggers: ClassVar[list[str]] = []
    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "medium"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = "auto"
    side_effects: ClassVar[list[SideEffect]] = [
        SideEffect.READS_WORKSPACE,
        SideEffect.READS_FILESYSTEM,
        SideEffect.CALLS_LLM,
        SideEffect.SENDS_TELEGRAM_MESSAGE,
        SideEffect.DELEGATES_TO_CODE_AGENT,
    ]
    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    def __init__(
        self,
        *,
        admin_user_id: int,
        workspace_root: Path,
        explorer_runner: ExplorerRunner,
        background_runner: BackgroundExplorerRunner | None = None,
    ) -> None:
        self._admin_user_id = admin_user_id
        self._workspace_root = workspace_root
        self._explorer_runner = explorer_runner
        self._background_runner = background_runner

    async def can_handle(self, message: str, context: AgentContext) -> float:
        if context.mode != "personal" or context.user_id != self._admin_user_id:
            return 0.0
        if message.strip().startswith("/"):
            return 0.0
        if _looks_like_codex_decision_only_goal_handoff(message):
            return 0.0
        return 0.86 if _looks_like_codebase_explorer_request(message) else 0.0

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        recent_messages = ContextLoader(self._workspace_root).load_recent_messages(
            chat_id=context.chat_id,
            mode=context.mode,
            exclude_text=message,
        )
        user_prompt = _build_user_prompt(
            message=message,
            workspace_root=self._workspace_root,
            recent_messages=recent_messages,
        )
        if _looks_like_incoming_material_preamble(message):
            create_awaiting = getattr(
                self._background_runner,
                "create_awaiting_input",
                None,
            )
            if create_awaiting is not None:
                try:
                    await create_awaiting(
                        system_prompt=_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                    )
                except Exception:
                    logger.warning(
                        "codebase_explorer_awaiting_input_create_failed",
                        exc_info=True,
                    )
            return SkillResult(
                success=True,
                response=(
                    "Кидай материал. Я не запускаю agent-задачу на одном "
                    "«сейчас скину»; как пришлёшь пост, файл или скрин, "
                    "прикреплю его к задаче и начну сравнение."
                ),
            )
        if (
            self._background_runner is not None
            and context.chat_id is not None
            and (context.bot is not None or _prefers_background_agent_job(context))
        ):
            try:
                job = await self._background_runner.start_background(
                    system_prompt=_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    completion_callback=_make_completion_callback(context)
                    if context.bot is not None
                    else _noop_completion_callback,
                    progress_callback=_make_progress_callback(context),
                )
                job_id = str(getattr(job, "id", "") or "").strip()
                suffix = f" Job: `{job_id}`." if job_id else ""
                return SkillResult(
                    success=True,
                    response=(
                        "взяла в фоновую agent-задачу. Чат не держу; когда "
                        "закончу проверку, пришлю отдельный ответ."
                        if context.bot is not None
                        else (
                            "взяла в фоновую read-only agent-задачу. Чат не "
                            f"держу; результат останется в audit trail.{suffix}"
                        )
                    ),
                    metadata={
                        "skill_name": self.name,
                        "agent_job_result_pending": True,
                        "agent_job_status": "running",
                        **({"agent_job_id": job_id} if job_id else {}),
                    },
                )
            except Exception:
                logger.warning(
                    "codebase_explorer_background_start_failed",
                    exc_info=True,
                )

        await _send_status(context, "Смотрю проект, логи и контекст read-only.")
        try:
            response = await self._explorer_runner(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                progress_callback=_make_progress_callback(context),
            )
        except Exception:
            logger.warning("codebase_explorer_failed", exc_info=True)
            return SkillResult(
                success=True,
                response=(
                    "Я попыталась открыть read-only анализ проекта, но Explorer "
                    "сейчас не ответил. Важное: без него я не буду делать вид, "
                    "что проверила кодовую базу."
                ),
            )

        response = response.strip()
        if not response:
            logger.warning("codebase_explorer_empty_response")
            return SkillResult(
                success=True,
                response=(
                    "Explorer вернул пустой ответ, поэтому честно: кодовую базу "
                    "сейчас не разобрала."
                ),
            )
        return SkillResult(success=True, response=response)


def _looks_like_codebase_explorer_request(text: str) -> bool:
    lower = text.strip().lower()
    if not lower:
        return False
    has_action = any(marker in lower for marker in _ACTION_MARKERS)
    has_codebase = any(marker in lower for marker in _CODEBASE_MARKERS)
    if has_action and has_codebase:
        return True
    has_source_only = bool(_SOURCE_ONLY_RE.search(lower))
    has_source = any(marker in lower for marker in _SOURCE_MARKERS)
    return has_action and has_source and has_codebase and not has_source_only


def _looks_like_codex_decision_only_goal_handoff(text: str) -> bool:
    lower = text.strip().lower()
    return (
        "codex/operator handoff" in lower
        and "operator_handoff_mode: decision_only_existing_agent_evidence" in lower
    )


def _prefers_background_agent_job(context: AgentContext) -> bool:
    metadata = context.metadata
    if metadata.get("digital_scenario_id"):
        return True
    if metadata.get("digital_scenario_action_kind"):
        return True
    return str(metadata.get("interface", "") or "") == "vscode"


async def _noop_completion_callback(text: str) -> None:
    del text


def _looks_like_incoming_material_preamble(text: str) -> bool:
    return bool(_INCOMING_MATERIAL_RE.search(text.strip().lower()))


def _build_user_prompt(
    *,
    message: str,
    workspace_root: Path,
    recent_messages: str,
) -> str:
    recent = recent_messages.strip() or "(нет доступного недавнего контекста)"
    return (
        "Никита написал обычный чат-запрос, но он требует живого read-only "
        "анализа проекта.\n\n"
        f"workspace_root: {workspace_root}\n\n"
        "Недавний чат, который может содержать пост/источник/скрин для сравнения:\n"
        f"{recent}\n\n"
        "Текущий запрос Никиты:\n"
        f"{message}\n\n"
        "Что сделать:\n"
        "- изучи нужные части репозитория и workspace/logs;\n"
        "- если запрос ссылается на пост/источник выше, сравни реальные механизмы "
        "в коде с тезисами поста;\n"
        "- не придумывай проверку внешних сайтов, если не открывала их;\n"
        "- дай конкретный вывод: что уже есть, чего нет, что стоит улучшить первым."
    )


def _make_progress_callback(
    context: AgentContext,
) -> Callable[[str], Awaitable[None]] | None:
    if context.bot is None or context.chat_id is None:
        return None

    async def progress(status: str) -> None:
        await _send_status(context, status)

    return progress


def _make_completion_callback(
    context: AgentContext,
) -> Callable[[str], Awaitable[None]]:
    async def completion(text: str) -> None:
        await _send_result(context, text)

    return completion


async def _send_status(context: AgentContext, status: str) -> None:
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
        logger.warning("codebase_explorer_status_send_failed", exc_info=True)


async def _send_result(context: AgentContext, text: str) -> None:
    if context.bot is None or context.chat_id is None:
        return
    clean = text.strip() or "Read-only agent job завершилась без текста."
    try:
        await send_long_message(
            context.bot,
            context.chat_id,
            clean,
            parse_mode=None,
        )
    except Exception:
        logger.warning("codebase_explorer_result_send_failed", exc_info=True)

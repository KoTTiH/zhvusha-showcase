"""Delegate skill — routes complex tasks to Codex."""

from __future__ import annotations

import asyncio
import shutil
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import structlog

from src.core.config import get_settings
from src.skills.base import (
    AgentContext,
    DelegatedSkill,
    ExecutionPlan,
    SideEffect,
    SkillResult,
)
from src.skills.code_agent.codex_cli import CodexCliBackend
from src.skills.code_agent.protocols import (
    CodeAgentExecutionError,
    CodeAgentUnavailableError,
    DelegateRequest,
)

if TYPE_CHECKING:
    from aiogram import Bot

logger = structlog.get_logger()

_DELEGATE_PREFIX = "/delegate"
_TG_MAX_LEN = 4000


class DelegateSkill(DelegatedSkill):
    """Delegates complex tasks to Codex."""

    name: ClassVar[str] = "delegate"
    description: ClassVar[str] = "Delegate tasks to Codex"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "strategist"

    triggers: ClassVar[list[str]] = [_DELEGATE_PREFIX]

    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "high"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = (
        "required"
    )

    side_effects: ClassVar[list[SideEffect]] = [
        SideEffect.DELEGATES_TO_CODE_AGENT,
        SideEffect.CALLS_LLM,
        SideEffect.CALLS_LLM_TIER_STRATEGIST,
        SideEffect.NETWORK_IO_EXTERNAL,
        SideEffect.SENDS_TELEGRAM_MESSAGE,
        SideEffect.SPAWNS_SUBPROCESS,
    ]

    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    executor: ClassVar[str] = "codex_cli"
    max_duration_seconds: ClassVar[float] = 300

    async def can_handle(self, message: str, context: AgentContext) -> float:
        """Handle /delegate commands in personal mode for the admin only."""
        settings = get_settings()
        if not settings.delegate_enabled:
            return 0.0
        if context.user_id != settings.admin_user_id:
            return 0.0
        if message.strip().lower().startswith(_DELEGATE_PREFIX):
            return 1.0
        return 0.0

    async def prepare(self, message: str, context: AgentContext) -> ExecutionPlan:
        """Build a delegation plan without spawning the backend session."""
        del context
        settings = get_settings()
        task = message.strip().removeprefix(_DELEGATE_PREFIX).strip()
        return ExecutionPlan(
            skill_name=self.name,
            skill_type="delegated",
            human_summary=f"Делегировать Codex: {task[:120]}",
            estimated_tokens=20000,
            estimated_cost_usd=Decimal("0.50"),
            estimated_duration_seconds=float(settings.delegate_timeout_seconds),
            side_effects_invoked=list(self.side_effects),
            delegated_to=self.executor,
            llm_calls_planned=1,
        )

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        """Parse /delegate command and run a Codex session."""
        settings = get_settings()

        task = message.strip().removeprefix(_DELEGATE_PREFIX).strip()
        if not task:
            return SkillResult(
                success=False,
                response=(
                    "Укажи задачу после /delegate.\n"
                    "Пример: /delegate прочитай логи за вчера и найди ошибки"
                ),
            )

        cwd = Path(settings.delegate_cwd).expanduser()
        timeout = settings.delegate_timeout_seconds
        model = settings.delegate_model or settings.code_agent_model
        codex_path = settings.codex_cli_path
        chat_id = context.chat_id
        bot = context.bot

        status_message_id: int | None = None
        if bot is not None and chat_id is not None:
            msg = await bot.send_message(
                chat_id, f"Делегирую Codex:\n{task[:200]}\n\nВыполняю..."
            )
            status_message_id = msg.message_id

        try:
            result = await asyncio.wait_for(
                _run_delegate(
                    task=task,
                    cwd=cwd,
                    model=model,
                    codex_path=codex_path,
                    bot=bot,
                    chat_id=chat_id,
                    status_message_id=status_message_id,
                ),
                timeout=timeout,
            )
        except TimeoutError:
            logger.warning("delegate_timeout", task=task[:100], timeout=timeout)
            return SkillResult(
                success=False,
                response=f"Таймаут делегирования ({timeout}s). Задача прервана.",
            )
        except CodeAgentUnavailableError as exc:
            return SkillResult(
                success=False,
                response=f"Codex backend недоступен: {exc.reason}",
            )
        except CodeAgentExecutionError as exc:
            logger.warning(
                "delegate_backend_failed", task=task[:100], reason=exc.reason
            )
            return SkillResult(
                success=False,
                response=f"Codex не завершил задачу: {exc.reason}",
            )
        except Exception:
            logger.exception("delegate_error", task=task[:100])
            return SkillResult(
                success=False,
                response="Ошибка при делегировании. Проверь логи.",
            )

        if len(result) > _TG_MAX_LEN:
            result = result[: _TG_MAX_LEN - 20] + "\n... (обрезано)"

        return SkillResult(success=True, response=result)


def _backend_available(codex_path: str = "codex") -> bool:
    """Check whether the configured Codex binary is on PATH."""
    return shutil.which(codex_path) is not None


async def _run_delegate(
    *,
    task: str,
    cwd: Path,
    model: str = "",
    codex_path: str = "codex",
    bot: Bot | None = None,
    chat_id: int | None = None,
    status_message_id: int | None = None,
) -> str:
    """Execute a Codex session and stream the final result to Telegram."""
    backend = CodexCliBackend(codex_path=codex_path, model=model)
    result = await backend.run_delegate(
        DelegateRequest(
            task=task,
            cwd=cwd,
            model=model,
        )
    )
    final = result.text or "(Codex не вернул текстового ответа)"

    if bot is not None and chat_id is not None and status_message_id is not None:
        await _safe_edit(
            bot,
            chat_id,
            status_message_id,
            f"Codex:\n\n{_truncate(final, _TG_MAX_LEN - 30)}",
        )

    return final


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... (обрезано)"


async def _safe_edit(bot: Bot, chat_id: int, message_id: int, text: str) -> None:
    """Edit a Telegram message, swallowing errors."""
    try:
        await bot.edit_message_text(
            text=text,
            chat_id=chat_id,
            message_id=message_id,
        )
    except Exception:
        logger.debug("delegate_edit_failed", exc_info=True)

"""Daemon decision engine — decides what to do with each signal.

Separate from src/core/decision.py (which handles chat System 1/2).
This engine is for autonomous signal processing.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

import structlog
from pydantic import BaseModel

from src.core.config import Tier  # noqa: TC001 — used in runtime cast("Tier", ...)
from src.llm.protocols import LLMRequest

if TYPE_CHECKING:
    from src.daemon.signals import Signal
    from src.daemon.tools.registry import ToolRegistry
    from src.llm.router import LLMRouter

logger = structlog.get_logger()


class DaemonDecisionType(StrEnum):
    """Five possible decisions for a signal."""

    IGNORE = "ignore"
    ACT_SILENT = "act_silent"
    ACT_NOTIFY = "act_notify"
    ASK = "ask"
    QUEUE = "queue"


class ActionSpec(BaseModel):
    """Specifies which tool to call and with what params."""

    tool: str
    params: dict[str, Any] = {}


class DaemonDecision(BaseModel):
    """Structured decision output from LLM."""

    decision: DaemonDecisionType
    reasoning: str = ""
    action: ActionSpec | None = None
    requires_approval: bool = False


_DECISION_SYSTEM = """\
Ты — движок решений Жвуши. Ты получаешь сигнал из внешнего источника \
и решаешь что с ним делать.

Доступные инструменты:
{tools_description}

Ответь ТОЛЬКО валидным JSON:
{{
    "decision": "ignore" | "act_silent" | "act_notify" | "ask" | "queue",
    "reasoning": "почему такое решение",
    "action": {{"tool": "название", "params": {{}}}} | null,
    "requires_approval": false
}}

Решения:
- ignore: записать, не действовать
- act_silent: сделать молча (обновить базу, записать эпизод)
- act_notify: сделать и сообщить Никите
- ask: спросить Никиту перед действием
- queue: отложить до утренней сессии
"""


class DaemonDecisionEngine:
    """Makes decisions about signals using LLM."""

    def __init__(
        self,
        llm_router: LLMRouter,
        tool_registry: ToolRegistry,
        *,
        tier: str = "analyst",
    ) -> None:
        self._llm_router = llm_router
        self._tool_registry = tool_registry
        self._tier = tier

    async def decide(self, signal: Signal) -> DaemonDecision:
        """Decide what to do with a signal."""
        system = _DECISION_SYSTEM.format(
            tools_description=self._tool_registry.format_for_llm()
        )

        prompt = (
            f"Сигнал:\n"
            f"- Источник: {signal.source}\n"
            f"- Тип: {signal.signal_type}\n"
            f"- Приоритет: {signal.priority}\n"
            f"- Содержимое: {json.dumps(signal.payload, ensure_ascii=False)}\n"
            f"- Требует ответа: {signal.requires_response}\n"
        )

        try:
            llm_response = await self._llm_router.generate(
                LLMRequest(
                    prompt=prompt,
                    system=system,
                    tier=cast("Tier", self._tier),
                    caller="daemon_decision",
                )
            )
            return self._parse_decision(llm_response.text)
        except Exception:
            logger.exception("daemon_decision_failed", signal_id=signal.id)
            return DaemonDecision(
                decision=DaemonDecisionType.IGNORE,
                reasoning="LLM call failed, ignoring signal",
            )

    @staticmethod
    def _parse_decision(response: str) -> DaemonDecision:
        """Parse LLM JSON response into DaemonDecision."""
        # Strip markdown code fences if present
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        try:
            data = json.loads(text)
            return DaemonDecision.model_validate(data)
        except Exception:
            logger.warning("daemon_decision_parse_failed", response=text[:200])
            return DaemonDecision(
                decision=DaemonDecisionType.IGNORE,
                reasoning=f"Parse failed: {text[:100]}",
            )

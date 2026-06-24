"""Tests for DaemonDecisionEngine."""

from __future__ import annotations

from unittest.mock import AsyncMock

from src.daemon.decision import (
    DaemonDecisionEngine,
    DaemonDecisionType,
)
from src.daemon.signals import Signal
from src.daemon.tools.registry import ToolRegistry
from src.llm.protocols import LLMResponse, LLMUsage


def _llm_resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="haiku", usage=LLMUsage())


class TestDaemonDecisionEngine:
    async def test_parse_valid_json(self) -> None:
        result = DaemonDecisionEngine._parse_decision(
            '{"decision": "ignore", "reasoning": "not relevant"}'
        )
        assert result.decision == DaemonDecisionType.IGNORE
        assert result.reasoning == "not relevant"

    async def test_parse_with_markdown_fences(self) -> None:
        result = DaemonDecisionEngine._parse_decision(
            '```json\n{"decision": "act_notify", "reasoning": "urgent"}\n```'
        )
        assert result.decision == DaemonDecisionType.ACT_NOTIFY

    async def test_parse_with_action(self) -> None:
        result = DaemonDecisionEngine._parse_decision(
            '{"decision": "act_silent", "reasoning": "save", '
            '"action": {"tool": "knowledge_store", "params": {"title": "X"}}}'
        )
        assert result.decision == DaemonDecisionType.ACT_SILENT
        assert result.action is not None
        assert result.action.tool == "knowledge_store"

    async def test_parse_invalid_json(self) -> None:
        result = DaemonDecisionEngine._parse_decision("not json at all")
        assert result.decision == DaemonDecisionType.IGNORE
        assert "Parse failed" in result.reasoning

    async def test_decide_calls_llm(self) -> None:
        llm_router = AsyncMock()
        llm_router.generate = AsyncMock(
            return_value=_llm_resp(
                '{"decision": "ignore", "reasoning": "low priority"}'
            )
        )

        registry = ToolRegistry()
        engine = DaemonDecisionEngine(llm_router, registry)

        signal = Signal(source="test", signal_type="test", payload={"data": "value"})
        decision = await engine.decide(signal)

        assert decision.decision == DaemonDecisionType.IGNORE
        llm_router.generate.assert_awaited_once()

    async def test_decide_handles_llm_error(self) -> None:
        llm_router = AsyncMock()
        llm_router.generate = AsyncMock(side_effect=RuntimeError("LLM down"))

        registry = ToolRegistry()
        engine = DaemonDecisionEngine(llm_router, registry)

        signal = Signal(source="test", signal_type="test")
        decision = await engine.decide(signal)

        assert decision.decision == DaemonDecisionType.IGNORE
        assert "failed" in decision.reasoning

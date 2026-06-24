"""CLI calls used to bypass caller tracking: the router called
``record_cli_call()`` without a caller argument, so strategist-tier
invocations never appeared in ``caller_counts`` — masking where Opus
spend came from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from src.llm.protocols import LLMRequest, LLMResponse, LLMUsage
from src.llm.router import LLMRouter
from src.monitoring.usage_tracker import UsageTracker

if TYPE_CHECKING:
    from pathlib import Path


async def test_record_cli_call_accepts_caller(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_cli_call(caller="morning_synthesis")
    tracker.record_cli_call(caller="morning_synthesis")
    tracker.record_cli_call(caller="daemon_strategist")

    today = tracker.get_today()
    assert today.cli_calls == 3
    assert today.caller_counts == {
        "morning_synthesis": 2,
        "daemon_strategist": 1,
    }


async def test_record_cli_session_accepts_caller(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "monitoring")
    tracker.record_cli_session(caller="morning_review")

    today = tracker.get_today()
    assert today.cli_sessions == 1
    assert today.caller_counts == {"morning_review": 1}


async def test_router_passes_caller_to_cli_record(tmp_path: Path) -> None:
    tracker = UsageTracker(tmp_path / "monitoring")

    cli_adapter = MagicMock()
    cli_adapter.name = "claude_cli"
    cli_adapter.default_model = "opus"

    async def fake_generate(req: LLMRequest) -> LLMResponse:
        return LLMResponse(text="ok", model="opus", usage=LLMUsage())

    cli_adapter.generate = fake_generate

    router = LLMRouter(
        adapters={"strategist": cli_adapter},  # type: ignore[arg-type]
        models={"strategist": "opus"},  # type: ignore[arg-type]
        providers_by_tier={"strategist": "claude_cli"},  # type: ignore[arg-type]
        usage_tracker=tracker,
    )

    with patch("src.llm.router.get_settings"):
        await router.generate(
            LLMRequest(prompt="hi", tier="strategist", caller="daemon_plan")
        )

    today = tracker.get_today()
    assert today.cli_calls == 1
    assert today.caller_counts.get("daemon_plan") == 1

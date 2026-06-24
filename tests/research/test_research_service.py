"""Contract tests for ResearchService — leaf module for spec-time research.

The service is injected with two callables (KB search, code search) so the
module stays a true leaf — no ``src.knowledge``/``src.memory``/``src.llm``
imports. Tests use plain dummy callables instead of mocks of capability
modules.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.mark.contract
class TestResearchServiceLeafBehavior:
    """ResearchService runs KB + code search per preset and aggregates."""

    async def test_service_calls_kb_and_code_search_for_api_integration(self) -> None:
        from src.research.protocols import Citation, ResearchResult
        from src.research.service import ResearchService

        kb_calls: list[str] = []
        code_calls: list[str] = []

        async def fake_kb_search(query: str) -> list[Citation]:
            kb_calls.append(query)
            return [Citation(source="kb", ref="kb_71", excerpt="spec-first workflow")]

        async def fake_code_search(query: str) -> list[Citation]:
            code_calls.append(query)
            return [
                Citation(
                    source="code", ref="src/skills/base.py:155", excerpt="BaseSkill"
                )
            ]

        service = ResearchService(
            kb_search=fake_kb_search,
            code_search=fake_code_search,
        )

        result = await service.research(
            query="add weather skill",
            preset="api_integration",
            budget_seconds=10.0,
        )

        assert isinstance(result, ResearchResult)
        assert kb_calls == ["add weather skill"]
        assert code_calls == ["add weather skill"]
        assert len(result.citations) == 2
        assert any(c.source == "kb" for c in result.citations)
        assert any(c.source == "code" for c in result.citations)
        assert result.elapsed_seconds >= 0.0
        assert result.truncated is False

    async def test_service_includes_extra_runtime_sources(self) -> None:
        from src.research.protocols import Citation
        from src.research.service import ResearchService

        runtime_calls: list[str] = []

        async def fake_kb_search(query: str) -> list[Citation]:
            del query
            return []

        async def fake_code_search(query: str) -> list[Citation]:
            del query
            return []

        async def runtime_source(query: str) -> list[Citation]:
            runtime_calls.append(query)
            return [
                Citation(
                    source="telegram_mcp",
                    ref="telegram://personal/dialog/42",
                    excerpt="read-only Telegram context",
                )
            ]

        service = ResearchService(
            kb_search=fake_kb_search,
            code_search=fake_code_search,
            runtime_sources=(runtime_source,),
        )

        result = await service.research(
            query="telegram context for self-coding",
            preset="api_integration",
            budget_seconds=10.0,
        )

        assert runtime_calls == ["telegram context for self-coding"]
        assert result.citations[-1].source == "telegram_mcp"
        assert "telegram_mcp" in result.sources_used
        assert result.truncated is False

    async def test_service_skips_code_search_for_foundational_preset(self) -> None:
        """Foundational preset is KB-only by spec (KB #73)."""
        from src.research.protocols import (
            Citation,  # noqa: TC002 — runtime use in callable
        )
        from src.research.service import ResearchService

        kb_calls: list[str] = []
        code_calls: list[str] = []

        async def fake_kb_search(query: str) -> list[Citation]:
            kb_calls.append(query)
            return []

        async def fake_code_search(query: str) -> list[Citation]:
            code_calls.append(query)
            return []

        service = ResearchService(
            kb_search=fake_kb_search, code_search=fake_code_search
        )

        await service.research(query="any", preset="foundational", budget_seconds=10.0)

        assert kb_calls == ["any"]
        assert code_calls == [], "foundational must not invoke code search"

    async def test_truncated_when_budget_exceeded(self) -> None:
        """If callable raises TimeoutError, ResearchResult.truncated is True."""
        from src.research.protocols import (
            Citation,
        )
        from src.research.service import ResearchService

        async def slow_kb(query: str) -> list[Citation]:
            raise TimeoutError("kb took too long")

        async def fake_code(query: str) -> list[Citation]:
            return [Citation(source="code", ref="x", excerpt="y")]

        service = ResearchService(kb_search=slow_kb, code_search=fake_code)

        result = await service.research(
            query="anything", preset="api_integration", budget_seconds=0.1
        )

        assert result.truncated is True
        assert all(c.source != "kb" for c in result.citations)

    async def test_budget_timeout_cancels_slow_source_and_returns_partial_result(
        self,
    ) -> None:
        """A hung source must not block spec drafting past the research budget."""
        from src.research.protocols import Citation
        from src.research.service import ResearchService

        slow_cancelled = asyncio.Event()

        async def fake_kb(query: str) -> list[Citation]:
            del query
            return [Citation(source="kb", ref="kb:ok", excerpt="fast context")]

        async def slow_code(query: str) -> list[Citation]:
            del query
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                slow_cancelled.set()
                raise
            return [Citation(source="code", ref="late", excerpt="too late")]

        service = ResearchService(kb_search=fake_kb, code_search=slow_code)

        result = await service.research(
            query="live browser verification",
            preset="api_integration",
            budget_seconds=0.01,
        )

        assert result.truncated is True
        assert slow_cancelled.is_set()
        assert [citation.source for citation in result.citations] == ["kb"]

    async def test_unknown_preset_raises_value_error(self) -> None:
        from src.research.protocols import (
            Citation,  # noqa: TC002 — runtime use in callable
        )
        from src.research.service import ResearchService

        async def fake_kb(query: str) -> list[Citation]:
            return []

        async def fake_code(query: str) -> list[Citation]:
            return []

        service = ResearchService(kb_search=fake_kb, code_search=fake_code)
        with pytest.raises(ValueError, match="preset"):
            await service.research(
                query="x", preset="unknown_preset", budget_seconds=1.0
            )


@pytest.mark.contract
class TestResearchPresets:
    """Each preset declares its source mix and trust thresholds."""

    def test_baseline_presets_present(self) -> None:
        from src.research.presets import PRESETS

        names = set(PRESETS.keys())
        assert {
            "foundational",
            "current_practices",
            "api_integration",
            "hot_topic",
        }.issubset(names)

    def test_each_preset_declares_kb_required(self) -> None:
        from src.research.presets import PRESETS

        for name, preset in PRESETS.items():
            assert preset.use_kb, f"{name} must include KB search per KB #72"

    def test_foundational_excludes_code_search(self) -> None:
        from src.research.presets import PRESETS

        assert PRESETS["foundational"].use_code_search is False

    def test_api_integration_includes_code_search(self) -> None:
        from src.research.presets import PRESETS

        assert PRESETS["api_integration"].use_code_search is True

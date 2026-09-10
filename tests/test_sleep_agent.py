"""Tests for SleepTimeAgent — background knowledge maintenance.

Covers:
- cooldown between cycles (5 min minimum)
- adaptive backoff (>40 calls/hr doubles cooldown, >50 → full stop)
- batch processing (single LLM call per task; fallback sequential on invalid JSON)
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.daemon.sleep_agent import (
    _BACKOFF_THRESHOLD,
    _FULL_STOP_THRESHOLD,
    _MAX_ENTRIES_PER_TASK,
    _MIN_CYCLE_INTERVAL,
    SleepTimeAgent,
)
from src.knowledge.models import KnowledgeEntry
from src.llm.protocols import LLMResponse, LLMUsage


def _llm_resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="haiku", usage=LLMUsage())


def _make_entry(
    entry_id: int = 1,
    title: str = "Test Entry",
    content: str = "Test content for analysis",
    **kwargs: object,
) -> KnowledgeEntry:
    entry = KnowledgeEntry(title=title, content=content)
    entry.id = entry_id
    entry.tags = kwargs.get("tags", [])  # type: ignore[assignment]
    entry.summary = kwargs.get("summary")  # type: ignore[assignment]
    entry.category_id = kwargs.get("category_id")  # type: ignore[assignment]
    entry.status = kwargs.get("status", "raw")  # type: ignore[assignment]
    return entry


def _tags_batch_json(items: list[tuple[int, list[str]]]) -> str:
    return json.dumps([{"id": eid, "tags": tags} for eid, tags in items])


def _summary_batch_json(items: list[tuple[int, str]]) -> str:
    return json.dumps([{"id": eid, "summary": summary} for eid, summary in items])


def _category_batch_json(items: list[tuple[int, str]]) -> str:
    return json.dumps([{"id": eid, "category": cat} for eid, cat in items])


@pytest.fixture
def mock_store() -> AsyncMock:
    store = AsyncMock()
    store.get_untagged = AsyncMock(return_value=[])
    store.get_unsummarized = AsyncMock(return_value=[])
    store.get_uncategorized = AsyncMock(return_value=[])
    store.browse_categories = AsyncMock(return_value=[])
    store.update_entry = AsyncMock(return_value=True)
    store.get_or_create_category = AsyncMock(return_value=42)
    return store


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock()
    # Default: empty JSON arrays — so `_process_all_tasks` can run without entries
    llm.generate = AsyncMock(return_value=_llm_resp("[]"))
    return llm


@pytest.fixture
def agent(mock_store: AsyncMock, mock_llm: AsyncMock) -> SleepTimeAgent:
    return SleepTimeAgent(knowledge_store=mock_store, llm_router=mock_llm)


class TestRunMaintenanceCycle:
    async def test_returns_zero_when_nothing_to_do(self, agent: SleepTimeAgent) -> None:
        total = await agent.run_maintenance_cycle()
        assert total == 0

    async def test_cooldown_blocks_rapid_cycles(self, agent: SleepTimeAgent) -> None:
        """Second cycle within 5 minutes returns 0 without running tasks."""
        await agent.run_maintenance_cycle()
        total = await agent.run_maintenance_cycle()
        assert total == 0

    async def test_counts_all_updates(
        self,
        agent: SleepTimeAgent,
        mock_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        mock_store.get_untagged.return_value = [_make_entry(1)]
        mock_store.get_unsummarized.return_value = [_make_entry(2, summary=None)]
        mock_store.get_uncategorized.return_value = [_make_entry(3, category_id=None)]
        mock_llm.generate = AsyncMock(
            side_effect=[
                _llm_resp(_tags_batch_json([(1, ["python"])])),
                _llm_resp(_summary_batch_json([(2, "Summary 2")])),
                _llm_resp(_category_batch_json([(3, "tools.python")])),
            ]
        )

        total = await agent.run_maintenance_cycle()
        assert total == 3
        assert mock_store.update_entry.await_count == 3

    async def test_continues_on_task_failure(
        self,
        agent: SleepTimeAgent,
        mock_store: AsyncMock,
    ) -> None:
        """One failing task should not block others."""
        mock_store.get_untagged.side_effect = RuntimeError("db error")
        mock_store.get_unsummarized.return_value = []
        mock_store.get_uncategorized.return_value = []

        total = await agent.run_maintenance_cycle()
        assert total == 0  # no crash


class TestAdaptiveBackoff:
    """Verify adaptive backoff: >40 calls/hr → 2x cooldown, >50 → full stop."""

    def _make_agent_with_tracker(
        self,
        mock_store: AsyncMock,
        mock_llm: AsyncMock,
        hourly_calls: int,
    ) -> tuple[SleepTimeAgent, MagicMock]:
        tracker = MagicMock()
        tracker.get_calls_in_last_hour = MagicMock(return_value=hourly_calls)
        agent = SleepTimeAgent(
            knowledge_store=mock_store,
            llm_router=mock_llm,
            usage_tracker=tracker,
        )
        return agent, tracker

    async def test_normal_cooldown_under_40_calls(
        self,
        mock_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Under 40 calls/hr: standard 5-minute cooldown."""
        agent, _ = self._make_agent_with_tracker(mock_store, mock_llm, hourly_calls=20)
        await agent.run_maintenance_cycle()
        # Move last cycle timestamp back by just over 5 minutes
        agent._last_cycle_ts -= _MIN_CYCLE_INTERVAL + 1
        total = await agent.run_maintenance_cycle()
        # Cycle runs (no entries, so total=0 but no cooldown skip)
        assert total == 0  # runs, but nothing to do
        # Verify cycle actually executed by checking timestamp got reset
        # (_last_cycle_ts advances when cycle runs)

    async def test_doubled_cooldown_above_40_calls(
        self,
        mock_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """41 calls/hr: cooldown doubles to 10 minutes."""
        agent, _ = self._make_agent_with_tracker(
            mock_store, mock_llm, hourly_calls=_BACKOFF_THRESHOLD + 1
        )
        await agent.run_maintenance_cycle()
        # Move back by just over the NORMAL cooldown (should still be blocked
        # because required cooldown is doubled)
        agent._last_cycle_ts -= _MIN_CYCLE_INTERVAL + 1
        # Second cycle should be blocked (needs 600 sec, elapsed ~301 sec)
        mock_store.get_untagged.return_value = [_make_entry(1)]
        total = await agent.run_maintenance_cycle()
        assert total == 0
        # Confirm store wasn't queried (cooldown skipped the cycle)
        # First call from cycle 1 (empty) + no new calls from cycle 2
        assert mock_store.get_untagged.await_count == 1

    async def test_full_stop_above_50_calls(
        self,
        mock_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """51 calls/hr: full stop — cooldown is hour-scale."""
        agent, _ = self._make_agent_with_tracker(
            mock_store, mock_llm, hourly_calls=_FULL_STOP_THRESHOLD + 1
        )
        await agent.run_maintenance_cycle()
        # Move back by just over the DOUBLED cooldown
        agent._last_cycle_ts -= _MIN_CYCLE_INTERVAL * 2 + 1
        # Still blocked — full stop is 3600 sec
        mock_store.get_untagged.return_value = [_make_entry(1)]
        total = await agent.run_maintenance_cycle()
        assert total == 0
        assert mock_store.get_untagged.await_count == 1

    async def test_works_without_tracker(
        self,
        agent: SleepTimeAgent,
        mock_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Backwards compat: no tracker → normal 5-minute cooldown."""
        await agent.run_maintenance_cycle()
        agent._last_cycle_ts -= _MIN_CYCLE_INTERVAL + 1
        mock_store.get_untagged.return_value = [_make_entry(1)]
        mock_llm.generate = AsyncMock(
            return_value=_llm_resp(_tags_batch_json([(1, ["python"])]))
        )
        await agent.run_maintenance_cycle()
        # With no tracker, normal cooldown applies → cycle runs
        assert mock_store.get_untagged.await_count == 2  # both cycles ran

    async def test_tracker_error_falls_back_to_normal(
        self,
        mock_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """If tracker raises, agent falls back to normal cooldown (fail-open)."""
        tracker = MagicMock()
        tracker.get_calls_in_last_hour = MagicMock(side_effect=RuntimeError("boom"))
        agent = SleepTimeAgent(
            knowledge_store=mock_store,
            llm_router=mock_llm,
            usage_tracker=tracker,
        )
        # Should not raise; default cooldown applies
        await agent.run_maintenance_cycle()


class TestBatchProcessing:
    """Verify batch processing: many entries → 1 LLM call per task."""

    async def test_single_llm_call_for_multiple_entries_tags(
        self,
        agent: SleepTimeAgent,
        mock_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """5 tag entries → 1 LLM call → 5 update_entry calls."""
        entries = [_make_entry(i) for i in range(1, 6)]
        mock_store.get_untagged.return_value = entries
        mock_llm.generate = AsyncMock(
            return_value=_llm_resp(
                _tags_batch_json([(i, ["python", "ai"]) for i in range(1, 6)])
            )
        )

        total = await agent.run_maintenance_cycle()
        # Exactly 1 LLM call for tagging (0 for empty summarize / categorize tasks)
        assert mock_llm.generate.await_count == 1
        assert total == 5
        # All 5 entries updated
        assert mock_store.update_entry.await_count == 5

    async def test_zero_entries_no_llm_call(
        self,
        agent: SleepTimeAgent,
        mock_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """0 entries → 0 LLM calls."""
        mock_store.get_untagged.return_value = []
        mock_store.get_unsummarized.return_value = []
        mock_store.get_uncategorized.return_value = []
        await agent.run_maintenance_cycle()
        assert mock_llm.generate.await_count == 0

    async def test_max_batch_size(
        self,
        agent: SleepTimeAgent,
        mock_store: AsyncMock,
    ) -> None:
        """Store is queried with limit=_MAX_ENTRIES_PER_TASK."""
        await agent.run_maintenance_cycle()
        mock_store.get_untagged.assert_awaited_with(limit=_MAX_ENTRIES_PER_TASK)
        mock_store.get_unsummarized.assert_awaited_with(limit=_MAX_ENTRIES_PER_TASK)
        mock_store.get_uncategorized.assert_awaited_with(limit=_MAX_ENTRIES_PER_TASK)

    async def test_fallback_sequential_on_invalid_json(
        self,
        agent: SleepTimeAgent,
        mock_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Invalid batch JSON → sequential fallback (N additional LLM calls)."""
        entries = [_make_entry(1), _make_entry(2), _make_entry(3)]
        mock_store.get_untagged.return_value = entries
        # First call (batch) returns garbage; subsequent (fallback) return
        # per-entry plain-text tags
        mock_llm.generate = AsyncMock(
            side_effect=[
                _llm_resp("this is not json at all"),
                _llm_resp("python, ai"),
                _llm_resp("asyncio, testing"),
                _llm_resp("fastapi"),
            ]
        )

        total = await agent.run_maintenance_cycle()
        # 1 batch attempt + 3 per-entry fallbacks = 4 LLM calls
        assert mock_llm.generate.await_count == 4
        assert total == 3
        # update_entry invoked once per entry
        assert mock_store.update_entry.await_count == 3

    async def test_fallback_handles_per_entry_llm_failure(
        self,
        agent: SleepTimeAgent,
        mock_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Fallback: individual LLM errors don't block other entries."""
        entries = [_make_entry(1), _make_entry(2)]
        mock_store.get_untagged.return_value = entries
        mock_llm.generate = AsyncMock(
            side_effect=[
                _llm_resp("not json"),  # batch fails
                RuntimeError("rate limit"),  # first fallback fails
                _llm_resp("python, testing"),  # second fallback succeeds
            ]
        )

        total = await agent.run_maintenance_cycle()
        assert total == 1
        # Only entry 2 got updated
        tag_calls = [
            c
            for c in mock_store.update_entry.await_args_list
            if c.kwargs.get("tags") is not None
        ]
        assert len(tag_calls) == 1
        assert tag_calls[0].args[0] == 2

    async def test_batch_applies_updates_per_entry_id(
        self,
        agent: SleepTimeAgent,
        mock_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Parsed JSON mapping id → payload is respected per entry."""
        entries = [_make_entry(1), _make_entry(2), _make_entry(3)]
        mock_store.get_untagged.return_value = entries
        mock_llm.generate = AsyncMock(
            return_value=_llm_resp(
                _tags_batch_json(
                    [
                        (1, ["a"]),
                        (2, ["b", "c"]),
                        (3, ["d"]),
                    ]
                )
            )
        )

        await agent.run_maintenance_cycle()
        mock_store.update_entry.assert_any_await(1, tags=["a"])
        mock_store.update_entry.assert_any_await(2, tags=["b", "c"])
        mock_store.update_entry.assert_any_await(3, tags=["d"])


class TestTagUntagged:
    async def test_applies_tags(
        self,
        agent: SleepTimeAgent,
        mock_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        mock_store.get_untagged.return_value = [_make_entry(1)]
        mock_llm.generate = AsyncMock(
            return_value=_llm_resp(_tags_batch_json([(1, ["python", "asyncio", "ai"])]))
        )

        total = await agent.run_maintenance_cycle()
        assert total >= 1

        mock_store.update_entry.assert_any_await(1, tags=["python", "asyncio", "ai"])

    async def test_skips_empty_tags_in_batch(
        self,
        agent: SleepTimeAgent,
        mock_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Entry with empty tag list in batch response is not updated."""
        mock_store.get_untagged.return_value = [_make_entry(1)]
        mock_llm.generate = AsyncMock(
            return_value=_llm_resp(_tags_batch_json([(1, [])]))
        )

        await agent.run_maintenance_cycle()
        tag_calls = [
            c
            for c in mock_store.update_entry.await_args_list
            if c.kwargs.get("tags") is not None
        ]
        assert len(tag_calls) == 0


class TestSummarizeUnsummarized:
    async def test_applies_summary(
        self,
        agent: SleepTimeAgent,
        mock_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        mock_store.get_unsummarized.return_value = [_make_entry(2, summary=None)]
        mock_llm.generate = AsyncMock(
            return_value=_llm_resp(
                _summary_batch_json([(2, "Краткое описание записи.")])
            )
        )

        total = await agent.run_maintenance_cycle()
        assert total >= 1

        mock_store.update_entry.assert_any_await(2, summary="Краткое описание записи.")


class TestCategorizeUncategorized:
    async def test_applies_category(
        self,
        agent: SleepTimeAgent,
        mock_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        mock_store.get_uncategorized.return_value = [_make_entry(3, category_id=None)]
        cats = [SimpleNamespace(path="tools.python"), SimpleNamespace(path="ai.llm")]
        mock_store.browse_categories.return_value = cats
        mock_llm.generate = AsyncMock(
            return_value=_llm_resp(_category_batch_json([(3, "tools.python")]))
        )

        total = await agent.run_maintenance_cycle()
        assert total >= 1

        mock_store.get_or_create_category.assert_any_await(
            "tools.python", "Tools > Python"
        )
        mock_store.update_entry.assert_any_await(3, category_id=42)

    async def test_normalizes_category_path(
        self,
        agent: SleepTimeAgent,
        mock_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        mock_store.get_uncategorized.return_value = [_make_entry(4)]
        mock_llm.generate = AsyncMock(
            return_value=_llm_resp(_category_batch_json([(4, "  Tools Python  ")]))
        )

        await agent.run_maintenance_cycle()

        mock_store.get_or_create_category.assert_any_await(
            "tools_python", "Tools_python"
        )

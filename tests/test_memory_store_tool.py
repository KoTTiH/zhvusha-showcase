"""Tests for daemon/tools/memory_store.py — MemoryStoreTool."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from src.daemon.tools.memory_store import MemoryStoreTool


@pytest.mark.asyncio
async def test_store_success() -> None:
    episodic = AsyncMock()
    episodic.record = AsyncMock(return_value=42)
    tool = MemoryStoreTool(episodic)

    result = await tool.execute({"content": "test episode", "source": "test"})
    assert result.success is True
    assert "42" in result.message
    episodic.record.assert_awaited_once()


@pytest.mark.asyncio
async def test_store_empty_content() -> None:
    episodic = AsyncMock()
    tool = MemoryStoreTool(episodic)

    result = await tool.execute({"content": ""})
    assert result.success is False
    assert "empty" in result.message.lower()


@pytest.mark.asyncio
async def test_store_default_params() -> None:
    episodic = AsyncMock()
    episodic.record = AsyncMock(return_value=1)
    tool = MemoryStoreTool(episodic)

    await tool.execute({"content": "data"})
    call_kwargs = episodic.record.call_args.kwargs
    assert call_kwargs["source"] == "daemon"
    assert call_kwargs["importance"] == 0.5


@pytest.mark.asyncio
async def test_store_db_error() -> None:
    episodic = AsyncMock()
    episodic.record = AsyncMock(side_effect=RuntimeError("db down"))
    tool = MemoryStoreTool(episodic)

    result = await tool.execute({"content": "data"})
    assert result.success is False
    assert "error" in result.message.lower()

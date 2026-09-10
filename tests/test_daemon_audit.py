"""Tests for daemon/audit.py — AuditLog."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.daemon.audit import AuditLog, AuditLogEntry


def _make_audit() -> tuple[AuditLog, AsyncMock]:
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    maker: Any = MagicMock(return_value=session)
    maker._session = session
    return AuditLog(maker), session


@pytest.mark.asyncio
async def test_record_creates_entry() -> None:
    audit, session = _make_audit()

    # Simulate flush setting the ID
    def _set_id() -> None:
        entry = session.add.call_args[0][0]
        entry.id = 1

    session.flush.side_effect = _set_id

    entry_id = await audit.record(
        signal_id="sig-1",
        decision="act_silent",
        reasoning="test reason",
        tool_name="send_telegram",
        tool_params={"text": "hello"},
        result="success",
        llm_tokens_used=100,
        llm_cost_usd=0.001,
    )

    assert entry_id == 1
    session.add.assert_called_once()
    session.commit.assert_awaited_once()
    added_entry = session.add.call_args[0][0]
    assert isinstance(added_entry, AuditLogEntry)
    assert added_entry.decision == "act_silent"
    assert added_entry.tool_name == "send_telegram"


@pytest.mark.asyncio
async def test_get_recent() -> None:
    audit, session = _make_audit()
    mock_entries = [MagicMock(spec=AuditLogEntry), MagicMock(spec=AuditLogEntry)]
    session.execute = AsyncMock(
        return_value=MagicMock(
            scalars=MagicMock(
                return_value=MagicMock(all=MagicMock(return_value=mock_entries))
            )
        )
    )

    result = await audit.get_recent(limit=10)
    assert len(result) == 2
    session.execute.assert_awaited_once()

"""Tests for dream proposal cooldown in ChatResponseSkill.

Covers:
- Cooldown prevents dream proposals within 1 hour
- After cooldown expires, dreams can be proposed again
- Cooldown timestamp updates on proposal
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from src.skills.chat_response.dream_extractor import DreamResult
from src.skills.chat_response.skill import (
    _DREAM_COOLDOWN_SECONDS,
    ChatResponseSkill,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def skill() -> ChatResponseSkill:
    return ChatResponseSkill()


@pytest.mark.asyncio
async def test_dream_check_skipped_during_cooldown(skill: ChatResponseSkill) -> None:
    """Dream check is skipped when within cooldown period."""
    # Simulate a recent dream proposal
    skill._last_dream_proposal_ts = time.monotonic()

    mock_bot = AsyncMock()
    mock_extractor = AsyncMock()
    # Should never be called
    mock_extractor.check = AsyncMock(
        return_value=DreamResult(has_dream=True, dream_text="test", confidence=0.9)
    )

    with patch(
        "src.skills.chat_response.skill.get_dream_extractor",
        return_value=mock_extractor,
    ):
        await skill._background_dream_check(
            bot_response="test",
            recent_context="",
            bot=mock_bot,
            chat_id=123,
        )

    # Extractor should not have been called due to cooldown
    mock_extractor.check.assert_not_awaited()
    mock_bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_dream_check_allowed_after_cooldown(
    skill: ChatResponseSkill, tmp_path: Path
) -> None:
    """Dream check proceeds when cooldown has expired."""
    # Set cooldown to far in the past
    skill._last_dream_proposal_ts = time.monotonic() - _DREAM_COOLDOWN_SECONDS - 1

    mock_bot = AsyncMock()
    mock_extractor = AsyncMock()
    mock_extractor.check = AsyncMock(
        return_value=DreamResult(has_dream=True, dream_text="new dream", confidence=0.8)
    )

    with (
        patch(
            "src.skills.chat_response.skill.get_dream_extractor",
            return_value=mock_extractor,
        ),
        patch(
            "src.skills.chat_response.skill.get_settings",
        ) as mock_settings,
    ):
        mock_settings.return_value.workspace_path = str(tmp_path / "test-ws")
        await skill._background_dream_check(
            bot_response="I want to learn painting",
            recent_context="",
            bot=mock_bot,
            chat_id=123,
        )

    mock_extractor.check.assert_awaited_once()
    mock_bot.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_dream_proposal_updates_cooldown_ts(
    skill: ChatResponseSkill, tmp_path: Path
) -> None:
    """Successful dream proposal updates the cooldown timestamp."""
    skill._last_dream_proposal_ts = time.monotonic() - _DREAM_COOLDOWN_SECONDS - 1

    mock_bot = AsyncMock()
    mock_extractor = AsyncMock()
    mock_extractor.check = AsyncMock(
        return_value=DreamResult(has_dream=True, dream_text="dream", confidence=0.8)
    )

    before = time.monotonic()
    with (
        patch(
            "src.skills.chat_response.skill.get_dream_extractor",
            return_value=mock_extractor,
        ),
        patch(
            "src.skills.chat_response.skill.get_settings",
        ) as mock_settings,
    ):
        mock_settings.return_value.workspace_path = str(tmp_path / "test-ws")
        await skill._background_dream_check(
            bot_response="test",
            recent_context="",
            bot=mock_bot,
            chat_id=123,
        )
    after = time.monotonic()

    assert skill._last_dream_proposal_ts >= before
    assert skill._last_dream_proposal_ts <= after


def test_cooldown_constant_is_one_hour() -> None:
    """Cooldown should be 1 hour (3600 seconds)."""
    assert _DREAM_COOLDOWN_SECONDS == 3600

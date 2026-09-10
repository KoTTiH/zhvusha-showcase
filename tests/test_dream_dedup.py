"""Tests for dream extractor dedup: existing dreams context in prompt.

Covers:
- Dream extractor receives existing dreams in prompt
- Dream extractor doesn't include tag when no existing dreams
- System prompt instructs LLM to skip duplicates
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from src.llm.protocols import LLMResponse, LLMUsage
from src.skills.chat_response.dream_extractor import DreamExtractor


def _llm_resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="haiku", usage=LLMUsage())


@pytest.fixture
def extractor() -> DreamExtractor:
    return DreamExtractor(tier="worker")


@pytest.mark.asyncio
async def test_existing_dreams_passed_to_prompt(extractor: DreamExtractor) -> None:
    """When existing_dreams is provided, it appears in the LLM prompt."""
    raw_json = '{"has_dream": false, "dream_text": "", "confidence": 0.9}'
    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(return_value=_llm_resp(raw_json))

    with patch(
        "src.skills.chat_response.dream_extractor.get_router",
        return_value=mock_router,
    ):
        await extractor.check(
            "test response",
            existing_dreams="- [2026-04-08] Научиться проверять факты",
        )

    request = mock_router.generate.call_args.args[0]
    prompt = request.prompt
    assert "Научиться проверять факты" in prompt
    assert "<EXISTING_DREAMS>" in prompt


@pytest.mark.asyncio
async def test_no_existing_dreams_tag_when_empty(extractor: DreamExtractor) -> None:
    """When existing_dreams is empty, no EXISTING_DREAMS tag in prompt."""
    raw_json = '{"has_dream": false, "dream_text": "", "confidence": 0.9}'
    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(return_value=_llm_resp(raw_json))

    with patch(
        "src.skills.chat_response.dream_extractor.get_router",
        return_value=mock_router,
    ):
        await extractor.check("test response", existing_dreams="")

    request = mock_router.generate.call_args.args[0]
    assert "<EXISTING_DREAMS>" not in request.prompt


@pytest.mark.asyncio
async def test_system_prompt_mentions_dedup_rule() -> None:
    """System prompt tells the LLM to not propose duplicates."""
    from src.skills.chat_response.dream_extractor import _DREAM_SYSTEM_PROMPT

    assert "EXISTING_DREAMS" in _DREAM_SYSTEM_PROMPT

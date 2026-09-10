"""Tests for DreamExtractor."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from src.llm.protocols import LLMResponse, LLMUsage
from src.skills.chat_response.dream_extractor import DreamExtractor, DreamResult


def _llm_resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="haiku", usage=LLMUsage())


@pytest.fixture
def extractor() -> DreamExtractor:
    return DreamExtractor(tier="worker")


@pytest.mark.asyncio
async def test_dream_found(extractor: DreamExtractor) -> None:
    """LLM detects a dream in the response."""
    raw_json = (
        '{"has_dream": true, "dream_text": "научиться рисовать", "confidence": 0.8}'
    )
    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(return_value=_llm_resp(raw_json))

    with patch(
        "src.skills.chat_response.dream_extractor.get_router", return_value=mock_router
    ):
        result = await extractor.check("Хочу научиться рисовать!")
    assert result is not None
    assert result.has_dream is True
    assert result.dream_text == "научиться рисовать"
    assert result.confidence == 0.8
    request = mock_router.generate.call_args.args[0]
    assert request.tier == "worker"


@pytest.mark.asyncio
async def test_no_dream(extractor: DreamExtractor) -> None:
    """LLM finds no dream in ordinary response."""
    raw_json = '{"has_dream": false, "dream_text": "", "confidence": 0.9}'
    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(return_value=_llm_resp(raw_json))

    with patch(
        "src.skills.chat_response.dream_extractor.get_router", return_value=mock_router
    ):
        result = await extractor.check("Привет, как дела?")
    assert result is not None
    assert result.has_dream is False


@pytest.mark.asyncio
async def test_llm_error_returns_none(extractor: DreamExtractor) -> None:
    """LLM failure returns None, never raises."""
    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(side_effect=RuntimeError("API down"))

    with patch(
        "src.skills.chat_response.dream_extractor.get_router", return_value=mock_router
    ):
        result = await extractor.check("test")
    assert result is None


@pytest.mark.asyncio
async def test_invalid_json_returns_none(extractor: DreamExtractor) -> None:
    """Invalid JSON from LLM returns None."""
    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(return_value=_llm_resp("not json at all"))

    with patch(
        "src.skills.chat_response.dream_extractor.get_router", return_value=mock_router
    ):
        result = await extractor.check("test")
    assert result is None


@pytest.mark.asyncio
async def test_low_confidence_still_returned(extractor: DreamExtractor) -> None:
    """Low confidence dream is returned — caller decides threshold."""
    raw_json = '{"has_dream": true, "dream_text": "что-то", "confidence": 0.3}'
    mock_router = AsyncMock()
    mock_router.generate = AsyncMock(return_value=_llm_resp(raw_json))

    with patch(
        "src.skills.chat_response.dream_extractor.get_router", return_value=mock_router
    ):
        result = await extractor.check("test")
    assert result is not None
    assert result.has_dream is True
    assert result.confidence == 0.3


def test_dream_result_validation() -> None:
    """DreamResult validates field constraints."""
    r = DreamResult(has_dream=True, dream_text="test", confidence=0.5)
    assert r.confidence == 0.5

    with pytest.raises(ValueError):
        DreamResult(has_dream=True, dream_text="test", confidence=1.5)

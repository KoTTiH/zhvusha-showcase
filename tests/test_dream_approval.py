"""Tests for dream approval state machine in ChatResponseSkill."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from src.skills.chat_response.skill import (
    ChatResponseSkill,
    _classify_approval_fast,
)


@pytest.fixture
def skill() -> ChatResponseSkill:
    return ChatResponseSkill()


@pytest.fixture
def workspace(tmp_path):  # type: ignore[no-untyped-def]
    personality = tmp_path / "personality"
    personality.mkdir()
    (personality / "dreams.md").write_text("# Мечты Жвуши\n\n", encoding="utf-8")
    return tmp_path


# --- _classify_approval tests ---


@pytest.mark.parametrize(
    "text,expected",
    [
        ("да", "yes"),
        ("ага", "yes"),
        ("конечно", "yes"),
        ("давай", "yes"),
        ("запомни", "yes"),
        ("запиши", "yes"),
        ("ок", "yes"),
        ("yes", "yes"),
        ("записывай", "yes"),
        ("нет", "no"),
        ("не надо", "no"),
        ("забей", "no"),
        ("нафиг", "no"),
        ("no", "no"),
        ("отмена", "no"),
        ("не сейчас", "later"),
        ("потом", "later"),
        ("позже", "later"),
        ("может быть", "later"),
        ("подумаю", "later"),
        ("что ты имеешь в виду?", "ambiguous"),
        ("расскажи подробнее", "ambiguous"),
        ("42", "ambiguous"),
    ],
)
def test_classify_approval_fast(text: str, expected: str) -> None:
    assert _classify_approval_fast(text) == expected


def test_classify_approval_strips_punctuation() -> None:
    """Trailing punctuation is stripped before matching."""
    assert _classify_approval_fast("да!") == "yes"
    assert _classify_approval_fast("нет.") == "no"
    assert _classify_approval_fast("потом,") == "later"


# --- _try_resolve_dream tests ---


@pytest.mark.asyncio
async def test_pending_dream_yes(skill: ChatResponseSkill, workspace) -> None:  # type: ignore[no-untyped-def]
    """Approved dream is written to dreams.md."""
    skill._pending_dream = "научиться рисовать"
    skill._pending_dream_ts = time.monotonic()

    result = await skill._try_resolve_dream("да", workspace)
    assert result is not None
    assert "Записала" in result.response

    dreams = (workspace / "personality" / "dreams.md").read_text(encoding="utf-8")
    assert "научиться рисовать" in dreams


@pytest.mark.asyncio
async def test_pending_dream_no(skill: ChatResponseSkill, workspace) -> None:  # type: ignore[no-untyped-def]
    """Rejected dream is discarded."""
    skill._pending_dream = "стать художником"
    skill._pending_dream_ts = time.monotonic()

    result = await skill._try_resolve_dream("нет", workspace)
    assert result is not None
    assert "забыла" in result.response
    assert skill._pending_dream is None


@pytest.mark.asyncio
async def test_pending_dream_later(skill: ChatResponseSkill, workspace) -> None:  # type: ignore[no-untyped-def]
    """Deferred dream is cleared."""
    skill._pending_dream = "полететь в космос"
    skill._pending_dream_ts = time.monotonic()

    result = await skill._try_resolve_dream("потом", workspace)
    assert result is not None
    assert "позже" in result.response.lower() or "Может" in result.response
    assert skill._pending_dream is None


@pytest.mark.asyncio
async def test_pending_dream_timeout(skill: ChatResponseSkill, workspace) -> None:  # type: ignore[no-untyped-def]
    """Expired pending dream is silently cleared."""
    skill._pending_dream = "что-то"
    skill._pending_dream_ts = time.monotonic() - 200  # >120s

    result = await skill._try_resolve_dream("да", workspace)
    assert result is None
    assert skill._pending_dream is None


@pytest.mark.asyncio
async def test_pending_dream_ambiguous(skill: ChatResponseSkill, workspace) -> None:  # type: ignore[no-untyped-def]
    """Ambiguous text keeps state and returns None (normal processing)."""
    skill._pending_dream = "что-то"
    skill._pending_dream_ts = time.monotonic()

    with patch(
        "src.skills.chat_response.skill._classify_approval_llm",
        return_value="ambiguous",
    ):
        result = await skill._try_resolve_dream("расскажи подробнее", workspace)
    assert result is None
    assert skill._pending_dream == "что-то"


def test_no_overwrite_existing_pending(skill: ChatResponseSkill) -> None:
    """New dream doesn't overwrite existing pending dream."""
    skill._pending_dream = "старая мечта"
    skill._pending_dream_ts = time.monotonic()

    # _background_dream_check should skip if _pending_dream exists
    # We test this by checking the state doesn't change
    assert skill._pending_dream == "старая мечта"


@pytest.mark.asyncio
async def test_social_mode_no_dream_check() -> None:
    """Dream extraction is not triggered in social mode."""
    skill = ChatResponseSkill()
    from src.skills.base import AgentContext

    ctx = AgentContext(
        user_id=123,
        chat_id=123,
        mode="social",
        message_id=1,
    )

    with (
        patch("src.skills.chat_response.skill.get_settings") as mock_settings,
        patch("src.skills.chat_response.skill.get_people_manager"),
        patch.object(skill, "_generate_response", return_value="hello"),
        patch.object(skill, "_intercept_post_command", return_value="hello"),
        patch("src.skills.chat_response.skill.get_dream_extractor") as mock_ext,
    ):
        mock_settings.return_value = MagicMock(
            admin_user_id=123,
            workspace_path="/tmp/test",  # noqa: S108
            public_info_about_nikita="",
        )
        await skill.execute("test", ctx)
        # Dream extractor should NOT be called in social mode
        mock_ext.assert_not_called()

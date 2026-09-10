"""Strong learning signals must reach only the creator, in personal mode.

``_background_enrich`` used to send a private memory proposal to whichever
``chat_id`` the message came from. For non-admin chats this leaked
Zhvusha's private memory workflow into a stranger's DM and put her state
machine (``_pending_learning``) into that chat.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.memory.sonnet_enricher import EnrichmentResult, LearningSignal
from src.skills.chat_response.skill import ChatResponseSkill

_PATCH_SETTINGS = "src.skills.chat_response.skill.get_settings"
_PATCH_ENRICHER = "src.skills.chat_response.skill.get_enricher"


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        workspace_path="unused_ws",
        claude_cli_path="claude",
        public_info_about_nikita="x",
        admin_user_id=42,
        chat_assistant_tier="analyst",
    )


def _strong_signal() -> LearningSignal:
    return LearningSignal(
        type="preference",
        scope="preferences",
        statement="Test preference statement",
        original_claim=None,
        apply_immediately=True,
        confidence=0.95,
    )


def _strong_enrichment() -> EnrichmentResult:
    return EnrichmentResult(
        importance=0.9,
        valence="positive",
        intent="statement",
        emotion="curious",
        confidence=0.9,
        is_feedback=False,
        feedback_strength=0.0,
        reasoning="test",
        learning_signal=_strong_signal(),
    )


def _make_skill() -> tuple[ChatResponseSkill, MagicMock, AsyncMock]:
    episodic = MagicMock()
    episodic.update_enrichment = AsyncMock()
    staging_writer = MagicMock()
    skill = ChatResponseSkill(episodic=episodic, staging_writer=staging_writer)
    return skill, staging_writer, AsyncMock()


def _run_enrich(
    skill: ChatResponseSkill,
    *,
    bot: object,
    chat_id: int,
    allow_proposals: bool,
    workspace_root: Path,
) -> None:
    enricher = MagicMock()
    enricher.enrich = AsyncMock(return_value=_strong_enrichment())
    with (
        patch(_PATCH_SETTINGS, return_value=_settings()),
        patch(_PATCH_ENRICHER, return_value=enricher),
        patch("src.personality.get_affective_state_manager"),
    ):
        asyncio.run(
            skill._background_enrich(
                episode_id=1,
                message="msg",
                recent_context="",
                prev_bot_response="",
                bot=bot,
                chat_id=chat_id,
                workspace_root=workspace_root,
                allow_proposals=allow_proposals,
            )
        )


def test_strong_signal_is_not_proposed_when_proposals_disabled(
    tmp_path: Path,
) -> None:
    skill, staging_writer, _ = _make_skill()
    bot = AsyncMock()

    _run_enrich(
        skill,
        bot=bot,
        chat_id=999,
        allow_proposals=False,
        workspace_root=tmp_path,
    )

    # No DM went to the non-admin chat
    bot.send_message.assert_not_awaited()
    # State machine must not be armed
    assert skill._pending_learning is None
    # Signal is instead staged silently so it isn't lost
    staging_writer.append.assert_called_once()


def test_strong_signal_is_proposed_when_allowed(tmp_path: Path) -> None:
    skill, staging_writer, _ = _make_skill()
    bot = AsyncMock()

    _run_enrich(
        skill,
        bot=bot,
        chat_id=42,
        allow_proposals=True,
        workspace_root=tmp_path,
    )

    # Proposal DM went out, state machine armed
    bot.send_message.assert_awaited_once()
    assert skill._pending_learning is not None
    # No silent staging in this path — user decides via approval
    staging_writer.append.assert_not_called()


def test_chat_self_coding_metadata_suppresses_background_proposals() -> None:
    from src.skills.base import AgentContext
    from src.skills.chat_response.skill import _suppress_background_proposals

    ctx = AgentContext(
        user_id=42,
        chat_id=42,
        mode="personal",
        metadata={"chat_self_coding": True},
    )

    assert _suppress_background_proposals(ctx) is True


# Mark pytest import used
_ = pytest

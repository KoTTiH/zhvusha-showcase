"""Tests for ChatResponseSkill background enrichment integration.

Covers: task scheduling, GC safety via _pending_tasks set, end-to-end
field update via stateful mock_episodic, and edge cases (social rate-limit,
episodic=None, enricher returns None, enricher raises).

Learning signal flow (approval-based):
- Strong signal (apply_immediately=True, confidence>0.8) → proposed to user
  via _propose_learning, stored in _pending_learning, NOT written to staging.
  User must approve ("да") via _try_resolve_learning before staging write.
- Weak signal → written to staging silently, no proposal.
- On approval → staging write + auto-apply corrections.
- On rejection → state cleared, nothing staged."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.memory.sonnet_enricher import EnrichmentResult, LearningSignal
from src.skills.base import AgentContext
from src.skills.chat_response.skill import ChatResponseSkill

_PATCH_SETTINGS = "src.skills.chat_response.skill.get_settings"
_PATCH_ROUTER = "src.skills.chat_response.skill.get_router"
_PATCH_PEOPLE = "src.skills.chat_response.skill.get_people_manager"
_PATCH_ENRICHER = "src.skills.chat_response.skill.get_enricher"


def _settings(tmp_path: str = "test_ws") -> SimpleNamespace:
    return SimpleNamespace(
        workspace_path=tmp_path,
        claude_cli_path="claude",
        public_info_about_nikita="Nikita is a developer.",
        admin_user_id=12345,
    )


def _context(mode: str = "personal", user_id: int = 12345) -> AgentContext:
    return AgentContext(
        user_id=user_id,
        chat_id=user_id,
        mode=mode,  # type: ignore[arg-type]
        message_id=1,
        bot=AsyncMock(),
    )


def _setup_workspace(root: str) -> None:
    ws = Path(root)
    (ws / "personality").mkdir(parents=True, exist_ok=True)
    (ws / "personality" / "core.md").write_text("I am Zhvusha.")
    (ws / "personality" / "genes.md").write_text("Curiosity: HIGH")
    (ws / "diary").mkdir(parents=True, exist_ok=True)
    (ws / "memory" / "people").mkdir(parents=True, exist_ok=True)


def _mock_people(*, interaction_count: int = 0) -> MagicMock:
    mgr = MagicMock()
    mgr.get_or_create_profile = MagicMock(return_value={"user_id": 12345})
    mgr.record_interaction = MagicMock(return_value=False)
    mgr.get_profile_for_context = MagicMock(return_value="")
    mgr.get_interaction_count = MagicMock(return_value=interaction_count)
    mgr.get_significance_level = MagicMock(return_value="admin")
    return mgr


def _sample_result(**overrides: object) -> EnrichmentResult:
    base: dict[str, object] = {
        "importance": 0.8,
        "valence": "positive",
        "intent": "statement",
        "emotion": "curious",
        "confidence": 0.9,
        "is_feedback": False,
        "feedback_strength": 0.0,
        "reasoning": "Nikita asking about his code.",
        "learning_signal": None,
    }
    base.update(overrides)
    return EnrichmentResult(**base)  # type: ignore[arg-type]


def _strong_learning_signal(**overrides: object) -> LearningSignal:
    base: dict[str, object] = {
        "type": "rule",
        "statement": "не писать формально в personal mode",
        "scope": "tone",
        "confidence": 0.92,
        "apply_immediately": True,
        "original_claim": None,
    }
    base.update(overrides)
    return LearningSignal(**base)  # type: ignore[arg-type]


def _weak_learning_signal(**overrides: object) -> LearningSignal:
    base: dict[str, object] = {
        "type": "preference",
        "statement": "кажется Никита предпочитает краткие ответы",
        "scope": "preferences",
        "confidence": 0.6,
        "apply_immediately": False,
        "original_claim": None,
    }
    base.update(overrides)
    return LearningSignal(**base)  # type: ignore[arg-type]


def _correction_learning_signal(**overrides: object) -> LearningSignal:
    base: dict[str, object] = {
        "type": "correction",
        "statement": "Kwork — единственный источник дохода",
        "scope": "personal_facts",
        "confidence": 0.92,
        "apply_immediately": True,
        "original_claim": "у Никиты есть основная работа в офисе",
    }
    base.update(overrides)
    return LearningSignal(**base)  # type: ignore[arg-type]


def _mock_consolidation_engine(
    *,
    return_value: object = None,
    side_effect: Exception | None = None,
) -> MagicMock:
    """Build a MagicMock ConsolidationEngine with stubbed handle_explicit_rejection.

    Pass `return_value=action` for a successful match, `return_value=None` for
    "no personality file matched", or `side_effect=Exception` to simulate
    failure inside handle_explicit_rejection.
    """
    engine = MagicMock()
    if side_effect is not None:
        engine.handle_explicit_rejection = AsyncMock(side_effect=side_effect)
    else:
        engine.handle_explicit_rejection = AsyncMock(return_value=return_value)
    return engine


_SENTINEL: object = object()


def _mock_enricher(
    *,
    return_value: EnrichmentResult | None | object = _SENTINEL,
    side_effect: Exception | None = None,
) -> AsyncMock:
    """Build an AsyncMock enricher.

    Pass `return_value=None` to simulate enrich() returning None explicitly
    (distinct from the default which is a valid EnrichmentResult).
    """
    enricher = AsyncMock()
    if side_effect is not None:
        enricher.enrich = AsyncMock(side_effect=side_effect)
    else:
        effective: EnrichmentResult | None = (
            _sample_result() if return_value is _SENTINEL else return_value  # type: ignore[assignment]
        )
        enricher.enrich = AsyncMock(return_value=effective)
    return enricher


def _mock_router_for_test(response: str = "ok") -> AsyncMock:
    """Build mock LLM router with agentic fallback to single-shot generate."""
    from src.llm.protocols import LLMResponse, LLMUsage

    r = AsyncMock()
    r.generate_with_tools = AsyncMock(side_effect=NotImplementedError)
    r.generate = AsyncMock(
        return_value=LLMResponse(text=response, model="sonnet", usage=LLMUsage())
    )
    return r


# --- Tests ---


async def test_execute_schedules_background_enrich_task(
    tmp_path: Path, mock_episodic: AsyncMock
) -> None:
    ws = str(tmp_path / "ws")
    _setup_workspace(ws)
    settings = _settings(ws)
    skill = ChatResponseSkill(episodic=mock_episodic)

    mock_router = _mock_router_for_test("Привет!")
    enricher = _mock_enricher()

    with (
        patch(_PATCH_SETTINGS, return_value=settings),
        patch(_PATCH_ROUTER, return_value=mock_router),
        patch(_PATCH_PEOPLE, return_value=_mock_people()),
        patch(_PATCH_ENRICHER, return_value=enricher),
    ):
        await skill.execute("привет", _context())
        # At least one background task was scheduled
        assert len(skill._pending_tasks) >= 1
        await asyncio.gather(*list(skill._pending_tasks))

    # After completion, done_callback should have drained the set
    assert len(skill._pending_tasks) == 0


async def test_execute_then_await_pending_applies_enrichment_end_to_end(
    tmp_path: Path,
    mock_episodic: AsyncMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Integration-ish test: execute -> enrichment task -> episode fields updated."""
    import logging

    ws = str(tmp_path / "ws")
    _setup_workspace(ws)
    settings = _settings(ws)
    skill = ChatResponseSkill(episodic=mock_episodic)

    mock_router = _mock_router_for_test("отвечаю")
    enricher = _mock_enricher(
        return_value=_sample_result(
            importance=0.95,
            valence="negative",
            intent="correction",
            emotion="frustrated",
            confidence=0.88,
        )
    )

    caplog.set_level(logging.INFO)

    with (
        patch(_PATCH_SETTINGS, return_value=settings),
        patch(_PATCH_ROUTER, return_value=mock_router),
        patch(_PATCH_PEOPLE, return_value=_mock_people()),
        patch(_PATCH_ENRICHER, return_value=enricher),
    ):
        result = await skill.execute("ты не так поняла", _context())
        await asyncio.gather(*list(skill._pending_tasks))

    assert result.success is True

    # First episode (user message) id is 1 from stateful mock_episodic
    user_episode = mock_episodic._episodes[1]
    assert user_episode["importance"] == 0.95
    assert user_episode["valence"] == "negative"
    assert user_episode["intent"] == "correction"
    assert user_episode["emotion"] == "frustrated"
    assert user_episode["confidence"] == 0.88

    # enricher.enrich called with the right arguments
    enricher.enrich.assert_awaited_once()
    call_kwargs = enricher.enrich.call_args.kwargs
    assert call_kwargs["message"] == "ты не так поняла"
    assert call_kwargs["prev_bot_response"] == "отвечаю"


async def test_background_enrich_swallows_enricher_exceptions(
    tmp_path: Path, mock_episodic: AsyncMock
) -> None:
    ws = str(tmp_path / "ws")
    _setup_workspace(ws)
    settings = _settings(ws)
    skill = ChatResponseSkill(episodic=mock_episodic)

    mock_router = _mock_router_for_test("ok")
    enricher = _mock_enricher(side_effect=RuntimeError("sonnet exploded"))

    with (
        patch(_PATCH_SETTINGS, return_value=settings),
        patch(_PATCH_ROUTER, return_value=mock_router),
        patch(_PATCH_PEOPLE, return_value=_mock_people()),
        patch(_PATCH_ENRICHER, return_value=enricher),
    ):
        result = await skill.execute("hello", _context())
        # Waiting on the task must not raise even if enricher did
        await asyncio.gather(*list(skill._pending_tasks), return_exceptions=False)

    assert result.success is True
    # Episode 1 still exists with baseline values
    assert mock_episodic._episodes[1]["valence"] == "neutral"


@pytest.mark.parametrize("scenario", ["episodic_none", "rate_limited"])
async def test_execute_skips_enrichment_when_episodic_unavailable_or_rate_limited(
    tmp_path: Path, scenario: str, mock_episodic: AsyncMock
) -> None:
    ws = str(tmp_path / "ws")
    _setup_workspace(ws)
    settings = _settings(ws)

    mock_router = _mock_router_for_test("ok")
    enricher = _mock_enricher()

    episodic_arg: AsyncMock | None
    if scenario == "episodic_none":
        episodic_arg = None
    else:
        episodic_arg = mock_episodic
        # Simulate social rate-limit: record returns -1
        mock_episodic.record = AsyncMock(return_value=-1)

    skill = ChatResponseSkill(episodic=episodic_arg)

    with (
        patch(_PATCH_SETTINGS, return_value=settings),
        patch(_PATCH_ROUTER, return_value=mock_router),
        patch(_PATCH_PEOPLE, return_value=_mock_people()),
        patch(_PATCH_ENRICHER, return_value=enricher),
    ):
        await skill.execute("hi", _context())
        if skill._pending_tasks:
            await asyncio.gather(*list(skill._pending_tasks))

    enricher.enrich.assert_not_awaited()


async def test_background_enrich_skipped_when_enricher_returns_none(
    tmp_path: Path, mock_episodic: AsyncMock
) -> None:
    """If enricher.enrich() returns None, update_enrichment must not be called."""
    ws = str(tmp_path / "ws")
    _setup_workspace(ws)
    settings = _settings(ws)
    skill = ChatResponseSkill(episodic=mock_episodic)

    mock_router = _mock_router_for_test("ok")
    enricher = _mock_enricher(return_value=None)  # type: ignore[arg-type]

    with (
        patch(_PATCH_SETTINGS, return_value=settings),
        patch(_PATCH_ROUTER, return_value=mock_router),
        patch(_PATCH_PEOPLE, return_value=_mock_people()),
        patch(_PATCH_ENRICHER, return_value=enricher),
    ):
        await skill.execute("hi", _context())
        await asyncio.gather(*list(skill._pending_tasks))

    # enrich was called, but update_enrichment was not
    enricher.enrich.assert_awaited_once()
    mock_episodic.update_enrichment.assert_not_awaited()
    # Episode 1 remains with baseline values
    assert mock_episodic._episodes[1]["valence"] == "neutral"


# --- Learning signals: proposals + staging ---


async def test_background_enrich_proposes_strong_signal(
    tmp_path: Path, mock_episodic: AsyncMock
) -> None:
    """EnrichmentResult with strong learning_signal -> proposed to user,
    stored in _pending_learning, NOT written to staging."""
    ws = str(tmp_path / "ws")
    _setup_workspace(ws)
    settings = _settings(ws)

    skill = ChatResponseSkill(episodic=mock_episodic)

    mock_router = _mock_router_for_test("отвечаю")
    enricher = _mock_enricher(
        return_value=_sample_result(learning_signal=_strong_learning_signal())
    )

    ctx = _context()

    with (
        patch(_PATCH_SETTINGS, return_value=settings),
        patch(_PATCH_ROUTER, return_value=mock_router),
        patch(_PATCH_PEOPLE, return_value=_mock_people()),
        patch(_PATCH_ENRICHER, return_value=enricher),
    ):
        await skill.execute("не пиши так формально", ctx)
        await asyncio.gather(*list(skill._pending_tasks))

    assert skill._pending_learning is not None
    assert skill._pending_learning.type == "rule"
    ctx.bot.send_message.assert_awaited_once()


async def test_background_enrich_proposal_text(
    tmp_path: Path, mock_episodic: AsyncMock
) -> None:
    """Proposal text format: contains the memory proposal and open decision."""
    ws = str(tmp_path / "ws")
    _setup_workspace(ws)
    settings = _settings(ws)

    skill = ChatResponseSkill(episodic=mock_episodic)

    mock_router = _mock_router_for_test("отвечаю")
    enricher = _mock_enricher(
        return_value=_sample_result(learning_signal=_strong_learning_signal())
    )

    ctx = _context()

    with (
        patch(_PATCH_SETTINGS, return_value=settings),
        patch(_PATCH_ROUTER, return_value=mock_router),
        patch(_PATCH_PEOPLE, return_value=_mock_people()),
        patch(_PATCH_ENRICHER, return_value=enricher),
    ):
        await skill.execute("не пиши формально", ctx)
        await asyncio.gather(*list(skill._pending_tasks))

    ctx.bot.send_message.assert_awaited_once()
    send_kwargs = ctx.bot.send_message.call_args.kwargs
    assert "хочу это запомнить" in send_kwargs["text"]
    assert "Сохранять?" in send_kwargs["text"]
    # No parse_mode — plaintext to avoid markdown injection
    assert "parse_mode" not in send_kwargs


async def test_background_enrich_skips_notification_for_weak_signal(
    tmp_path: Path, mock_episodic: AsyncMock
) -> None:
    ws = str(tmp_path / "ws")
    _setup_workspace(ws)
    settings = _settings(ws)

    staging_writer = MagicMock()
    staging_writer.append = MagicMock(
        return_value=tmp_path
        / "ws"
        / "personality"
        / ".staging"
        / "learnings_pending.md"
    )

    skill = ChatResponseSkill(episodic=mock_episodic, staging_writer=staging_writer)

    mock_router = _mock_router_for_test("ok")
    # Weak: confidence=0.6, apply_immediately=False -> no notification
    enricher = _mock_enricher(
        return_value=_sample_result(learning_signal=_weak_learning_signal())
    )

    ctx = _context()

    with (
        patch(_PATCH_SETTINGS, return_value=settings),
        patch(_PATCH_ROUTER, return_value=mock_router),
        patch(_PATCH_PEOPLE, return_value=_mock_people()),
        patch(_PATCH_ENRICHER, return_value=enricher),
    ):
        await skill.execute("hi", ctx)
        await asyncio.gather(*list(skill._pending_tasks))

    # Staging append still happens (routed to pending)
    staging_writer.append.assert_called_once()
    # But no notification
    ctx.bot.send_message.assert_not_awaited()


async def test_background_enrich_skips_notification_when_apply_immediately_false(
    tmp_path: Path, mock_episodic: AsyncMock
) -> None:
    """High confidence but apply_immediately=False -> no notification,
    staging routes to pending."""
    ws = str(tmp_path / "ws")
    _setup_workspace(ws)
    settings = _settings(ws)

    staging_writer = MagicMock()
    staging_writer.append = MagicMock(
        return_value=tmp_path
        / "ws"
        / "personality"
        / ".staging"
        / "learnings_pending.md"
    )

    skill = ChatResponseSkill(episodic=mock_episodic, staging_writer=staging_writer)

    mock_router = _mock_router_for_test("ok")
    enricher = _mock_enricher(
        return_value=_sample_result(
            learning_signal=_strong_learning_signal(
                confidence=0.95, apply_immediately=False
            )
        )
    )

    ctx = _context()

    with (
        patch(_PATCH_SETTINGS, return_value=settings),
        patch(_PATCH_ROUTER, return_value=mock_router),
        patch(_PATCH_PEOPLE, return_value=_mock_people()),
        patch(_PATCH_ENRICHER, return_value=enricher),
    ):
        await skill.execute("hi", ctx)
        await asyncio.gather(*list(skill._pending_tasks))

    ctx.bot.send_message.assert_not_awaited()


async def test_background_enrich_proposal_failure_clears_pending(
    tmp_path: Path, mock_episodic: AsyncMock
) -> None:
    """If bot.send_message raises, the background task must complete without
    propagating. _pending_learning is cleared on failure."""
    ws = str(tmp_path / "ws")
    _setup_workspace(ws)
    settings = _settings(ws)

    skill = ChatResponseSkill(episodic=mock_episodic)

    mock_router = _mock_router_for_test("ok")
    enricher = _mock_enricher(
        return_value=_sample_result(learning_signal=_strong_learning_signal())
    )

    ctx = _context()
    ctx.bot.send_message = AsyncMock(side_effect=RuntimeError("telegram down"))

    with (
        patch(_PATCH_SETTINGS, return_value=settings),
        patch(_PATCH_ROUTER, return_value=mock_router),
        patch(_PATCH_PEOPLE, return_value=_mock_people()),
        patch(_PATCH_ENRICHER, return_value=enricher),
    ):
        result = await skill.execute("hi", ctx)
        await asyncio.gather(*list(skill._pending_tasks), return_exceptions=False)

    assert result.success is True
    assert skill._pending_learning is None  # cleared on failure


async def test_background_enrich_rate_limits_proposals() -> None:
    """4 proposals in a short window -> only 3 send_message awaits."""
    skill = ChatResponseSkill()
    mock_bot = AsyncMock()

    for _ in range(4):
        signal = _strong_learning_signal()
        skill._pending_learning = signal
        await skill._propose_learning(bot=mock_bot, chat_id=12345, signal=signal)

    assert mock_bot.send_message.await_count == 3


async def test_propose_learning_skips_when_bot_is_none() -> None:
    """Direct unit test: bot=None -> early return, nothing sent."""
    skill = ChatResponseSkill()
    signal = _strong_learning_signal()
    skill._pending_learning = signal
    await skill._propose_learning(bot=None, chat_id=12345, signal=signal)
    # No exception, no rate-limit slot consumed
    assert len(skill._notification_times) == 0


async def test_propose_learning_skips_when_chat_id_is_none() -> None:
    """Direct unit test: chat_id=None -> early return, nothing sent."""
    skill = ChatResponseSkill()
    mock_bot = AsyncMock()
    signal = _strong_learning_signal()
    skill._pending_learning = signal
    await skill._propose_learning(bot=mock_bot, chat_id=None, signal=signal)
    mock_bot.send_message.assert_not_awaited()
    assert len(skill._notification_times) == 0


async def test_background_enrich_skips_staging_when_learning_signal_none(
    tmp_path: Path, mock_episodic: AsyncMock
) -> None:
    """learning_signal=None -> StagingWriter.append NOT called."""
    ws = str(tmp_path / "ws")
    _setup_workspace(ws)
    settings = _settings(ws)

    staging_writer = MagicMock()
    staging_writer.append = MagicMock()

    skill = ChatResponseSkill(episodic=mock_episodic, staging_writer=staging_writer)

    mock_router = _mock_router_for_test("ok")
    # No learning_signal (default None)
    enricher = _mock_enricher(return_value=_sample_result())

    ctx = _context()

    with (
        patch(_PATCH_SETTINGS, return_value=settings),
        patch(_PATCH_ROUTER, return_value=mock_router),
        patch(_PATCH_PEOPLE, return_value=_mock_people()),
        patch(_PATCH_ENRICHER, return_value=enricher),
    ):
        await skill.execute("hi", ctx)
        await asyncio.gather(*list(skill._pending_tasks))

    staging_writer.append.assert_not_called()
    ctx.bot.send_message.assert_not_awaited()


async def test_propose_learning_sends_correct_text() -> None:
    """Proposal text format: memo + statement + open decision."""
    skill = ChatResponseSkill()
    mock_bot = AsyncMock()
    signal = _strong_learning_signal()
    skill._pending_learning = signal  # set so it can be cleared on success

    await skill._propose_learning(bot=mock_bot, chat_id=12345, signal=signal)

    mock_bot.send_message.assert_awaited_once()
    kwargs = mock_bot.send_message.call_args.kwargs
    assert (
        kwargs["text"]
        == f"\U0001f4dd хочу это запомнить: {signal.statement}\nСохранять?"
    )
    assert "parse_mode" not in kwargs


# --- Learning approval flow ---


async def test_try_resolve_learning_approves_and_stages(tmp_path: Path) -> None:
    """User says 'да' -> signal written to staging, response '📝 записала!'."""
    ws = tmp_path / "ws"
    _setup_workspace(str(ws))

    staging_writer = MagicMock()
    staging_writer.append = MagicMock(
        return_value=ws / "personality" / ".staging" / "learnings_immediate.md"
    )

    skill = ChatResponseSkill(staging_writer=staging_writer)
    signal = _strong_learning_signal()
    skill._pending_learning = signal
    skill._pending_learning_episode_id = 42
    skill._pending_learning_ts = time.monotonic()
    skill._pending_learning_chat_id = 12345

    result = await skill._try_resolve_learning("да", ws, chat_id=12345)

    assert result is not None
    assert result.response == "\U0001f4dd записала!"
    assert skill._pending_learning is None
    staging_writer.append.assert_called_once()
    args = staging_writer.append.call_args
    assert args.args[0].type == "rule"


async def test_try_resolve_learning_rejects(tmp_path: Path) -> None:
    """User says 'нет' -> nothing staged, response 'Ок, не записываю.'."""
    ws = tmp_path / "ws"
    _setup_workspace(str(ws))

    staging_writer = MagicMock()

    skill = ChatResponseSkill(staging_writer=staging_writer)
    skill._pending_learning = _strong_learning_signal()
    skill._pending_learning_ts = time.monotonic()
    skill._pending_learning_chat_id = 12345

    result = await skill._try_resolve_learning("нет", ws, chat_id=12345)

    assert result is not None
    assert result.response == "Ок, не записываю."
    staging_writer.append.assert_not_called()


async def test_try_resolve_learning_correction_triggers_rejection_handler(
    tmp_path: Path,
) -> None:
    """Approved correction -> staging write + handle_explicit_rejection called."""
    ws = tmp_path / "ws"
    _setup_workspace(str(ws))

    staging_writer = MagicMock()
    engine = _mock_consolidation_engine(return_value=MagicMock(file_path="profile.md"))

    skill = ChatResponseSkill(
        staging_writer=staging_writer, consolidation_engine=engine
    )
    signal = _correction_learning_signal()
    skill._pending_learning = signal
    skill._pending_learning_episode_id = 7
    skill._pending_learning_ts = time.monotonic()
    skill._pending_learning_chat_id = 12345

    result = await skill._try_resolve_learning("да", ws, chat_id=12345)

    assert result is not None
    assert result.response == "\U0001f4dd записала!"
    staging_writer.append.assert_called_once()
    engine.handle_explicit_rejection.assert_awaited_once_with(
        rejected_conclusion="у Никиты есть основная работа в офисе",
        nikita_correction="Kwork — единственный источник дохода",
    )


async def test_try_resolve_learning_correction_engine_none(tmp_path: Path) -> None:
    """consolidation_engine=None -> staging write succeeds, no crash."""
    ws = tmp_path / "ws"
    _setup_workspace(str(ws))

    staging_writer = MagicMock()

    skill = ChatResponseSkill(staging_writer=staging_writer, consolidation_engine=None)
    skill._pending_learning = _correction_learning_signal()
    skill._pending_learning_episode_id = 7
    skill._pending_learning_ts = time.monotonic()
    skill._pending_learning_chat_id = 12345

    result = await skill._try_resolve_learning("да", ws, chat_id=12345)

    assert result is not None
    assert result.response == "\U0001f4dd записала!"
    staging_writer.append.assert_called_once()


async def test_try_resolve_learning_correction_exception_swallowed(
    tmp_path: Path,
) -> None:
    """handle_explicit_rejection raises -> staging still written, no crash."""
    ws = tmp_path / "ws"
    _setup_workspace(str(ws))

    staging_writer = MagicMock()
    engine = _mock_consolidation_engine(side_effect=RuntimeError("boom"))

    skill = ChatResponseSkill(
        staging_writer=staging_writer, consolidation_engine=engine
    )
    skill._pending_learning = _correction_learning_signal()
    skill._pending_learning_episode_id = 7
    skill._pending_learning_ts = time.monotonic()
    skill._pending_learning_chat_id = 12345

    result = await skill._try_resolve_learning("да", ws, chat_id=12345)

    assert result is not None
    assert result.response == "\U0001f4dd записала!"
    staging_writer.append.assert_called_once()


async def test_try_resolve_learning_timeout_clears_state(tmp_path: Path) -> None:
    """After timeout, pending learning is cleared and None returned."""
    skill = ChatResponseSkill()
    skill._pending_learning = _strong_learning_signal()
    skill._pending_learning_ts = time.monotonic() - 200  # expired
    skill._pending_learning_chat_id = 12345

    result = await skill._try_resolve_learning("да", tmp_path, chat_id=12345)

    assert result is None
    assert skill._pending_learning is None

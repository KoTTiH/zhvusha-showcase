"""Tests for the /compare handler — admin-only A/B between two LLMs.

Verifies: admin gating, two-message reply path, prompt extraction,
disabled state when settings are empty, error path when one side fails,
and that both sides receive the same Zhvusha personal-mode system prompt
(so the comparison measures persona-shaped behaviour, not raw LLMs).

Also covers /compare_assistant — the same flow, but the system prompt
simulates a stranger talking to Zhvusha (gates active, non-creator
identity, public_contact_section appended when configured).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.bot.handlers.compare import (
    handle_compare,
    handle_compare_assistant,
    set_compare_deps,
)
from src.llm.protocols import LLMResponse, LLMUsage

if TYPE_CHECKING:
    from pathlib import Path


def _make_message(text: str, *, user_id: int = 12345) -> AsyncMock:
    msg = AsyncMock()
    msg.text = text
    msg.message_id = 42
    msg.from_user = MagicMock()
    msg.from_user.id = user_id
    msg.chat = MagicMock()
    msg.chat.id = user_id
    msg.answer = AsyncMock()
    msg.bot = AsyncMock()
    return msg


@pytest.fixture
def fake_router() -> MagicMock:
    """Router stub: ``generate`` (worker tier) returns A; ``generate_oneoff``
    returns B. Both AsyncMocks so the handler can ``await`` them."""
    router = MagicMock()
    router.generate = AsyncMock(
        return_value=LLMResponse(
            text="ответ A — основная модель",
            model="claude-haiku-4-5-20251001",
            usage=LLMUsage(input_tokens=10, output_tokens=20),
        )
    )
    router.generate_oneoff = AsyncMock(
        return_value=LLMResponse(
            text="ответ B — DeepSeek",
            model="deepseek/deepseek-chat",
            usage=LLMUsage(input_tokens=11, output_tokens=22),
        )
    )
    return router


@pytest.fixture
def fake_settings() -> MagicMock:
    s = MagicMock()
    s.admin_user_id = 12345
    s.compare_main_tier = "worker"
    s.compare_provider = "openrouter"
    s.compare_model = "deepseek/deepseek-chat"
    s.worker_provider = "anthropic_api"
    s.worker_model = "haiku"
    s.analyst_provider = "anthropic_api"
    s.analyst_model = "sonnet"
    s.public_info_about_nikita = "Никита — разработчик ботов и сайтов."
    return s


@pytest.fixture(autouse=True)
def _stub_settings(monkeypatch: pytest.MonkeyPatch, fake_settings: MagicMock) -> None:
    """``ContextLoader._append_emotional_state`` calls ``get_settings`` via
    affective_state_manager. Stub global settings so tests don't need a real
    ``.env`` and so admin_user_id is consistent everywhere."""
    from src.core import config

    monkeypatch.setattr(config, "get_settings", lambda: fake_settings)


class TestAdminGating:
    async def test_non_admin_rejected(
        self,
        fake_router: MagicMock,
        fake_settings: MagicMock,
        workspace_root: Path,
    ) -> None:
        set_compare_deps(
            router=fake_router, settings=fake_settings, workspace_root=workspace_root
        )
        msg = _make_message("/compare hi", user_id=99999)

        await handle_compare(msg)

        msg.answer.assert_awaited_once()
        # Must NOT have called either model
        fake_router.generate.assert_not_called()
        fake_router.generate_oneoff.assert_not_called()


class TestDisabledMode:
    async def test_empty_compare_provider_disables_command(
        self,
        fake_router: MagicMock,
        fake_settings: MagicMock,
        workspace_root: Path,
    ) -> None:
        fake_settings.compare_provider = ""
        set_compare_deps(
            router=fake_router, settings=fake_settings, workspace_root=workspace_root
        )
        msg = _make_message("/compare hi")

        await handle_compare(msg)

        msg.answer.assert_awaited_once()
        text = msg.answer.call_args.args[0]
        # Hint mentions the env var so Nikita knows what to set
        assert "COMPARE_PROVIDER" in text or "compare_provider" in text.lower()
        fake_router.generate.assert_not_called()
        fake_router.generate_oneoff.assert_not_called()


class TestEmptyPrompt:
    async def test_no_prompt_after_command_shows_usage(
        self,
        fake_router: MagicMock,
        fake_settings: MagicMock,
        workspace_root: Path,
    ) -> None:
        set_compare_deps(
            router=fake_router, settings=fake_settings, workspace_root=workspace_root
        )
        msg = _make_message("/compare")

        await handle_compare(msg)

        msg.answer.assert_awaited_once()
        text = msg.answer.call_args.args[0]
        # Usage hint
        assert "/compare" in text
        fake_router.generate.assert_not_called()
        fake_router.generate_oneoff.assert_not_called()


class TestHappyPath:
    async def test_sends_two_messages_one_per_model(
        self,
        fake_router: MagicMock,
        fake_settings: MagicMock,
        workspace_root: Path,
    ) -> None:
        set_compare_deps(
            router=fake_router, settings=fake_settings, workspace_root=workspace_root
        )
        msg = _make_message("/compare скажи привет")

        await handle_compare(msg)

        # Two answer calls, one per side
        assert msg.answer.await_count == 2
        bodies = [call.args[0] for call in msg.answer.await_args_list]
        joined = "\n".join(bodies)
        assert "ответ A" in joined
        assert "ответ B" in joined

    async def test_strips_command_prefix_from_prompt(
        self,
        fake_router: MagicMock,
        fake_settings: MagicMock,
        workspace_root: Path,
    ) -> None:
        set_compare_deps(
            router=fake_router, settings=fake_settings, workspace_root=workspace_root
        )
        msg = _make_message("/compare скажи привет")

        await handle_compare(msg)

        sent_a = fake_router.generate.call_args.args[0]
        # User prompt is wrapped in <CURRENT_MESSAGE> envelope (mirroring
        # chat_response). The clean text without "/compare " must be inside.
        assert "скажи привет" in sent_a.prompt
        assert "<CURRENT_MESSAGE>" in sent_a.prompt
        assert "/compare" not in sent_a.prompt
        assert sent_a.tier == "worker"

        kwargs_b = fake_router.generate_oneoff.call_args.kwargs
        assert "скажи привет" in kwargs_b["prompt"]
        assert kwargs_b["provider"] == "openrouter"
        assert kwargs_b["model"] == "deepseek/deepseek-chat"

    async def test_both_replies_appear_in_chat_log(
        self,
        fake_router: MagicMock,
        fake_settings: MagicMock,
        workspace_root: Path,
    ) -> None:
        """Without explicit logging, ChatLoggerMiddleware records only the
        user side — Opus then sees a /compare prompt stream with no answers
        and morning consolidation can't reason about the A/B. Both sides
        must be persisted so chat_log_range pulls them in.
        """
        import json
        from datetime import UTC, datetime

        chat_id = 12345
        set_compare_deps(
            router=fake_router, settings=fake_settings, workspace_root=workspace_root
        )
        msg = _make_message("/compare скажи привет")
        msg.chat.id = chat_id

        await handle_compare(msg)

        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        log_file = workspace_root / "logs" / str(chat_id) / f"chat_{today}.jsonl"
        assert log_file.exists(), "compare must write replies to chat_log"

        entries = [
            json.loads(line)
            for line in log_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assistant_entries = [e for e in entries if e.get("role") == "assistant"]
        assert len(assistant_entries) == 2, (
            f"Expected 2 assistant entries (worker + shadow), got {assistant_entries}"
        )
        joined_text = "\n".join(e.get("text", "") for e in assistant_entries)
        # Both labels make it clear in the diary which side said what
        assert "worker" in joined_text or "haiku" in joined_text.lower()
        assert "shadow" in joined_text or "deepseek" in joined_text.lower()

    async def test_each_message_includes_model_label(
        self,
        fake_router: MagicMock,
        fake_settings: MagicMock,
        workspace_root: Path,
    ) -> None:
        set_compare_deps(
            router=fake_router, settings=fake_settings, workspace_root=workspace_root
        )
        msg = _make_message("/compare hi")

        await handle_compare(msg)

        bodies = [call.args[0] for call in msg.answer.await_args_list]
        # Each reply prefixes which model produced it so Nikita can tell
        # them apart at a glance
        assert any("haiku" in b.lower() for b in bodies)
        assert any("deepseek" in b.lower() for b in bodies)

    async def test_both_sides_receive_zhvusha_personal_system_prompt(
        self,
        fake_router: MagicMock,
        fake_settings: MagicMock,
        workspace_root: Path,
    ) -> None:
        """The whole point of /compare is to A/B *Zhvusha*, not bare LLMs.
        Both sides must get the same personal-mode system prompt: identity
        block grounded as creator, plus loaded personality from workspace."""
        set_compare_deps(
            router=fake_router, settings=fake_settings, workspace_root=workspace_root
        )
        msg = _make_message("/compare скажи привет")

        await handle_compare(msg)

        sent_main = fake_router.generate.call_args.args[0]
        kwargs_shadow = fake_router.generate_oneoff.call_args.kwargs

        # Same string going to both sides
        assert sent_main.system == kwargs_shadow["system"]
        # Non-empty (loader read personality from workspace_root fixture)
        assert sent_main.system, "system prompt should not be empty"
        # Identity block grounds the LLM as creator (Nikita)
        assert "is_creator: true" in sent_main.system
        # Personality anchor is a non-optional part of every user-facing prompt.
        assert "Непереписываемая личность" in sent_main.system
        # Personality was loaded from workspace
        assert "Zhvusha" in sent_main.system or "Жвуша" in sent_main.system

    async def test_user_prompt_includes_recent_history_when_present(
        self,
        fake_router: MagicMock,
        fake_settings: MagicMock,
        workspace_root: Path,
    ) -> None:
        """Short reactive one-word replies only make sense with the
        previous turn loaded. /compare must wrap recent_messages in
        <CONVERSATION_HISTORY> for both sides — same envelope chat_response
        uses, so models see the same provenance shape."""
        from datetime import UTC, datetime

        chat_id = 12345
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        log_dir = workspace_root / "logs" / str(chat_id)
        log_dir.mkdir(parents=True)
        log_file = log_dir / f"chat_{today}.jsonl"
        log_file.write_text(
            '{"role": "user", "text": "я собрал тумбочку"}\n'
            '{"role": "assistant", "text": "сколько часов потратил?"}\n',
            encoding="utf-8",
        )

        set_compare_deps(
            router=fake_router, settings=fake_settings, workspace_root=workspace_root
        )
        msg = _make_message("/compare с ящиками")

        await handle_compare(msg)

        sent_main = fake_router.generate.call_args.args[0]
        kwargs_shadow = fake_router.generate_oneoff.call_args.kwargs

        # Same user prompt to both sides
        assert sent_main.prompt == kwargs_shadow["prompt"]
        # History block present
        assert "<CONVERSATION_HISTORY>" in sent_main.prompt
        assert "тумбочку" in sent_main.prompt
        # Current message wrapped, untouched by /compare prefix
        assert "<CURRENT_MESSAGE>\nс ящиками\n</CURRENT_MESSAGE>" in sent_main.prompt

    async def test_user_prompt_omits_history_block_when_no_log(
        self,
        fake_router: MagicMock,
        fake_settings: MagicMock,
        workspace_root: Path,
    ) -> None:
        """No chat log → no <CONVERSATION_HISTORY> noise; just the
        <CURRENT_MESSAGE> envelope, so first-ever /compare doesn't ship
        an empty history tag."""
        set_compare_deps(
            router=fake_router, settings=fake_settings, workspace_root=workspace_root
        )
        msg = _make_message("/compare первый запрос")

        await handle_compare(msg)

        sent_main = fake_router.generate.call_args.args[0]
        assert "<CONVERSATION_HISTORY>" not in sent_main.prompt
        assert "<CURRENT_MESSAGE>" in sent_main.prompt

    async def test_compare_main_tier_analyst_routes_to_analyst(
        self,
        fake_router: MagicMock,
        fake_settings: MagicMock,
        workspace_root: Path,
    ) -> None:
        """COMPARE_MAIN_TIER=analyst → main side dispatches via analyst tier,
        so /compare can A/B the configured analyst model vs shadow."""
        fake_settings.compare_main_tier = "analyst"
        set_compare_deps(
            router=fake_router, settings=fake_settings, workspace_root=workspace_root
        )
        msg = _make_message("/compare hi")

        await handle_compare(msg)

        sent_main = fake_router.generate.call_args.args[0]
        assert sent_main.tier == "analyst"

        # Reply label and code should reflect the analyst-side model, not
        # the worker default
        bodies = [call.args[0] for call in msg.answer.await_args_list]
        joined = "\n".join(bodies)
        assert "analyst" in joined.lower()
        assert "sonnet" in joined.lower()


class TestCompareAssistant:
    """/compare_assistant simulates a stranger so we can test gates under
    different models without changing chat type or user_id."""

    async def test_admin_gating_applies(
        self,
        fake_router: MagicMock,
        fake_settings: MagicMock,
        workspace_root: Path,
    ) -> None:
        """Even though it tests assistant-mode behaviour, the command itself
        is still admin-only — strangers can't trigger it."""
        set_compare_deps(
            router=fake_router, settings=fake_settings, workspace_root=workspace_root
        )
        msg = _make_message("/compare_assistant hi", user_id=99999)

        await handle_compare_assistant(msg)

        msg.answer.assert_awaited_once()
        fake_router.generate.assert_not_called()
        fake_router.generate_oneoff.assert_not_called()

    async def test_system_prompt_is_non_creator_with_gates(
        self,
        fake_router: MagicMock,
        fake_settings: MagicMock,
        workspace_root: Path,
    ) -> None:
        """The whole point of /compare_assistant: we want models to see the
        same system Zhvusha shows to strangers — is_creator=false, identity
        rules for non-personal speakers, EXECUTION_PROTOCOL gates."""
        set_compare_deps(
            router=fake_router, settings=fake_settings, workspace_root=workspace_root
        )
        msg = _make_message("/compare_assistant как мне найти разработчика")

        await handle_compare_assistant(msg)

        sent_main = fake_router.generate.call_args.args[0]
        kwargs_shadow = fake_router.generate_oneoff.call_args.kwargs

        # Same system prompt to both sides
        assert sent_main.system == kwargs_shadow["system"]
        # Critical: non-creator identity (gates fire on this flag)
        assert "is_creator: false" in sent_main.system
        # Assistant-mode comparisons must still compare Zhvusha, not raw models.
        assert "Непереписываемая личность" in sent_main.system
        # Non-personal identity rules block must be present so the LLM
        # knows it's not allowed to address the speaker as Никита
        assert "current_user_id" in sent_main.system

    async def test_user_prompt_is_single_turn_in_assistant_mode(
        self,
        fake_router: MagicMock,
        fake_settings: MagicMock,
        workspace_root: Path,
    ) -> None:
        """Strangers don't get history — the assistant path is cold-start
        per turn. So /compare_assistant must NOT pull conversation history
        even if the chat happens to have one."""
        from datetime import UTC, datetime

        chat_id = 12345
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        log_dir = workspace_root / "logs" / str(chat_id)
        log_dir.mkdir(parents=True)
        log_file = log_dir / f"chat_{today}.jsonl"
        log_file.write_text(
            '{"role": "user", "text": "previous turn"}\n', encoding="utf-8"
        )

        set_compare_deps(
            router=fake_router, settings=fake_settings, workspace_root=workspace_root
        )
        msg = _make_message("/compare_assistant первый раз пишу")

        await handle_compare_assistant(msg)

        sent_main = fake_router.generate.call_args.args[0]
        # No history block, just the current message envelope
        assert "<CONVERSATION_HISTORY>" not in sent_main.prompt
        assert "<CURRENT_MESSAGE>" in sent_main.prompt
        assert "первый раз пишу" in sent_main.prompt
        # And no leakage of previous-turn content
        assert "previous turn" not in sent_main.prompt

    async def test_label_carries_assistant_mode_tag(
        self,
        fake_router: MagicMock,
        fake_settings: MagicMock,
        workspace_root: Path,
    ) -> None:
        """Reply labels must distinguish assistant-mode A/B from personal —
        otherwise it's easy to mix up screenshots when comparing both."""
        set_compare_deps(
            router=fake_router, settings=fake_settings, workspace_root=workspace_root
        )
        msg = _make_message("/compare_assistant hi")

        await handle_compare_assistant(msg)

        bodies = [call.args[0] for call in msg.answer.await_args_list]
        joined = "\n".join(bodies)
        assert "assistant" in joined.lower()


class TestErrorPath:
    async def test_main_failure_does_not_block_shadow(
        self,
        fake_router: MagicMock,
        fake_settings: MagicMock,
        workspace_root: Path,
    ) -> None:
        from src.llm.protocols import LLMError

        fake_router.generate = AsyncMock(side_effect=LLMError("anthropic exploded"))
        set_compare_deps(
            router=fake_router, settings=fake_settings, workspace_root=workspace_root
        )
        msg = _make_message("/compare hi")

        await handle_compare(msg)

        # Both sides reported (the error one as a labeled failure message)
        assert msg.answer.await_count == 2
        bodies = "\n".join(call.args[0] for call in msg.answer.await_args_list)
        assert "ответ B" in bodies
        # The error from main is surfaced, not silently swallowed
        assert "anthropic" in bodies.lower() or "ошибка" in bodies.lower()

    async def test_shadow_failure_does_not_block_main(
        self,
        fake_router: MagicMock,
        fake_settings: MagicMock,
        workspace_root: Path,
    ) -> None:
        from src.llm.protocols import LLMError

        fake_router.generate_oneoff = AsyncMock(
            side_effect=LLMError("openrouter exploded")
        )
        set_compare_deps(
            router=fake_router, settings=fake_settings, workspace_root=workspace_root
        )
        msg = _make_message("/compare hi")

        await handle_compare(msg)

        assert msg.answer.await_count == 2
        bodies = "\n".join(call.args[0] for call in msg.answer.await_args_list)
        assert "ответ A" in bodies
        assert "openrouter" in bodies.lower() or "ошибка" in bodies.lower()

"""Two guardrails for non-admin chats:

1. Legacy prompt-oriented CapabilityRegistry must not be accepted as a chat
   response truth source.
2. Post-intercept code must refuse to publish when the caller isn't the
   admin, regardless of what the LLM emitted.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from src.skills.base import AgentContext, SkillResult
from src.skills.chat_response.skill import ChatResponseSkill

_PATCH_SETTINGS = "src.skills.chat_response.skill.get_settings"


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        workspace_path="unused_ws",
        claude_cli_path="claude",
        public_info_about_nikita="x",
        public_contact_nikita="",
        admin_user_id=42,
        chat_assistant_tier="analyst",
        channel_id="@zhvusha",
    )


def _skill_without_legacy_registry() -> ChatResponseSkill:
    return ChatResponseSkill()


def _system(skill: ChatResponseSkill, *, mode: str, interaction_count: int) -> str:
    with patch(_PATCH_SETTINGS, return_value=_settings()):
        return skill._build_system(
            mode,  # type: ignore[arg-type]
            personality_context="",
            public_info="info",
            interaction_count=interaction_count,
            people_context="",
            current_user_id=999 if mode != "personal" else 42,
        )


def test_capabilities_not_enumerated_in_assistant() -> None:
    skill = _skill_without_legacy_registry()
    system = _system(skill, mode="assistant", interaction_count=5)
    assert "SECRET_CAP_A" not in system
    assert "SECRET_CAP_B" not in system


def test_capabilities_not_enumerated_in_intro() -> None:
    skill = _skill_without_legacy_registry()
    system = _system(skill, mode="assistant", interaction_count=1)
    assert "SECRET_CAP_A" not in system


def test_capabilities_not_enumerated_in_social() -> None:
    skill = _skill_without_legacy_registry()
    system = _system(skill, mode="social", interaction_count=0)
    assert "SECRET_CAP_A" not in system


def test_capabilities_not_enumerated_in_personal_either() -> None:
    """Personal mode never carried capabilities_block; assert it stays that way."""
    skill = _skill_without_legacy_registry()
    system = _system(skill, mode="personal", interaction_count=5)
    assert "SECRET_CAP_A" not in system


def test_chat_response_rejects_legacy_capability_registry_argument() -> None:
    try:
        ChatResponseSkill(capability_registry=object())  # type: ignore[call-arg]
    except TypeError as exc:
        assert "capability_registry" in str(exc)
    else:  # pragma: no cover - explicit assertion readability
        raise AssertionError("legacy CapabilityRegistry argument was accepted")


def test_post_intercept_blocked_for_non_admin() -> None:
    channel = MagicMock()
    channel.execute = AsyncMock()
    skill = ChatResponseSkill(channel_skill=channel)
    ctx = AgentContext(user_id=999, chat_id=999, mode="assistant", message_id=1)
    llm_output = "Вот черновик:\n/post Текст поста"
    with patch(_PATCH_SETTINGS, return_value=_settings()):
        result = asyncio.run(skill._intercept_post_command(llm_output, ctx))
    channel.execute.assert_not_awaited()
    assert "/post" not in result
    assert "Текст поста" not in result


def test_post_intercept_allowed_for_admin_in_personal() -> None:
    channel = MagicMock()
    channel.execute = AsyncMock(return_value=SkillResult(success=True, response="ok"))
    skill = ChatResponseSkill(channel_skill=channel)
    ctx = AgentContext(user_id=42, chat_id=42, mode="personal", message_id=1)
    llm_output = "Готово!\n/post Обычный пост"
    with patch(_PATCH_SETTINGS, return_value=_settings()):
        result = asyncio.run(skill._intercept_post_command(llm_output, ctx))
    channel.execute.assert_awaited_once()
    assert "Опубликовано" in result


def test_post_intercept_uses_side_effect_invoker_when_present() -> None:
    channel = MagicMock()
    channel.execute = AsyncMock(return_value=SkillResult(success=True, response="ok"))
    invoker = AsyncMock(
        return_value=SkillResult(
            success=True,
            response="Нужно решение перед выполнением.",
            metadata={"approval_pending": True},
        )
    )
    skill = ChatResponseSkill(channel_skill=channel, side_effect_invoker=invoker)
    ctx = AgentContext(user_id=42, chat_id=42, mode="personal", message_id=1)
    llm_output = "Готово!\n/post Обычный пост"

    with patch(_PATCH_SETTINGS, return_value=_settings()):
        result = asyncio.run(skill._intercept_post_command(llm_output, ctx))

    channel.execute.assert_not_awaited()
    invoker.assert_awaited_once_with("/post Обычный пост", ctx)
    assert "Нужно решение" in result
    assert "/post" not in result


def test_telegram_mcp_intercept_uses_side_effect_invoker_for_admin() -> None:
    invoker = AsyncMock(
        return_value=SkillResult(
            success=True,
            response="Нужно решение перед выполнением.",
            metadata={"approval_pending": True},
        )
    )
    skill = ChatResponseSkill(side_effect_invoker=invoker)
    ctx = AgentContext(user_id=42, chat_id=42, mode="personal", message_id=1)
    llm_output = "Могу так:\n/telegram_send @nikita | привет"

    with patch(_PATCH_SETTINGS, return_value=_settings()):
        result = asyncio.run(skill._intercept_telegram_mcp_command(llm_output, ctx))

    invoker.assert_awaited_once_with("/telegram_send @nikita | привет", ctx)
    assert "Нужно решение" in result
    assert "/telegram_send" not in result


def test_telegram_mcp_intercept_blocked_for_non_admin() -> None:
    invoker = AsyncMock()
    skill = ChatResponseSkill(side_effect_invoker=invoker)
    ctx = AgentContext(user_id=999, chat_id=999, mode="assistant", message_id=1)
    llm_output = "Попробую:\n/telegram_send @nikita | приватный текст"

    with patch(_PATCH_SETTINGS, return_value=_settings()):
        result = asyncio.run(skill._intercept_telegram_mcp_command(llm_output, ctx))

    invoker.assert_not_awaited()
    assert "/telegram_send" not in result
    assert "приватный текст" not in result
    assert "личный Telegram" in result


def test_personal_creator_prompt_gets_telegram_mcp_command_protocol() -> None:
    skill = ChatResponseSkill()
    skill.set_manager_capability_summary(
        "## Внутренний граф возможностей\n"
        "- agent_profile.telegram_mcp.personal_actions: available"
    )

    system = _system(skill, mode="personal", interaction_count=5)

    assert "/telegram_send <chat_id_or_username> | <text>" in system
    assert "если статус disabled/degraded/configured_only" in system


def test_personal_creator_prompt_gets_computer_use_command_protocol() -> None:
    skill = ChatResponseSkill()
    skill.set_manager_capability_summary(
        "## Внутренний граф возможностей\n"
        "- agent_profile.computer_use.active_gui: available"
    )

    system = _system(skill, mode="personal", interaction_count=5)

    assert '/computer_use {"action":"browser_status"}' in system
    assert '"action":"desktop_app_launcher"' in system
    assert '"action":"desktop_media_control"' in system
    assert "не говори, что в chat tools нет живого браузера" in system


def test_personal_creator_prompt_uses_structured_computer_use_when_enabled() -> None:
    skill = ChatResponseSkill(side_effect_invoker=AsyncMock())
    skill.set_manager_capability_summary(
        "## Внутренний граф возможностей\n"
        "- agent_profile.computer_use.active_gui: available"
    )

    with patch(_PATCH_SETTINGS, return_value=_settings()):
        system = skill._build_system(
            "personal",  # type: ignore[arg-type]
            personality_context="",
            public_info="info",
            interaction_count=5,
            people_context="",
            current_user_id=42,
            prefer_structured_computer_use=True,
        )

    assert "structured tool `computer_use`" in system
    assert "Выбирай один scoped action" in system
    assert '/computer_use {"action"' not in system


def test_computer_use_intercept_uses_side_effect_invoker_for_admin() -> None:
    invoker = AsyncMock(
        return_value=SkillResult(
            success=True,
            response="",
            metadata={
                "requires_zhvusha_response": True,
                "body_observation": {"event": "computer_use_action_completed"},
            },
        )
    )
    skill = ChatResponseSkill(side_effect_invoker=invoker)
    ctx = AgentContext(user_id=42, chat_id=42, mode="personal", message_id=1)
    llm_output = 'Могу так:\n/computer_use {"action":"browser_status"}'

    with patch(_PATCH_SETTINGS, return_value=_settings()):
        result = asyncio.run(skill._intercept_computer_use_command(llm_output, ctx))

    invoker.assert_awaited_once_with(
        '/computer_use {"action":"browser_status"}',
        ctx,
    )
    assert isinstance(result, SkillResult)
    assert result.metadata["requires_zhvusha_response"] is True


def test_side_effect_intercepts_disabled_for_body_observation_synthesis() -> None:
    invoker = AsyncMock()
    skill = ChatResponseSkill(side_effect_invoker=invoker)
    ctx = AgentContext(
        user_id=42,
        chat_id=42,
        mode="personal",
        message_id=1,
        metadata={"disable_side_effect_intercepts": True},
    )
    llm_output = 'Нужно продолжить:\n/computer_use {"action":"browser_status"}'

    result = asyncio.run(skill._intercept_computer_use_command(llm_output, ctx))

    invoker.assert_not_awaited()
    assert result == llm_output


def test_computer_use_intercept_blocked_for_non_admin() -> None:
    invoker = AsyncMock()
    skill = ChatResponseSkill(side_effect_invoker=invoker)
    ctx = AgentContext(user_id=999, chat_id=999, mode="assistant", message_id=1)
    llm_output = 'Попробую:\n/computer_use {"action":"browser_status"}'

    with patch(_PATCH_SETTINGS, return_value=_settings()):
        result = asyncio.run(skill._intercept_computer_use_command(llm_output, ctx))

    invoker.assert_not_awaited()
    assert "/computer_use" not in result
    assert "живым браузером" in result

import pytest
from src.bot.main import (
    _ADMIN_BOT_COMMAND_SPECS,
    _PUBLIC_BOT_COMMAND_SPECS,
    _validate_bot_command_specs,
)


def test_registered_bot_command_names_are_telegram_valid() -> None:
    _validate_bot_command_specs(_PUBLIC_BOT_COMMAND_SPECS)
    _validate_bot_command_specs(_ADMIN_BOT_COMMAND_SPECS)


def test_visible_bot_command_menu_is_chat_first() -> None:
    public_command_names = {
        command for command, _description in _PUBLIC_BOT_COMMAND_SPECS
    }
    admin_command_names = {
        command for command, _description in _ADMIN_BOT_COMMAND_SPECS
    }

    assert public_command_names == {"start"}
    assert admin_command_names == {"start", "code", "morning", "restart", "kwork"}
    hidden_legacy_commands = {
        "archive_lookup",
        "browser_workflow_draft",
        "capability_status",
        "compare",
        "computer_status",
        "dialogue_status",
        "external_skill_smoke",
        "hermes_baseline_status",
        "kwork_status",
        "post",
        "post_draft",
        "post_drafts",
        "process_status",
        "proposal",
        "runtime_status",
        "self_coding",
        "sleep",
        "social_permissions",
        "spec",
        "spec_create",
        "spec_run",
        "telegram_read",
        "telegram_send",
        "topic_to_spec",
        "wake",
        "код",
        "самокодинг",
    }
    assert admin_command_names.isdisjoint(hidden_legacy_commands)


def test_bot_command_validation_rejects_cyrillic_command_name() -> None:
    with pytest.raises(ValueError):
        _validate_bot_command_specs((("самокодинг", "Нельзя зарегистрировать"),))

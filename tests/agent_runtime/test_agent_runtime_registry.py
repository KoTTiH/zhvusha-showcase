"""Registry contract tests for Agent Runtime profiles and capabilities."""

from __future__ import annotations

import pytest


def test_agent_registry_builds_invocation_profile_from_definition() -> None:
    from src.agent_runtime.models import AgentDefinition
    from src.agent_runtime.registry import AgentRegistry

    registry = AgentRegistry(
        agents=(
            AgentDefinition(
                id="source_compare",
                purpose="compare source with repo",
                default_worker="codex_cli",
                allowed_capabilities=("read_code", "read_attachments"),
            ),
        )
    )

    profile = registry.make_invocation_profile(
        "source_compare",
        allowed_capabilities=("read_code",),
        denied_capabilities=("write_files",),
        metadata={"model": "gpt-5.5"},
    )

    assert profile.id == "source_compare.v1"
    assert profile.worker == "codex_cli"
    assert profile.allowed_capabilities == ("read_code",)
    assert profile.denied_capabilities == ("write_files",)
    assert profile.metadata["model"] == "gpt-5.5"


def test_agent_registry_rejects_capabilities_outside_definition() -> None:
    from src.agent_runtime.models import AgentDefinition
    from src.agent_runtime.registry import AgentRegistry

    registry = AgentRegistry(
        agents=(
            AgentDefinition(
                id="source_compare",
                purpose="compare source with repo",
                default_worker="codex_cli",
                allowed_capabilities=("read_code",),
            ),
        )
    )

    with pytest.raises(ValueError, match="not allowed"):
        registry.make_invocation_profile(
            "source_compare",
            allowed_capabilities=("read_code", "write_files"),
        )


def test_capability_registry_validates_profile_capabilities() -> None:
    from src.agent_runtime.models import CapabilityDefinition, InvocationProfile
    from src.agent_runtime.registry import CapabilityRegistry

    registry = CapabilityRegistry(
        capabilities=(
            CapabilityDefinition(id="read_code"),
            CapabilityDefinition(id="read_attachments"),
        )
    )

    registry.validate_profile(
        InvocationProfile(
            id="readonly",
            allowed_capabilities=("read_code",),
            denied_capabilities=("read_attachments",),
        )
    )

    with pytest.raises(ValueError, match="unknown capability"):
        registry.validate_profile(
            InvocationProfile(id="bad", allowed_capabilities=("restart",))
        )


def test_builtin_registries_validate_all_profiles() -> None:
    from src.agent_runtime.profiles import (
        AGENCY_READONLY_DRAFT,
        COMPUTER_USE_ACTIVE_GUI,
        COMPUTER_USE_AGENT,
        COMPUTER_USE_APPROVED_SHELL,
        DAIVINCHIK_AUTOLIKE_AGENT,
        DAIVINCHIK_AUTOLIKE_BOT_COMMAND,
        DAIVINCHIK_AUTOLIKE_MVP,
        DAIVINCHIK_TASTE_PROFILE_AGENT,
        DAIVINCHIK_TASTE_PROFILE_READONLY,
        DESKTOP_CONTROL_AGENT,
        DESKTOP_CONTROL_CONVENIENCE,
        LIFE_REFLECTION_READONLY,
        LIFE_RUNTIME_AGENT,
        SOURCE_COMPARE_READONLY,
        TELEGRAM_MCP_PERSONAL_READONLY,
        WEB_RESEARCH_READONLY,
        build_builtin_agent_registry,
        build_builtin_capability_registry,
    )

    agent_registry = build_builtin_agent_registry()
    capability_registry = build_builtin_capability_registry()

    assert {agent.id for agent in agent_registry.all()} >= {
        "source_compare",
        "self_coding",
        "telegram_mcp_personal",
        "daivinchik_taste_profile",
        "daivinchik_autolike",
        "web_research",
        "agency",
        "life_runtime",
        "desktop_control",
        "computer_use",
    }
    assert COMPUTER_USE_AGENT.default_worker == "computer_use"
    assert COMPUTER_USE_ACTIVE_GUI.worker == "computer_use"
    assert COMPUTER_USE_APPROVED_SHELL.worker == "computer_use"
    assert DESKTOP_CONTROL_AGENT.default_worker == "desktop_control"
    assert DESKTOP_CONTROL_CONVENIENCE.worker == "desktop_control"
    assert DAIVINCHIK_AUTOLIKE_AGENT.default_worker == "daivinchik_taste_profile"
    assert DAIVINCHIK_AUTOLIKE_MVP.worker == "daivinchik_taste_profile"
    assert AGENCY_READONLY_DRAFT.worker == "agency"
    assert DAIVINCHIK_TASTE_PROFILE_AGENT.default_worker == ("daivinchik_taste_profile")
    assert DAIVINCHIK_TASTE_PROFILE_READONLY.worker == "daivinchik_taste_profile"
    assert DAIVINCHIK_AUTOLIKE_BOT_COMMAND.worker == "daivinchik_taste_profile"
    assert LIFE_RUNTIME_AGENT.default_worker == "life_runtime"
    assert LIFE_REFLECTION_READONLY.worker == "life_runtime"
    assert SOURCE_COMPARE_READONLY.worker == "source_compare"
    assert WEB_RESEARCH_READONLY.worker == "web_research"
    assert TELEGRAM_MCP_PERSONAL_READONLY.worker == "telegram_mcp"
    capability_registry.validate_profile(AGENCY_READONLY_DRAFT)
    capability_registry.validate_profile(COMPUTER_USE_ACTIVE_GUI)
    capability_registry.validate_profile(COMPUTER_USE_APPROVED_SHELL)
    capability_registry.validate_profile(DESKTOP_CONTROL_CONVENIENCE)
    capability_registry.validate_profile(LIFE_REFLECTION_READONLY)
    capability_registry.validate_profile(WEB_RESEARCH_READONLY)
    capability_registry.validate_profile(TELEGRAM_MCP_PERSONAL_READONLY)
    capability_registry.validate_profile(DAIVINCHIK_TASTE_PROFILE_READONLY)
    capability_registry.validate_profile(DAIVINCHIK_AUTOLIKE_MVP)
    capability_registry.validate_profile(DAIVINCHIK_AUTOLIKE_BOT_COMMAND)
    assert "send_message" in AGENCY_READONLY_DRAFT.denied_capabilities
    assert "desktop.shell" in DESKTOP_CONTROL_CONVENIENCE.denied_capabilities
    assert "desktop.powershell" in DESKTOP_CONTROL_CONVENIENCE.denied_capabilities
    assert "desktop.shell" in COMPUTER_USE_ACTIVE_GUI.denied_capabilities
    assert "desktop.powershell" in COMPUTER_USE_ACTIVE_GUI.denied_capabilities
    assert "desktop.shell" in COMPUTER_USE_APPROVED_SHELL.allowed_capabilities
    assert "browser_click" in COMPUTER_USE_APPROVED_SHELL.denied_capabilities
    assert "browser_click" in COMPUTER_USE_ACTIVE_GUI.allowed_capabilities
    assert "desktop_input" in COMPUTER_USE_ACTIVE_GUI.allowed_capabilities
    assert "send_message" in LIFE_REFLECTION_READONLY.denied_capabilities
    assert "write_files" in LIFE_REFLECTION_READONLY.denied_capabilities
    assert "telegram_mcp_send" in AGENCY_READONLY_DRAFT.denied_capabilities
    assert "browser_submit" in WEB_RESEARCH_READONLY.denied_capabilities
    assert "telegram_mcp_send" in TELEGRAM_MCP_PERSONAL_READONLY.denied_capabilities
    assert "telegram_mcp_send" in DAIVINCHIK_TASTE_PROFILE_READONLY.denied_capabilities
    assert "telegram_mcp_send" in DAIVINCHIK_AUTOLIKE_MVP.denied_capabilities
    assert "telegram_mcp_modify" in DAIVINCHIK_AUTOLIKE_MVP.denied_capabilities
    assert "telegram_mcp_daivinchik_button" in (
        DAIVINCHIK_AUTOLIKE_MVP.allowed_capabilities
    )
    assert "telegram_mcp_daivinchik_reply_button" in (
        DAIVINCHIK_AUTOLIKE_MVP.allowed_capabilities
    )
    assert "telegram_mcp_daivinchik_notify" in (
        DAIVINCHIK_AUTOLIKE_MVP.allowed_capabilities
    )
    assert "telegram_mcp_daivinchik_button" in (
        DAIVINCHIK_AUTOLIKE_BOT_COMMAND.allowed_capabilities
    )
    assert "telegram_mcp_daivinchik_reply_button" in (
        DAIVINCHIK_AUTOLIKE_BOT_COMMAND.allowed_capabilities
    )
    assert "telegram_mcp_daivinchik_notify" in (
        DAIVINCHIK_AUTOLIKE_BOT_COMMAND.allowed_capabilities
    )
    assert "telegram_mcp_media_read" in (
        DAIVINCHIK_TASTE_PROFILE_READONLY.allowed_capabilities
    )
    assert "personal" in TELEGRAM_MCP_PERSONAL_READONLY.metadata["account"]

"""Unified CapabilityGraph contract tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from src.agent_runtime.models import InvocationProfile
from src.agent_runtime.profiles import (
    AGENCY_AGENT,
    AGENCY_READONLY_DRAFT,
    COMPUTER_USE_ACTIVE_GUI,
    COMPUTER_USE_AGENT,
    DESKTOP_CONTROL_AGENT,
    DESKTOP_CONTROL_CONVENIENCE,
    LIFE_REFLECTION_READONLY,
    LIFE_RUNTIME_AGENT,
    SELF_CODING_READONLY,
    SELF_IMPROVEMENT_AGENT,
    SELF_IMPROVEMENT_AUTONOMOUS,
    TELEGRAM_MCP_PERSONAL_ACTIONS,
    TELEGRAM_MCP_PERSONAL_AGENT,
    TELEGRAM_MCP_PERSONAL_READONLY,
)
from src.agent_runtime.tools import FunctionAgentTool, ToolGateway


def _settings(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "telegram_mcp_enabled": False,
        "telegram_mcp_session_string_personal": "",
        "telegram_mcp_session_name_personal": "",
        "daemon_enabled": False,
        "news_sources_enabled": False,
        "self_coding_enabled": False,
        "autonomous_self_coding_enabled": False,
        "agency_runtime_enabled": False,
        "agency_social_autonomy_enabled": False,
        "life_runtime_enabled": False,
        "voice_gateway_enabled": False,
        "voice_stt_provider": "",
        "voice_tts_enabled": False,
        "desktop_control_enabled": False,
        "computer_use_enabled": False,
        "live_browser_control_enabled": False,
        "live_browser_auto_launch": False,
        "computer_use_shell_enabled": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_telegram_mcp_profile_is_configured_only_without_worker() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityStatus,
        build_capability_graph,
    )

    graph = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(
            telegram_mcp_enabled=True,
            telegram_mcp_session_string_personal="super_secret_session",
        ),
        active_skill_names=(),
        startup_skill_names=(),
        agent_definitions=(TELEGRAM_MCP_PERSONAL_AGENT,),
        invocation_profiles=(
            TELEGRAM_MCP_PERSONAL_READONLY,
            TELEGRAM_MCP_PERSONAL_ACTIONS,
        ),
        registered_worker_names=(),
        tool_gateways=(),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )

    readonly = graph.require("agent_profile.telegram_mcp.personal_readonly")
    actions = graph.require("agent_profile.telegram_mcp.personal_actions")
    agent = graph.require("agent_definition.telegram_mcp_personal")

    assert readonly.status is CapabilityStatus.CONFIGURED_ONLY
    assert actions.status is CapabilityStatus.CONFIGURED_ONLY
    assert agent.status is CapabilityStatus.CONFIGURED_ONLY
    assert "telegram_mcp" in readonly.reason
    assert "super_secret_session" not in graph.format_manager_summary()
    graph.assert_available_profiles_have_registered_workers()


def test_telegram_mcp_degrades_when_worker_exists_but_personal_session_is_missing() -> (
    None
):
    from src.agent_runtime.capability_graph import (
        CapabilityStatus,
        build_capability_graph,
    )

    graph = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(telegram_mcp_enabled=True),
        active_skill_names=(),
        startup_skill_names=(),
        invocation_profiles=(TELEGRAM_MCP_PERSONAL_READONLY,),
        registered_worker_names=("telegram_mcp",),
        tool_gateways=(),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )

    node = graph.require("agent_profile.telegram_mcp.personal_readonly")

    assert node.status is CapabilityStatus.DEGRADED
    assert "session" in node.reason.lower()


def test_telegram_mcp_profile_available_when_worker_session_and_tools_exist() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityStatus,
        build_capability_graph,
    )

    async def noop(_payload: dict[str, Any]) -> str:
        return "ok"

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("telegram_mcp_call_read", "telegram_mcp_read", noop),
            FunctionAgentTool("telegram_mcp_send_message", "telegram_mcp_send", noop),
            FunctionAgentTool("telegram_mcp_call_modify", "telegram_mcp_modify", noop),
            FunctionAgentTool("telegram_mcp_call_admin", "telegram_mcp_admin", noop),
            FunctionAgentTool(
                "telegram_mcp_call_media",
                "telegram_mcp_media_files",
                noop,
            ),
        )
    )
    graph = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(
            telegram_mcp_enabled=True,
            telegram_mcp_session_string_personal="super_secret_session",
        ),
        active_skill_names=("telegram_mcp_personal",),
        startup_skill_names=("telegram_mcp_personal",),
        agent_definitions=(TELEGRAM_MCP_PERSONAL_AGENT,),
        invocation_profiles=(
            TELEGRAM_MCP_PERSONAL_READONLY,
            TELEGRAM_MCP_PERSONAL_ACTIONS,
        ),
        registered_worker_names=("telegram_mcp",),
        tool_gateways=(gateway,),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )

    assert graph.require("agent_profile.telegram_mcp.personal_readonly").status is (
        CapabilityStatus.AVAILABLE
    )
    assert graph.require("agent_profile.telegram_mcp.personal_actions").status is (
        CapabilityStatus.AVAILABLE
    )
    assert graph.require(
        "agent_capability.telegram_mcp.personal_actions.telegram_mcp_send"
    ).status is (CapabilityStatus.AVAILABLE)
    assert graph.require_tool("telegram_mcp_send_message").requires_approval is True
    assert "super_secret_session" not in graph.format_manager_summary()
    graph.assert_available_profiles_have_registered_workers()


def test_available_agent_profile_requires_registered_worker_and_tool_surface() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityStatus,
        build_capability_graph,
    )

    async def read_workspace(_payload: dict[str, Any]) -> str:
        return "ok"

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool(
                "read_workspace_file",
                "read_workspace",
                read_workspace,
            ),
        )
    )
    graph = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(),
        active_skill_names=(),
        startup_skill_names=(),
        invocation_profiles=(
            SELF_CODING_READONLY,
            InvocationProfile(
                id="demo.missing_worker",
                worker="missing_worker",
                allowed_capabilities=("read_workspace",),
            ),
        ),
        registered_worker_names=("codex_cli",),
        tool_gateways=(gateway,),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )

    assert graph.require("agent_profile.self_coding.readonly_discussion").status is (
        CapabilityStatus.AVAILABLE
    )
    assert graph.require("agent_profile.demo.missing_worker").status is (
        CapabilityStatus.CONFIGURED_ONLY
    )
    assert graph.require_tool("read_workspace_file").status is (
        CapabilityStatus.AVAILABLE
    )
    assert graph.tool_graph().require("read_workspace_file").status is (
        CapabilityStatus.AVAILABLE
    )
    graph.assert_available_profiles_have_registered_workers()


def test_skill_manifest_orphan_detection_requires_registration_or_disabled_status(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityStatus,
        build_capability_graph,
    )

    skill_root = tmp_path / "skills"
    (skill_root / "orphan").mkdir(parents=True)
    (skill_root / "orphan" / "skill.yaml").write_text(
        "\n".join(
            [
                "name: orphan_skill",
                "description: Orphan production skill",
                'version: "0.1.0"',
                "type: inline",
                "llm_tier: worker",
                "source: manual",
            ]
        ),
        encoding="utf-8",
    )
    (skill_root / "disabled").mkdir()
    (skill_root / "disabled" / "skill.yaml").write_text(
        "\n".join(
            [
                "name: disabled_skill",
                "description: Disabled skill",
                'version: "0.1.0"',
                "type: inline",
                "llm_tier: worker",
                "source: manual",
                "status: disabled",
                "disabled_reason: intentionally off",
            ]
        ),
        encoding="utf-8",
    )

    graph = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(),
        active_skill_names=(),
        startup_skill_names=(),
        skill_manifest_root=skill_root,
        invocation_profiles=(),
        registered_worker_names=(),
        tool_gateways=(),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )

    assert graph.require("skill.orphan_skill").status is CapabilityStatus.ORPHANED
    assert graph.require("skill.disabled_skill").status is CapabilityStatus.DISABLED
    with pytest.raises(AssertionError, match="orphan_skill"):
        graph.assert_no_required_skill_orphans()


def test_active_experimental_skill_manifest_is_reported_as_available(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityStatus,
        build_capability_graph,
    )

    skill_root = tmp_path / "skills"
    (skill_root / "external_skill_acquisition").mkdir(parents=True)
    (skill_root / "external_skill_acquisition" / "skill.yaml").write_text(
        "\n".join(
            [
                "name: external_skill_acquisition",
                "description: External skill acquisition",
                'version: "0.1.0"',
                "type: inline",
                "llm_tier: worker",
                "source: manual",
                "status: experimental",
            ]
        ),
        encoding="utf-8",
    )

    graph = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(),
        active_skill_names=("external_skill_acquisition",),
        startup_skill_names=("external_skill_acquisition",),
        skill_manifest_root=skill_root,
        invocation_profiles=(),
        registered_worker_names=(),
        tool_gateways=(),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )

    node = graph.require("skill.external_skill_acquisition")
    assert node.status is CapabilityStatus.AVAILABLE
    assert "active bot dispatcher" in node.reason
    assert "experimental" in node.reason


def test_external_skill_records_are_visible_but_not_available_until_approved(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityStatus,
        build_capability_graph,
    )
    from src.skills.external_skill_loader.loader import (
        ExternalSkillAuditReport,
        ExternalSkillInventory,
        ExternalSkillPackage,
        ExternalSkillSource,
        ExternalSkillStatus,
        PersonalSkillRegistryRecord,
        QuarantinedExternalSkill,
    )

    source = ExternalSkillSource(
        source_type="local_folder",
        locator=str(tmp_path / "source"),
    )
    package = ExternalSkillPackage(
        skill_id="kube-debug",
        name="kube-debug",
        source=source,
        root_path=str(tmp_path),
        inventory=ExternalSkillInventory(skill_markdown="SKILL.md"),
    )
    quarantined = QuarantinedExternalSkill(
        skill_id="kube-debug",
        name="kube-debug",
        source=source,
        quarantine_path=str(tmp_path / "quarantine" / "kube-debug"),
        content_hash="abc",
        status=ExternalSkillStatus.QUARANTINED,
        package=package,
    )
    needs_review = PersonalSkillRegistryRecord(
        skill_id="needs-review",
        name="needs-review",
        source=source,
        quarantine_path=str(tmp_path / "quarantine" / "needs-review"),
        content_hash="def",
        status=ExternalSkillStatus.NEEDS_REVIEW,
        audit_report=ExternalSkillAuditReport(
            skill_id="needs-review",
            name="needs-review",
            status=ExternalSkillStatus.NEEDS_REVIEW,
            risk_level="medium",
            read_only_allowed=True,
            execution_allowed=False,
        ),
    )
    blocked = PersonalSkillRegistryRecord(
        skill_id="blocked",
        name="blocked",
        source=source,
        quarantine_path=str(tmp_path / "quarantine" / "blocked"),
        content_hash="ghi",
        status=ExternalSkillStatus.BLOCKED,
        audit_report=ExternalSkillAuditReport(
            skill_id="blocked",
            name="blocked",
            status=ExternalSkillStatus.BLOCKED,
            risk_level="blocked",
            blocked=True,
            read_only_allowed=False,
            execution_allowed=False,
        ),
    )
    approved = PersonalSkillRegistryRecord(
        skill_id="approved",
        name="approved",
        source=source,
        quarantine_path=str(tmp_path / "quarantine" / "approved"),
        content_hash="jkl",
        status=ExternalSkillStatus.APPROVED_READONLY,
        audit_report=ExternalSkillAuditReport(
            skill_id="approved",
            name="approved",
            status=ExternalSkillStatus.NEEDS_REVIEW,
            risk_level="low",
            read_only_allowed=True,
            execution_allowed=False,
        ),
        readonly_approval_id="approval-readonly",
    )

    graph = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(),
        active_skill_names=(),
        startup_skill_names=(),
        invocation_profiles=(),
        registered_worker_names=(),
        tool_gateways=(),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
        external_skill_records=(quarantined, needs_review, blocked, approved),
    )

    assert graph.require("external_skill.kube-debug").status is (
        CapabilityStatus.QUARANTINED
    )
    assert graph.require("external_skill.needs-review").status is (
        CapabilityStatus.NEEDS_REVIEW
    )
    assert graph.require("external_skill.blocked").status is CapabilityStatus.BLOCKED
    assert graph.require("external_skill.approved").status is CapabilityStatus.AVAILABLE
    assert "external_skill.needs-review" in graph.format_manager_summary()


def test_known_hermes_gap_capabilities_are_in_truth_graph() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityStatus,
        build_capability_graph,
    )

    graph = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(),
        active_skill_names=(),
        startup_skill_names=(),
        invocation_profiles=(),
        registered_worker_names=(),
        tool_gateways=(),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )

    node = graph.require("agent_capability.hermes_gap.kubernetes_debug")

    assert node.status is CapabilityStatus.DISABLED
    assert node.capability_id == "kubernetes_debug"
    assert "Hermes roadmap example" in node.reason


def test_curated_external_skill_records_are_visible_but_inactive(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityStatus,
        build_capability_graph,
    )
    from src.skills.external_skill_loader.loader import ExternalSkillStatus

    graph = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(),
        active_skill_names=(),
        startup_skill_names=(),
        invocation_profiles=(),
        registered_worker_names=(),
        tool_gateways=(),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
        external_skill_records=(
            _external_skill_record(
                tmp_path,
                "rejected",
                ExternalSkillStatus.REJECTED,
            ),
            _external_skill_record(
                tmp_path,
                "superseded",
                ExternalSkillStatus.SUPERSEDED,
            ),
            _external_skill_record(
                tmp_path,
                "native-converted",
                ExternalSkillStatus.NATIVE_CONVERTED,
            ),
        ),
    )

    rejected = graph.require("external_skill.rejected")
    superseded = graph.require("external_skill.superseded")
    native_converted = graph.require("external_skill.native-converted")
    assert rejected.status is CapabilityStatus.DISABLED
    assert superseded.status is CapabilityStatus.DISABLED
    assert native_converted.status is CapabilityStatus.DISABLED
    assert "rejected" in rejected.reason
    assert "superseded" in superseded.reason
    assert "native skill" in native_converted.reason


def test_relevant_config_flags_are_consumed_by_graph_nodes() -> None:
    from src.agent_runtime.capability_graph import (
        RELEVANT_CONFIG_FLAGS,
        build_capability_graph,
    )

    graph = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(daemon_enabled=True, news_sources_enabled=True),
        active_skill_names=(),
        startup_skill_names=(),
        invocation_profiles=(TELEGRAM_MCP_PERSONAL_READONLY,),
        registered_worker_names=(),
        tool_gateways=(),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )

    assert {flag.name for flag in graph.config_flags} == set(RELEVANT_CONFIG_FLAGS)
    assert all(flag.consumer_ids for flag in graph.config_flags)
    graph.assert_relevant_config_flags_consumed()


def test_voice_gateway_capability_status_tracks_stt_provider() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityStatus,
        build_capability_graph,
    )

    disabled = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(),
        active_skill_names=(),
        startup_skill_names=(),
        invocation_profiles=(),
        registered_worker_names=(),
        tool_gateways=(),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )
    degraded = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(voice_gateway_enabled=True),
        active_skill_names=(),
        startup_skill_names=(),
        invocation_profiles=(),
        registered_worker_names=(),
        tool_gateways=(),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )
    available = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(
            voice_gateway_enabled=True,
            voice_stt_provider="local_whisper",
            voice_tts_enabled=True,
        ),
        active_skill_names=(),
        startup_skill_names=(),
        invocation_profiles=(),
        registered_worker_names=(),
        tool_gateways=(),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )

    assert disabled.require("config.voice_gateway").status is CapabilityStatus.DISABLED
    assert degraded.require("config.voice_gateway").status is CapabilityStatus.DEGRADED
    assert "STT provider" in degraded.require("config.voice_gateway").reason
    assert (
        available.require("config.voice_gateway").status is CapabilityStatus.AVAILABLE
    )
    assert "voice" in available.format_manager_summary(max_items=80)


def test_desktop_control_profile_is_disabled_until_enabled_and_worker_registered() -> (
    None
):
    from src.agent_runtime.capability_graph import (
        CapabilityStatus,
        build_capability_graph,
    )
    from src.agent_runtime.profiles import DESKTOP_CONTROL_CONVENIENCE
    from src.agent_runtime.tools import FunctionAgentTool, ToolGateway

    async def media_pause(_payload: dict[str, object]) -> str:
        return "paused"

    disabled = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(),
        active_skill_names=(),
        startup_skill_names=(),
        invocation_profiles=(DESKTOP_CONTROL_CONVENIENCE,),
        registered_worker_names=(),
        tool_gateways=(),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )
    configured = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(desktop_control_enabled=True),
        active_skill_names=(),
        startup_skill_names=(),
        invocation_profiles=(DESKTOP_CONTROL_CONVENIENCE,),
        registered_worker_names=(),
        tool_gateways=(),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )
    missing_tool = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(desktop_control_enabled=True),
        active_skill_names=(),
        startup_skill_names=(),
        invocation_profiles=(DESKTOP_CONTROL_CONVENIENCE,),
        registered_worker_names=("desktop_control",),
        tool_gateways=(),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )
    available = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(desktop_control_enabled=True),
        active_skill_names=(),
        startup_skill_names=(),
        invocation_profiles=(DESKTOP_CONTROL_CONVENIENCE,),
        registered_worker_names=("desktop_control",),
        tool_gateways=(
            ToolGateway(
                tools=(
                    FunctionAgentTool(
                        "desktop_media_control",
                        "desktop_media_control",
                        media_pause,
                    ),
                )
            ),
        ),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )

    assert disabled.require("agent_profile.desktop_control.convenience").status is (
        CapabilityStatus.DISABLED
    )
    assert configured.require("agent_profile.desktop_control.convenience").status is (
        CapabilityStatus.CONFIGURED_ONLY
    )
    assert configured.require("config.desktop_control").status is (
        CapabilityStatus.DEGRADED
    )
    assert missing_tool.require("config.desktop_control").status is (
        CapabilityStatus.DEGRADED
    )
    assert (
        missing_tool.require(
            "agent_capability.desktop_control.convenience.desktop_media_control"
        ).status
        is CapabilityStatus.CONFIGURED_ONLY
    )
    assert available.require("config.desktop_control").status is (
        CapabilityStatus.AVAILABLE
    )
    assert (
        available.require(
            "agent_capability.desktop_control.convenience.desktop_media_control"
        ).status
        is CapabilityStatus.AVAILABLE
    )


def test_computer_use_profile_tracks_config_worker_and_live_browser_tools() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityStatus,
        build_capability_graph,
    )
    from src.agent_runtime.tools import FunctionAgentTool, ToolGateway

    async def action(_payload: dict[str, object]) -> str:
        return "ok"

    disabled = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(),
        active_skill_names=(),
        startup_skill_names=(),
        agent_definitions=(COMPUTER_USE_AGENT,),
        invocation_profiles=(COMPUTER_USE_ACTIVE_GUI,),
        registered_worker_names=("computer_use",),
        tool_gateways=(),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )
    missing_tools = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(
            computer_use_enabled=True,
            live_browser_control_enabled=True,
        ),
        active_skill_names=(),
        startup_skill_names=(),
        agent_definitions=(COMPUTER_USE_AGENT,),
        invocation_profiles=(COMPUTER_USE_ACTIVE_GUI,),
        registered_worker_names=("computer_use",),
        tool_gateways=(),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )
    available = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(
            computer_use_enabled=True,
            live_browser_control_enabled=True,
        ),
        active_skill_names=(),
        startup_skill_names=(),
        agent_definitions=(COMPUTER_USE_AGENT,),
        invocation_profiles=(COMPUTER_USE_ACTIVE_GUI,),
        registered_worker_names=("computer_use",),
        tool_gateways=(
            ToolGateway(
                tools=(
                    FunctionAgentTool(
                        "browser_live_click",
                        "browser_click",
                        action,
                    ),
                    FunctionAgentTool(
                        "browser_live_status",
                        "browser_live_control",
                        action,
                    ),
                )
            ),
        ),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )
    auto_launch_ready = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(
            computer_use_enabled=True,
            live_browser_control_enabled=True,
            live_browser_auto_launch=True,
        ),
        active_skill_names=(),
        startup_skill_names=(),
        agent_definitions=(COMPUTER_USE_AGENT,),
        invocation_profiles=(COMPUTER_USE_ACTIVE_GUI,),
        registered_worker_names=("computer_use",),
        tool_gateways=(
            ToolGateway(
                tools=(
                    FunctionAgentTool(
                        "browser_live_click",
                        "browser_click",
                        action,
                    ),
                    FunctionAgentTool(
                        "browser_live_status",
                        "browser_live_control",
                        action,
                    ),
                )
            ),
        ),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )

    assert disabled.require("agent_profile.computer_use.active_gui").status is (
        CapabilityStatus.DISABLED
    )
    assert disabled.require("config.computer_use").status is CapabilityStatus.DISABLED
    assert missing_tools.require("config.computer_use").status is (
        CapabilityStatus.DEGRADED
    )
    assert missing_tools.require("config.live_browser_control").status is (
        CapabilityStatus.AVAILABLE
    )
    assert missing_tools.require("runtime.live_browser_adapter").status is (
        CapabilityStatus.CONFIGURED_ONLY
    )
    assert missing_tools.require("runtime.live_browser_health").status is (
        CapabilityStatus.CONFIGURED_ONLY
    )
    assert available.require("agent_profile.computer_use.active_gui").status is (
        CapabilityStatus.AVAILABLE
    )
    assert available.require("config.computer_use").status is (
        CapabilityStatus.AVAILABLE
    )
    assert available.require("config.live_browser_control").status is (
        CapabilityStatus.AVAILABLE
    )
    assert available.require("runtime.live_browser_adapter").status is (
        CapabilityStatus.AVAILABLE
    )
    assert available.require("runtime.live_browser_health").status is (
        CapabilityStatus.DEGRADED
    )
    assert auto_launch_ready.require("runtime.live_browser_health").status is (
        CapabilityStatus.AVAILABLE
    )
    assert (
        available.require(
            "agent_capability.computer_use.active_gui.browser_click"
        ).status
        is CapabilityStatus.AVAILABLE
    )


def test_agency_runtime_profile_is_disabled_by_default() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityStatus,
        build_capability_graph,
    )

    graph = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(),
        active_skill_names=(),
        startup_skill_names=(),
        invocation_profiles=(AGENCY_READONLY_DRAFT,),
        registered_worker_names=("agency",),
        tool_gateways=(),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )

    assert graph.require("agent_profile.agency.readonly_draft").status is (
        CapabilityStatus.DISABLED
    )
    assert graph.require("config.agency_runtime").status is CapabilityStatus.DISABLED
    assert graph.require("config.agency_social_autonomy").status is (
        CapabilityStatus.DISABLED
    )
    graph.assert_available_profiles_have_registered_workers()


def test_life_runtime_profiles_are_not_available_without_worker() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityStatus,
        build_capability_graph,
    )

    graph = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(life_runtime_enabled=True),
        active_skill_names=(),
        startup_skill_names=(),
        agent_definitions=(LIFE_RUNTIME_AGENT,),
        invocation_profiles=(LIFE_REFLECTION_READONLY,),
        registered_worker_names=(),
        tool_gateways=(),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )

    assert graph.require("agent_definition.life_runtime").status is (
        CapabilityStatus.CONFIGURED_ONLY
    )
    assert graph.require("agent_profile.life_reflection.readonly").status is (
        CapabilityStatus.CONFIGURED_ONLY
    )
    assert graph.require("config.life_runtime").status is CapabilityStatus.DEGRADED
    graph.assert_available_profiles_have_registered_workers()


def test_life_runtime_profile_is_disabled_by_default() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityStatus,
        build_capability_graph,
    )

    graph = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(),
        active_skill_names=(),
        startup_skill_names=(),
        invocation_profiles=(LIFE_REFLECTION_READONLY,),
        registered_worker_names=("life_runtime",),
        tool_gateways=(),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )

    assert graph.require("agent_profile.life_reflection.readonly").status is (
        CapabilityStatus.DISABLED
    )
    assert graph.require("config.life_runtime").status is CapabilityStatus.DISABLED


def test_flag_gated_agent_definitions_are_disabled_when_feature_flag_is_off() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityStatus,
        build_capability_graph,
    )

    graph = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(),
        active_skill_names=(),
        startup_skill_names=(),
        agent_definitions=(
            SELF_IMPROVEMENT_AGENT,
            AGENCY_AGENT,
            LIFE_RUNTIME_AGENT,
            DESKTOP_CONTROL_AGENT,
        ),
        invocation_profiles=(
            SELF_IMPROVEMENT_AUTONOMOUS,
            AGENCY_READONLY_DRAFT,
            LIFE_REFLECTION_READONLY,
            DESKTOP_CONTROL_CONVENIENCE,
        ),
        registered_worker_names=(
            "self_improvement",
            "agency",
            "life_runtime",
            "desktop_control",
        ),
        tool_gateways=(),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )

    assert graph.require("agent_definition.self_improvement").status is (
        CapabilityStatus.DISABLED
    )
    assert graph.require("agent_definition.agency").status is (
        CapabilityStatus.DISABLED
    )
    assert graph.require("agent_definition.life_runtime").status is (
        CapabilityStatus.DISABLED
    )
    assert graph.require("agent_definition.desktop_control").status is (
        CapabilityStatus.DISABLED
    )


def test_agency_runtime_profile_available_with_flag_and_worker() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityStatus,
        build_capability_graph,
    )

    graph = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(
            agency_runtime_enabled=True,
            agency_social_autonomy_enabled=True,
        ),
        active_skill_names=(),
        startup_skill_names=(),
        invocation_profiles=(AGENCY_READONLY_DRAFT,),
        registered_worker_names=("agency",),
        tool_gateways=(),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )

    assert graph.require("agent_profile.agency.readonly_draft").status is (
        CapabilityStatus.AVAILABLE
    )
    assert graph.require(
        "agent_capability.agency.readonly_draft.agency_intent_plan"
    ).status is (CapabilityStatus.AVAILABLE)
    assert graph.require("config.agency_runtime").status is CapabilityStatus.AVAILABLE
    assert graph.require("config.agency_social_autonomy").status is (
        CapabilityStatus.AVAILABLE
    )
    graph.assert_relevant_config_flags_consumed()


def test_manager_capability_summary_is_private_to_creator_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.skills.chat_response.skill import ChatResponseSkill

    monkeypatch.setattr(
        "src.skills.chat_response.skill.get_settings",
        lambda: SimpleNamespace(
            admin_user_id=12345,
            public_contact_nikita="",
        ),
    )
    skill = ChatResponseSkill()
    skill.set_manager_capability_summary(
        "## Внутренний граф возможностей\n- agent_profile.telegram_mcp.personal_readonly: configured_only"
    )

    creator_system = skill._build_system(
        "personal",
        personality_context="",
        public_info="",
        people_context="Никита",
        current_user_id=12345,
    )
    stranger_system = skill._build_system(
        "assistant",
        personality_context="",
        public_info="",
        interaction_count=3,
        current_user_id=999,
    )

    assert "agent_profile.telegram_mcp.personal_readonly" in creator_system
    assert "agent_profile.telegram_mcp.personal_readonly" not in stranger_system


def test_manager_summary_pins_available_telegram_mcp_profiles() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityGraph,
        CapabilityKind,
        CapabilityNode,
        CapabilityStatus,
    )

    noisy_nodes = tuple(
        CapabilityNode(
            id=f"agent_capability.noisy.profile_{index}.read_workspace",
            label=f"noise {index}",
            kind=CapabilityKind.AGENT_CAPABILITY,
            status=CapabilityStatus.CONFIGURED_ONLY,
            reason="noisy configured-only capability",
        )
        for index in range(40)
    )
    graph = CapabilityGraph(
        capabilities=(
            *noisy_nodes,
            CapabilityNode(
                id="agent_profile.telegram_mcp.personal_actions",
                label="telegram_mcp.personal_actions",
                kind=CapabilityKind.AGENT_PROFILE,
                status=CapabilityStatus.AVAILABLE,
                reason="worker, MCP server and personal session are configured",
            ),
            CapabilityNode(
                id="agent_capability.telegram_mcp.personal_actions.telegram_mcp_send",
                label="telegram_mcp_send",
                kind=CapabilityKind.AGENT_CAPABILITY,
                status=CapabilityStatus.AVAILABLE,
                reason="profile and required runtime surface are registered",
            ),
            CapabilityNode(
                id="config.telegram_mcp",
                label="Telegram MCP personal account",
                kind=CapabilityKind.CONFIG_FLAG,
                status=CapabilityStatus.CONFIGURED_ONLY,
                reason="TELEGRAM_MCP_ENABLED controls telegram_mcp profiles",
            ),
            CapabilityNode(
                id="mcp_server.telegram-mcp-personal",
                label="telegram-mcp-personal",
                kind=CapabilityKind.MCP_SERVER,
                status=CapabilityStatus.CONFIGURED_ONLY,
                reason=".mcp.json config exists",
            ),
        )
    )

    summary = graph.format_manager_summary(max_items=80)

    assert "agent_profile.telegram_mcp.personal_actions: available" in summary
    assert (
        "agent_capability.telegram_mcp.personal_actions.telegram_mcp_send: available"
        in summary
    )
    assert "config.telegram_mcp" not in summary
    assert "mcp_server.telegram-mcp-personal" not in summary


def test_manager_summary_pins_external_skill_control_surfaces() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityGraph,
        CapabilityKind,
        CapabilityNode,
        CapabilityStatus,
    )

    noisy_nodes = tuple(
        CapabilityNode(
            id=f"agent_capability.noisy.blocker_{index}.web_search_sources",
            label=f"noise {index}",
            kind=CapabilityKind.AGENT_CAPABILITY,
            status=CapabilityStatus.BLOCKED,
            reason="noisy blocker",
        )
        for index in range(12)
    )
    graph = CapabilityGraph(
        capabilities=(
            *noisy_nodes,
            CapabilityNode(
                id="skill.external_skill_acquisition",
                label="external_skill_acquisition",
                kind=CapabilityKind.SKILL,
                status=CapabilityStatus.AVAILABLE,
                reason="registered in active bot dispatcher; manifest status is experimental",
            ),
            CapabilityNode(
                id="skill.external_skill_runtime",
                label="external_skill_runtime",
                kind=CapabilityKind.SKILL,
                status=CapabilityStatus.AVAILABLE,
                reason="registered in active bot dispatcher; manifest status is experimental",
            ),
        )
    )

    summary = graph.format_manager_summary(max_items=2)

    assert "skill.external_skill_acquisition: available" in summary
    assert "skill.external_skill_runtime: available" in summary


def _external_skill_record(
    tmp_path: Path,
    skill_id: str,
    status: Any,
):
    from src.skills.external_skill_loader.loader import (
        ExternalSkillAuditReport,
        ExternalSkillSource,
        PersonalSkillRegistryRecord,
    )

    return PersonalSkillRegistryRecord(
        skill_id=skill_id,
        name=skill_id,
        source=ExternalSkillSource(
            source_type="local_folder",
            locator=str(tmp_path / "source" / skill_id),
        ),
        quarantine_path=str(tmp_path / "quarantine" / skill_id),
        content_hash=f"hash-{skill_id}",
        status=status,
        audit_report=ExternalSkillAuditReport(
            skill_id=skill_id,
            name=skill_id,
            status=status,
            risk_level="low",
            read_only_allowed=True,
        ),
    )

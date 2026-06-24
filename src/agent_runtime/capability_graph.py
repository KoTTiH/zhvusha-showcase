"""Unified capability and tool graph for ZHVUSHA runtime surfaces."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel

from src.agent_runtime.digital_scenarios import BUILTIN_DIGITAL_SCENARIOS
from src.agent_runtime.tools import SIDE_EFFECT_CAPABILITIES

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from src.agent_runtime.models import AgentDefinition, InvocationProfile


RELEVANT_CONFIG_FLAGS: tuple[str, ...] = (
    "TELEGRAM_MCP_ENABLED",
    "DAEMON_ENABLED",
    "NEWS_SOURCES_ENABLED",
    "SELF_CODING_ENABLED",
    "AUTONOMOUS_SELF_CODING_ENABLED",
    "AGENCY_RUNTIME_ENABLED",
    "AGENCY_SOCIAL_AUTONOMY_ENABLED",
    "LIFE_RUNTIME_ENABLED",
    "VOICE_GATEWAY_ENABLED",
    "DESKTOP_CONTROL_ENABLED",
    "COMPUTER_USE_ENABLED",
    "LIVE_BROWSER_CONTROL_ENABLED",
    "COMPUTER_USE_SHELL_ENABLED",
)
KNOWN_HERMES_GAP_CAPABILITIES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "kubernetes_debug",
        "Kubernetes debug workflow",
        "Hermes roadmap example; no native or approved external skill is registered",
        ("docs/hermes-skill-compatibility-roadmap.md:84",),
    ),
)

_CONFIG_ATTR_BY_FLAG: dict[str, str] = {
    "TELEGRAM_MCP_ENABLED": "telegram_mcp_enabled",
    "DAEMON_ENABLED": "daemon_enabled",
    "NEWS_SOURCES_ENABLED": "news_sources_enabled",
    "SELF_CODING_ENABLED": "self_coding_enabled",
    "AUTONOMOUS_SELF_CODING_ENABLED": "autonomous_self_coding_enabled",
    "AGENCY_RUNTIME_ENABLED": "agency_runtime_enabled",
    "AGENCY_SOCIAL_AUTONOMY_ENABLED": "agency_social_autonomy_enabled",
    "LIFE_RUNTIME_ENABLED": "life_runtime_enabled",
    "VOICE_GATEWAY_ENABLED": "voice_gateway_enabled",
    "DESKTOP_CONTROL_ENABLED": "desktop_control_enabled",
    "COMPUTER_USE_ENABLED": "computer_use_enabled",
    "LIVE_BROWSER_CONTROL_ENABLED": "live_browser_control_enabled",
    "COMPUTER_USE_SHELL_ENABLED": "computer_use_shell_enabled",
}


class CapabilityStatus(StrEnum):
    """Runtime availability status for one graph node."""

    AVAILABLE = "available"
    CONFIGURED_ONLY = "configured_only"
    DEGRADED = "degraded"
    DISABLED = "disabled"
    ORPHANED = "orphaned"
    QUARANTINED = "quarantined"
    NEEDS_REVIEW = "needs_review"
    BLOCKED = "blocked"


class CapabilityKind(StrEnum):
    """Surface type represented by the graph."""

    SKILL = "skill"
    AGENT_DEFINITION = "agent_definition"
    AGENT_PROFILE = "agent_profile"
    AGENT_CAPABILITY = "agent_capability"
    DAEMON = "daemon"
    DAEMON_TOOL = "daemon_tool"
    MCP_SERVER = "mcp_server"
    CONFIG_FLAG = "config_flag"
    RUNTIME = "runtime"
    NEWS_SOURCE = "news_source"
    EXTERNAL_SKILL = "external_skill"
    DIGITAL_SCENARIO = "digital_scenario"


class ToolKind(StrEnum):
    """Tool plane represented by the graph."""

    AGENT_RUNTIME = "agent_runtime"
    DAEMON = "daemon"
    MCP_SERVER = "mcp_server"


class CapabilityNode(BaseModel):
    """Single capability surface and its honest runtime status."""

    id: str
    label: str
    kind: CapabilityKind
    status: CapabilityStatus
    reason: str = ""
    evidence: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    worker: str = ""
    profile_id: str = ""
    capability_id: str = ""
    manager_visible: bool = True


class ToolNode(BaseModel):
    """Single callable or configured tool across runtime planes."""

    name: str
    kind: ToolKind
    capability: str = ""
    status: CapabilityStatus
    requires_approval: bool = False
    evidence: tuple[str, ...] = ()


class ConfigFlagNode(BaseModel):
    """Config flag tracked by the graph and the graph nodes that consume it."""

    name: str
    enabled: bool
    consumer_ids: tuple[str, ...] = ()


class ToolGraph(BaseModel):
    """Unified inventory of tools across runtime planes."""

    tools: tuple[ToolNode, ...] = ()

    def require(self, name: str) -> ToolNode:
        """Return a tool node by tool name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise KeyError(f"unknown tool node: {name}")


class CapabilityGraph(BaseModel):
    """Unified inventory for capabilities, tools and config switches."""

    capabilities: tuple[CapabilityNode, ...] = ()
    tools: tuple[ToolNode, ...] = ()
    config_flags: tuple[ConfigFlagNode, ...] = ()

    def require(self, node_id: str) -> CapabilityNode:
        """Return a capability node or raise KeyError with a useful message."""
        for node in self.capabilities:
            if node.id == node_id:
                return node
        raise KeyError(f"unknown capability node: {node_id}")

    def require_tool(self, name: str) -> ToolNode:
        """Return a tool node by tool name."""
        return self.tool_graph().require(name)

    def tool_graph(self) -> ToolGraph:
        """Return the ToolGraph view of this capability graph."""
        return ToolGraph(tools=self.tools)

    def assert_available_profiles_have_registered_workers(self) -> None:
        """Fail if an available Agent Runtime profile has no worker evidence."""
        offenders = [
            node.id
            for node in self.capabilities
            if node.kind is CapabilityKind.AGENT_PROFILE
            and node.status is CapabilityStatus.AVAILABLE
            and "worker_registered" not in node.evidence
        ]
        if offenders:
            raise AssertionError(
                "available Agent Runtime profiles without registered worker: "
                + ", ".join(sorted(offenders))
            )

    def assert_no_required_skill_orphans(self) -> None:
        """Fail on non-disabled production skill manifests that are unreachable."""
        offenders = [
            node.id
            for node in self.capabilities
            if node.kind is CapabilityKind.SKILL
            and node.status is CapabilityStatus.ORPHANED
        ]
        if offenders:
            raise AssertionError(
                "orphaned non-disabled skill manifests: " + ", ".join(sorted(offenders))
            )

    def assert_relevant_config_flags_consumed(self) -> None:
        """Fail when a tracked runtime flag has no graph consumer."""
        by_name = {flag.name: flag for flag in self.config_flags}
        missing = [name for name in RELEVANT_CONFIG_FLAGS if name not in by_name]
        empty = [
            name
            for name in RELEVANT_CONFIG_FLAGS
            if name in by_name and not by_name[name].consumer_ids
        ]
        offenders = [*missing, *empty]
        if offenders:
            raise AssertionError(
                "relevant config flags without graph consumers: "
                + ", ".join(sorted(offenders))
            )

    def format_manager_summary(self, *, max_items: int = 30) -> str:
        """Return a secret-free private manager summary for Жвуша."""
        visible = [node for node in self.capabilities if _manager_summary_visible(node)]
        if not visible:
            return ""
        priority = {
            CapabilityStatus.BLOCKED: 0,
            CapabilityStatus.QUARANTINED: 1,
            CapabilityStatus.NEEDS_REVIEW: 2,
            CapabilityStatus.ORPHANED: 3,
            CapabilityStatus.DEGRADED: 4,
            CapabilityStatus.CONFIGURED_ONLY: 5,
            CapabilityStatus.DISABLED: 6,
            CapabilityStatus.AVAILABLE: 7,
        }
        sorted_visible = sorted(
            visible, key=lambda node: (priority[node.status], node.id)
        )
        pinned_rows = [node for node in sorted_visible if _manager_summary_pinned(node)]
        remaining_rows = [
            node for node in sorted_visible if not _manager_summary_pinned(node)
        ]
        rows = [*pinned_rows[:max_items]]
        if len(rows) < max_items:
            rows.extend(remaining_rows[: max_items - len(rows)])
        lines = [
            "## Внутренний граф возможностей",
            "Статусы показывают только реальные runtime-пути; секреты и raw env не включены.",
        ]
        for node in rows:
            reason = f" — {node.reason}" if node.reason else ""
            lines.append(f"- {node.id}: {node.status.value}{reason}")
        return "\n".join(lines)


def _manager_summary_pinned(node: CapabilityNode) -> bool:
    """Keep decision-critical personal-account paths visible in short summaries."""
    return node.id.startswith(
        (
            "agent_profile.telegram_mcp.",
            "agent_capability.telegram_mcp.",
            "agent_profile.agency.",
            "agent_capability.agency.",
            "config.agency_",
            "agent_profile.computer_use.",
            "agent_capability.computer_use.",
            "config.computer_use",
            "config.live_browser_control",
            "runtime.live_browser_",
            "skill.external_skill_",
            "external_skill.",
            "digital_scenario.",
        )
    )


def _manager_summary_visible(node: CapabilityNode) -> bool:
    """Hide low-level wiring nodes that can contradict the callable profile view."""
    if not node.manager_visible:
        return False
    if node.kind is CapabilityKind.MCP_SERVER:
        return False
    return node.id != "config.telegram_mcp"


def build_capability_graph(
    *,
    project_root: Path,
    settings: Any,
    active_skill_names: Sequence[str],
    startup_skill_names: Sequence[str],
    invocation_profiles: Sequence[InvocationProfile],
    registered_worker_names: Sequence[str],
    tool_gateways: Sequence[Any],
    daemon_tool_names: Sequence[str],
    agent_definitions: Sequence[AgentDefinition] = (),
    mcp_config_path: Path | None = None,
    skill_manifest_root: Path | None = None,
    external_skill_records: Sequence[Any] = (),
    daemon_active: bool = False,
    news_monitor_active: bool = False,
) -> CapabilityGraph:
    """Build the graph from real runtime registrations and repo manifests."""
    skill_root = skill_manifest_root or project_root / "src" / "skills"
    mcp_path = mcp_config_path or project_root / ".mcp.json"
    active_skills = set(active_skill_names)
    startup_skills = set(startup_skill_names)
    workers = set(registered_worker_names)
    tools = [
        *_agent_tool_nodes(tool_gateways),
        *_daemon_tool_nodes(
            daemon_tool_names, daemon_enabled=_flag(settings, "DAEMON_ENABLED")
        ),
        *_mcp_tool_nodes(mcp_path),
    ]

    capabilities: list[CapabilityNode] = []
    capabilities.extend(
        _skill_nodes(
            skill_root=skill_root,
            active_skill_names=active_skills,
            startup_skill_names=startup_skills,
        )
    )
    capabilities.extend(
        _agent_definition_nodes(
            agents=agent_definitions,
            profiles=invocation_profiles,
            registered_worker_names=workers,
            settings=settings,
        )
    )
    capabilities.extend(
        _agent_profile_nodes(
            profiles=invocation_profiles,
            registered_worker_names=workers,
            settings=settings,
            mcp_config_path=mcp_path,
        )
    )
    capabilities.extend(
        _agent_capability_nodes(
            profiles=invocation_profiles,
            profile_nodes=capabilities,
            tools=tools,
        )
    )
    capabilities.extend(_known_hermes_gap_nodes(capabilities))
    capabilities.extend(
        _mcp_server_nodes(mcp_path),
    )
    capabilities.extend(_external_skill_nodes(external_skill_records))
    capabilities.extend(
        _config_surface_nodes(
            settings=settings,
            profiles=invocation_profiles,
            registered_worker_names=workers,
            tools=tools,
            daemon_active=daemon_active,
            daemon_tool_names=daemon_tool_names,
            news_monitor_active=news_monitor_active,
        )
    )
    capabilities.extend(_digital_scenario_nodes(capabilities))

    config_flags = tuple(
        ConfigFlagNode(
            name=name,
            enabled=_flag(settings, name),
            consumer_ids=tuple(
                node.id for node in capabilities if name in set(node.flags)
            ),
        )
        for name in RELEVANT_CONFIG_FLAGS
    )
    return CapabilityGraph(
        capabilities=tuple(_dedupe_capabilities(capabilities)),
        tools=tuple(_dedupe_tools(tools)),
        config_flags=config_flags,
    )


def _digital_scenario_nodes(
    existing_capabilities: Sequence[CapabilityNode],
) -> list[CapabilityNode]:
    """Represent generalized digital-agent polygons in the runtime truth graph."""
    by_id = {node.id: node for node in existing_capabilities}
    nodes: list[CapabilityNode] = []
    for scenario in BUILTIN_DIGITAL_SCENARIOS:
        status, reason = _digital_scenario_status(
            required_ids=scenario.required_capability_nodes,
            capabilities_by_id=by_id,
        )
        nodes.append(
            CapabilityNode(
                id=f"digital_scenario.{scenario.id}",
                label=scenario.title,
                kind=CapabilityKind.DIGITAL_SCENARIO,
                status=status,
                reason=reason,
                evidence=(
                    "digital_scenario_registry",
                    f"eval_cases={len(scenario.eval_cases)}",
                    *scenario.required_capability_nodes[:6],
                ),
                capability_id=scenario.id,
            )
        )
    return nodes


def _digital_scenario_status(
    *,
    required_ids: tuple[str, ...],
    capabilities_by_id: dict[str, CapabilityNode],
) -> tuple[CapabilityStatus, str]:
    required = list(required_ids)
    existing = [
        capabilities_by_id[node_id]
        for node_id in required
        if node_id in capabilities_by_id
    ]
    available = [node for node in existing if node.status is CapabilityStatus.AVAILABLE]
    blocked = [
        node
        for node in existing
        if node.status
        in {
            CapabilityStatus.BLOCKED,
            CapabilityStatus.QUARANTINED,
            CapabilityStatus.NEEDS_REVIEW,
        }
    ]
    if required and len(available) == len(required):
        return (
            CapabilityStatus.AVAILABLE,
            "all required runtime surfaces are available for representative evals",
        )
    if blocked:
        return (
            CapabilityStatus.BLOCKED,
            "blocked required surfaces: " + ", ".join(node.id for node in blocked[:4]),
        )
    unavailable = [
        node_id
        for node_id in required
        if capabilities_by_id.get(node_id) is None
        or capabilities_by_id[node_id].status is not CapabilityStatus.AVAILABLE
    ]
    if available:
        return (
            CapabilityStatus.DEGRADED,
            (
                f"{len(available)}/{len(required)} required runtime surfaces "
                "available; missing/degraded: " + ", ".join(unavailable[:4])
            ),
        )
    if existing or unavailable:
        return (
            CapabilityStatus.DISABLED,
            "no required runtime surfaces are available; first gaps: "
            + ", ".join(unavailable[:4]),
        )
    return CapabilityStatus.DISABLED, "scenario has no required runtime surfaces"


def _skill_nodes(
    *,
    skill_root: Path,
    active_skill_names: set[str],
    startup_skill_names: set[str],
) -> list[CapabilityNode]:
    nodes: list[CapabilityNode] = []
    for manifest_path in sorted(skill_root.glob("*/skill.yaml")):
        data = _load_yaml_mapping(manifest_path)
        name = str(data.get("name", manifest_path.parent.name))
        explicit_status = str(data.get("status", "")).strip().lower()
        source = str(data.get("source", "")).strip().lower()
        disabled_reason = str(data.get("disabled_reason", "")).strip()
        enabled = data.get("enabled", None)
        experimental = explicit_status == "experimental" or source == "experimental"
        if explicit_status == "disabled" or enabled is False:
            status = CapabilityStatus.DISABLED
            reason = disabled_reason or "skill explicitly disabled in manifest"
        elif name in active_skill_names:
            status = CapabilityStatus.AVAILABLE
            reason = (
                "registered in active bot dispatcher; manifest status is experimental"
                if experimental
                else "registered in active bot dispatcher"
            )
        elif experimental:
            status = CapabilityStatus.DISABLED
            reason = "experimental skill is not active in this process"
        elif name in startup_skill_names:
            status = CapabilityStatus.CONFIGURED_ONLY
            reason = "manifest is startup-known but not active in this process"
        else:
            status = CapabilityStatus.ORPHANED
            reason = "skill.yaml exists but startup routing does not register it"
        nodes.append(
            CapabilityNode(
                id=f"skill.{name}",
                label=name,
                kind=CapabilityKind.SKILL,
                status=status,
                reason=reason,
                evidence=(str(manifest_path.relative_to(skill_root.parent.parent)),),
            )
        )
    return nodes


def _agent_profile_nodes(
    *,
    profiles: Sequence[InvocationProfile],
    registered_worker_names: set[str],
    settings: Any,
    mcp_config_path: Path,
) -> list[CapabilityNode]:
    mcp_servers = _load_mcp_servers(mcp_config_path)
    nodes: list[CapabilityNode] = []
    for profile in profiles:
        flags = _flags_for_profile(profile.id)
        worker_registered = profile.worker in registered_worker_names
        evidence: list[str] = [f"profile={profile.id}", f"worker={profile.worker}"]
        if worker_registered:
            evidence.append("worker_registered")

        if profile.id.startswith("telegram_mcp."):
            status, reason = _telegram_mcp_profile_status(
                profile=profile,
                settings=settings,
                worker_registered=worker_registered,
                mcp_servers=mcp_servers,
            )
        elif (
            disabled_status := _disabled_profile_status(profile.id, settings)
        ) is not None:
            status, reason = disabled_status
        elif worker_registered:
            status = CapabilityStatus.AVAILABLE
            reason = "worker registered in AgentRuntime"
        else:
            status = CapabilityStatus.CONFIGURED_ONLY
            reason = f"profile points to missing worker {profile.worker}"

        nodes.append(
            CapabilityNode(
                id=f"agent_profile.{profile.id}",
                label=profile.id,
                kind=CapabilityKind.AGENT_PROFILE,
                status=status,
                reason=reason,
                evidence=tuple(evidence),
                flags=flags,
                worker=profile.worker,
                profile_id=profile.id,
            )
        )
    return nodes


def _agent_definition_nodes(
    *,
    agents: Sequence[AgentDefinition],
    profiles: Sequence[InvocationProfile],
    registered_worker_names: set[str],
    settings: Any,
) -> list[CapabilityNode]:
    profile_workers = {profile.worker for profile in profiles}
    nodes: list[CapabilityNode] = []
    for agent in agents:
        default_worker = agent.default_worker
        flag = _flag_for_agent_definition(agent.id)
        flags = (flag,) if flag else ()
        evidence: tuple[str, ...]
        if flag and not _flag(settings, flag):
            status = CapabilityStatus.DISABLED
            reason = f"{flag}=false"
            evidence = (f"feature_flag={flag}",)
        elif default_worker in registered_worker_names:
            status = CapabilityStatus.AVAILABLE
            reason = "default worker registered in AgentRuntime"
            evidence = ("worker_registered",)
        elif default_worker in profile_workers:
            status = CapabilityStatus.CONFIGURED_ONLY
            reason = f"agent/profile configured, missing worker {default_worker}"
            evidence = ()
        else:
            status = CapabilityStatus.ORPHANED
            reason = f"default worker {default_worker} is not referenced by profiles"
            evidence = ()
        nodes.append(
            CapabilityNode(
                id=f"agent_definition.{agent.id}",
                label=agent.id,
                kind=CapabilityKind.AGENT_DEFINITION,
                status=status,
                reason=reason,
                evidence=evidence,
                flags=flags,
                worker=default_worker,
            )
        )
    return nodes


def _agent_capability_nodes(
    *,
    profiles: Sequence[InvocationProfile],
    profile_nodes: list[CapabilityNode],
    tools: Sequence[ToolNode],
) -> list[CapabilityNode]:
    profile_status = {
        node.profile_id: node.status
        for node in profile_nodes
        if node.kind is CapabilityKind.AGENT_PROFILE and node.profile_id
    }
    tool_capabilities = {
        tool.capability
        for tool in tools
        if tool.kind is ToolKind.AGENT_RUNTIME
        and tool.status is CapabilityStatus.AVAILABLE
        and tool.capability
    }
    nodes: list[CapabilityNode] = []
    seen: set[tuple[str, str]] = set()
    for profile in profiles:
        parent_status = profile_status.get(profile.id, CapabilityStatus.CONFIGURED_ONLY)
        for capability in profile.allowed_capabilities:
            key = (profile.id, capability)
            if key in seen:
                continue
            seen.add(key)
            requires_tool = capability in tool_capabilities or capability.startswith(
                (
                    "browser_",
                    "web_search",
                    "read_workspace",
                    "telegram_mcp_",
                    "desktop.",
                    "desktop_",
                )
            )
            status: CapabilityStatus
            if parent_status is not CapabilityStatus.AVAILABLE:
                status = parent_status
                reason = f"profile status is {parent_status.value}"
            elif requires_tool and capability not in tool_capabilities:
                status = CapabilityStatus.CONFIGURED_ONLY
                reason = f"no ToolGateway tool registered for {capability}"
            else:
                status = CapabilityStatus.AVAILABLE
                reason = "profile and required runtime surface are registered"
            nodes.append(
                CapabilityNode(
                    id=f"agent_capability.{profile.id}.{capability}",
                    label=capability,
                    kind=CapabilityKind.AGENT_CAPABILITY,
                    status=status,
                    reason=reason,
                    profile_id=profile.id,
                    capability_id=capability,
                )
            )
    return nodes


def _telegram_mcp_profile_status(
    *,
    profile: InvocationProfile,
    settings: Any,
    worker_registered: bool,
    mcp_servers: dict[str, dict[str, Any]],
) -> tuple[CapabilityStatus, str]:
    del profile
    if not _flag(settings, "TELEGRAM_MCP_ENABLED"):
        return CapabilityStatus.DISABLED, "TELEGRAM_MCP_ENABLED=false"
    if "telegram-mcp-personal" not in mcp_servers:
        return CapabilityStatus.DEGRADED, "telegram-mcp-personal missing from .mcp.json"
    if not worker_registered:
        return (
            CapabilityStatus.CONFIGURED_ONLY,
            "profile/.mcp/config exist but telegram_mcp worker is not registered",
        )
    session = str(getattr(settings, "telegram_mcp_session_string_personal", ""))
    session_name = str(getattr(settings, "telegram_mcp_session_name_personal", ""))
    if not session and not session_name:
        return CapabilityStatus.DEGRADED, "personal Telegram MCP session is missing"
    return (
        CapabilityStatus.AVAILABLE,
        "worker, MCP server and personal session are configured",
    )


def _disabled_profile_status(
    profile_id: str,
    settings: Any,
) -> tuple[CapabilityStatus, str] | None:
    gated_profiles: tuple[tuple[bool, str, str], ...] = (
        (
            profile_id == "self_coding.implementation",
            "SELF_CODING_ENABLED",
            "SELF_CODING_ENABLED=false",
        ),
        (
            profile_id == "self_improvement.autonomous",
            "AUTONOMOUS_SELF_CODING_ENABLED",
            "AUTONOMOUS_SELF_CODING_ENABLED=false",
        ),
        (
            profile_id.startswith("agency."),
            "AGENCY_RUNTIME_ENABLED",
            "AGENCY_RUNTIME_ENABLED=false",
        ),
        (
            profile_id.startswith("life_"),
            "LIFE_RUNTIME_ENABLED",
            "LIFE_RUNTIME_ENABLED=false",
        ),
        (
            profile_id.startswith("desktop_control."),
            "DESKTOP_CONTROL_ENABLED",
            "DESKTOP_CONTROL_ENABLED=false",
        ),
        (
            profile_id.startswith("computer_use."),
            "COMPUTER_USE_ENABLED",
            "COMPUTER_USE_ENABLED=false",
        ),
    )
    for matches, flag, reason in gated_profiles:
        if matches and not _flag(settings, flag):
            return CapabilityStatus.DISABLED, reason
    return None


def _mcp_server_nodes(mcp_config_path: Path) -> list[CapabilityNode]:
    return [
        CapabilityNode(
            id=f"mcp_server.{name}",
            label=name,
            kind=CapabilityKind.MCP_SERVER,
            status=CapabilityStatus.CONFIGURED_ONLY,
            reason=".mcp.json config exists; bot runtime adapter decides reachability",
            evidence=("mcp_config",),
            flags=("TELEGRAM_MCP_ENABLED",) if name == "telegram-mcp-personal" else (),
        )
        for name in sorted(_load_mcp_servers(mcp_config_path))
    ]


def _known_hermes_gap_nodes(
    existing_capabilities: Sequence[CapabilityNode],
) -> list[CapabilityNode]:
    """Represent doc-backed Hermes parity gaps in the runtime truth graph."""
    known_capability_ids = {
        node.capability_id
        for node in existing_capabilities
        if node.kind is CapabilityKind.AGENT_CAPABILITY and node.capability_id
    }
    nodes: list[CapabilityNode] = []
    for capability_id, label, reason, evidence in KNOWN_HERMES_GAP_CAPABILITIES:
        if capability_id in known_capability_ids:
            continue
        nodes.append(
            CapabilityNode(
                id=f"agent_capability.hermes_gap.{capability_id}",
                label=label,
                kind=CapabilityKind.AGENT_CAPABILITY,
                status=CapabilityStatus.DISABLED,
                reason=reason,
                evidence=evidence,
                capability_id=capability_id,
            )
        )
    return nodes


def _config_surface_nodes(
    *,
    settings: Any,
    profiles: Sequence[InvocationProfile],
    registered_worker_names: set[str],
    tools: Sequence[ToolNode],
    daemon_active: bool,
    daemon_tool_names: Sequence[str],
    news_monitor_active: bool,
) -> list[CapabilityNode]:
    profile_ids = {profile.id for profile in profiles}
    worker_names = set(registered_worker_names)
    nodes: list[CapabilityNode] = []
    nodes.append(
        CapabilityNode(
            id="config.telegram_mcp",
            label="Telegram MCP personal account",
            kind=CapabilityKind.CONFIG_FLAG,
            status=CapabilityStatus.CONFIGURED_ONLY
            if _flag(settings, "TELEGRAM_MCP_ENABLED")
            else CapabilityStatus.DISABLED,
            reason="TELEGRAM_MCP_ENABLED controls telegram_mcp profiles and MCP config",
            flags=("TELEGRAM_MCP_ENABLED",),
        )
    )
    nodes.append(
        CapabilityNode(
            id="config.daemon",
            label="embedded daemon",
            kind=CapabilityKind.DAEMON,
            status=_daemon_status(
                enabled=_flag(settings, "DAEMON_ENABLED"),
                daemon_active=daemon_active,
                daemon_tool_names=daemon_tool_names,
            ),
            reason=_daemon_reason(
                enabled=_flag(settings, "DAEMON_ENABLED"),
                daemon_active=daemon_active,
                daemon_tool_names=daemon_tool_names,
            ),
            flags=("DAEMON_ENABLED",),
        )
    )
    nodes.extend(
        CapabilityNode(
            id=f"daemon_tool.{name}",
            label=name,
            kind=CapabilityKind.DAEMON_TOOL,
            status=CapabilityStatus.AVAILABLE
            if _flag(settings, "DAEMON_ENABLED") and daemon_active
            else CapabilityStatus.DISABLED
            if not _flag(settings, "DAEMON_ENABLED")
            else CapabilityStatus.DEGRADED,
            reason="registered in daemon ToolRegistry",
            flags=("DAEMON_ENABLED",),
        )
        for name in sorted(daemon_tool_names)
    )
    nodes.append(
        CapabilityNode(
            id="config.news_sources",
            label="news/topic source monitor",
            kind=CapabilityKind.NEWS_SOURCE,
            status=CapabilityStatus.AVAILABLE
            if news_monitor_active
            else CapabilityStatus.DISABLED
            if not _flag(settings, "NEWS_SOURCES_ENABLED")
            else CapabilityStatus.DEGRADED,
            reason="NEWS_SOURCES_ENABLED controls NewsMonitor startup",
            flags=("NEWS_SOURCES_ENABLED",),
        )
    )
    nodes.append(
        CapabilityNode(
            id="config.self_coding",
            label="self-coding implementation gate",
            kind=CapabilityKind.CONFIG_FLAG,
            status=CapabilityStatus.AVAILABLE
            if _flag(settings, "SELF_CODING_ENABLED")
            and "self_coding.implementation" in profile_ids
            and "self_coding_native" in worker_names
            else CapabilityStatus.DISABLED
            if not _flag(settings, "SELF_CODING_ENABLED")
            else CapabilityStatus.DEGRADED,
            reason="SELF_CODING_ENABLED gates implementation runs",
            flags=("SELF_CODING_ENABLED",),
        )
    )
    nodes.append(
        CapabilityNode(
            id="config.autonomous_self_coding",
            label="autonomous self-coding loop",
            kind=CapabilityKind.CONFIG_FLAG,
            status=CapabilityStatus.AVAILABLE
            if _flag(settings, "AUTONOMOUS_SELF_CODING_ENABLED")
            and "self_improvement.autonomous" in profile_ids
            and "self_improvement" in worker_names
            else CapabilityStatus.DISABLED
            if not _flag(settings, "AUTONOMOUS_SELF_CODING_ENABLED")
            else CapabilityStatus.DEGRADED,
            reason="AUTONOMOUS_SELF_CODING_ENABLED gates scheduled self-work",
            flags=("AUTONOMOUS_SELF_CODING_ENABLED",),
        )
    )
    agency_runtime_available = (
        _flag(settings, "AGENCY_RUNTIME_ENABLED")
        and "agency.readonly_draft" in profile_ids
        and "agency" in worker_names
    )
    nodes.append(
        CapabilityNode(
            id="config.agency_runtime",
            label="agency self-complexification runtime",
            kind=CapabilityKind.CONFIG_FLAG,
            status=CapabilityStatus.AVAILABLE
            if agency_runtime_available
            else CapabilityStatus.DISABLED
            if not _flag(settings, "AGENCY_RUNTIME_ENABLED")
            else CapabilityStatus.DEGRADED,
            reason="AGENCY_RUNTIME_ENABLED gates agency planning jobs",
            flags=("AGENCY_RUNTIME_ENABLED",),
        )
    )
    nodes.append(
        CapabilityNode(
            id="config.agency_social_autonomy",
            label="agency social autonomy policy",
            kind=CapabilityKind.CONFIG_FLAG,
            status=CapabilityStatus.AVAILABLE
            if agency_runtime_available
            and _flag(settings, "AGENCY_SOCIAL_AUTONOMY_ENABLED")
            else CapabilityStatus.DISABLED
            if not _flag(settings, "AGENCY_SOCIAL_AUTONOMY_ENABLED")
            else CapabilityStatus.DEGRADED,
            reason=(
                "AGENCY_SOCIAL_AUTONOMY_ENABLED gates permission-backed social "
                "judgement; tool side effects remain enforced separately"
            ),
            flags=("AGENCY_SOCIAL_AUTONOMY_ENABLED", "AGENCY_RUNTIME_ENABLED"),
        )
    )
    life_runtime_available = (
        _flag(settings, "LIFE_RUNTIME_ENABLED")
        and "life_reflection.readonly" in profile_ids
        and "life_runtime" in worker_names
    )
    nodes.append(
        CapabilityNode(
            id="config.life_runtime",
            label="LifeRuntime read-only inner loop",
            kind=CapabilityKind.CONFIG_FLAG,
            status=CapabilityStatus.AVAILABLE
            if life_runtime_available
            else CapabilityStatus.DISABLED
            if not _flag(settings, "LIFE_RUNTIME_ENABLED")
            else CapabilityStatus.DEGRADED,
            reason="LIFE_RUNTIME_ENABLED gates read-only LifeRuntime daemon ticks",
            flags=("LIFE_RUNTIME_ENABLED",),
        )
    )
    nodes.append(
        CapabilityNode(
            id="config.voice_gateway",
            label="voice input/output gateway",
            kind=CapabilityKind.CONFIG_FLAG,
            status=_voice_gateway_status(settings),
            reason=_voice_gateway_reason(settings),
            flags=("VOICE_GATEWAY_ENABLED",),
        )
    )
    desktop_profile_enabled = "desktop_control.convenience" in profile_ids
    desktop_worker_registered = "desktop_control" in worker_names
    desktop_tool_registered = any(
        tool.kind is ToolKind.AGENT_RUNTIME
        and tool.status is CapabilityStatus.AVAILABLE
        and (
            tool.capability.startswith("desktop_")
            or tool.capability.startswith("desktop.")
        )
        for tool in tools
    )
    nodes.append(
        CapabilityNode(
            id="config.desktop_control",
            label="Desktop Control Skill Pack",
            kind=CapabilityKind.CONFIG_FLAG,
            status=CapabilityStatus.AVAILABLE
            if _flag(settings, "DESKTOP_CONTROL_ENABLED")
            and desktop_profile_enabled
            and desktop_worker_registered
            and desktop_tool_registered
            else CapabilityStatus.DISABLED
            if not _flag(settings, "DESKTOP_CONTROL_ENABLED")
            else CapabilityStatus.DEGRADED,
            reason=(
                "DESKTOP_CONTROL_ENABLED gates narrow desktop capabilities; "
                "worker and at least one desktop ToolGateway tool must be registered; "
                "shell/powershell are outside the convenience pack"
            ),
            flags=("DESKTOP_CONTROL_ENABLED",),
        )
    )
    computer_use_profile_enabled = "computer_use.active_gui" in profile_ids
    computer_use_shell_profile_enabled = "computer_use.approved_shell" in profile_ids
    computer_use_worker_registered = "computer_use" in worker_names
    computer_use_tool_registered = any(
        tool.kind is ToolKind.AGENT_RUNTIME
        and tool.status is CapabilityStatus.AVAILABLE
        and (
            tool.capability.startswith("browser_")
            or tool.capability.startswith("desktop_")
        )
        for tool in tools
    )
    live_browser_tool_registered = any(
        tool.kind is ToolKind.AGENT_RUNTIME
        and tool.status is CapabilityStatus.AVAILABLE
        and tool.capability
        in {
            "browser_live_control",
            "browser_navigate",
            "browser_click",
            "browser_type",
            "browser_scroll",
            "browser_tab_control",
            "browser_form_draft",
            "browser_interactive_task",
        }
        for tool in tools
    )
    live_browser_enabled = _flag(settings, "LIVE_BROWSER_CONTROL_ENABLED")
    live_browser_auto_launch = bool(
        getattr(settings, "live_browser_auto_launch", False)
    )
    nodes.append(
        CapabilityNode(
            id="config.computer_use",
            label="Computer-use active GUI profile",
            kind=CapabilityKind.CONFIG_FLAG,
            status=CapabilityStatus.AVAILABLE
            if _flag(settings, "COMPUTER_USE_ENABLED")
            and computer_use_profile_enabled
            and computer_use_worker_registered
            and computer_use_tool_registered
            else CapabilityStatus.DISABLED
            if not _flag(settings, "COMPUTER_USE_ENABLED")
            else CapabilityStatus.DEGRADED,
            reason=(
                "COMPUTER_USE_ENABLED gates active GUI computer-use; worker, "
                "profile and ToolGateway surfaces must be registered; shell has "
                "a separate approved profile"
            ),
            flags=("COMPUTER_USE_ENABLED", "COMPUTER_USE_SHELL_ENABLED"),
        )
    )
    nodes.append(
        CapabilityNode(
            id="config.live_browser_control",
            label="Live Google Chrome control",
            kind=CapabilityKind.CONFIG_FLAG,
            status=CapabilityStatus.AVAILABLE
            if live_browser_enabled
            else CapabilityStatus.DISABLED,
            reason=(
                "LIVE_BROWSER_CONTROL_ENABLED flag only; backend adapter and "
                "runtime health are tracked by runtime.live_browser_* nodes"
            ),
            flags=("LIVE_BROWSER_CONTROL_ENABLED", "COMPUTER_USE_ENABLED"),
        )
    )
    nodes.append(
        CapabilityNode(
            id="runtime.live_browser_adapter",
            label="Live browser ToolGateway adapter",
            kind=CapabilityKind.RUNTIME,
            status=CapabilityStatus.AVAILABLE
            if live_browser_enabled and live_browser_tool_registered
            else CapabilityStatus.DISABLED
            if not live_browser_enabled
            else CapabilityStatus.CONFIGURED_ONLY,
            reason=(
                "live-browser ToolGateway/backend adapter is registered"
                if live_browser_tool_registered
                else "live-browser flag is on, but backend/tool plane is not registered"
            ),
            flags=("LIVE_BROWSER_CONTROL_ENABLED", "COMPUTER_USE_ENABLED"),
        )
    )
    nodes.append(
        CapabilityNode(
            id="runtime.live_browser_health",
            label="Live browser attach health",
            kind=CapabilityKind.RUNTIME,
            status=CapabilityStatus.DISABLED
            if not live_browser_enabled
            else CapabilityStatus.CONFIGURED_ONLY
            if not live_browser_tool_registered
            else CapabilityStatus.AVAILABLE
            if live_browser_auto_launch
            else CapabilityStatus.DEGRADED,
            reason=(
                "LIVE_BROWSER_CONTROL_ENABLED=false"
                if not live_browser_enabled
                else "backend/tool plane is not registered"
                if not live_browser_tool_registered
                else "managed auto-launch path is configured"
                if live_browser_auto_launch
                else (
                    "manual Chrome attach path requires an existing remote debug "
                    "endpoint; managed auto-launch is disabled"
                )
            ),
            flags=("LIVE_BROWSER_CONTROL_ENABLED", "COMPUTER_USE_ENABLED"),
        )
    )
    shell_tool_registered = any(
        tool.kind is ToolKind.AGENT_RUNTIME
        and tool.status is CapabilityStatus.AVAILABLE
        and tool.capability == "desktop.shell"
        for tool in tools
    )
    nodes.append(
        CapabilityNode(
            id="config.computer_use_shell",
            label="Computer-use shell capability",
            kind=CapabilityKind.CONFIG_FLAG,
            status=CapabilityStatus.AVAILABLE
            if _flag(settings, "COMPUTER_USE_SHELL_ENABLED")
            and computer_use_worker_registered
            and computer_use_shell_profile_enabled
            and shell_tool_registered
            else CapabilityStatus.DISABLED
            if not _flag(settings, "COMPUTER_USE_SHELL_ENABLED")
            else CapabilityStatus.DEGRADED,
            reason=(
                "COMPUTER_USE_SHELL_ENABLED gates the separate high-risk "
                "structured argv shell profile; Desktop Control still excludes "
                "shell and powershell"
            ),
            flags=("COMPUTER_USE_SHELL_ENABLED",),
        )
    )
    return nodes


def _external_skill_nodes(records: Sequence[Any]) -> list[CapabilityNode]:
    nodes: list[CapabilityNode] = []
    for record in records:
        skill_id = str(getattr(record, "skill_id", "")).strip()
        if not skill_id:
            continue
        raw_status = _status_value(getattr(record, "status", ""))
        status, reason = _external_skill_status(raw_status)
        evidence: list[str] = []
        quarantine_path = str(getattr(record, "quarantine_path", "")).strip()
        content_hash = str(getattr(record, "content_hash", "")).strip()
        readonly_approval_id = str(getattr(record, "readonly_approval_id", "")).strip()
        execution_approval_id = str(
            getattr(record, "execution_approval_id", "")
        ).strip()
        if quarantine_path:
            evidence.append(f"quarantine_path={quarantine_path}")
        if content_hash:
            evidence.append(f"content_hash={content_hash[:12]}")
        if readonly_approval_id:
            evidence.append("readonly_approval")
        if execution_approval_id:
            evidence.append("execution_approval")
        nodes.append(
            CapabilityNode(
                id=f"external_skill.{skill_id}",
                label=str(getattr(record, "name", skill_id)) or skill_id,
                kind=CapabilityKind.EXTERNAL_SKILL,
                status=status,
                reason=reason,
                evidence=tuple(evidence),
            )
        )
    return nodes


def _external_skill_status(raw_status: str) -> tuple[CapabilityStatus, str]:
    if raw_status == "quarantined":
        return (
            CapabilityStatus.QUARANTINED,
            "external skill is imported into quarantine and is not usable yet",
        )
    if raw_status == "needs_review":
        return (
            CapabilityStatus.NEEDS_REVIEW,
            "external skill audit exists but Никита has not approved read-only use",
        )
    if raw_status == "blocked":
        return CapabilityStatus.BLOCKED, "external skill audit blocked this package"
    if raw_status == "approved_readonly":
        return (
            CapabilityStatus.AVAILABLE,
            "external skill is approved for read-only procedural use only",
        )
    if raw_status == "execution_approved":
        return (
            CapabilityStatus.AVAILABLE,
            "external skill has an execution approval profile; ToolGateway still gates tools",
        )
    if raw_status == "native_conversion_candidate":
        return (
            CapabilityStatus.AVAILABLE,
            "external skill is approved and marked for spec-first native conversion",
        )
    if raw_status == "rejected":
        return (
            CapabilityStatus.DISABLED,
            "external skill was rejected by operator curation",
        )
    if raw_status == "superseded":
        return (
            CapabilityStatus.DISABLED,
            "external skill was superseded by a native or newer skill",
        )
    if raw_status == "native_converted":
        return (
            CapabilityStatus.DISABLED,
            "external skill was converted into a native skill and is no longer active",
        )
    return CapabilityStatus.DEGRADED, f"unknown external skill status: {raw_status}"


def _status_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).strip().lower()


def _daemon_status(
    *,
    enabled: bool,
    daemon_active: bool,
    daemon_tool_names: Sequence[str],
) -> CapabilityStatus:
    if not enabled:
        return CapabilityStatus.DISABLED
    if daemon_active and daemon_tool_names:
        return CapabilityStatus.AVAILABLE
    return CapabilityStatus.DEGRADED


def _daemon_reason(
    *,
    enabled: bool,
    daemon_active: bool,
    daemon_tool_names: Sequence[str],
) -> str:
    if not enabled:
        return "DAEMON_ENABLED=false"
    if daemon_active and daemon_tool_names:
        return "daemon built with registered tools"
    return "DAEMON_ENABLED=true but daemon runtime/tools are not active"


def _voice_gateway_status(settings: Any) -> CapabilityStatus:
    if not _flag(settings, "VOICE_GATEWAY_ENABLED"):
        return CapabilityStatus.DISABLED
    if not str(getattr(settings, "voice_stt_provider", "")).strip():
        return CapabilityStatus.DEGRADED
    return CapabilityStatus.AVAILABLE


def _voice_gateway_reason(settings: Any) -> str:
    if not _flag(settings, "VOICE_GATEWAY_ENABLED"):
        return "VOICE_GATEWAY_ENABLED=false"
    provider = str(getattr(settings, "voice_stt_provider", "")).strip()
    if not provider:
        return "VOICE_GATEWAY_ENABLED=true but STT provider is not configured"
    tts = (
        "TTS enabled"
        if bool(getattr(settings, "voice_tts_enabled", False))
        else "TTS disabled"
    )
    return f"voice gateway normalized input enabled with STT provider {provider}; {tts}"


def _agent_tool_nodes(tool_gateways: Sequence[Any]) -> list[ToolNode]:
    nodes: list[ToolNode] = []
    for gateway in tool_gateways:
        for tool in _tools_from_gateway(gateway):
            capability = str(getattr(tool, "capability", ""))
            name = str(getattr(tool, "name", ""))
            if not name:
                continue
            nodes.append(
                ToolNode(
                    name=name,
                    kind=ToolKind.AGENT_RUNTIME,
                    capability=capability,
                    status=CapabilityStatus.AVAILABLE,
                    requires_approval=capability in SIDE_EFFECT_CAPABILITIES,
                    evidence=("ToolGateway",),
                )
            )
    return nodes


def _daemon_tool_nodes(
    daemon_tool_names: Sequence[str],
    *,
    daemon_enabled: bool,
) -> list[ToolNode]:
    status = (
        CapabilityStatus.CONFIGURED_ONLY
        if daemon_enabled
        else CapabilityStatus.DISABLED
    )
    return [
        ToolNode(
            name=name,
            kind=ToolKind.DAEMON,
            status=status,
            evidence=("DaemonToolRegistry",),
        )
        for name in sorted(daemon_tool_names)
    ]


def _mcp_tool_nodes(mcp_config_path: Path) -> list[ToolNode]:
    return [
        ToolNode(
            name=name,
            kind=ToolKind.MCP_SERVER,
            status=CapabilityStatus.CONFIGURED_ONLY,
            evidence=("mcp_config",),
        )
        for name in sorted(_load_mcp_servers(mcp_config_path))
    ]


def _tools_from_gateway(gateway: Any) -> tuple[Any, ...]:
    method = getattr(gateway, "registered_tools", None)
    if callable(method):
        tools = method()
        return tuple(tools)
    raw = getattr(gateway, "_tools", {})
    if isinstance(raw, dict):
        return tuple(raw.values())
    return ()


def _load_mcp_servers(path: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    servers = data.get("mcpServers", {})
    if not isinstance(servers, dict):
        return {}
    return {
        str(name): dict(value)
        for name, value in servers.items()
        if isinstance(value, dict)
    }


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError:
        return {}
    return data if isinstance(data, dict) else {}


def _flag(settings: Any, env_name: str) -> bool:
    attr = _CONFIG_ATTR_BY_FLAG[env_name]
    value = getattr(settings, attr, False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _flags_for_profile(profile_id: str) -> tuple[str, ...]:
    if profile_id.startswith("telegram_mcp."):
        return ("TELEGRAM_MCP_ENABLED",)
    if profile_id == "self_coding.implementation":
        return ("SELF_CODING_ENABLED",)
    if profile_id == "self_improvement.autonomous":
        return ("AUTONOMOUS_SELF_CODING_ENABLED",)
    if profile_id.startswith("agency."):
        return ("AGENCY_RUNTIME_ENABLED",)
    if profile_id.startswith("life_"):
        return ("LIFE_RUNTIME_ENABLED",)
    if profile_id.startswith("desktop_control."):
        return ("DESKTOP_CONTROL_ENABLED",)
    if profile_id.startswith("computer_use."):
        return ("COMPUTER_USE_ENABLED", "LIVE_BROWSER_CONTROL_ENABLED")
    return ()


def _flag_for_agent_definition(agent_id: str) -> str:
    return {
        "self_improvement": "AUTONOMOUS_SELF_CODING_ENABLED",
        "agency": "AGENCY_RUNTIME_ENABLED",
        "life_runtime": "LIFE_RUNTIME_ENABLED",
        "desktop_control": "DESKTOP_CONTROL_ENABLED",
        "computer_use": "COMPUTER_USE_ENABLED",
    }.get(agent_id, "")


def _dedupe_capabilities(nodes: list[CapabilityNode]) -> list[CapabilityNode]:
    result: dict[str, CapabilityNode] = {}
    for node in nodes:
        result[node.id] = node
    return [result[key] for key in sorted(result)]


def _dedupe_tools(nodes: list[ToolNode]) -> list[ToolNode]:
    result: dict[tuple[ToolKind, str], ToolNode] = {}
    for node in nodes:
        result[(node.kind, node.name)] = node
    return [
        result[key] for key in sorted(result, key=lambda item: (item[0].value, item[1]))
    ]

"""Bot command surface for CapabilityGraph status."""

from __future__ import annotations


def test_capability_status_command_is_admin_only() -> None:
    from src.agent_runtime.capability_graph import CapabilityGraph
    from src.bot.main import _capability_status_reply
    from src.skills.base import AgentContext

    reply = _capability_status_reply(
        "/capability_status",
        AgentContext(user_id=2, chat_id=1, mode="personal"),
        admin_user_id=1,
        capability_graph=CapabilityGraph(),
    )

    assert reply == "Эта команда доступна только Никите."


def test_capability_status_command_renders_secret_free_graph_summary() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityGraph,
        CapabilityKind,
        CapabilityNode,
        CapabilityStatus,
    )
    from src.bot.main import _capability_status_reply
    from src.skills.base import AgentContext

    graph = CapabilityGraph(
        capabilities=(
            CapabilityNode(
                id="agent_profile.telegram_mcp.personal_actions",
                label="personal Telegram actions",
                kind=CapabilityKind.AGENT_PROFILE,
                status=CapabilityStatus.DEGRADED,
                reason="personal Telegram MCP session is missing",
                evidence=("super_secret_session",),
            ),
        )
    )

    reply = _capability_status_reply(
        "/capability_status",
        AgentContext(user_id=1, chat_id=1, mode="personal"),
        admin_user_id=1,
        capability_graph=graph,
    )

    assert reply is not None
    assert "Внутренний граф возможностей" in reply
    assert "agent_profile.telegram_mcp.personal_actions: degraded" in reply
    assert "personal Telegram MCP session is missing" in reply
    assert "super_secret_session" not in reply


def test_digital_scenarios_command_is_admin_only() -> None:
    from src.agent_runtime.capability_graph import CapabilityGraph
    from src.bot.main import _digital_scenarios_reply
    from src.skills.base import AgentContext

    reply = _digital_scenarios_reply(
        "/digital_scenarios",
        AgentContext(user_id=2, chat_id=1, mode="personal"),
        admin_user_id=1,
        capability_graph=CapabilityGraph(),
    )

    assert reply == "Эта команда доступна только Никите."


def test_digital_scenarios_command_renders_eval_coverage() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityGraph,
        CapabilityKind,
        CapabilityNode,
        CapabilityStatus,
    )
    from src.agent_runtime.digital_scenarios import BUILTIN_DIGITAL_SCENARIOS
    from src.bot.main import _digital_scenarios_reply
    from src.skills.base import AgentContext

    scenario = next(
        item for item in BUILTIN_DIGITAL_SCENARIOS if item.id == "ai_cto_projects"
    )
    graph = CapabilityGraph(
        capabilities=(
            CapabilityNode(
                id="digital_scenario.ai_cto_projects",
                label="AI-CTO для проектов",
                kind=CapabilityKind.DIGITAL_SCENARIO,
                status=CapabilityStatus.AVAILABLE,
                reason="all required runtime surfaces are available",
            ),
            *(
                CapabilityNode(
                    id=node_id,
                    label=node_id,
                    kind=CapabilityKind.SKILL,
                    status=CapabilityStatus.AVAILABLE,
                )
                for node_id in scenario.required_capability_nodes
            ),
        )
    )

    reply = _digital_scenarios_reply(
        "/digital_scenarios ai_cto_projects",
        AgentContext(user_id=1, chat_id=1, mode="personal"),
        admin_user_id=1,
        capability_graph=graph,
    )

    assert reply is not None
    assert "## Digital scenario coverage" in reply
    assert "digital_scenario.ai_cto_projects: available; eval 7/7" in reply
    assert "Artifacts:" in reply
    assert "Approval:" in reply
    assert "Chat surface: natural_language_user_flow" in reply


def test_digital_scenarios_command_renders_live_matrix_artifact() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityGraph,
        CapabilityKind,
        CapabilityNode,
        CapabilityStatus,
    )
    from src.agent_runtime.digital_scenarios import BUILTIN_DIGITAL_SCENARIOS
    from src.bot.main import _digital_scenarios_reply
    from src.skills.base import AgentContext

    scenario = next(
        item
        for item in BUILTIN_DIGITAL_SCENARIOS
        if item.id == "autonomous_niche_researcher"
    )
    graph = CapabilityGraph(
        capabilities=(
            CapabilityNode(
                id="digital_scenario.autonomous_niche_researcher",
                label="Автономный исследователь ниши",
                kind=CapabilityKind.DIGITAL_SCENARIO,
                status=CapabilityStatus.AVAILABLE,
                reason="all required runtime surfaces are available",
            ),
            *(
                CapabilityNode(
                    id=node_id,
                    label=node_id,
                    kind=CapabilityKind.AGENT_CAPABILITY,
                    status=CapabilityStatus.AVAILABLE,
                )
                for node_id in scenario.required_capability_nodes
            ),
        )
    )

    reply = _digital_scenarios_reply(
        "/digital_scenarios autonomous_niche_researcher matrix",
        AgentContext(user_id=1, chat_id=1, mode="personal"),
        admin_user_id=1,
        capability_graph=graph,
    )

    assert reply is not None
    assert "## Digital scenario live matrix" in reply
    assert "digital_scenario.autonomous_niche_researcher" in reply
    assert "- happy_path:" in reply
    assert "runtime_evidence" in reply
    assert "source_actor_or_test_path" in reply


def test_digital_scenarios_command_renders_live_evidence_summary() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityGraph,
        CapabilityKind,
        CapabilityNode,
        CapabilityStatus,
    )
    from src.agent_runtime.digital_scenario_coverage import DigitalScenarioLiveEvidence
    from src.agent_runtime.digital_scenarios import BUILTIN_DIGITAL_SCENARIOS
    from src.bot.main import _digital_scenarios_reply
    from src.skills.base import AgentContext

    scenario = next(
        item
        for item in BUILTIN_DIGITAL_SCENARIOS
        if item.id == "autonomous_niche_researcher"
    )
    graph = CapabilityGraph(
        capabilities=(
            CapabilityNode(
                id="digital_scenario.autonomous_niche_researcher",
                label="Автономный исследователь ниши",
                kind=CapabilityKind.DIGITAL_SCENARIO,
                status=CapabilityStatus.AVAILABLE,
                reason="all required runtime surfaces are available",
            ),
            *(
                CapabilityNode(
                    id=node_id,
                    label=node_id,
                    kind=CapabilityKind.AGENT_CAPABILITY,
                    status=CapabilityStatus.AVAILABLE,
                )
                for node_id in scenario.required_capability_nodes
            ),
        )
    )

    reply = _digital_scenarios_reply(
        "/digital_scenarios autonomous_niche_researcher evidence",
        AgentContext(user_id=1, chat_id=1, mode="personal"),
        admin_user_id=1,
        capability_graph=graph,
        live_evidence_records=(
            DigitalScenarioLiveEvidence(
                scenario_id="autonomous_niche_researcher",
                variant="happy_path",
                source_actor="codex_operator",
                chat_message_id="vscode-chat:123",
                runtime_evidence=("job=web_research:done",),
                structured_observation_or_result="Context Capsule returned findings.",
                limitations_or_unknowns="No external submissions were attempted.",
                artifact_refs=("reports/niche-research.md",),
                approval_boundary_respected=True,
            ),
        ),
    )

    assert reply is not None
    assert "## Digital scenario live evidence" in reply
    assert "digital_scenario.autonomous_niche_researcher" in reply
    assert "Covered variants: 1/7" in reply
    assert "- happy_path: complete" in reply
    assert "- paraphrase: missing result" in reply


def test_digital_scenarios_command_loads_persisted_live_evidence(tmp_path) -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityGraph,
        CapabilityKind,
        CapabilityNode,
        CapabilityStatus,
    )
    from src.agent_runtime.digital_scenario_coverage import (
        DigitalScenarioLiveEvidence,
        append_digital_scenario_live_evidence,
    )
    from src.agent_runtime.digital_scenarios import BUILTIN_DIGITAL_SCENARIOS
    from src.bot.main import _digital_scenarios_reply
    from src.skills.base import AgentContext

    scenario = next(
        item for item in BUILTIN_DIGITAL_SCENARIOS if item.id == "ai_cto_projects"
    )
    graph = CapabilityGraph(
        capabilities=(
            CapabilityNode(
                id="digital_scenario.ai_cto_projects",
                label="AI-CTO для проектов",
                kind=CapabilityKind.DIGITAL_SCENARIO,
                status=CapabilityStatus.AVAILABLE,
            ),
            *(
                CapabilityNode(
                    id=node_id,
                    label=node_id,
                    kind=CapabilityKind.SKILL,
                    status=CapabilityStatus.AVAILABLE,
                )
                for node_id in scenario.required_capability_nodes
            ),
        )
    )
    append_digital_scenario_live_evidence(
        tmp_path,
        DigitalScenarioLiveEvidence(
            scenario_id="ai_cto_projects",
            variant="happy_path",
            source_actor="codex_operator",
            chat_message_id="vscode:-7331:42",
            runtime_evidence=("skill=codebase_explorer", "success=true"),
            structured_observation_or_result="Structured audit result.",
            limitations_or_unknowns="No writes were attempted.",
            declared_no_artifact=True,
            approval_boundary_respected=True,
        ),
    )

    reply = _digital_scenarios_reply(
        "/digital_scenarios ai_cto_projects evidence",
        AgentContext(user_id=1, chat_id=1, mode="personal"),
        admin_user_id=1,
        capability_graph=graph,
        workspace_root=tmp_path,
    )

    assert reply is not None
    assert "Covered variants: 1/7" in reply
    assert "- happy_path: complete" in reply
    assert "skill=codebase_explorer" in reply


def test_external_skill_registry_records_are_loaded_for_capability_graph(
    tmp_path,
) -> None:
    from src.bot.main import _external_skill_records_for_capability_graph
    from src.skills.external_skill_loader.loader import (
        ExternalSkillAuditReport,
        ExternalSkillSource,
        ExternalSkillStatus,
        FilePersonalSkillRegistry,
        PersonalSkillRegistryRecord,
    )

    registry = FilePersonalSkillRegistry(tmp_path / "registry")
    registry._write(
        PersonalSkillRegistryRecord(
            skill_id="kube-debug",
            name="kube-debug",
            source=ExternalSkillSource(source_type="local_folder", locator="source"),
            quarantine_path=str(tmp_path / "quarantine" / "kube-debug"),
            content_hash="abc",
            status=ExternalSkillStatus.APPROVED_READONLY,
            audit_report=ExternalSkillAuditReport(
                skill_id="kube-debug",
                name="kube-debug",
                status=ExternalSkillStatus.NEEDS_REVIEW,
                risk_level="low",
                read_only_allowed=True,
            ),
            readonly_approval_id="approval-readonly",
        )
    )

    records = _external_skill_records_for_capability_graph(registry)

    assert len(records) == 1
    assert records[0].skill_id == "kube-debug"


def test_corrupt_external_skill_registry_does_not_break_capability_graph_startup(
    tmp_path,
) -> None:
    from src.bot.main import _external_skill_records_for_capability_graph
    from src.skills.external_skill_loader.loader import FilePersonalSkillRegistry

    registry = FilePersonalSkillRegistry(tmp_path / "registry")
    (tmp_path / "registry" / "broken.json").write_text("{not json", encoding="utf-8")

    assert _external_skill_records_for_capability_graph(registry) == ()

"""Digital-agent polygon coverage contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any


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
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_builtin_digital_scenarios_cover_objective_polygons_and_eval_variants() -> None:
    from src.agent_runtime.digital_scenarios import (
        BUILTIN_DIGITAL_SCENARIOS,
        NATURAL_LANGUAGE_CHAT_SURFACE,
        REQUIRED_DIGITAL_SCENARIO_IDS,
        REQUIRED_EVAL_VARIANTS,
    )

    by_id = {scenario.id: scenario for scenario in BUILTIN_DIGITAL_SCENARIOS}

    assert set(by_id) == set(REQUIRED_DIGITAL_SCENARIO_IDS)
    for scenario in BUILTIN_DIGITAL_SCENARIOS:
        variants = {case.variant for case in scenario.eval_cases}
        assert variants == set(REQUIRED_EVAL_VARIANTS)
        assert len(scenario.task_family) > 30
        assert scenario.invariants
        assert scenario.required_capability_nodes
        assert scenario.approval_boundaries
        assert scenario.chat_surface == NATURAL_LANGUAGE_CHAT_SURFACE
        assert not any(
            "одна фраза" in invariant.lower() for invariant in scenario.invariants
        )
        assert not any(
            case.prompt.strip().startswith("/") for case in scenario.eval_cases
        )
        assert not any(story.strip().startswith("/") for story in scenario.user_stories)


def test_goal_polygons_use_chat_ready_requirements_not_future_daemon_flags() -> None:
    from src.agent_runtime.digital_scenarios import BUILTIN_DIGITAL_SCENARIOS

    goal_polygons = {
        "personal_ops_hq",
        "ai_cto_projects",
        "agent_designer",
        "digital_twin_work_style",
        "external_skill_lab",
        "autonomous_niche_researcher",
        "project_archivist_biographer",
        "execution_partner",
    }
    future_autonomy_nodes = {
        "agent_definition.agency",
        "agent_profile.agency.readonly_draft",
        "agent_profile.life_reflection.readonly",
        "config.life_runtime",
        "config.daemon",
    }

    by_id = {scenario.id: scenario for scenario in BUILTIN_DIGITAL_SCENARIOS}

    for scenario_id in goal_polygons:
        scenario = by_id[scenario_id]
        assert not set(scenario.required_capability_nodes) & future_autonomy_nodes


def test_capability_graph_adds_digital_scenario_nodes_and_derives_status() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityKind,
        CapabilityStatus,
        build_capability_graph,
    )
    from src.agent_runtime.profiles import WEB_RESEARCH_READONLY
    from src.agent_runtime.tools import FunctionAgentTool, ToolGateway

    async def noop(_payload: dict[str, Any]) -> str:
        return "ok"

    unavailable = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(),
        active_skill_names=(),
        startup_skill_names=(),
        invocation_profiles=(WEB_RESEARCH_READONLY,),
        registered_worker_names=(),
        tool_gateways=(),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )
    available = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(),
        active_skill_names=("web_research",),
        startup_skill_names=("web_research",),
        invocation_profiles=(WEB_RESEARCH_READONLY,),
        registered_worker_names=("web_research",),
        tool_gateways=(
            ToolGateway(
                tools=(
                    FunctionAgentTool("web_search_sources", "web_search_sources", noop),
                    FunctionAgentTool("browser_read_url", "browser_read", noop),
                    FunctionAgentTool(
                        "browser_screenshot_url",
                        "browser_screenshot",
                        noop,
                    ),
                ),
            ),
        ),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )

    unavailable_node = unavailable.require(
        "digital_scenario.autonomous_niche_researcher"
    )
    available_node = available.require("digital_scenario.autonomous_niche_researcher")

    assert unavailable_node.kind is CapabilityKind.DIGITAL_SCENARIO
    assert unavailable_node.status is not CapabilityStatus.AVAILABLE
    assert available_node.status is CapabilityStatus.AVAILABLE
    assert "eval_cases=7" in available_node.evidence


def test_manager_summary_pins_digital_scenario_nodes() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityGraph,
        CapabilityKind,
        CapabilityNode,
        CapabilityStatus,
    )

    graph = CapabilityGraph(
        capabilities=(
            CapabilityNode(
                id="agent_capability.noisy.blocker_1.web_search_sources",
                label="noise",
                kind=CapabilityKind.AGENT_CAPABILITY,
                status=CapabilityStatus.BLOCKED,
                reason="noisy blocker",
            ),
            CapabilityNode(
                id="digital_scenario.ai_cto_projects",
                label="AI-CTO для проектов",
                kind=CapabilityKind.DIGITAL_SCENARIO,
                status=CapabilityStatus.DEGRADED,
                reason="2/5 runtime surfaces available",
            ),
        )
    )

    summary = graph.format_manager_summary(max_items=1)

    assert "digital_scenario.ai_cto_projects: degraded" in summary


def test_digital_scenario_coverage_summary_exposes_eval_and_gaps() -> None:
    from src.agent_runtime.capability_graph import build_capability_graph
    from src.agent_runtime.digital_scenario_coverage import (
        build_digital_scenario_coverage,
        render_digital_scenario_coverage_summary,
    )
    from src.agent_runtime.profiles import WEB_RESEARCH_READONLY
    from src.agent_runtime.tools import FunctionAgentTool, ToolGateway

    async def noop(_payload: dict[str, Any]) -> str:
        return "ok"

    graph = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(),
        active_skill_names=("web_research",),
        startup_skill_names=("web_research",),
        invocation_profiles=(WEB_RESEARCH_READONLY,),
        registered_worker_names=("web_research",),
        tool_gateways=(
            ToolGateway(
                tools=(
                    FunctionAgentTool("web_search_sources", "web_search_sources", noop),
                    FunctionAgentTool("browser_read_url", "browser_read", noop),
                    FunctionAgentTool(
                        "browser_screenshot_url",
                        "browser_screenshot",
                        noop,
                    ),
                ),
            ),
        ),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )

    coverage = build_digital_scenario_coverage(graph)
    by_id = {item.id: item for item in coverage}
    summary = render_digital_scenario_coverage_summary(coverage)

    assert by_id["autonomous_niche_researcher"].ready_for_live_matrix is True
    assert by_id["autonomous_niche_researcher"].case_count == 7
    assert by_id["personal_ops_hq"].ready_for_live_matrix is False
    assert "skill.morning_digest" in by_id["personal_ops_hq"].missing_required_nodes
    assert (
        "digital_scenario.autonomous_niche_researcher: available; eval 7/7" in summary
    )
    assert "digital_scenario.personal_ops_hq: disabled; eval 7/7" in summary
    assert "missing skill.morning_digest" in summary
    detail = render_digital_scenario_coverage_summary(
        coverage,
        scenario_id="autonomous_niche_researcher",
    )
    assert "Chat surface: natural_language_user_flow" in detail


def test_digital_scenario_matrix_artifact_lists_cases_and_required_evidence() -> None:
    from src.agent_runtime.capability_graph import build_capability_graph
    from src.agent_runtime.digital_scenario_coverage import (
        build_digital_scenario_coverage,
        build_digital_scenario_matrix_artifact,
        render_digital_scenario_matrix_artifact,
    )
    from src.agent_runtime.digital_scenarios import REQUIRED_EVAL_VARIANTS
    from src.agent_runtime.profiles import WEB_RESEARCH_READONLY
    from src.agent_runtime.tools import FunctionAgentTool, ToolGateway

    async def noop(_payload: dict[str, Any]) -> str:
        return "ok"

    graph = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(),
        active_skill_names=("web_research",),
        startup_skill_names=("web_research",),
        invocation_profiles=(WEB_RESEARCH_READONLY,),
        registered_worker_names=("web_research",),
        tool_gateways=(
            ToolGateway(
                tools=(
                    FunctionAgentTool("web_search_sources", "web_search_sources", noop),
                    FunctionAgentTool("browser_read_url", "browser_read", noop),
                    FunctionAgentTool(
                        "browser_screenshot_url",
                        "browser_screenshot",
                        noop,
                    ),
                ),
            ),
        ),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )

    coverage = build_digital_scenario_coverage(graph)
    ready_artifact = build_digital_scenario_matrix_artifact(
        coverage,
        "autonomous_niche_researcher",
    )
    blocked_artifact = build_digital_scenario_matrix_artifact(
        coverage,
        "personal_ops_hq",
    )

    assert ready_artifact is not None
    assert ready_artifact.ready_for_live_matrix is True
    assert {case.variant for case in ready_artifact.cases} == set(
        REQUIRED_EVAL_VARIANTS
    )
    assert all(
        "runtime_evidence" in case.required_evidence for case in ready_artifact.cases
    )
    assert all(
        "source_actor_or_test_path" in case.required_evidence
        for case in ready_artifact.cases
    )
    assert blocked_artifact is not None
    assert blocked_artifact.ready_for_live_matrix is False
    assert "skill.morning_digest" in blocked_artifact.missing_required_nodes

    rendered = render_digital_scenario_matrix_artifact(ready_artifact)

    assert "## Digital scenario live matrix" in rendered
    assert "digital_scenario.autonomous_niche_researcher" in rendered
    assert "- happy_path:" in rendered
    assert "Required evidence:" in rendered


def test_digital_scenario_live_evidence_summary_requires_all_variants() -> None:
    from src.agent_runtime.capability_graph import build_capability_graph
    from src.agent_runtime.digital_scenario_coverage import (
        DigitalScenarioLiveEvidence,
        build_digital_scenario_coverage,
        build_digital_scenario_live_evidence_summary,
        render_digital_scenario_live_evidence_summary,
    )
    from src.agent_runtime.profiles import WEB_RESEARCH_READONLY
    from src.agent_runtime.tools import FunctionAgentTool, ToolGateway

    async def noop(_payload: dict[str, Any]) -> str:
        return "ok"

    graph = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(),
        active_skill_names=("web_research",),
        startup_skill_names=("web_research",),
        invocation_profiles=(WEB_RESEARCH_READONLY,),
        registered_worker_names=("web_research",),
        tool_gateways=(
            ToolGateway(
                tools=(
                    FunctionAgentTool("web_search_sources", "web_search_sources", noop),
                    FunctionAgentTool("browser_read_url", "browser_read", noop),
                    FunctionAgentTool(
                        "browser_screenshot_url",
                        "browser_screenshot",
                        noop,
                    ),
                ),
            ),
        ),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )
    coverage = build_digital_scenario_coverage(graph)
    evidence = DigitalScenarioLiveEvidence(
        scenario_id="autonomous_niche_researcher",
        variant="happy_path",
        source_actor="codex_operator",
        chat_message_id="vscode-chat:123",
        runtime_evidence=("job=web_research:done", "sources=3"),
        structured_observation_or_result="Context Capsule returned findings.",
        limitations_or_unknowns="No external submissions were attempted.",
        artifact_refs=("reports/niche-research.md",),
        approval_boundary_respected=True,
    )

    summary = build_digital_scenario_live_evidence_summary(
        coverage,
        records=(evidence,),
        scenario_id="autonomous_niche_researcher",
    )

    assert summary is not None
    assert summary.scenario_complete is False
    assert summary.covered_variants == ("happy_path",)
    assert "paraphrase" in summary.missing_variants
    assert summary.variant_statuses[0].complete is True
    assert summary.variant_statuses[1].complete is False

    rendered = render_digital_scenario_live_evidence_summary(summary)

    assert "## Digital scenario live evidence" in rendered
    assert "Covered variants: 1/7" in rendered
    assert "- happy_path: complete" in rendered
    assert "- paraphrase: missing result" in rendered


def test_digital_scenario_live_evidence_rejects_user_impersonation_record() -> None:
    from src.agent_runtime.capability_graph import build_capability_graph
    from src.agent_runtime.digital_scenario_coverage import (
        DigitalScenarioLiveEvidence,
        build_digital_scenario_coverage,
        build_digital_scenario_live_evidence_summary,
    )
    from src.agent_runtime.profiles import WEB_RESEARCH_READONLY
    from src.agent_runtime.tools import FunctionAgentTool, ToolGateway

    async def noop(_payload: dict[str, Any]) -> str:
        return "ok"

    graph = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(),
        active_skill_names=("web_research",),
        startup_skill_names=("web_research",),
        invocation_profiles=(WEB_RESEARCH_READONLY,),
        registered_worker_names=("web_research",),
        tool_gateways=(
            ToolGateway(
                tools=(
                    FunctionAgentTool("web_search_sources", "web_search_sources", noop),
                    FunctionAgentTool("browser_read_url", "browser_read", noop),
                    FunctionAgentTool(
                        "browser_screenshot_url",
                        "browser_screenshot",
                        noop,
                    ),
                ),
            ),
        ),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )
    coverage = build_digital_scenario_coverage(graph)
    impersonated = DigitalScenarioLiveEvidence(
        scenario_id="autonomous_niche_researcher",
        variant="happy_path",
        source_actor="user",
        chat_message_id="vscode-chat:124",
        runtime_evidence=("job=web_research:done",),
        structured_observation_or_result="Context Capsule returned findings.",
        limitations_or_unknowns="No external submissions were attempted.",
        artifact_refs=("reports/niche-research.md",),
        approval_boundary_respected=True,
    )

    summary = build_digital_scenario_live_evidence_summary(
        coverage,
        records=(impersonated,),
        scenario_id="autonomous_niche_researcher",
    )

    assert summary is not None
    happy_path = summary.variant_statuses[0]
    assert happy_path.complete is False
    assert "source_actor_or_test_path" in happy_path.missing_evidence


def test_digital_scenario_live_evidence_marks_complete_only_when_all_variants_proved() -> (
    None
):
    from src.agent_runtime.capability_graph import build_capability_graph
    from src.agent_runtime.digital_scenario_coverage import (
        DigitalScenarioLiveEvidence,
        build_digital_scenario_coverage,
        build_digital_scenario_live_evidence_summary,
    )
    from src.agent_runtime.digital_scenarios import REQUIRED_EVAL_VARIANTS
    from src.agent_runtime.profiles import WEB_RESEARCH_READONLY
    from src.agent_runtime.tools import FunctionAgentTool, ToolGateway

    async def noop(_payload: dict[str, Any]) -> str:
        return "ok"

    graph = build_capability_graph(
        project_root=_project_root(),
        settings=_settings(),
        active_skill_names=("web_research",),
        startup_skill_names=("web_research",),
        invocation_profiles=(WEB_RESEARCH_READONLY,),
        registered_worker_names=("web_research",),
        tool_gateways=(
            ToolGateway(
                tools=(
                    FunctionAgentTool("web_search_sources", "web_search_sources", noop),
                    FunctionAgentTool("browser_read_url", "browser_read", noop),
                    FunctionAgentTool(
                        "browser_screenshot_url",
                        "browser_screenshot",
                        noop,
                    ),
                ),
            ),
        ),
        daemon_tool_names=(),
        mcp_config_path=_project_root() / ".mcp.json",
    )
    coverage = build_digital_scenario_coverage(graph)
    records = tuple(
        DigitalScenarioLiveEvidence(
            scenario_id="autonomous_niche_researcher",
            variant=variant,
            source_actor="automated_test",
            test_path=f"tests/live/{variant}.md",
            chat_message_id=f"vscode-chat:{variant}",
            runtime_evidence=(f"variant={variant}",),
            structured_observation_or_result="Structured result captured.",
            limitations_or_unknowns="No external submissions were attempted.",
            artifact_refs=(),
            declared_no_artifact=True,
            approval_boundary_respected=True,
        )
        for variant in REQUIRED_EVAL_VARIANTS
    )

    summary = build_digital_scenario_live_evidence_summary(
        coverage,
        records=records,
        scenario_id="autonomous_niche_researcher",
    )

    assert summary is not None
    assert summary.scenario_complete is True
    assert summary.covered_variants == REQUIRED_EVAL_VARIANTS
    assert summary.missing_variants == ()


def test_digital_scenario_live_evidence_store_roundtrips_recent_records(
    tmp_path,
) -> None:
    from src.agent_runtime.digital_scenario_coverage import (
        DigitalScenarioLiveEvidence,
        append_digital_scenario_live_evidence,
        digital_scenario_live_evidence_path,
        load_digital_scenario_live_evidence,
    )

    record = DigitalScenarioLiveEvidence(
        scenario_id="ai_cto_projects",
        variant="happy_path",
        source_actor="codex_operator",
        chat_message_id="vscode:-7331:42",
        runtime_evidence=("skill=codebase_explorer", "success=true"),
        structured_observation_or_result="Structured audit result.",
        limitations_or_unknowns="No writes were attempted.",
        declared_no_artifact=True,
        approval_boundary_respected=True,
        created_at="2026-05-24T00:00:00+00:00",
    )

    path = append_digital_scenario_live_evidence(tmp_path, record)
    path.write_text(
        path.read_text(encoding="utf-8") + "not json\n",
        encoding="utf-8",
    )

    loaded = load_digital_scenario_live_evidence(tmp_path)

    assert path == digital_scenario_live_evidence_path(tmp_path)
    assert len(loaded) == 1
    assert loaded[0] == record

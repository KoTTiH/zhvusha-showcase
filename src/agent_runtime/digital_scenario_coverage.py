"""Coverage view for generalized digital-agent scenario contracts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.agent_runtime.capability_graph import (
    CapabilityGraph,
    CapabilityNode,
    CapabilityStatus,
)
from src.agent_runtime.digital_scenarios import (
    BUILTIN_DIGITAL_SCENARIOS,
    REQUIRED_EVAL_VARIANTS,
    DigitalScenarioDefinition,
)

_BLOCKING_STATUSES = {
    CapabilityStatus.BLOCKED,
    CapabilityStatus.QUARANTINED,
    CapabilityStatus.NEEDS_REVIEW,
}
_LIVE_EVIDENCE_RELATIVE_PATH = (
    Path("runtime") / "digital_scenarios" / "live_evidence.jsonl"
)


@dataclass(frozen=True)
class DigitalScenarioCoverage:
    """Runtime coverage fact for one digital-agent scenario family."""

    id: str
    title: str
    status: CapabilityStatus
    reason: str
    task_family: str
    user_stories: tuple[str, ...]
    invariants: tuple[str, ...]
    memory_surfaces: tuple[str, ...]
    artifact_types: tuple[str, ...]
    approval_boundaries: tuple[str, ...]
    chat_surface: str
    case_count: int
    required_case_count: int
    available_required_nodes: tuple[str, ...]
    missing_required_nodes: tuple[str, ...]
    blocked_required_nodes: tuple[str, ...]

    @property
    def ready_for_live_matrix(self) -> bool:
        """Return whether all declared runtime surfaces and matrix cases exist."""
        return (
            self.status is CapabilityStatus.AVAILABLE
            and self.case_count == self.required_case_count
            and not self.missing_required_nodes
            and not self.blocked_required_nodes
        )


@dataclass(frozen=True)
class DigitalScenarioMatrixCase:
    """One live-check case with required evidence for a scenario family."""

    variant: str
    prompt: str
    expected_behavior: str
    required_evidence: tuple[str, ...]


@dataclass(frozen=True)
class DigitalScenarioMatrixArtifact:
    """Operator-facing artifact that makes scenario live checks auditable."""

    scenario_id: str
    title: str
    status: CapabilityStatus
    ready_for_live_matrix: bool
    missing_required_nodes: tuple[str, ...]
    approval_boundaries: tuple[str, ...]
    cases: tuple[DigitalScenarioMatrixCase, ...]


@dataclass(frozen=True)
class DigitalScenarioLiveEvidence:
    """One sanitized live result record for a matrix variant."""

    scenario_id: str
    variant: str
    source_actor: str = ""
    test_path: str = ""
    chat_message_id: str = ""
    runtime_evidence: tuple[str, ...] = ()
    structured_observation_or_result: str = ""
    limitations_or_unknowns: str = ""
    artifact_refs: tuple[str, ...] = ()
    declared_no_artifact: bool = False
    approval_boundary_respected: bool = False
    created_at: str = ""


@dataclass(frozen=True)
class DigitalScenarioLiveVariantStatus:
    """Evidence completeness status for one matrix variant."""

    scenario_id: str
    variant: str
    complete: bool
    missing_evidence: tuple[str, ...]
    evidence: DigitalScenarioLiveEvidence | None = None


@dataclass(frozen=True)
class DigitalScenarioLiveEvidenceSummary:
    """Auditable scenario-level proof state from live result records."""

    scenario_id: str
    title: str
    ready_for_live_matrix: bool
    scenario_complete: bool
    record_count: int
    required_variant_count: int
    covered_variants: tuple[str, ...]
    missing_variants: tuple[str, ...]
    variant_statuses: tuple[DigitalScenarioLiveVariantStatus, ...]


def digital_scenario_live_evidence_path(workspace_root: Path) -> Path:
    """Return the append-only live evidence ledger path for digital scenarios."""
    return workspace_root.expanduser().resolve() / _LIVE_EVIDENCE_RELATIVE_PATH


def append_digital_scenario_live_evidence(
    workspace_root: Path,
    record: DigitalScenarioLiveEvidence,
) -> Path:
    """Append one sanitized live evidence record and return the ledger path."""
    path = digital_scenario_live_evidence_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True))
        handle.write("\n")
    return path


def load_digital_scenario_live_evidence(
    workspace_root: Path,
    *,
    max_records: int = 500,
) -> tuple[DigitalScenarioLiveEvidence, ...]:
    """Load recent sanitized live evidence records from the append-only ledger."""
    path = digital_scenario_live_evidence_path(workspace_root)
    if not path.is_file():
        return ()
    records: list[DigitalScenarioLiveEvidence] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        records.append(_live_evidence_from_mapping(payload))
    return tuple(records[-max_records:])


def build_digital_scenario_coverage(
    graph: CapabilityGraph,
) -> tuple[DigitalScenarioCoverage, ...]:
    """Build an auditable coverage matrix from registry definitions and graph facts."""
    by_id = {node.id: node for node in graph.capabilities}
    return tuple(
        _coverage_for_scenario(scenario=scenario, capabilities_by_id=by_id)
        for scenario in BUILTIN_DIGITAL_SCENARIOS
    )


def render_digital_scenario_coverage_summary(
    coverage: tuple[DigitalScenarioCoverage, ...],
    *,
    scenario_id: str = "",
) -> str:
    """Render a secret-free operator summary for VS Code/Telegram admin chat."""
    normalized_id = scenario_id.strip().removeprefix("digital_scenario.")
    selected = (
        tuple(item for item in coverage if item.id == normalized_id)
        if normalized_id
        else coverage
    )
    lines = [
        "## Digital scenario coverage",
        "Показывает классы задач, matrix cases и реальные runtime gaps; это не proof of full completion.",
    ]
    if normalized_id and not selected:
        lines.append(f"Сценарий `{normalized_id}` не найден.")
        return "\n".join(lines)
    for item in selected:
        readiness = "live-ready" if item.ready_for_live_matrix else "needs gaps"
        missing = _format_node_list(item.missing_required_nodes)
        blocked = _format_node_list(item.blocked_required_nodes)
        suffix = f"; {readiness}"
        if missing:
            suffix += f"; missing {missing}"
        if blocked:
            suffix += f"; blocked {blocked}"
        lines.append(
            f"- digital_scenario.{item.id}: {item.status.value}; "
            f"{'eval'} {item.case_count}/{item.required_case_count}{suffix}"
        )
        if normalized_id:
            lines.extend(_render_detail_lines(item))
    return "\n".join(lines)


def build_digital_scenario_live_evidence_summary(
    coverage: tuple[DigitalScenarioCoverage, ...],
    *,
    records: tuple[DigitalScenarioLiveEvidence, ...],
    scenario_id: str,
) -> DigitalScenarioLiveEvidenceSummary | None:
    """Validate live records against the full matrix for one scenario family."""
    normalized_id = scenario_id.strip().removeprefix("digital_scenario.")
    if not normalized_id:
        return None
    coverage_by_id = {item.id: item for item in coverage}
    scenario_by_id = {item.id: item for item in BUILTIN_DIGITAL_SCENARIOS}
    item = coverage_by_id.get(normalized_id)
    scenario = scenario_by_id.get(normalized_id)
    if item is None or scenario is None:
        return None

    relevant_records = tuple(
        record
        for record in records
        if record.scenario_id.strip().removeprefix("digital_scenario.") == normalized_id
    )
    required_evidence = _required_live_evidence(item)
    variant_statuses = tuple(
        _best_live_status_for_variant(
            scenario_id=normalized_id,
            variant=case.variant,
            records=relevant_records,
            required_evidence=required_evidence,
        )
        for case in scenario.eval_cases
    )
    covered_variants = tuple(
        status.variant for status in variant_statuses if status.complete
    )
    missing_variants = tuple(
        status.variant for status in variant_statuses if not status.complete
    )
    scenario_complete = (
        item.ready_for_live_matrix
        and len(covered_variants) == len(scenario.eval_cases)
        and not missing_variants
    )
    return DigitalScenarioLiveEvidenceSummary(
        scenario_id=normalized_id,
        title=item.title,
        ready_for_live_matrix=item.ready_for_live_matrix,
        scenario_complete=scenario_complete,
        record_count=len(relevant_records),
        required_variant_count=len(scenario.eval_cases),
        covered_variants=covered_variants,
        missing_variants=missing_variants,
        variant_statuses=variant_statuses,
    )


def render_digital_scenario_live_evidence_summary(
    summary: DigitalScenarioLiveEvidenceSummary,
) -> str:
    """Render sanitized live proof state for operator/admin chat."""
    proof = "complete" if summary.scenario_complete else "incomplete"
    readiness = "ready" if summary.ready_for_live_matrix else "blocked_by_gaps"
    lines = [
        "## Digital scenario live evidence",
        f"Scenario: digital_scenario.{summary.scenario_id} ({summary.title})",
        f"Proof: {proof}; matrix readiness: {readiness}",
        (
            f"Covered variants: {len(summary.covered_variants)}/"
            f"{summary.required_variant_count}"
        ),
    ]
    if summary.record_count == 0:
        lines.append("Live result records: none")
    for status in summary.variant_statuses:
        if status.complete:
            lines.append(f"- {status.variant}: complete")
        elif status.evidence is None:
            lines.append(f"- {status.variant}: missing result")
        else:
            missing = ", ".join(status.missing_evidence)
            lines.append(f"- {status.variant}: incomplete; missing {missing}")
        if status.evidence is not None:
            lines.append(f"  Evidence: {_render_live_evidence_record(status.evidence)}")
    return "\n".join(lines)


def build_digital_scenario_matrix_artifact(
    coverage: tuple[DigitalScenarioCoverage, ...],
    scenario_id: str,
) -> DigitalScenarioMatrixArtifact | None:
    """Build a concrete live-check artifact for one scenario family."""
    normalized_id = scenario_id.strip().removeprefix("digital_scenario.")
    coverage_by_id = {item.id: item for item in coverage}
    scenario_by_id = {item.id: item for item in BUILTIN_DIGITAL_SCENARIOS}
    item = coverage_by_id.get(normalized_id)
    scenario = scenario_by_id.get(normalized_id)
    if item is None or scenario is None:
        return None
    return DigitalScenarioMatrixArtifact(
        scenario_id=item.id,
        title=item.title,
        status=item.status,
        ready_for_live_matrix=item.ready_for_live_matrix,
        missing_required_nodes=item.missing_required_nodes,
        approval_boundaries=item.approval_boundaries,
        cases=tuple(
            DigitalScenarioMatrixCase(
                variant=case.variant,
                prompt=case.prompt,
                expected_behavior=case.expected_behavior,
                required_evidence=_required_live_evidence(item),
            )
            for case in scenario.eval_cases
        ),
    )


def render_digital_scenario_matrix_artifact(
    artifact: DigitalScenarioMatrixArtifact,
) -> str:
    """Render a live-check artifact without secrets or raw internal logs."""
    readiness = "ready" if artifact.ready_for_live_matrix else "blocked_by_gaps"
    lines = [
        "## Digital scenario live matrix",
        f"Scenario: digital_scenario.{artifact.scenario_id} ({artifact.title})",
        f"Status: {artifact.status.value}; readiness: {readiness}",
        f"Approval boundaries: {', '.join(artifact.approval_boundaries)}",
    ]
    if artifact.missing_required_nodes:
        lines.append(
            "Missing runtime nodes: " + ", ".join(artifact.missing_required_nodes)
        )
    for case in artifact.cases:
        lines.append(f"- {case.variant}: {case.prompt}")
        lines.append(f"  Expected: {case.expected_behavior}")
        lines.append(f"  Required evidence: {', '.join(case.required_evidence)}")
    return "\n".join(lines)


def _best_live_status_for_variant(
    *,
    scenario_id: str,
    variant: str,
    records: tuple[DigitalScenarioLiveEvidence, ...],
    required_evidence: tuple[str, ...],
) -> DigitalScenarioLiveVariantStatus:
    variant_records = tuple(record for record in records if record.variant == variant)
    if not variant_records:
        return DigitalScenarioLiveVariantStatus(
            scenario_id=scenario_id,
            variant=variant,
            complete=False,
            missing_evidence=("live_result_record",),
        )
    statuses = tuple(
        _status_for_live_record(
            scenario_id=scenario_id,
            variant=variant,
            record=record,
            required_evidence=required_evidence,
        )
        for record in variant_records
    )
    complete_statuses = tuple(status for status in statuses if status.complete)
    if complete_statuses:
        return complete_statuses[-1]
    return statuses[-1]


def _status_for_live_record(
    *,
    scenario_id: str,
    variant: str,
    record: DigitalScenarioLiveEvidence,
    required_evidence: tuple[str, ...],
) -> DigitalScenarioLiveVariantStatus:
    missing = tuple(
        field
        for field in required_evidence
        if not _live_record_satisfies_field(record, field)
    )
    return DigitalScenarioLiveVariantStatus(
        scenario_id=scenario_id,
        variant=variant,
        complete=not missing,
        missing_evidence=missing,
        evidence=record,
    )


def _coverage_for_scenario(
    *,
    scenario: DigitalScenarioDefinition,
    capabilities_by_id: dict[str, CapabilityNode],
) -> DigitalScenarioCoverage:
    scenario_node = capabilities_by_id.get(f"digital_scenario.{scenario.id}")
    required_nodes = tuple(
        capabilities_by_id.get(node_id)
        for node_id in scenario.required_capability_nodes
    )
    available = tuple(
        node.id
        for node in required_nodes
        if node is not None and node.status is CapabilityStatus.AVAILABLE
    )
    blocked = tuple(
        node.id
        for node in required_nodes
        if node is not None and node.status in _BLOCKING_STATUSES
    )
    missing = tuple(
        node_id
        for node_id in scenario.required_capability_nodes
        if capabilities_by_id.get(node_id) is None
        or capabilities_by_id[node_id].status is not CapabilityStatus.AVAILABLE
    )
    status = (
        scenario_node.status
        if scenario_node is not None
        else _derive_missing_scenario_status(available=available, missing=missing)
    )
    reason = (
        scenario_node.reason
        if scenario_node is not None
        else "digital_scenario node is missing from CapabilityGraph"
    )
    return DigitalScenarioCoverage(
        id=scenario.id,
        title=scenario.title,
        status=status,
        reason=reason,
        task_family=scenario.task_family,
        user_stories=scenario.user_stories,
        invariants=scenario.invariants,
        memory_surfaces=scenario.memory_surfaces,
        artifact_types=scenario.artifact_types,
        approval_boundaries=scenario.approval_boundaries,
        chat_surface=scenario.chat_surface,
        case_count=len(scenario.eval_cases),
        required_case_count=len(REQUIRED_EVAL_VARIANTS),
        available_required_nodes=available,
        missing_required_nodes=missing,
        blocked_required_nodes=blocked,
    )


def _required_live_evidence(item: DigitalScenarioCoverage) -> tuple[str, ...]:
    evidence = [
        "source_actor_or_test_path",
        "chat_message_id",
        "runtime_evidence",
        "structured_observation_or_result",
        "limitations_or_unknowns",
    ]
    if item.artifact_types:
        evidence.append("artifact_or_declared_no_artifact")
    if item.approval_boundaries:
        evidence.append("approval_boundary_respected")
    return tuple(evidence)


def _live_record_satisfies_field(
    record: DigitalScenarioLiveEvidence,
    field: str,
) -> bool:
    if field == "source_actor_or_test_path":
        return _has_allowed_source_actor_or_test_path(record)
    if field == "chat_message_id":
        return _has_text(record.chat_message_id)
    if field == "runtime_evidence":
        return _has_any_text(record.runtime_evidence)
    if field == "structured_observation_or_result":
        return _has_text(record.structured_observation_or_result)
    if field == "limitations_or_unknowns":
        return _has_text(record.limitations_or_unknowns)
    if field == "artifact_or_declared_no_artifact":
        return _has_any_text(record.artifact_refs) or record.declared_no_artifact
    if field == "approval_boundary_respected":
        return record.approval_boundary_respected
    return False


def _live_evidence_from_mapping(
    payload: dict[str, object],
) -> DigitalScenarioLiveEvidence:
    return DigitalScenarioLiveEvidence(
        scenario_id=str(payload.get("scenario_id", "") or ""),
        variant=str(payload.get("variant", "") or ""),
        source_actor=str(payload.get("source_actor", "") or ""),
        test_path=str(payload.get("test_path", "") or ""),
        chat_message_id=str(payload.get("chat_message_id", "") or ""),
        runtime_evidence=_tuple_of_text(payload.get("runtime_evidence")),
        structured_observation_or_result=str(
            payload.get("structured_observation_or_result", "") or ""
        ),
        limitations_or_unknowns=str(payload.get("limitations_or_unknowns", "") or ""),
        artifact_refs=_tuple_of_text(payload.get("artifact_refs")),
        declared_no_artifact=bool(payload.get("declared_no_artifact", False)),
        approval_boundary_respected=bool(
            payload.get("approval_boundary_respected", False)
        ),
        created_at=str(payload.get("created_at", "") or ""),
    )


def _tuple_of_text(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, list | tuple):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _has_allowed_source_actor_or_test_path(
    record: DigitalScenarioLiveEvidence,
) -> bool:
    if _has_text(record.test_path):
        return True
    actor = record.source_actor.strip().casefold()
    if not actor:
        return False
    return actor not in {"user", "human", "nikita", "никита"}


def _has_text(value: str) -> bool:
    return bool(value.strip())


def _has_any_text(values: tuple[str, ...]) -> bool:
    return any(value.strip() for value in values)


def _render_live_evidence_record(record: DigitalScenarioLiveEvidence) -> str:
    parts = [f"message={record.chat_message_id or 'missing'}"]
    if record.source_actor:
        parts.append(f"actor={record.source_actor}")
    if record.test_path:
        parts.append(f"test={record.test_path}")
    if record.runtime_evidence:
        parts.append(f"runtime={_format_node_list(record.runtime_evidence)}")
    if record.artifact_refs:
        parts.append(f"artifacts={_format_node_list(record.artifact_refs)}")
    elif record.declared_no_artifact:
        parts.append("artifacts=declared_none")
    return "; ".join(parts)


def _derive_missing_scenario_status(
    *,
    available: tuple[str, ...],
    missing: tuple[str, ...],
) -> CapabilityStatus:
    if missing and available:
        return CapabilityStatus.DEGRADED
    if missing:
        return CapabilityStatus.DISABLED
    return CapabilityStatus.AVAILABLE


def _render_detail_lines(item: DigitalScenarioCoverage) -> list[str]:
    return [
        f"  Family: {item.task_family}",
        f"  Stories: {'; '.join(item.user_stories)}",
        f"  Invariants: {'; '.join(item.invariants)}",
        f"  Memory: {', '.join(item.memory_surfaces)}",
        f"  Artifacts: {', '.join(item.artifact_types)}",
        f"  Approval: {', '.join(item.approval_boundaries)}",
        f"  Chat surface: {item.chat_surface}",
        f"  Reason: {item.reason}",
    ]


def _format_node_list(nodes: tuple[str, ...], *, limit: int = 4) -> str:
    if not nodes:
        return ""
    rendered = ", ".join(nodes[:limit])
    if len(nodes) > limit:
        rendered += f", +{len(nodes) - limit}"
    return rendered

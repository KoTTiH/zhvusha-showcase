"""Doctor/status surface for the external skill registry."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ValidationError

from src.skills.external_skill_loader.loader import (
    ExternalSkillStatus,
    PersonalSkillRegistryRecord,
)


class ExternalSkillDoctorSeverity(StrEnum):
    """Severity for external skill doctor findings."""

    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    BLOCKER = "blocker"


class ExternalSkillDoctorFinding(BaseModel):
    """One operator-safe status finding."""

    code: str
    severity: ExternalSkillDoctorSeverity
    message: str
    skill_id: str = ""
    evidence: tuple[str, ...] = ()


class ExternalSkillRecoveryAction(BaseModel):
    """One recovery action or recommendation."""

    code: str
    description: str
    automatic: bool = False
    completed: bool = False
    affected_paths: tuple[str, ...] = ()


class ExternalSkillRegistrySummary(BaseModel):
    """Aggregated registry health summary."""

    total_records: int = 0
    active_records: int = 0
    corrupt_registry_files: int = 0
    missing_quarantine_paths: int = 0
    blocked_records: int = 0
    needs_review_records: int = 0
    approved_readonly_records: int = 0
    execution_approved_records: int = 0
    native_conversion_candidate_records: int = 0
    rejected_records: int = 0
    superseded_records: int = 0
    native_converted_records: int = 0


class ExternalSkillDoctorReport(BaseModel):
    """Complete doctor report for status/recovery surfaces."""

    summary: ExternalSkillRegistrySummary
    findings: tuple[ExternalSkillDoctorFinding, ...] = ()
    recovery_actions: tuple[ExternalSkillRecoveryAction, ...] = ()

    def render_for_operator(self) -> str:
        """Render a secret-free operator status report."""
        lines = [
            "External Skill Registry status",
            f"- total_records: {self.summary.total_records}",
            f"- active_records: {self.summary.active_records}",
            f"- corrupt_registry_files: {self.summary.corrupt_registry_files}",
            f"- missing_quarantine_paths: {self.summary.missing_quarantine_paths}",
            f"- blocked_records: {self.summary.blocked_records}",
            f"- needs_review_records: {self.summary.needs_review_records}",
            f"- approved_readonly_records: {self.summary.approved_readonly_records}",
            f"- execution_approved_records: {self.summary.execution_approved_records}",
            "- native_conversion_candidate_records: "
            f"{self.summary.native_conversion_candidate_records}",
            f"- rejected_records: {self.summary.rejected_records}",
            f"- superseded_records: {self.summary.superseded_records}",
            f"- native_converted_records: {self.summary.native_converted_records}",
        ]
        if self.findings:
            lines.append("")
            lines.append("Findings:")
            for finding in self.findings:
                skill = f" [{finding.skill_id}]" if finding.skill_id else ""
                lines.append(f"- {finding.severity.value}:{skill} {finding.code}")
                if finding.message:
                    lines.append(f"  {finding.message}")
        if self.recovery_actions:
            lines.append("")
            lines.append("Recovery actions:")
            for action in self.recovery_actions:
                status = "done" if action.completed else "manual"
                if action.automatic and not action.completed:
                    status = "available"
                lines.append(f"- {action.code}: {status}")
                lines.append(f"  {action.description}")
        return "\n".join(lines)


@dataclass(frozen=True)
class _LoadedRegistryState:
    records: tuple[PersonalSkillRegistryRecord, ...]
    corrupt_paths: tuple[Path, ...]


class ExternalSkillDoctor:
    """Inspect and recover the personal external skill registry."""

    def __init__(self, *, registry_root: Path, quarantine_root: Path) -> None:
        self.registry_root = registry_root.expanduser().resolve()
        self.quarantine_root = quarantine_root.expanduser().resolve()

    def inspect(self) -> ExternalSkillDoctorReport:
        """Return current external skill registry status without side effects."""
        state = self._load_registry_state()
        findings: list[ExternalSkillDoctorFinding] = []
        recovery_actions: list[ExternalSkillRecoveryAction] = []

        if state.corrupt_paths:
            findings.append(
                ExternalSkillDoctorFinding(
                    code="corrupt_registry_record",
                    severity=ExternalSkillDoctorSeverity.BLOCKER,
                    message=(
                        "One or more registry JSON files are unreadable; they are "
                        "ignored until recovered."
                    ),
                    evidence=_safe_path_labels(state.corrupt_paths),
                )
            )
            recovery_actions.append(
                ExternalSkillRecoveryAction(
                    code="recover_corrupt_registry_files",
                    description=(
                        "Move corrupt registry JSON files into registry/corrupt/ "
                        "so healthy records can still be inspected."
                    ),
                    automatic=True,
                    affected_paths=_safe_path_labels(state.corrupt_paths),
                )
            )

        for record in state.records:
            self._inspect_record(
                record=record,
                findings=findings,
                recovery_actions=recovery_actions,
            )

        return ExternalSkillDoctorReport(
            summary=_build_summary(
                records=state.records,
                corrupt_count=len(state.corrupt_paths),
                missing_quarantine_count=sum(
                    1
                    for finding in findings
                    if finding.code == "missing_quarantine_path"
                ),
            ),
            findings=tuple(findings),
            recovery_actions=tuple(recovery_actions),
        )

    def recover_corrupt_registry_files(self) -> ExternalSkillDoctorReport:
        """Move corrupt registry files aside, then return a fresh status report."""
        state = self._load_registry_state()
        moved: list[Path] = []
        corrupt_root = self.registry_root / "corrupt"
        corrupt_root.mkdir(parents=True, exist_ok=True)
        for path in state.corrupt_paths:
            destination = _unique_destination(corrupt_root / path.name)
            shutil.move(str(path), str(destination))
            moved.append(destination)
        report = self.inspect()
        if not moved:
            return report
        action = ExternalSkillRecoveryAction(
            code="corrupt_registry_files_moved",
            description="Corrupt registry JSON files were moved aside.",
            automatic=True,
            completed=True,
            affected_paths=_safe_path_labels(tuple(moved)),
        )
        return report.model_copy(
            update={"recovery_actions": (*report.recovery_actions, action)}
        )

    def _load_registry_state(self) -> _LoadedRegistryState:
        records: list[PersonalSkillRegistryRecord] = []
        corrupt_paths: list[Path] = []
        if not self.registry_root.exists():
            return _LoadedRegistryState(records=(), corrupt_paths=())
        for path in sorted(self.registry_root.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                records.append(PersonalSkillRegistryRecord(**data))
            except (OSError, json.JSONDecodeError, ValidationError):
                corrupt_paths.append(path)
        return _LoadedRegistryState(
            records=tuple(records),
            corrupt_paths=tuple(corrupt_paths),
        )

    def _inspect_record(
        self,
        *,
        record: PersonalSkillRegistryRecord,
        findings: list[ExternalSkillDoctorFinding],
        recovery_actions: list[ExternalSkillRecoveryAction],
    ) -> None:
        quarantine_path = Path(record.quarantine_path)
        if not quarantine_path.exists():
            findings.append(
                ExternalSkillDoctorFinding(
                    code="missing_quarantine_path",
                    severity=ExternalSkillDoctorSeverity.BLOCKER,
                    message="Registry record points to a missing quarantine folder.",
                    skill_id=record.skill_id,
                )
            )
            recovery_actions.append(
                ExternalSkillRecoveryAction(
                    code="reimport_or_block_missing_quarantine",
                    description=(
                        "Re-import the external skill from its approved source into "
                        "quarantine, or mark the registry record blocked."
                    ),
                    automatic=False,
                )
            )
        elif not _is_relative_to(quarantine_path, self.quarantine_root):
            findings.append(
                ExternalSkillDoctorFinding(
                    code="quarantine_path_outside_store",
                    severity=ExternalSkillDoctorSeverity.BLOCKER,
                    message="Quarantine path is outside the configured quarantine store.",
                    skill_id=record.skill_id,
                )
            )

        if record.status is ExternalSkillStatus.APPROVED_READONLY:
            _inspect_readonly_approval(record=record, findings=findings)

        if record.status is ExternalSkillStatus.EXECUTION_APPROVED:
            _inspect_readonly_approval(record=record, findings=findings)
            _inspect_execution_approval(record=record, findings=findings)

        if record.status is ExternalSkillStatus.NATIVE_CONVERSION_CANDIDATE:
            _inspect_readonly_approval(record=record, findings=findings)
            _inspect_native_conversion(record=record, findings=findings)

        if record.status is ExternalSkillStatus.NEEDS_REVIEW:
            findings.append(
                ExternalSkillDoctorFinding(
                    code="needs_review",
                    severity=ExternalSkillDoctorSeverity.WARNING,
                    message="Audit exists; read-only use still needs Никита approval.",
                    skill_id=record.skill_id,
                )
            )
        if record.status is ExternalSkillStatus.BLOCKED:
            findings.append(
                ExternalSkillDoctorFinding(
                    code="blocked_by_audit",
                    severity=ExternalSkillDoctorSeverity.INFO,
                    message="Record is blocked and not active.",
                    skill_id=record.skill_id,
                )
            )
        _inspect_curated_inactive_status(record=record, findings=findings)


def _build_summary(
    *,
    records: tuple[PersonalSkillRegistryRecord, ...],
    corrupt_count: int,
    missing_quarantine_count: int,
) -> ExternalSkillRegistrySummary:
    counts: Counter[str] = Counter(record.status.value for record in records)
    active_statuses = {
        ExternalSkillStatus.APPROVED_READONLY,
        ExternalSkillStatus.EXECUTION_APPROVED,
        ExternalSkillStatus.NATIVE_CONVERSION_CANDIDATE,
    }
    return ExternalSkillRegistrySummary(
        total_records=len(records),
        active_records=sum(1 for record in records if record.status in active_statuses),
        corrupt_registry_files=corrupt_count,
        missing_quarantine_paths=missing_quarantine_count,
        blocked_records=counts[ExternalSkillStatus.BLOCKED.value],
        needs_review_records=counts[ExternalSkillStatus.NEEDS_REVIEW.value],
        approved_readonly_records=counts[ExternalSkillStatus.APPROVED_READONLY.value],
        execution_approved_records=counts[ExternalSkillStatus.EXECUTION_APPROVED.value],
        native_conversion_candidate_records=counts[
            ExternalSkillStatus.NATIVE_CONVERSION_CANDIDATE.value
        ],
        rejected_records=counts[ExternalSkillStatus.REJECTED.value],
        superseded_records=counts[ExternalSkillStatus.SUPERSEDED.value],
        native_converted_records=counts[ExternalSkillStatus.NATIVE_CONVERTED.value],
    )


def _inspect_readonly_approval(
    *,
    record: PersonalSkillRegistryRecord,
    findings: list[ExternalSkillDoctorFinding],
) -> None:
    if not record.readonly_approval_id:
        findings.append(
            ExternalSkillDoctorFinding(
                code="approved_without_readonly_approval",
                severity=ExternalSkillDoctorSeverity.BLOCKER,
                message="Read-only approved record has no approval id.",
                skill_id=record.skill_id,
            )
        )
    if record.audit_report.blocked or not record.audit_report.read_only_allowed:
        findings.append(
            ExternalSkillDoctorFinding(
                code="approved_but_audit_blocks_readonly",
                severity=ExternalSkillDoctorSeverity.BLOCKER,
                message="Record is approved but its audit does not allow read-only use.",
                skill_id=record.skill_id,
            )
        )


def _inspect_execution_approval(
    *,
    record: PersonalSkillRegistryRecord,
    findings: list[ExternalSkillDoctorFinding],
) -> None:
    if not record.execution_approval_id:
        findings.append(
            ExternalSkillDoctorFinding(
                code="execution_without_execution_approval",
                severity=ExternalSkillDoctorSeverity.BLOCKER,
                message="Execution-approved record has no execution approval id.",
                skill_id=record.skill_id,
            )
        )
    if not record.approved_capabilities:
        findings.append(
            ExternalSkillDoctorFinding(
                code="execution_without_approved_capabilities",
                severity=ExternalSkillDoctorSeverity.BLOCKER,
                message="Execution-approved record has no scoped capabilities.",
                skill_id=record.skill_id,
            )
        )
    requested = set(record.audit_report.requested_capabilities)
    unknown = tuple(
        sorted(
            capability
            for capability in record.approved_capabilities
            if capability not in requested
        )
    )
    if unknown:
        findings.append(
            ExternalSkillDoctorFinding(
                code="execution_capability_not_requested_by_audit",
                severity=ExternalSkillDoctorSeverity.BLOCKER,
                message="Execution approval grants capability not requested by audit.",
                skill_id=record.skill_id,
                evidence=unknown,
            )
        )


def _inspect_native_conversion(
    *,
    record: PersonalSkillRegistryRecord,
    findings: list[ExternalSkillDoctorFinding],
) -> None:
    if not record.native_conversion_approval_id:
        findings.append(
            ExternalSkillDoctorFinding(
                code="native_conversion_without_approval",
                severity=ExternalSkillDoctorSeverity.BLOCKER,
                message="Native conversion candidate has no approval id.",
                skill_id=record.skill_id,
            )
        )
    if record.use_count <= 0:
        findings.append(
            ExternalSkillDoctorFinding(
                code="native_conversion_without_successful_use",
                severity=ExternalSkillDoctorSeverity.BLOCKER,
                message="Native conversion candidate has no successful use history.",
                skill_id=record.skill_id,
            )
        )
    if not record.native_conversion_reason:
        findings.append(
            ExternalSkillDoctorFinding(
                code="native_conversion_without_reason",
                severity=ExternalSkillDoctorSeverity.WARNING,
                message="Native conversion candidate has no operator-readable reason.",
                skill_id=record.skill_id,
            )
        )


def _inspect_curated_inactive_status(
    *,
    record: PersonalSkillRegistryRecord,
    findings: list[ExternalSkillDoctorFinding],
) -> None:
    messages = {
        ExternalSkillStatus.REJECTED: (
            "rejected_by_operator",
            "Record was rejected and is not active.",
        ),
        ExternalSkillStatus.SUPERSEDED: (
            "superseded_by_native_or_newer_skill",
            "Record was superseded and is not active.",
        ),
        ExternalSkillStatus.NATIVE_CONVERTED: (
            "native_converted",
            "Record was converted into a native skill and is not active.",
        ),
    }
    item = messages.get(record.status)
    if item is None:
        return
    code, message = item
    findings.append(
        ExternalSkillDoctorFinding(
            code=code,
            severity=ExternalSkillDoctorSeverity.INFO,
            message=message,
            skill_id=record.skill_id,
        )
    )


def _safe_path_labels(paths: tuple[Path, ...]) -> tuple[str, ...]:
    return tuple(path.name for path in paths)


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}.{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"cannot allocate recovery path for {path.name}")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True

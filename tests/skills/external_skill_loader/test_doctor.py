"""External skill registry doctor/status contracts."""

from __future__ import annotations

from pathlib import Path


def test_external_skill_doctor_renders_secret_free_status_and_recovery_hints(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_loader.doctor import (
        ExternalSkillDoctor,
        ExternalSkillDoctorSeverity,
    )
    from src.skills.external_skill_loader.loader import (
        ExternalSkillAuditReport,
        ExternalSkillSource,
        ExternalSkillStatus,
        FilePersonalSkillRegistry,
        PersonalSkillRegistryRecord,
    )

    registry = FilePersonalSkillRegistry(tmp_path / "registry")
    source = ExternalSkillSource(
        source_type="local_folder",
        locator=str(tmp_path / "source"),
        acquisition_approval_id="approval-acquire",
        approved_by_user_id=1291112109,
    )
    registry._write(
        PersonalSkillRegistryRecord(
            skill_id="approved",
            name="approved",
            source=source,
            quarantine_path=str(tmp_path / "quarantine" / "approved"),
            content_hash="abc",
            status=ExternalSkillStatus.APPROVED_READONLY,
            audit_report=ExternalSkillAuditReport(
                skill_id="approved",
                name="approved",
                status=ExternalSkillStatus.NEEDS_REVIEW,
                risk_level="high",
                requested_env_vars=("OPENAI_API_KEY",),
                read_only_allowed=True,
            ),
            readonly_approval_id="approval-readonly",
        )
    )
    registry._write(
        PersonalSkillRegistryRecord(
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
            ),
        )
    )
    (tmp_path / "registry" / "broken.json").write_text("{not-json", encoding="utf-8")

    report = ExternalSkillDoctor(
        registry_root=tmp_path / "registry",
        quarantine_root=tmp_path / "quarantine",
    ).inspect()
    rendered = report.render_for_operator()

    assert report.summary.total_records == 2
    assert report.summary.corrupt_registry_files == 1
    assert report.summary.missing_quarantine_paths == 2
    assert any(
        finding.severity is ExternalSkillDoctorSeverity.BLOCKER
        and finding.code == "corrupt_registry_record"
        for finding in report.findings
    )
    assert any(
        action.code == "recover_corrupt_registry_files"
        for action in report.recovery_actions
    )
    assert "External Skill Registry status" in rendered
    assert "OPENAI_API_KEY" not in rendered
    assert "approval-readonly" not in rendered


def test_external_skill_doctor_recovers_corrupt_registry_files(tmp_path: Path) -> None:
    from src.skills.external_skill_loader.doctor import ExternalSkillDoctor

    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    corrupt_path = registry_root / "broken.json"
    corrupt_path.write_text("{not-json", encoding="utf-8")

    report = ExternalSkillDoctor(
        registry_root=registry_root,
        quarantine_root=tmp_path / "quarantine",
    ).recover_corrupt_registry_files()

    assert not corrupt_path.exists()
    assert (registry_root / "corrupt" / "broken.json").exists()
    assert report.summary.corrupt_registry_files == 0
    assert any(
        action.code == "corrupt_registry_files_moved"
        for action in report.recovery_actions
    )


def test_external_skill_doctor_flags_approved_record_without_readonly_approval(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_loader.doctor import (
        ExternalSkillDoctor,
        ExternalSkillDoctorSeverity,
    )
    from src.skills.external_skill_loader.loader import (
        ExternalSkillAuditReport,
        ExternalSkillSource,
        ExternalSkillStatus,
        FilePersonalSkillRegistry,
        PersonalSkillRegistryRecord,
    )

    quarantine_path = tmp_path / "quarantine" / "approved"
    quarantine_path.mkdir(parents=True)
    registry = FilePersonalSkillRegistry(tmp_path / "registry")
    source = ExternalSkillSource(source_type="local_folder", locator=str(tmp_path))
    registry._write(
        PersonalSkillRegistryRecord(
            skill_id="approved",
            name="approved",
            source=source,
            quarantine_path=str(quarantine_path),
            content_hash="abc",
            status=ExternalSkillStatus.APPROVED_READONLY,
            audit_report=ExternalSkillAuditReport(
                skill_id="approved",
                name="approved",
                status=ExternalSkillStatus.NEEDS_REVIEW,
                risk_level="low",
                read_only_allowed=True,
            ),
        )
    )

    report = ExternalSkillDoctor(
        registry_root=tmp_path / "registry",
        quarantine_root=tmp_path / "quarantine",
    ).inspect()

    assert any(
        finding.code == "approved_without_readonly_approval"
        and finding.severity is ExternalSkillDoctorSeverity.BLOCKER
        for finding in report.findings
    )


def test_external_skill_doctor_flags_invalid_execution_approval_record(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_loader.doctor import (
        ExternalSkillDoctor,
        ExternalSkillDoctorSeverity,
    )
    from src.skills.external_skill_loader.loader import (
        ExternalSkillAuditReport,
        ExternalSkillSource,
        ExternalSkillStatus,
        FilePersonalSkillRegistry,
        PersonalSkillRegistryRecord,
    )

    quarantine_path = tmp_path / "quarantine" / "execution"
    quarantine_path.mkdir(parents=True)
    registry = FilePersonalSkillRegistry(tmp_path / "registry")
    source = ExternalSkillSource(source_type="local_folder", locator=str(tmp_path))
    registry._write(
        PersonalSkillRegistryRecord(
            skill_id="execution",
            name="execution",
            source=source,
            quarantine_path=str(quarantine_path),
            content_hash="abc",
            status=ExternalSkillStatus.EXECUTION_APPROVED,
            audit_report=ExternalSkillAuditReport(
                skill_id="execution",
                name="execution",
                status=ExternalSkillStatus.NEEDS_REVIEW,
                risk_level="medium",
                requested_capabilities=("browser_read",),
                read_only_allowed=True,
            ),
            readonly_approval_id="approval-readonly",
            approved_capabilities=("telegram_mcp_send",),
        )
    )

    report = ExternalSkillDoctor(
        registry_root=tmp_path / "registry",
        quarantine_root=tmp_path / "quarantine",
    ).inspect()

    assert report.summary.execution_approved_records == 1
    assert any(
        finding.code == "execution_without_execution_approval"
        and finding.severity is ExternalSkillDoctorSeverity.BLOCKER
        for finding in report.findings
    )
    assert any(
        finding.code == "execution_capability_not_requested_by_audit"
        and finding.severity is ExternalSkillDoctorSeverity.BLOCKER
        for finding in report.findings
    )


def test_external_skill_doctor_counts_native_conversion_candidates(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_loader.doctor import ExternalSkillDoctor
    from src.skills.external_skill_loader.loader import (
        ExternalSkillAuditReport,
        ExternalSkillSource,
        ExternalSkillStatus,
        FilePersonalSkillRegistry,
        PersonalSkillRegistryRecord,
    )

    quarantine_path = tmp_path / "quarantine" / "native"
    quarantine_path.mkdir(parents=True)
    registry = FilePersonalSkillRegistry(tmp_path / "registry")
    source = ExternalSkillSource(source_type="local_folder", locator=str(tmp_path))
    registry._write(
        PersonalSkillRegistryRecord(
            skill_id="native",
            name="native",
            source=source,
            quarantine_path=str(quarantine_path),
            content_hash="abc",
            status=ExternalSkillStatus.NATIVE_CONVERSION_CANDIDATE,
            audit_report=ExternalSkillAuditReport(
                skill_id="native",
                name="native",
                status=ExternalSkillStatus.NEEDS_REVIEW,
                risk_level="low",
                read_only_allowed=True,
            ),
            readonly_approval_id="approval-readonly",
            native_conversion_approval_id="approval-native",
            native_conversion_reason="Repeated useful runs.",
            use_count=3,
        )
    )

    report = ExternalSkillDoctor(
        registry_root=tmp_path / "registry",
        quarantine_root=tmp_path / "quarantine",
    ).inspect()
    rendered = report.render_for_operator()

    assert report.summary.native_conversion_candidate_records == 1
    assert report.summary.active_records == 1
    assert "native_conversion_candidate_records: 1" in rendered
    assert "approval-native" not in rendered

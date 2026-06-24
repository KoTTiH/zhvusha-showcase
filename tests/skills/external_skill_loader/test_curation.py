"""Personal external skill curation lifecycle contracts."""

from __future__ import annotations

from pathlib import Path


def test_personal_registry_curation_states_are_not_runtime_active(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_loader.loader import (
        ExternalSkillStatus,
        FilePersonalSkillRegistry,
    )

    registry = FilePersonalSkillRegistry(tmp_path / "registry")
    rejected = _registered_record(tmp_path, registry=registry, name="reject-me")
    superseded = _registered_record(tmp_path, registry=registry, name="old-flow")
    converted = _registered_record(tmp_path, registry=registry, name="convert-me")

    rejected_record = registry.reject(
        rejected.skill_id,
        approval_id="approval-reject-secret",
        approved_by_user_id=12345,
        reason="Unsafe source.",
    )
    registry.approve_readonly(
        superseded.skill_id,
        approval_id="approval-readonly-secret",
        approved_by_user_id=12345,
    )
    superseded_record = registry.mark_superseded(
        superseded.skill_id,
        approval_id="approval-supersede-secret",
        approved_by_user_id=12345,
        superseded_by_skill_id="native.old_flow",
        reason="Native skill now owns this workflow.",
    )
    registry.approve_readonly(
        converted.skill_id,
        approval_id="approval-readonly-secret",
        approved_by_user_id=12345,
    )
    registry.record_successful_use(converted.skill_id)
    registry.mark_native_conversion_candidate(
        converted.skill_id,
        approval_id="approval-native-candidate-secret",
        approved_by_user_id=12345,
        minimum_successful_uses=1,
    )
    converted_record = registry.mark_native_converted(
        converted.skill_id,
        approval_id="approval-native-converted-secret",
        approved_by_user_id=12345,
        native_skill_name="convert_me",
        reason="Spec implementation landed.",
    )

    assert rejected_record.status is ExternalSkillStatus.REJECTED
    assert superseded_record.status is ExternalSkillStatus.SUPERSEDED
    assert converted_record.status is ExternalSkillStatus.NATIVE_CONVERTED
    assert registry.active_records() == ()
    assert registry.get(converted.skill_id).native_skill_name == "convert_me"


def test_doctor_counts_curated_inactive_records(tmp_path: Path) -> None:
    from src.skills.external_skill_loader.doctor import ExternalSkillDoctor
    from src.skills.external_skill_loader.loader import FilePersonalSkillRegistry

    registry = FilePersonalSkillRegistry(tmp_path / "registry")
    rejected = _registered_record(tmp_path, registry=registry, name="reject-me")
    registry.reject(
        rejected.skill_id,
        approval_id="approval-reject-secret",
        approved_by_user_id=12345,
        reason="Unsafe source.",
    )

    report = ExternalSkillDoctor(
        registry_root=tmp_path / "registry",
        quarantine_root=tmp_path / "quarantine",
    ).inspect()

    assert report.summary.rejected_records == 1
    assert report.summary.active_records == 0
    rendered = report.render_for_operator()
    assert "rejected_records: 1" in rendered
    assert "approval-" not in rendered


def _registered_record(
    tmp_path: Path,
    *,
    registry,
    name: str,
):
    from src.skills.external_skill_loader.loader import (
        ExternalSkillSource,
        FileExternalSkillQuarantineStore,
        audit_external_skill_package,
    )

    source_root = _write_external_skill_folder(tmp_path / "source" / name, name=name)
    quarantine = FileExternalSkillQuarantineStore(tmp_path / "quarantine")
    quarantined = quarantine.import_folder(
        source_root,
        source=ExternalSkillSource(
            source_type="local_folder",
            locator=str(source_root),
            acquisition_approval_id="approval-import-secret",
            approved_by_user_id=12345,
        ),
    )
    return registry.register_quarantined(
        quarantined,
        audit_report=audit_external_skill_package(quarantined.package),
    )


def _write_external_skill_folder(root: Path, *, name: str) -> Path:
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                "description: Curated external workflow",
                "tools: [browser_read]",
                "---",
                "# Curated external workflow",
                "1. Inspect facts.",
            ]
        ),
        encoding="utf-8",
    )
    return root

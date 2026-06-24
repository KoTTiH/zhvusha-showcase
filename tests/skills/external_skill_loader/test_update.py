"""External skill update and re-audit contracts."""

from __future__ import annotations

from pathlib import Path


def test_external_skill_update_checker_blocks_new_capabilities_before_approval(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_loader.update import ExternalSkillUpdateChecker

    record, source_root = _approved_record(tmp_path, tools=("browser_read",))
    _write_external_skill_folder(
        source_root,
        tools=("browser_read", "browser_submit"),
        overwrite=True,
    )

    report = ExternalSkillUpdateChecker().check_local_folder_update(
        record,
        source_root,
    )

    assert report.changed is True
    assert report.requires_reapproval is True
    assert report.blocks_update is True
    assert report.activation_allowed is False
    assert report.new_requested_capabilities == ("browser_submit",)
    rendered = report.render_for_operator()
    assert "new_requested_capabilities: browser_submit" in rendered
    assert "approval-" not in rendered


def test_external_skill_update_checker_allows_unchanged_source(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_loader.update import ExternalSkillUpdateChecker

    record, source_root = _approved_record(tmp_path, tools=("browser_read",))

    report = ExternalSkillUpdateChecker().check_local_folder_update(
        record,
        source_root,
    )

    assert report.changed is False
    assert report.requires_reapproval is False
    assert report.blocks_update is False
    assert report.activation_allowed is True
    assert report.new_requested_capabilities == ()


def _approved_record(tmp_path: Path, *, tools: tuple[str, ...]):
    from src.skills.external_skill_loader.loader import (
        ExternalSkillSource,
        FileExternalSkillQuarantineStore,
        FilePersonalSkillRegistry,
        audit_external_skill_package,
    )

    source_root = _write_external_skill_folder(tmp_path / "source", tools=tools)
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
    registry = FilePersonalSkillRegistry(tmp_path / "registry")
    registry.register_quarantined(
        quarantined,
        audit_report=audit_external_skill_package(quarantined.package),
    )
    return (
        registry.approve_readonly(
            quarantined.skill_id,
            approval_id="approval-readonly-secret",
            approved_by_user_id=12345,
        ),
        source_root,
    )


def _write_external_skill_folder(
    root: Path,
    *,
    tools: tuple[str, ...],
    overwrite: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=overwrite)
    (root / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: kube-debug",
                "description: Investigate Kubernetes ingress",
                f"tools: [{', '.join(tools)}]",
                "---",
                "# Kubernetes ingress debug",
                "1. Inspect ingress symptoms.",
                "2. Compare service and ingress annotations.",
            ]
        ),
        encoding="utf-8",
    )
    return root

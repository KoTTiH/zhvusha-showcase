"""Native conversion spec generation for useful external skills."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml


def test_native_conversion_spec_generator_creates_valid_task_yaml_without_approval_ids(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_loader.loader import ExternalSkillStatus
    from src.skills.external_skill_loader.native_conversion import (
        NativeSkillConversionSpecGenerator,
    )
    from src.skills.spec_command.parser import SpecModel

    record = _native_candidate_record(
        tmp_path,
        tools=("browser_read",),
        minimum_successful_uses=2,
    )

    draft = NativeSkillConversionSpecGenerator().generate(
        record,
        created_at=datetime(2026, 5, 19, 12, 0, tzinfo=UTC),
    )

    spec = SpecModel.model_validate(yaml.safe_load(draft.yaml_text))
    assert record.status is ExternalSkillStatus.NATIVE_CONVERSION_CANDIDATE
    assert draft.filename == ("tasks/2026-05-19-convert-external-skill-kube-debug.yaml")
    assert spec.slug == "convert-external-skill-kube-debug"
    assert spec.tier == 2
    assert spec.created_by == "zhvusha"
    assert spec.failing_test.file == "tests/skills/kube_debug/test_skill.py"
    assert "src/skills/kube_debug/skill.py" in spec.whitelist_paths
    assert spec.source_provenance
    assert "approval-" not in draft.yaml_text
    assert "approval-" not in draft.migration_note_markdown


def test_native_conversion_spec_generator_classifies_high_risk_tool_as_tier3(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_loader.native_conversion import (
        NativeSkillConversionSpecGenerator,
    )
    from src.skills.spec_command.parser import SpecModel

    record = _native_candidate_record(
        tmp_path,
        tools=("browser_submit",),
        minimum_successful_uses=1,
    )

    draft = NativeSkillConversionSpecGenerator().generate(
        record,
        created_at=datetime(2026, 5, 19, 12, 0, tzinfo=UTC),
    )

    spec = SpecModel.model_validate(yaml.safe_load(draft.yaml_text))
    assert spec.tier == 3
    assert any("browser_submit" in item for item in spec.blast_radius)
    assert "Никита approval" in spec.rationale


def _native_candidate_record(
    tmp_path: Path,
    *,
    tools: tuple[str, ...],
    minimum_successful_uses: int,
):
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
    registry.approve_readonly(
        quarantined.skill_id,
        approval_id="approval-readonly-secret",
        approved_by_user_id=12345,
    )
    for _ in range(minimum_successful_uses):
        registry.record_successful_use(quarantined.skill_id)
    return registry.mark_native_conversion_candidate(
        quarantined.skill_id,
        approval_id="approval-native-secret",
        approved_by_user_id=12345,
        minimum_successful_uses=minimum_successful_uses,
    )


def _write_external_skill_folder(root: Path, *, tools: tuple[str, ...]) -> Path:
    root.mkdir(parents=True)
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

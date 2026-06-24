"""Read-only update checks and re-audit for external skills."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.skills.external_skill_loader.loader import (
    ExternalSkillAuditReport,
    ExternalSkillSource,
    PersonalSkillRegistryRecord,
    audit_external_skill_package,
    hash_external_skill_tree,
    parse_external_skill_folder,
)

if TYPE_CHECKING:
    from pathlib import Path


class ExternalSkillUpdateReport(BaseModel):
    """Result of checking a candidate update without activating it."""

    skill_id: str
    changed: bool
    current_hash: str
    candidate_hash: str
    requested_capabilities: tuple[str, ...] = ()
    new_requested_capabilities: tuple[str, ...] = ()
    removed_requested_capabilities: tuple[str, ...] = ()
    requires_reapproval: bool = False
    blocks_update: bool = False
    activation_allowed: bool = False
    audit_report: ExternalSkillAuditReport | None = None

    def render_for_operator(self) -> str:
        """Render a concise operator-safe update report."""
        lines = [
            "External Skill Update status",
            f"- skill_id: {self.skill_id}",
            f"- changed: {_yes_no(self.changed)}",
            f"- requires_reapproval: {_yes_no(self.requires_reapproval)}",
            f"- blocks_update: {_yes_no(self.blocks_update)}",
            f"- activation_allowed: {_yes_no(self.activation_allowed)}",
            "- requested_capabilities: "
            f"{_format_capabilities(self.requested_capabilities)}",
            "- new_requested_capabilities: "
            f"{_format_capabilities(self.new_requested_capabilities)}",
            "- removed_requested_capabilities: "
            f"{_format_capabilities(self.removed_requested_capabilities)}",
        ]
        if self.audit_report is not None:
            lines.extend(
                (
                    f"- audit_risk: {self.audit_report.risk_level}",
                    f"- audit_blocked: {_yes_no(self.audit_report.blocked)}",
                )
            )
        return "\n".join(lines)


class ExternalSkillUpdateChecker:
    """Check candidate updates and fail closed before activation."""

    def check_local_folder_update(
        self,
        record: PersonalSkillRegistryRecord,
        source_root: Path,
    ) -> ExternalSkillUpdateReport:
        """Re-audit a local candidate folder without changing registry state."""
        candidate_root = source_root.expanduser().resolve()
        candidate_hash = hash_external_skill_tree(candidate_root)
        current_capabilities = tuple(record.audit_report.requested_capabilities)
        if candidate_hash == record.content_hash:
            return ExternalSkillUpdateReport(
                skill_id=record.skill_id,
                changed=False,
                current_hash=record.content_hash,
                candidate_hash=candidate_hash,
                requested_capabilities=current_capabilities,
                activation_allowed=True,
            )

        package = parse_external_skill_folder(
            candidate_root,
            source=ExternalSkillSource(
                source_type="local_folder",
                locator=str(candidate_root),
                acquisition_approval_id=record.source.acquisition_approval_id,
                approved_by_user_id=record.source.approved_by_user_id,
            ),
        ).model_copy(update={"skill_id": record.skill_id})
        audit_report = audit_external_skill_package(package)
        candidate_capabilities = tuple(audit_report.requested_capabilities)
        new_capabilities = _sorted_difference(
            candidate_capabilities,
            current_capabilities,
        )
        removed_capabilities = _sorted_difference(
            current_capabilities,
            candidate_capabilities,
        )
        blocks_update = audit_report.blocked or bool(new_capabilities)
        return ExternalSkillUpdateReport(
            skill_id=record.skill_id,
            changed=True,
            current_hash=record.content_hash,
            candidate_hash=candidate_hash,
            requested_capabilities=candidate_capabilities,
            new_requested_capabilities=new_capabilities,
            removed_requested_capabilities=removed_capabilities,
            requires_reapproval=True,
            blocks_update=blocks_update,
            activation_allowed=False,
            audit_report=audit_report,
        )


def _sorted_difference(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(sorted(set(left) - set(right)))


def _format_capabilities(capabilities: tuple[str, ...]) -> str:
    return ", ".join(capabilities) if capabilities else "none"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"

"""Bot command surface for external skill registry status."""

from __future__ import annotations

from pathlib import Path


def test_external_skill_status_command_is_admin_only(tmp_path: Path) -> None:
    from src.bot.main import _external_skill_status_reply
    from src.skills.base import AgentContext

    reply = _external_skill_status_reply(
        "/external_skill_status",
        AgentContext(user_id=2, chat_id=1, mode="personal"),
        admin_user_id=1,
        workspace_root=tmp_path,
    )

    assert reply == "Эта команда доступна только Никите."


def test_external_skill_status_command_renders_registry_status_without_secrets(
    tmp_path: Path,
) -> None:
    from src.bot.main import _external_skill_status_reply
    from src.skills.base import AgentContext
    from src.skills.external_skill_loader.loader import (
        ExternalSkillAuditReport,
        ExternalSkillSource,
        ExternalSkillStatus,
        FilePersonalSkillRegistry,
        PersonalSkillRegistryRecord,
    )

    workspace = tmp_path / "workspace"
    registry = FilePersonalSkillRegistry(workspace / "skills" / "external" / "registry")
    quarantine_path = workspace / "skills" / "external" / "quarantine" / "skill"
    quarantine_path.mkdir(parents=True)
    registry._write(
        PersonalSkillRegistryRecord(
            skill_id="skill",
            name="skill",
            source=ExternalSkillSource(source_type="local_folder", locator="source"),
            quarantine_path=str(quarantine_path),
            content_hash="abc",
            status=ExternalSkillStatus.APPROVED_READONLY,
            audit_report=ExternalSkillAuditReport(
                skill_id="skill",
                name="skill",
                status=ExternalSkillStatus.NEEDS_REVIEW,
                risk_level="high",
                requested_env_vars=("OPENAI_API_KEY",),
                read_only_allowed=True,
            ),
            readonly_approval_id="approval-secret-ish",
        )
    )

    reply = _external_skill_status_reply(
        "/external_skill_status",
        AgentContext(user_id=1, chat_id=1, mode="personal"),
        admin_user_id=1,
        workspace_root=workspace,
    )

    assert reply is not None
    assert "External Skill Registry status" in reply
    assert "approved_readonly_records: 1" in reply
    assert "OPENAI_API_KEY" not in reply
    assert "approval-secret-ish" not in reply

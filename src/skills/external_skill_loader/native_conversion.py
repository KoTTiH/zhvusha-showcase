"""Generate spec-first native conversion drafts for external skills."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Literal

import yaml
from pydantic import BaseModel

from src.skills.external_skill_loader.loader import (
    ExternalSkillStatus,
    PersonalSkillRegistryRecord,
)
from src.skills.spec_command.parser import (
    FailingTest,
    ResearchFinding,
    SourceProvenance,
    SpecKind,
    SpecModel,
    SpecStatus,
)

_HIGH_RISK_CAPABILITIES: frozenset[str] = frozenset(
    {
        "browser_submit",
        "commit",
        "commit_after_gate",
        "delete",
        "edit_env",
        "external_skill_execute",
        "login",
        "publish",
        "purchase",
        "restart",
        "send_message",
        "telegram_mcp_admin",
        "telegram_mcp_daivinchik_button",
        "telegram_mcp_daivinchik_notify",
        "telegram_mcp_daivinchik_reply_button",
        "telegram_mcp_modify",
        "telegram_mcp_send",
        "write_files",
        "write_whitelisted_files_after_approval",
    }
)


class NativeSkillConversionSpecDraft(BaseModel):
    """Generated task YAML and migration note for operator review."""

    filename: str
    yaml_text: str
    migration_note_markdown: str
    spec: SpecModel


class NativeSkillConversionSpecGenerator:
    """Build a self-coding task draft from a repeated-use external skill."""

    def generate(
        self,
        record: PersonalSkillRegistryRecord,
        *,
        created_at: datetime | None = None,
    ) -> NativeSkillConversionSpecDraft:
        """Return a validated spec draft without writing it to tasks/."""
        if record.status is not ExternalSkillStatus.NATIVE_CONVERSION_CANDIDATE:
            raise ValueError(
                "native conversion spec requires native_conversion_candidate status"
            )
        if not record.native_conversion_reason:
            raise ValueError("native conversion record has no conversion reason")

        timestamp = created_at or datetime.now(UTC)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        date_prefix = timestamp.astimezone(UTC).date().isoformat()
        skill_slug = _slugify(record.name or record.skill_id)
        task_slug = _bounded_slug(f"convert-external-skill-{skill_slug}")
        module_slug = _module_slug(skill_slug)
        filename = f"tasks/{date_prefix}-{task_slug}.yaml"
        requested_capabilities = tuple(record.audit_report.requested_capabilities)
        tier: Literal[2, 3] = 3 if _requires_tier3(requested_capabilities) else 2
        tier_note = (
            "High-risk mapped capabilities require explicit Никита approval "
            "before implementation."
            if tier == 3
            else "Read-only or low-risk mapped capabilities keep this conversion Tier 2."
        )

        spec = SpecModel(
            slug=task_slug,
            title=_bounded_title(
                record.native_conversion_spec_title
                or f"Convert external skill {record.name} to native ZHVUSHA skill"
            ),
            created_at=timestamp,
            created_by="zhvusha",
            tier=tier,
            kind=SpecKind.FEATURE,
            goal=(
                f"Convert approved external skill `{record.skill_id}` into a native "
                "ZHVUSHA skill through spec-first self-coding. External content "
                "remains source material only; no script, prompt or tool authority "
                "is inherited from the imported package."
            ),
            failing_test=FailingTest(
                file=f"tests/skills/{module_slug}/test_skill.py",
                name="test_native_skill_preserves_audited_workflow_without_external_authority",
                spec=(
                    "Native skill exposes the audited workflow as "
                    "ZHVUSHA-controlled behavior, returns source-grounded output, "
                    "and cannot execute external scripts or bypass ToolGateway."
                ),
            ),
            whitelist_paths=[
                filename,
                f"src/skills/{module_slug}/__init__.py",
                f"src/skills/{module_slug}/skill.py",
                f"src/skills/{module_slug}/skill.yaml",
                f"tests/skills/{module_slug}/test_skill.py",
            ],
            read_only_paths=[
                "docs/hermes-skill-compatibility-roadmap.md",
                record.quarantine_path,
            ],
            blast_radius=[
                "Adds a native ZHVUSHA skill candidate derived from an audited "
                "external procedure.",
                "Does not grant new runtime authority; any tool use must remain "
                "behind SkillInvocationService, Agent Runtime and ToolGateway gates.",
                f"Mapped capabilities under review: {_format_capabilities(requested_capabilities)}.",
                tier_note,
            ],
            rollback_path=[
                f"Remove src/skills/{module_slug}/ and tests/skills/{module_slug}/.",
                "Keep or reject the original external skill registry record without "
                "changing its approval state.",
                "Run targeted skill, registry and external skill tests after removal.",
            ],
            preserve_behavior=[
                "External skill content stays untrusted source material and is never "
                "promoted into system instructions.",
                "No external scripts, templates or declared tools execute directly.",
                "Side effects stay capability scoped and approval gated through "
                "ToolGateway.",
                "Useful lessons become memory candidates or spec artifacts, not "
                "direct core memory writes.",
            ],
            allowed_simplifications=[],
            existing_tests_to_update=[],
            research_findings=[
                ResearchFinding(
                    source=f"external_skill_registry:{record.skill_id}",
                    excerpt=record.native_conversion_reason,
                    relevance=(
                        "Repeated successful use is the reason to convert the "
                        "foreign procedure into a native Жвуша skill."
                    ),
                ),
                ResearchFinding(
                    source=f"external_skill_audit:{record.skill_id}",
                    excerpt=(
                        "Requested capabilities: "
                        f"{_format_capabilities(requested_capabilities)}"
                    ),
                    relevance=(
                        "The conversion spec must preserve capability boundaries "
                        "instead of inheriting broad external permissions."
                    ),
                ),
            ],
            source_provenance=[
                SourceProvenance(
                    url=record.quarantine_path,
                    source_type=_source_type_for_spec(record),
                    trust_tier="direct",
                    claim=(
                        "Quarantined external skill package is the source material "
                        "for native conversion, not runtime authority."
                    ),
                )
            ],
            chat_context=[
                (
                    f"External skill `{record.skill_id}` reached "
                    f"{record.use_count} successful uses and was marked for "
                    "native conversion."
                )
            ],
            rationale=(
                f"{record.native_conversion_reason} {tier_note} The generated "
                "spec keeps approval gates, audit trail and Жвуша's cognitive "
                "loop as the execution boundary."
            ),
            status=SpecStatus.PENDING_APPROVAL,
        )
        yaml_text = yaml.safe_dump(
            spec.model_dump(mode="json"),
            allow_unicode=True,
            sort_keys=False,
        )
        return NativeSkillConversionSpecDraft(
            filename=filename,
            yaml_text=yaml_text,
            migration_note_markdown=_render_migration_note(
                record=record,
                filename=filename,
                requested_capabilities=requested_capabilities,
            ),
            spec=spec,
        )


def _requires_tier3(capabilities: tuple[str, ...]) -> bool:
    return bool(set(capabilities) & _HIGH_RISK_CAPABILITIES)


def _render_migration_note(
    *,
    record: PersonalSkillRegistryRecord,
    filename: str,
    requested_capabilities: tuple[str, ...],
) -> str:
    return "\n".join(
        [
            f"# Native conversion migration note: {record.name}",
            "",
            f"- external_skill_id: `{record.skill_id}`",
            f"- source_type: `{record.source.source_type}`",
            f"- successful_uses: `{record.use_count}`",
            f"- requested_capabilities: `{_format_capabilities(requested_capabilities)}`",
            f"- spec_draft: `{filename}`",
            "",
            "External skill content remains source material. The native skill must "
            "be implemented through the normal spec-first self-coding gates, with "
            "ToolGateway enforcement and memory staging preserved.",
        ]
    )


def _source_type_for_spec(
    record: PersonalSkillRegistryRecord,
) -> Literal["local_repo", "github", "other"]:
    if record.source.source_type in {"local_folder", "local_archive"}:
        return "local_repo"
    if record.source.source_type == "git":
        return "github"
    return "other"


def _format_capabilities(capabilities: tuple[str, ...]) -> str:
    return ", ".join(capabilities) if capabilities else "none"


def _module_slug(slug: str) -> str:
    module = slug.replace("-", "_")
    if not module or module[0].isdigit():
        module = f"skill_{module}"
    return module


def _bounded_slug(slug: str, *, max_length: int = 60) -> str:
    cleaned = _slugify(slug)
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[:max_length].rstrip("-") or "external-skill-conversion"


def _bounded_title(title: str, *, max_length: int = 200) -> str:
    cleaned = " ".join(title.strip().split())
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[:max_length].rstrip()


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return cleaned or "external-skill"

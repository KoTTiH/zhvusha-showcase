#!/usr/bin/env python3
"""Verify the Codex-only self-coding MVP contract.

This is a lightweight architectural gate, not a replacement for pytest, mypy,
ruff, import-linter, whitelist checks, or Tier 3 protection.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


def check_self_coding_mvp(repo_root: Path) -> list[str]:
    errors: list[str] = []
    errors.extend(_check_codex_registry())
    errors.extend(_check_spec_contract())
    errors.extend(_check_news_pillars_contract(repo_root))
    errors.extend(_check_proposal_archive_contract(repo_root))
    errors.extend(_check_posts_reports_contract(repo_root))
    errors.extend(_check_plan_completion_foundations(repo_root))
    errors.extend(_check_autonomous_self_coding_contract(repo_root))
    errors.extend(_check_zhvusha_specs(repo_root))
    return errors


def _check_codex_registry() -> list[str]:
    from src.skills.code_agent.registry import (
        CODEX_BACKEND,
        CodeAgentRegistry,
    )

    errors: list[str] = []
    if CODEX_BACKEND != "codex_cli":
        errors.append(f"CODEX_BACKEND must be codex_cli, got {CODEX_BACKEND!r}")
    for backend in ("claude_agent_sdk", "claude_code_sdk", "claude_cli"):
        try:
            CodeAgentRegistry(backends={}, backend=backend)
        except ValueError:
            continue
        errors.append(f"blocked backend {backend!r} was accepted")
    registry = CodeAgentRegistry(backends={}, backend="codex_cli")
    if registry.backend_order != ("codex_cli",):
        errors.append(
            f"backend_order must be codex-only, got {registry.backend_order!r}"
        )
    return errors


def _check_spec_contract() -> list[str]:
    from src.skills.spec_command.parser import SpecModel

    errors: list[str] = []
    for field in (
        "rationale",
        "source_provenance",
        "chat_context",
        "preserve_behavior",
        "allowed_simplifications",
        "autonomous_approval_reason",
    ):
        if field not in SpecModel.model_fields:
            errors.append(f"SpecModel missing {field}")
    schema = SpecModel.model_json_schema()
    properties = schema.get("properties", {})
    if "source_provenance" not in properties:
        errors.append("SpecModel JSON schema missing source_provenance")
    if "chat_context" not in properties:
        errors.append("SpecModel JSON schema missing chat_context")
    if "preserve_behavior" not in properties:
        errors.append("SpecModel JSON schema missing preserve_behavior")
    if "allowed_simplifications" not in properties:
        errors.append("SpecModel JSON schema missing allowed_simplifications")
    if "autonomous_approval_reason" not in properties:
        errors.append("SpecModel JSON schema missing autonomous_approval_reason")
    return errors


def _check_news_pillars_contract(repo_root: Path) -> list[str]:
    errors: list[str] = []
    errors.extend(_check_required_files(repo_root, _NEWS_PILLARS_REQUIRED_PATHS))
    errors.extend(_check_news_migration(repo_root))
    errors.extend(_check_importlinter_news_pillars(repo_root))
    errors.extend(_check_tier3_pillars_protection(repo_root))
    return errors


_NEWS_PILLARS_REQUIRED_PATHS = [
    "src/news/models.py",
    "src/news/dedup.py",
    "src/news/clustering.py",
    "src/news/store.py",
    "src/news/pipeline.py",
    "src/pillars/models.py",
    "src/pillars/reader.py",
    "src/collectors/rss.py",
    "src/collectors/arxiv.py",
    "src/collectors/github_trending.py",
    "src/collectors/huggingface.py",
    "src/collectors/lmarena.py",
    "src/collectors/source_items.py",
    "src/skills/topic_to_spec/provider.py",
    "src/skills/topic_to_spec/skill.py",
    "src/skills/topic_to_spec/skill.yaml",
    "src/skills/morning_digest/provider.py",
    "src/skills/morning_digest/skill.py",
    "src/skills/morning_digest/skill.yaml",
    "alembic/versions/006_add_news_pipeline_tables.py",
]


def _check_proposal_archive_contract(repo_root: Path) -> list[str]:
    errors: list[str] = []
    errors.extend(_check_required_files(repo_root, _PROPOSAL_ARCHIVE_REQUIRED_PATHS))
    errors.extend(_check_proposal_model())
    errors.extend(_check_archive_migration(repo_root))
    errors.extend(_check_topic_to_spec_writes_proposals())
    return errors


_PROPOSAL_ARCHIVE_REQUIRED_PATHS = [
    "proposals/.gitkeep",
    "src/skills/proposal_command/models.py",
    "src/skills/proposal_command/store.py",
    "src/skills/proposal_command/skill.py",
    "src/skills/proposal_command/writer.py",
    "src/skills/proposal_command/skill.yaml",
    "src/archive/models.py",
    "src/archive/store.py",
    "src/archive/files.py",
    "src/skills/cycle_analyzer/analyzer.py",
    "src/skills/cycle_analyzer/skill.py",
    "src/skills/cycle_analyzer/skill.yaml",
    "src/skills/topic_to_spec/__init__.py",
    "alembic/versions/007_add_archive_nodes.py",
]


def _check_proposal_model() -> list[str]:
    from src.skills.proposal_command.models import ProposalModel, ProposalStatus

    errors: list[str] = []
    for field in ("rationale", "source_provenance", "files_likely_touched"):
        if field not in ProposalModel.model_fields:
            errors.append(f"ProposalModel missing {field}")
    if ProposalStatus.APPROVED.value != "approved":
        errors.append("ProposalStatus approved value drifted")
    return errors


def _check_archive_migration(repo_root: Path) -> list[str]:
    errors: list[str] = []
    migration = repo_root / "alembic/versions/007_add_archive_nodes.py"
    if not migration.exists():
        return errors
    text = migration.read_text(encoding="utf-8")
    for table_or_column in (
        "archive_nodes",
        "source_evidence",
        "model_config",
        "embedding",
    ):
        if table_or_column not in text:
            errors.append(f"archive migration missing {table_or_column}")
    return errors


def _check_topic_to_spec_writes_proposals() -> list[str]:
    from src.skills.base import SideEffect
    from src.skills.topic_to_spec.builder import build_candidate_from_topic
    from src.skills.topic_to_spec.models import TopicRecord
    from src.skills.topic_to_spec.skill import TopicToSpecSkill

    errors: list[str] = []
    if SideEffect.WRITES_FILESYSTEM not in TopicToSpecSkill.side_effects:
        errors.append("TopicToSpecSkill must declare writes_filesystem for proposals")
    candidate = build_candidate_from_topic(
        TopicRecord(
            cluster_key="mvp-gate",
            title="Self-growth enrichment guard",
            summary="Future backlog topics must preserve existing Жвуша behaviour.",
            top_terms=("self-coding", "personality"),
            final_priority=100.0,
        )
    )
    if not candidate.preserve_behavior:
        errors.append("TopicCandidate missing preserve_behavior contract")
    if candidate.allowed_simplifications:
        errors.append("TopicCandidate must default allowed_simplifications to empty")
    return errors


def _check_posts_reports_contract(repo_root: Path) -> list[str]:
    errors: list[str] = []
    errors.extend(_check_required_files(repo_root, _POSTS_REPORTS_REQUIRED_PATHS))
    errors.extend(_check_channel_writer_draft_publish())
    return errors


_POSTS_REPORTS_REQUIRED_PATHS = [
    "src/skills/post_drafts/models.py",
    "src/skills/post_drafts/store.py",
    "src/skills/post_drafts/provider.py",
    "src/skills/post_drafts/skill.py",
    "src/skills/post_drafts/skill.yaml",
    "src/skills/weekly_report/formatter.py",
    "src/skills/weekly_report/provider.py",
    "src/skills/weekly_report/skill.py",
    "src/skills/weekly_report/skill.yaml",
    "tests/skills/post_drafts/test_models.py",
    "tests/skills/post_drafts/test_skill.py",
    "tests/skills/weekly_report/test_formatter.py",
    "tests/skills/weekly_report/test_skill.py",
]


def _check_plan_completion_foundations(repo_root: Path) -> list[str]:
    errors: list[str] = []
    errors.extend(_check_required_files(repo_root, _PLAN_COMPLETION_REQUIRED_PATHS))
    errors.extend(_check_adversarial_generator_contract())
    return errors


_PLAN_COMPLETION_REQUIRED_PATHS = [
    "src/skills/adversarial_test_gen/generator.py",
    "src/skills/adversarial_test_gen/skill.py",
    "src/skills/adversarial_test_gen/skill.yaml",
    "src/skills/ideation_to_spec/self_critique.py",
    "src/skills/implement_spec/formal_gates.py",
    "src/skills/implement_spec/reviewer.py",
    "src/skills/chat_self_coding/merge.py",
    "src/knowledge/summarization.py",
    "src/skills/code_agent/worktrees.py",
    "src/skills/external_skill_loader/loader.py",
    "src/skills/prompt_optimizer/optimizer.py",
    "src/llm/openai_compatible.py",
    "tests/test_llm_providers.py",
    "tests/skills/prompt_optimizer/test_optimizer.py",
    "tests/skills/adversarial_test_gen/test_generator.py",
    "tests/skills/ideation_to_spec/test_self_critique.py",
    "tests/skills/implement_spec/test_formal_gates.py",
    "tests/skills/implement_spec/test_reviewer.py",
    "tests/skills/chat_self_coding/test_merge.py",
    "tests/knowledge/test_summarization.py",
    "tests/skills/code_agent/test_worktrees.py",
    "tests/skills/external_skill_loader/test_loader.py",
]


def _check_adversarial_generator_contract() -> list[str]:
    from datetime import UTC, datetime

    from src.archive.models import ArchiveNode, ArchiveStatus
    from src.skills.adversarial_test_gen.generator import generate_adversarial_tests

    node = ArchiveNode(
        slug="gate-failure",
        tier=2,
        status=ArchiveStatus.FAILED,
        created_at=datetime(2026, 5, 7, tzinfo=UTC),
        diff_summary="failed",
        tests_summary="failed",
        insight="real failure",
    )
    drafts = generate_adversarial_tests([node])
    if not drafts or drafts[0].archive_slug != "gate-failure":
        return ["adversarial generator must anchor drafts to archive nodes"]
    return []


def _check_autonomous_self_coding_contract(repo_root: Path) -> list[str]:
    from src.agent_runtime.profiles import SELF_IMPROVEMENT_AUTONOMOUS

    errors: list[str] = []
    errors.extend(_check_required_files(repo_root, _AUTONOMOUS_SELF_CODING_PATHS))
    if SELF_IMPROVEMENT_AUTONOMOUS.worker != "self_improvement":
        errors.append("SELF_IMPROVEMENT_AUTONOMOUS must use self_improvement worker")
    for denied in (
        "write_files",
        "edit_env",
        "restart",
        "publish",
        "browser_submit",
        "send_message",
        "commit",
    ):
        if denied not in SELF_IMPROVEMENT_AUTONOMOUS.denied_capabilities:
            errors.append(f"self_improvement profile must deny {denied}")
    if (
        "self_approve_low_risk_specs"
        not in SELF_IMPROVEMENT_AUTONOMOUS.allowed_capabilities
    ):
        errors.append("self_improvement profile missing self_approve_low_risk_specs")
    tier3_request_capability = "request_tier3_specs_for_nikita_approval"
    if tier3_request_capability not in SELF_IMPROVEMENT_AUTONOMOUS.allowed_capabilities:
        errors.append(f"self_improvement profile missing {tier3_request_capability}")
    if "self_approve_tier3_specs" in SELF_IMPROVEMENT_AUTONOMOUS.allowed_capabilities:
        errors.append("self_improvement profile must not self-approve Tier 3 specs")
    return errors


_AUTONOMOUS_SELF_CODING_PATHS = [
    "src/skills/autonomous_self_coding/__init__.py",
    "src/skills/autonomous_self_coding/planner.py",
    "src/skills/autonomous_self_coding/skill.py",
    "src/skills/autonomous_self_coding/skill.yaml",
    "src/agent_runtime/workers/self_improvement.py",
    "tests/skills/autonomous_self_coding/__init__.py",
    "tests/skills/autonomous_self_coding/test_skill.py",
    "tests/agent_runtime/test_self_improvement_worker.py",
    "tests/skills/spec_command/test_autonomous_approval.py",
    "tests/test_autonomous_self_coding_config.py",
]


def _check_channel_writer_draft_publish() -> list[str]:
    from src.skills.channel_writer.skill import ChannelWriterSkill

    errors: list[str] = []
    if not any(
        trigger.startswith("/post_draft") for trigger in ChannelWriterSkill.triggers
    ):
        errors.append("ChannelWriterSkill must expose /post_draft publish trigger")
    return errors


def _check_required_files(repo_root: Path, paths: list[str]) -> list[str]:
    errors: list[str] = []
    for rel in paths:
        if not (repo_root / rel).exists():
            errors.append(f"missing self-coding contract file: {rel}")
    return errors


def _check_news_migration(repo_root: Path) -> list[str]:
    errors: list[str] = []
    migration = repo_root / "alembic/versions/006_add_news_pipeline_tables.py"
    if not migration.exists():
        return errors
    text = migration.read_text(encoding="utf-8")
    for table in ("news_items", "topic_clusters"):
        if table not in text:
            errors.append(f"migration missing table {table}")
    return errors


def _check_importlinter_news_pillars(repo_root: Path) -> list[str]:
    errors: list[str] = []
    importlinter = repo_root / ".importlinter"
    if not importlinter.exists():
        return errors
    text = importlinter.read_text(encoding="utf-8")
    for module in ("src.news", "src.pillars"):
        if module not in text:
            errors.append(f".importlinter missing leaf module {module}")
    return errors


def _check_tier3_pillars_protection(repo_root: Path) -> list[str]:
    errors: list[str] = []
    tier3 = repo_root / "scripts/check_tier3_protection.sh"
    if not tier3.exists():
        return errors
    text = tier3.read_text(encoding="utf-8")
    if "personality/pillars.md" not in text:
        errors.append("Tier 3 protection missing personality/pillars.md")
    return errors


def _check_zhvusha_specs(repo_root: Path) -> list[str]:
    errors: list[str] = []
    tasks_dir = repo_root / "tasks"
    if not tasks_dir.exists():
        return errors
    for path in sorted(tasks_dir.glob("*.yaml")):
        raw = _load_yaml(path)
        if raw.get("created_by") != "zhvusha":
            continue
        if not str(raw.get("rationale", "")).strip():
            errors.append(f"{path.relative_to(repo_root)} missing rationale")
        source_provenance = raw.get("source_provenance")
        if not isinstance(source_provenance, list) or not source_provenance:
            errors.append(f"{path.relative_to(repo_root)} missing source_provenance")
        preserve_behavior = raw.get("preserve_behavior")
        if not isinstance(preserve_behavior, list) or not preserve_behavior:
            errors.append(f"{path.relative_to(repo_root)} missing preserve_behavior")
        allowed_simplifications = raw.get("allowed_simplifications")
        if allowed_simplifications is None:
            errors.append(
                f"{path.relative_to(repo_root)} missing allowed_simplifications"
            )
        elif not isinstance(allowed_simplifications, list):
            errors.append(
                f"{path.relative_to(repo_root)} allowed_simplifications is not a list"
            )
    return errors


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        return {"_error": str(exc)}
    if not isinstance(raw, dict):
        return {}
    return raw


def main() -> int:
    repo_root = Path.cwd()
    errors = check_self_coding_mvp(repo_root)
    if errors:
        sys.stderr.write("Self-coding MVP gate failed:\n")
        for error in errors:
            sys.stderr.write(f"- {error}\n")
        return 1
    sys.stdout.write("Self-coding MVP gate passed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

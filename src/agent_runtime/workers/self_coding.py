"""Self-coding worker bridge for the existing spec-first implementation flow."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel

from src.agent_runtime.models import ContextCapsule
from src.skills.base import AgentContext, SkillResult

if TYPE_CHECKING:
    from pathlib import Path

    from src.agent_runtime.models import AgentJob, ContextPack


class LegacySkillExecutor(Protocol):
    """Narrow protocol for running the existing ImplementSpecSkill."""

    async def __call__(self, message: str, context: AgentContext) -> SkillResult: ...


class SpecImplementationEngine(Protocol):
    """Native implementation engine shared by `/код` and `/spec_run`."""

    async def run_slug(self, slug: str, context: AgentContext) -> SkillResult: ...


class SelfCodingWorkerError(RuntimeError):
    """Worker failure that preserves SkillResult metadata for `/код` recovery."""

    def __init__(self, response: str, metadata: dict[str, Any]) -> None:
        super().__init__(response)
        self.metadata = {
            str(key): str(value) for key, value in metadata.items() if value is not None
        }


class SelfCodingRunSummary(BaseModel):
    """Structured self-coding run output for Agent Runtime capsules."""

    slug: str
    status: str = "completed"
    summary: str
    reasoning_summary: str = ""
    changed_files: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    failed_repairs: tuple[str, ...] = ()
    risk_review: str = ""
    approval_summary: str = ""
    decision_transcript: tuple[str, ...] = ()
    research_sources: tuple[str, ...] = ()
    commit_sha: str = ""
    branch: str = ""
    spec_path: str = ""
    backend: str = ""
    memory_candidates: tuple[str, ...] = ()
    archive_candidates: tuple[str, ...] = ()
    next_actions: tuple[str, ...] = ()
    quality_warnings: tuple[str, ...] = ()


class SelfCodingRunSummaryArchiveWriter(Protocol):
    """Persists self-coding run summaries as durable workspace artifacts."""

    def write(
        self,
        *,
        run_summary: SelfCodingRunSummary,
        markdown_report: str,
    ) -> tuple[str, ...]: ...


class FileSelfCodingRunSummaryArchive:
    """Write run summaries into a workspace-scoped self-coding summary archive."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        artifact_dir: str = "self_coding_summaries/agent_runtime",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._workspace_root = workspace_root.expanduser().resolve()
        self._archive_root = (self._workspace_root / artifact_dir).resolve()
        if not self._archive_root.is_relative_to(self._workspace_root):
            raise ValueError("self-coding summary archive escapes workspace root")
        self._clock = clock or (lambda: datetime.now(tz=UTC))

    def write(
        self,
        *,
        run_summary: SelfCodingRunSummary,
        markdown_report: str,
    ) -> tuple[str, ...]:
        created_at = _aware_utc(self._clock())
        stem = _archive_stem(run_summary, created_at)
        self._archive_root.mkdir(parents=True, exist_ok=True)
        markdown_path = self._archive_root / f"{stem}.md"
        json_path = self._archive_root / f"{stem}.json"
        markdown_path.write_text(
            _archive_markdown(created_at=created_at, markdown_report=markdown_report),
            encoding="utf-8",
        )
        json_path.write_text(
            run_summary.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        return (
            f"self_coding_run_summary_markdown_path: {self._relative(markdown_path)}",
            f"self_coding_run_summary_json_path: {self._relative(json_path)}",
        )

    def _relative(self, path: Path) -> str:
        resolved = path.resolve()
        if not resolved.is_relative_to(self._workspace_root):
            raise RuntimeError("self-coding summary artifact escaped workspace root")
        return resolved.relative_to(self._workspace_root).as_posix()


class SelfCodingNativeWorkerBackend:
    """Native Agent Runtime worker for approved self-coding implementation."""

    name = "self_coding_native"

    def __init__(
        self,
        *,
        implementation_engine: SpecImplementationEngine,
        bot: Any | None = None,
        summary_archive: SelfCodingRunSummaryArchiveWriter | None = None,
    ) -> None:
        self._implementation_engine = implementation_engine
        self._bot = bot
        self._summary_archive = summary_archive

    async def run(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
    ) -> ContextCapsule:
        slug = _extract_spec_slug(context_pack.user_request)
        if not slug:
            raise RuntimeError("self-coding job is missing spec slug")
        result = await self._implementation_engine.run_slug(
            slug,
            AgentContext(
                user_id=job.owner_user_id,
                chat_id=job.chat_id,
                mode="personal",
                bot=self._bot,
                metadata=_worker_metadata(job.id, context_pack),
            ),
        )
        if not result.success:
            raise SelfCodingWorkerError(
                result.response or "self-coding implementation failed",
                result.metadata,
            )
        return _capsule_from_skill_result(
            result,
            slug,
            context_pack=job.context_pack,
            summary_archive=self._summary_archive,
        )

    async def cancel(self, job_id: str) -> bool:
        del job_id
        return False


class SelfCodingLegacyWorkerBackend:
    """Agent Runtime worker that preserves existing self-coding gates.

    The worker does not implement edits itself. It delegates to the current
    `/spec_run <slug>` skill, which already owns approval, whitelist, env guard,
    tests, commit and no-downgrade behavior.
    """

    name = "self_coding_legacy"

    def __init__(
        self,
        *,
        legacy_execute: LegacySkillExecutor,
        bot: Any | None = None,
        summary_archive: SelfCodingRunSummaryArchiveWriter | None = None,
    ) -> None:
        self._legacy_execute = legacy_execute
        self._bot = bot
        self._summary_archive = summary_archive

    async def run(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
    ) -> ContextCapsule:
        slug = _extract_spec_slug(context_pack.user_request)
        if not slug:
            raise RuntimeError("self-coding job is missing spec slug")
        result = await self._legacy_execute(
            f"/spec_run {slug}",
            AgentContext(
                user_id=job.owner_user_id,
                chat_id=job.chat_id,
                mode="personal",
                bot=self._bot,
                metadata=_worker_metadata(job.id, context_pack),
            ),
        )
        if not result.success:
            raise SelfCodingWorkerError(
                result.response or "self-coding implementation failed",
                result.metadata,
            )
        return _capsule_from_skill_result(
            result,
            slug,
            context_pack=job.context_pack,
            summary_archive=self._summary_archive,
        )

    async def cancel(self, job_id: str) -> bool:
        del job_id
        return False


def _extract_spec_slug(text: str) -> str:
    parts = text.strip().split()
    if len(parts) >= 2 and parts[0] in {"/spec_run", "spec_run"}:
        return parts[1].strip()
    if len(parts) == 1:
        return parts[0].strip()
    return ""


def _worker_metadata(job_id: str, context_pack: ContextPack) -> dict[str, str]:
    metadata = {"agent_job_id": job_id}
    task_id = _extract_active_value(context_pack.active_code_state, "code_task_id")
    if task_id:
        metadata["chat_self_coding_code_task_id"] = task_id
    for source_key, target_key in _EDITOR_RESUME_METADATA_KEYS.items():
        value = _extract_active_value(context_pack.active_code_state, source_key)
        if value:
            metadata[target_key] = value
    return metadata


_EDITOR_RESUME_METADATA_KEYS = {
    "editor_codex_session_id": "chat_self_coding_editor_codex_session_id",
    "failed_worktree_path": "chat_self_coding_failed_worktree_path",
    "failed_worktree_label": "chat_self_coding_failed_worktree_label",
    "failed_worktree_base_branch": "chat_self_coding_failed_worktree_base_branch",
    "failed_worktree_base_sha": "chat_self_coding_failed_worktree_base_sha",
}


def _extract_active_value(active_code_state: str, wanted_key: str) -> str:
    for line in active_code_state.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() == wanted_key:
            return value.strip()
    return ""


def _capsule_from_skill_result(
    result: SkillResult,
    slug: str,
    *,
    context_pack: ContextPack | None = None,
    summary_archive: SelfCodingRunSummaryArchiveWriter | None = None,
) -> ContextCapsule:
    run_summary = _run_summary_from_skill_result(
        result,
        slug,
        context_pack=context_pack,
    )
    markdown_report = _render_run_summary_markdown(
        run_summary,
        original_response=result.response,
    )
    archive_artifacts = _write_summary_archive_artifacts(
        summary_archive,
        run_summary=run_summary,
        markdown_report=markdown_report,
    )
    return ContextCapsule(
        summary=run_summary.summary,
        processed_context=result.response,
        sources=(run_summary.spec_path,),
        artifacts=(
            *_extract_artifacts(result.metadata),
            *archive_artifacts,
            f"self_coding_run_summary: {run_summary.model_dump_json()}",
        ),
        memory_candidates=run_summary.memory_candidates,
        next_actions=run_summary.next_actions,
        markdown_report=markdown_report,
    )


def _write_summary_archive_artifacts(
    summary_archive: SelfCodingRunSummaryArchiveWriter | None,
    *,
    run_summary: SelfCodingRunSummary,
    markdown_report: str,
) -> tuple[str, ...]:
    if summary_archive is None:
        return ()
    try:
        return summary_archive.write(
            run_summary=run_summary,
            markdown_report=markdown_report,
        )
    except Exception as exc:
        return (f"self_coding_run_summary_archive_failed: {type(exc).__name__}",)


def _run_summary_from_skill_result(
    result: SkillResult,
    slug: str,
    *,
    context_pack: ContextPack | None = None,
) -> SelfCodingRunSummary:
    metadata = result.metadata
    context_metadata = context_pack.metadata if context_pack is not None else {}
    changed_files = _metadata_items(metadata, "changed_files")
    tests = _metadata_items(metadata, "tests")
    risk_review = _metadata_text(metadata, "risk_review")
    quality_warnings = _quality_warnings(
        changed_files=changed_files,
        tests=tests,
        risk_review=risk_review,
    )
    next_actions = _unique_items(
        _metadata_items(metadata, "next_actions"),
        _extract_next_actions(result.response),
        _quality_warning_next_actions(quality_warnings),
    )
    return SelfCodingRunSummary(
        slug=slug,
        summary=_summary_from_response(result.response, slug),
        reasoning_summary=_metadata_text(metadata, "reasoning_summary"),
        changed_files=changed_files,
        tests=tests,
        failed_repairs=_unique_items(
            _metadata_items(metadata, "failed_repairs"),
            _metadata_items(metadata, "repair_notes"),
        ),
        risk_review=risk_review,
        approval_summary=_metadata_text(metadata, "approval_summary")
        or _metadata_text(context_metadata, "approval_summary"),
        decision_transcript=_unique_items(
            _metadata_items(metadata, "decision_transcript"),
            _metadata_items(context_metadata, "decision_transcript"),
        ),
        research_sources=_unique_items(
            _metadata_items(metadata, "research_sources"),
            _metadata_items(context_metadata, "research_sources"),
        ),
        commit_sha=_metadata_text(metadata, "commit_sha"),
        branch=_metadata_text(metadata, "branch"),
        spec_path=_metadata_text(metadata, "spec_path") or f"tasks/{slug}.yaml",
        backend=_metadata_text(metadata, "backend"),
        memory_candidates=_metadata_items(metadata, "memory_candidates"),
        archive_candidates=_metadata_items(metadata, "archive_candidates"),
        next_actions=next_actions,
        quality_warnings=quality_warnings,
    )


def _summary_from_response(response: str, slug: str) -> str:
    first = response.strip().splitlines()[0].strip() if response.strip() else ""
    return first or f"Self-coding `{slug}` завершён."


def _extract_artifacts(metadata: dict[str, Any]) -> list[str]:
    artifacts: list[str] = []
    for key in ("commit", "commit_sha", "branch", "spec_path", "env_audit_path"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            artifacts.append(f"{key}: {value}")
    return artifacts


def _extract_sources(metadata: dict[str, Any], slug: str) -> list[str]:
    spec_path = metadata.get("spec_path")
    if isinstance(spec_path, str) and spec_path:
        return [spec_path]
    return [f"tasks/{slug}.yaml"]


def _extract_next_actions(response: str) -> list[str]:
    if "dry-run" in response.lower():
        return ["Включить self-coding или запустить после снятия dry-run gate."]
    return []


def _metadata_text(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _metadata_items(metadata: dict[str, Any], key: str) -> tuple[str, ...]:
    value = metadata.get(key)
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, Iterable):
        return tuple(item_text for item in value if (item_text := str(item).strip()))
    item_text = str(value).strip()
    return (item_text,) if item_text else ()


def _unique_items(*groups: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        for item in group:
            if item and item not in seen:
                seen.add(item)
                result.append(item)
    return tuple(result)


def _quality_warnings(
    *,
    changed_files: tuple[str, ...],
    tests: tuple[str, ...],
    risk_review: str,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if not changed_files:
        warnings.append("missing_changed_files")
    if not tests:
        warnings.append("missing_tests")
    if not risk_review:
        warnings.append("missing_risk_review")
    return tuple(warnings)


def _quality_warning_next_actions(warnings: tuple[str, ...]) -> tuple[str, ...]:
    if not warnings:
        return ()
    return (
        "Review self-coding summary quality: missing audit fields "
        f"({', '.join(warnings)}) before treating this run as parity-complete.",
    )


_ARCHIVE_STEM_RE = re.compile(r"[^a-z0-9._-]+")


def _archive_stem(run_summary: SelfCodingRunSummary, created_at: datetime) -> str:
    slug = _safe_archive_slug(run_summary.slug)
    digest = hashlib.sha256(run_summary.model_dump_json().encode("utf-8")).hexdigest()[
        :12
    ]
    return f"{created_at.strftime('%Y%m%dT%H%M%SZ')}-{slug}-{digest}"


def _safe_archive_slug(value: str) -> str:
    safe = _ARCHIVE_STEM_RE.sub("-", value.lower()).strip("-._")
    return safe[:80] or "self-coding"


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _archive_markdown(*, created_at: datetime, markdown_report: str) -> str:
    return (
        "# SelfCodingRunSummary archive\n\n"
        f"- created_at: `{created_at.isoformat()}`\n"
        "- artifact_scope: `self_coding_run_summary`\n\n"
        f"{markdown_report.rstrip()}\n"
    )


def _render_run_summary_markdown(
    run_summary: SelfCodingRunSummary,
    *,
    original_response: str,
) -> str:
    lines = [
        "## Сводка self-coding",
        f"- Spec: `{run_summary.slug}`",
        f"- Итог: {run_summary.summary}",
    ]
    if run_summary.branch:
        lines.append(f"- Branch: `{run_summary.branch}`")
    if run_summary.commit_sha:
        lines.append(f"- Commit: `{run_summary.commit_sha}`")
    if run_summary.backend:
        lines.append(f"- Backend: `{run_summary.backend}`")

    _append_list_section(lines, "Изменённые файлы", run_summary.changed_files)
    _append_text_section(lines, "Reasoning summary", run_summary.reasoning_summary)
    _append_text_section(lines, "Approval summary", run_summary.approval_summary)
    _append_list_section(lines, "Decision transcript", run_summary.decision_transcript)
    _append_list_section(lines, "Research sources", run_summary.research_sources)
    _append_list_section(lines, "Проверки", run_summary.tests)
    _append_list_section(lines, "Repair history", run_summary.failed_repairs)
    _append_text_section(lines, "Risk review", run_summary.risk_review)
    _append_list_section(
        lines,
        "Summary quality warnings",
        run_summary.quality_warnings,
    )
    _append_list_section(lines, "Memory candidates", run_summary.memory_candidates)
    _append_list_section(lines, "Archive candidates", run_summary.archive_candidates)
    _append_list_section(lines, "Next actions", run_summary.next_actions)
    if original_response.strip():
        _append_text_section(lines, "Исходный ответ", original_response.strip())
    return "\n".join(lines)


def _append_text_section(lines: list[str], title: str, text: str) -> None:
    if text:
        lines.extend(("", f"### {title}", text))


def _append_list_section(lines: list[str], title: str, items: Iterable[str]) -> None:
    values = tuple(items)
    if not values:
        return
    lines.extend(("", f"### {title}"))
    lines.extend(f"- {item}" for item in values)

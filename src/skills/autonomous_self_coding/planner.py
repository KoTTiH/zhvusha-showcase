"""Planner for Жвуша's autonomous self-coding loop."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel

from src.agent_runtime.self_work_context import sanitize_self_work_text
from src.skills.base import AgentContext, SkillResult
from src.skills.spec_command.parser import SpecModel, SpecStatus
from src.skills.spec_command.store import (
    find_spec_path,
    list_spec_files,
    load_spec_raw,
    save_spec_raw,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from src.agent_runtime.models import AgentJob, ContextPack

_MAX_CONTEXT_CHARS = 9000
_MAX_FILE_CHARS = 1800
_MAX_ARCHIVE_FILES = 4


class SelfImprovementCycleResult(BaseModel):
    """Structured result returned by one autonomous self-work cycle."""

    status: Literal[
        "started_implementation",
        "created_pending",
        "blocked",
        "skipped",
        "failed",
    ]
    summary: str
    spec_slug: str = ""
    implementation_job_id: str = ""
    change_summary_path: str = ""
    memory_candidates: tuple[str, ...] = ()
    next_actions: tuple[str, ...] = ()
    details: str = ""
    needs_user_confirmation: bool = False


class IdeationSkillProtocol(Protocol):
    """Minimal Architect surface used by the autonomous planner."""

    async def execute(self, message: str, context: AgentContext) -> SkillResult: ...


class ImplementationRunnerProtocol(Protocol):
    """Minimal Agent Runtime self-coding runner surface."""

    async def start_background(
        self,
        *,
        slug: str,
        context: AgentContext,
        recent_messages: tuple[str, ...] = (),
        completion_callback: Callable[[Any], Awaitable[None]],
    ) -> Any: ...


class AutonomousSelfCodingEngine:
    """Turn Жвуша's self-work context into one safe self-coding cycle."""

    def __init__(
        self,
        *,
        tasks_dir: Path,
        workspace_root: Path,
        admin_user_id: int,
        ideation_skill: IdeationSkillProtocol,
        implementation_runner: ImplementationRunnerProtocol,
        max_autonomous_tier: int = 1,
        change_summary_dir: Path | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._tasks_dir = tasks_dir
        self._workspace_root = workspace_root
        self._admin_user_id = admin_user_id
        self._ideation = ideation_skill
        self._implementation = implementation_runner
        self._max_autonomous_tier = max(0, max_autonomous_tier)
        self._change_summary_dir = (
            change_summary_dir
            if change_summary_dir is not None
            else workspace_root / "self_coding_summaries" / "autonomous"
        )
        self._clock = clock or (lambda: datetime.now(tz=UTC))

    async def run_once(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
    ) -> SelfImprovementCycleResult:
        """Run one autonomous discovery/approval/implementation pass."""
        existing = self._next_existing_spec()
        if existing is not None:
            spec, _raw, path = existing
            return _existing_spec_needs_fresh_confirmation(
                spec=spec,
                path=path,
                blocker=self._autonomous_blocker(spec),
            )

        ideation = await self._create_spec(job, context_pack=context_pack)
        if not ideation.success:
            return SelfImprovementCycleResult(
                status="failed",
                summary=ideation.response
                or "Autonomous Architect did not create a spec.",
                next_actions=("Inspect Architect failure before retrying.",),
            )
        slug = str(ideation.metadata.get("slug") or "").strip()
        if not slug:
            return SelfImprovementCycleResult(
                status="failed",
                summary="Autonomous Architect returned no spec slug.",
                next_actions=("Fix Architect metadata contract before retrying.",),
            )
        generated_path = find_spec_path(self._tasks_dir, slug)
        if generated_path is None:
            return SelfImprovementCycleResult(
                status="failed",
                summary=f"Autonomous Architect reported `{slug}`, but no spec file exists.",
                spec_slug=slug,
            )
        raw = load_spec_raw(generated_path)
        spec = SpecModel.model_validate(raw)
        blocker = self._autonomous_blocker(spec)
        if blocker:
            return SelfImprovementCycleResult(
                status="created_pending",
                summary=f"Created `{slug}` but left it pending: {blocker}.",
                spec_slug=slug,
                next_actions=_approval_next_actions(spec=spec, blocker=blocker),
                details=generated_path.as_posix(),
            )
        spec = self._approve(raw=raw, path=generated_path, spec=spec)
        return await self._start_implementation(spec=spec, job=job)

    def _next_existing_spec(self) -> tuple[SpecModel, dict[str, Any], Path] | None:
        """Return old runnable specs for fresh chat confirmation, not execution."""
        pending: tuple[SpecModel, dict[str, Any], Path] | None = None
        for path in list_spec_files(self._tasks_dir):
            try:
                raw = load_spec_raw(path)
                spec = SpecModel.model_validate(raw)
            except (KeyError, ValueError):
                continue
            if spec.status is SpecStatus.APPROVED:
                return spec, raw, path
            if (
                pending is None
                and spec.status is SpecStatus.PENDING_APPROVAL
                and spec.created_by == "zhvusha"
            ):
                pending = (spec, raw, path)
        return pending

    async def _create_spec(
        self,
        job: AgentJob,
        *,
        context_pack: ContextPack,
    ) -> SkillResult:
        context = self._context(job)
        request = _build_ideation_request(
            self._workspace_root, context_pack=context_pack
        )
        return await self._ideation.execute(f"/spec_create {request}", context)

    def _approve(
        self,
        *,
        raw: dict[str, Any],
        path: Path,
        spec: SpecModel,
    ) -> SpecModel:
        raw["status"] = SpecStatus.APPROVED.value
        raw["approved_at"] = self._clock().astimezone(UTC).isoformat()
        raw["approved_by"] = "zhvusha"
        mandate = (
            "Tier 3 autonomous mandate from Никита; post-implementation "
            "architecture/safety review remains mandatory; "
            if spec.tier == 3
            else ""
        )
        raw["autonomous_approval_reason"] = (
            f"Tier {spec.tier}; {mandate}Жвуша-created; within autonomous max tier "
            f"{self._max_autonomous_tier}; no live env activation; existing "
            "ImplementSpec whitelist, tests, reviewer/formal and no-downgrade "
            "gates remain mandatory."
        )
        approved = SpecModel.model_validate(raw)
        save_spec_raw(path, raw)
        return approved

    async def _start_implementation(
        self,
        *,
        spec: SpecModel,
        job: AgentJob,
    ) -> SelfImprovementCycleResult:
        change_summary_path = self._change_summary_path(spec)
        implementation_job_id_box: dict[str, str] = {}

        async def _write_completion_summary(result: Any) -> None:
            self._write_change_summary(
                spec=spec,
                implementation_job_id=implementation_job_id_box.get("id", ""),
                result=result,
                path=change_summary_path,
            )

        implementation_job = await self._implementation.start_background(
            slug=spec.slug,
            context=self._context(job),
            recent_messages=(
                "Autonomous self-work cycle approved this spec under the standing "
                f"Tier {spec.tier} autonomous mandate. Preserve all normal "
                "self-coding gates, and for Tier 3 keep the architecture/safety "
                "review strict.",
            ),
            completion_callback=_write_completion_summary,
        )
        implementation_job_id = str(getattr(implementation_job, "id", ""))
        implementation_job_id_box["id"] = implementation_job_id
        return SelfImprovementCycleResult(
            status="started_implementation",
            summary=f"Started autonomous implementation for `{spec.slug}`.",
            spec_slug=spec.slug,
            implementation_job_id=implementation_job_id,
            change_summary_path=change_summary_path.as_posix(),
            memory_candidates=(
                f"Autonomous self-coding selected `{spec.slug}` and started "
                "implementation through the normal gated self-coding runner. "
                f"Detailed change summary path: {change_summary_path.as_posix()}",
            ),
            next_actions=(
                "Wait for the self-coding implementation job to finish.",
                f"Review the detailed change summary at {change_summary_path.as_posix()}.",
            ),
        )

    def _context(self, job: AgentJob) -> AgentContext:
        return AgentContext(
            user_id=self._admin_user_id,
            chat_id=job.chat_id,
            mode="personal",
            bot=None,
            metadata={
                "autonomous_self_coding": True,
                "agent_job_id": job.id,
                "autonomous_self_coding_source_message_id": job.source_message_id,
            },
        )

    def _autonomous_blocker(self, spec: SpecModel) -> str:
        if spec.status not in {SpecStatus.PENDING_APPROVAL, SpecStatus.APPROVED}:
            return f"status `{spec.status.value}` is not runnable"
        if spec.tier >= 3 and spec.approved_by != "nikita":
            return (
                "Tier 3 requires Никита's explicit chat approval and discussion "
                "before implementation"
            )
        if spec.tier > self._max_autonomous_tier:
            return (
                f"Tier {spec.tier} is above autonomous max tier "
                f"{self._max_autonomous_tier}"
            )
        if spec.live_env_activation:
            return "live env activation is not allowed in autonomous self-coding"
        return ""

    def _change_summary_path(self, spec: SpecModel) -> Path:
        stamp = self._clock().astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        return self._change_summary_dir / f"{stamp}-{spec.slug}.md"

    def _write_change_summary(
        self,
        *,
        spec: SpecModel,
        implementation_job_id: str,
        result: Any,
        path: Path,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        response = str(getattr(result, "response", "") or "").strip()
        success = bool(getattr(result, "success", False))
        raw_metadata = getattr(result, "metadata", {}) or {}
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        resolved_job_id = implementation_job_id or str(metadata.get("agent_job_id", ""))
        path.write_text(
            _render_change_summary(
                created_at=self._clock().astimezone(UTC),
                spec=spec,
                implementation_job_id=resolved_job_id,
                success=success,
                response=response,
                metadata=metadata,
            ),
            encoding="utf-8",
        )


def _existing_spec_needs_fresh_confirmation(
    *,
    spec: SpecModel,
    path: Path,
    blocker: str = "",
) -> SelfImprovementCycleResult:
    blocker_note = f" Текущий safety-блокер: {blocker}." if blocker else ""
    return SelfImprovementCycleResult(
        status="blocked",
        summary=(
            f"Нашла старую runnable-спеку `{spec.slug}`, но автономно не запускаю: "
            "нужно свежее подтверждение Никиты, что это всё ещё надо делать."
            f"{blocker_note}"
        ),
        spec_slug=spec.slug,
        next_actions=(
            f"Спросить Никиту, актуальна ли `{spec.slug}`.",
            "Если да — запускать через обычный approval/spec_run path.",
            "Если нет — отклонить, архивировать или переписать спеку перед новым запуском.",
        ),
        details=path.as_posix(),
        needs_user_confirmation=True,
    )


def _build_ideation_request(
    workspace_root: Path,
    *,
    context_pack: ContextPack | None = None,
) -> str:
    context = _read_context_pack_self_work_context(context_pack)
    if not context:
        context = _read_self_work_context(workspace_root)
    return (
        "Autonomous self-work cycle. Choose exactly one small, high-confidence "
        "improvement for Жвуша herself and write a normal tasks/*.yaml spec. "
        "The improvement must increase capability, context, checks, reliability "
        "or useful complexity without flattening personality or removing gates. "
        "Prefer the lowest sufficient tier. Tier 3 is allowed only when the "
        "improvement truly changes an architectural/personality/safety contract "
        "and must remain pending for Никита's short chat approval; do not "
        "self-approve Tier 3. Do not create live "
        ".env activation, restart/publish/send-message side effects, "
        "auth/billing/secrets changes, or broad rewrites. Preserve all /код, "
        "Agent Runtime, whitelist, test, reviewer/formal, memory staging, "
        "detailed change summary and no-downgrade behavior.\n\n"
        "Self-work context Жвуша collected:\n"
        f"{context}"
    )


def _read_context_pack_self_work_context(context_pack: ContextPack | None) -> str:
    if context_pack is None:
        return ""
    has_capsule = context_pack.metadata.get("self_work_context_capsule") == "true"
    parts: list[str] = []
    if has_capsule or "Self-work Context Capsule" in context_pack.active_code_state:
        if context_pack.active_code_state.strip():
            parts.append(context_pack.active_code_state.strip())
        parts.extend(item.strip() for item in context_pack.chat_context if item.strip())
        structured_candidates = _structured_self_work_candidates(context_pack)
        if structured_candidates:
            parts.append(structured_candidates)
    joined = "\n\n".join(parts).strip()
    if not joined:
        return ""
    if len(joined) <= _MAX_CONTEXT_CHARS:
        return joined
    return joined[:_MAX_CONTEXT_CHARS].rstrip() + "\n... [truncated]"


def _structured_self_work_candidates(context_pack: ContextPack) -> str:
    raw = context_pack.metadata.get("self_work_safe_spec_candidates", "")
    candidates = tuple(
        sanitize_self_work_text(item)
        for line in raw.splitlines()
        if (item := line.strip())
    )
    if not candidates:
        return ""
    lines = ["## Structured self-work candidates"]
    lines.extend(
        f"- safe_spec_candidate: {candidate}"
        for candidate in candidates[:_MAX_ARCHIVE_FILES]
    )
    return "\n".join(lines)


def _read_self_work_context(workspace_root: Path) -> str:
    sections: list[str] = []
    for label, path in (
        ("personality pillars", workspace_root / "personality" / "pillars.md"),
        ("personality core", workspace_root / "personality" / "core.md"),
        ("consolidation", workspace_root / "inbox" / "consolidation_results.md"),
    ):
        excerpt = _read_excerpt(path)
        if excerpt:
            sections.append(f"## {label}\n{excerpt}")

    archive_dir = workspace_root / "self_coding_archive"
    if archive_dir.exists():
        archive_paths = sorted(
            archive_dir.rglob("*.md"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )[:_MAX_ARCHIVE_FILES]
        for path in archive_paths:
            excerpt = _read_excerpt(path)
            if excerpt:
                sections.append(f"## self_coding_archive/{path.name}\n{excerpt}")

    joined = "\n\n".join(sections).strip()
    if not joined:
        return "No local self-work context was available; choose a minimal test/docs/check improvement."
    if len(joined) <= _MAX_CONTEXT_CHARS:
        return joined
    return joined[:_MAX_CONTEXT_CHARS].rstrip() + "\n... [truncated]"


def _approval_next_actions(
    *,
    spec: SpecModel,
    blocker: str,
) -> tuple[str, ...]:
    if spec.tier >= 3:
        return (
            (
                f"Ask Никита briefly in chat: Это Tier 3 по `{spec.slug}`. "
                "Разрешаешь запуск или сначала обсудим правки?"
            ),
            "Do not paste the full spec unless Никита asks for details.",
            "Classify Никита's free-text decision through the AI approval classifier.",
        )
    return (f"Wait for manual approval: {blocker}.",)


def _read_excerpt(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""
    cleaned = text.strip()
    if len(cleaned) <= _MAX_FILE_CHARS:
        return cleaned
    return cleaned[:_MAX_FILE_CHARS].rstrip() + "\n... [truncated]"


def _render_change_summary(
    *,
    created_at: datetime,
    spec: SpecModel,
    implementation_job_id: str,
    success: bool,
    response: str,
    metadata: dict[str, Any],
) -> str:
    metadata_lines = _metadata_lines(metadata)
    whitelist = "\n".join(f"- `{path}`" for path in spec.whitelist_paths)
    preserve = "\n".join(f"- {entry}" for entry in spec.preserve_behavior)
    simplifications = "\n".join(f"- {entry}" for entry in spec.allowed_simplifications)
    return (
        "# Сводка автономного изменения\n\n"
        f"- created_at: `{created_at.isoformat()}`\n"
        f"- spec: `{spec.slug}`\n"
        f"- tier: `Tier {spec.tier}`\n"
        f"- status: `{'success' if success else 'failed'}`\n"
        f"- implementation_job_id: `{implementation_job_id or 'unknown'}`\n\n"
        "## Цель\n"
        f"{spec.goal}\n\n"
        "## Обоснование автономного approval\n"
        f"{spec.autonomous_approval_reason or 'Не указано.'}\n\n"
        "## Затронутые файлы по spec\n"
        f"{whitelist or '- Нет whitelist_paths.'}\n\n"
        "## Контракт сохранения поведения\n"
        f"{preserve or '- Не указан.'}\n\n"
        "## Допустимые упрощения\n"
        f"{simplifications or '- Нет.'}\n\n"
        "## Метаданные реализации\n"
        f"{metadata_lines or '- Нет metadata.'}\n\n"
        "## Итог Codex/ImplementSpec\n"
        f"{response or 'Нет ответа от implementation runner.'}\n"
    )


def _metadata_lines(metadata: dict[str, Any]) -> str:
    lines: list[str] = []
    for key in sorted(metadata):
        value = metadata[key]
        if value is None or value == "":
            continue
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)

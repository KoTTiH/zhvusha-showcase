"""ImplementSpecSkill — Editor half of the Architect/Editor split (Phase 13).

Picks up an *approved* spec from ``tasks/<slug>.yaml``, runs it through
the configured Codex code-agent backend in a temporary detached worktree, and
cherry-picks the green diff back as a ``zhvusha-coder`` commit.

Hard gates, in order:

1. Admin-only, personal mode, ``/spec_run`` or ``/implement_spec`` command.
2. Spec exists and parses against ``SpecModel``.
3. Spec ``status == APPROVED``.
4. ``spec.tier <= self_coding_max_tier``. Tier 3 is allowed only when
   Никита explicitly raised the configured max tier and approved the spec.
5. ``CapsEnforcer.check`` allows the invocation (KB #90 ban-safe rate).
6. ``self_coding_enabled and not context.is_dry_run`` — otherwise return
   a dry-run preview without touching git or the code-agent backend.

On the happy path: temporary worktree → record cap → backend → commit gate →
cherry-pick to the live branch → write status back. Failures
(IsolatedWorkspaceError, SDKUnavailableError, CommitRunnerError) mark the spec
``failed`` and surface a chat-friendly reason.
"""

from __future__ import annotations

import contextlib
import re
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

import structlog

from src.skills.base import (
    AgentContext,
    DelegatedSkill,
    ExecutionPlan,
    SideEffect,
    SkillResult,
)
from src.skills.chat_self_coding.events import (
    BlockEvent,
    BlockEventType,
    NoopBlockPublisher,
)
from src.skills.implement_spec.commit_runner import (
    CommitResult,
    CommitRunner,
    CommitRunnerError,
)
from src.skills.implement_spec.dry_run import simulate
from src.skills.implement_spec.env_guard import (
    EnvGuard,
    EnvGuardResult,
    LiveEnvActivationResult,
    LiveEnvActivator,
    format_protected_env_prompt,
)
from src.skills.implement_spec.formal_gates import (
    FormalGateInputs,
    FormalGateVerdict,
)
from src.skills.implement_spec.reviewer import ReviewRequest, ReviewVerdict
from src.skills.implement_spec.sdk_runner import (
    EditorSdkResult,
    SDKUnavailableError,
)
from src.skills.implement_spec.workspace import (
    AppliedCommit,
    IsolatedWorkspace,
    IsolatedWorkspaceError,
    IsolatedWorkspaceManager,
    PreservedFailureWorkspace,
)
from src.skills.spec_command.parser import (
    ExistingTestUpdate,
    SpecKind,
    SpecModel,
    SpecStatus,
)
from src.skills.spec_command.store import (
    find_spec_path,
    load_spec_raw,
    save_spec_raw,
)
from src.utils.subprocess_env import clean_env_for_git_subprocess

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.skills.chat_self_coding.events import BlockPublisher
    from src.skills.implement_spec.branch_manager import BranchManager
    from src.skills.implement_spec.caps_enforcer import CapsEnforcer
    from src.skills.spec_command.parser import SpecModel as _SpecModel

logger = structlog.get_logger()

_TRIGGERS: tuple[str, ...] = ("/spec_run", "/implement_spec")
_NATURAL_SPEC_RUN_PREFIXES: tuple[str, ...] = (
    "запусти approved spec",
    "реализуй approved spec",
    "запусти spec",
    "реализуй spec",
)
_REPAIR_RETRY_LIMIT = 1
_REPAIR_CONTEXT_MAX_CHARS = 6000
_REPAIR_CONTEXT_EDGE_CHARS = 2500


def _normalize_chat_route_text(text: str) -> str:
    return " ".join(text.strip().lower().replace("ё", "е").split())


def _strip_natural_tail(original: str, prefix: str) -> str:
    pattern = r"^\s*" + r"\s+".join(re.escape(part) for part in prefix.split())
    pattern += r"\s*[:\-—]?\s*"
    return re.sub(pattern, "", original, count=1, flags=re.I).strip(" \t\n\r:-—")


def _extract_natural_spec_slug(message: str) -> str | None:
    text = message.strip()
    normalized = _normalize_chat_route_text(text)
    for prefix in _NATURAL_SPEC_RUN_PREFIXES:
        if (
            normalized == prefix
            or normalized.startswith(prefix + " ")
            or normalized.startswith(prefix + ":")
        ):
            tail = _strip_natural_tail(text, prefix)
            slug = tail.split(maxsplit=1)[0] if tail else ""
            return slug
    return None


@dataclass(frozen=True)
class _EditorAttemptOutcome:
    success: bool
    retryable: bool
    gate: str
    reason: str
    response: str
    error_reason: str
    next_step: str
    commit: CommitResult | None = None
    sdk_outcome: EditorSdkResult | None = None
    formal_result: FormalGateVerdict | None = None
    current_test_count: int = 0
    changed_files: tuple[str, ...] = ()
    repair_notes: tuple[str, ...] = ()
    attempt: int = 0
    max_attempts: int = 0
    repair_note: str = ""
    needs_user_decision: bool = False
    decision_question: str = ""


@dataclass(frozen=True)
class _EditorResumeContext:
    session_id: str
    worktree_path: Path
    worktree_label: str
    base_branch: str
    base_sha: str


class _EditorSdkProto(Protocol):
    """Callable that runs the Codex Editor backend.

    Production binds :func:`src.skills.implement_spec.sdk_runner.run_editor_sdk`.
    Tests pass an ``AsyncMock`` whose side effect writes whitelist files
    to simulate the real Editor's diff.
    """

    async def __call__(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        cwd: Path,
        project_root: Path,
        whitelist_paths: list[str],
        existing_tests_to_update_paths: list[str] | None = None,
        progress_callback: Callable[[str], Awaitable[None]] | None = None,
        session_id: str = "",
        persist_session: bool = False,
    ) -> EditorSdkResult: ...


class _CycleAnalyzerProto(Protocol):
    async def record_success(
        self,
        *,
        spec: _SpecModel,
        spec_path: Path,
        branch_name: str,
        commit_sha: str,
        sdk_summary: str,
        backend: str,
    ) -> object: ...

    async def record_failure(
        self,
        *,
        spec: _SpecModel,
        spec_path: Path,
        branch_name: str | None,
        reason: str,
        backend: str = "codex_cli",
    ) -> object: ...


class _FormalGateProto(Protocol):
    def __call__(self, inputs: FormalGateInputs) -> FormalGateVerdict: ...


class _ReviewerProto(Protocol):
    async def review(self, request: ReviewRequest) -> ReviewVerdict: ...


class _AdversarialProviderProto(Protocol):
    async def generate(self, query: str, *, limit: int = 5) -> list[Any]: ...


class _EnvGuardProto(Protocol):
    instruction_text: str

    def enforce(self, project_root: Path) -> EnvGuardResult: ...


class _WorkspaceManagerProto(Protocol):
    def create_workspace(self, slug: str) -> IsolatedWorkspace: ...

    def reopen_preserved_workspace(
        self,
        *,
        path: Path,
        label: str,
        base_branch: str,
        base_sha: str,
    ) -> IsolatedWorkspace: ...

    def apply_commit(self, commit_sha: str) -> AppliedCommit: ...

    def preserve_failure(
        self, workspace: IsolatedWorkspace, *, reason: str
    ) -> PreservedFailureWorkspace: ...

    def cleanup(self, workspace: IsolatedWorkspace) -> None: ...


class ImplementSpecSkill(DelegatedSkill):
    """Run an approved tasks/<slug>.yaml through the Editor backend."""

    name: ClassVar[str] = "implement_spec"
    description: ClassVar[str] = "Editor: run approved tasks/<slug>.yaml spec via Codex"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "strategist"

    triggers: ClassVar[list[str]] = list(_TRIGGERS)

    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "high"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = (
        "required"
    )

    side_effects: ClassVar[list[SideEffect]] = [
        SideEffect.DELEGATES_TO_CODE_AGENT,
        SideEffect.CALLS_LLM,
        SideEffect.CALLS_LLM_TIER_STRATEGIST,
        SideEffect.READS_FILESYSTEM,
        SideEffect.WRITES_FILESYSTEM,
        SideEffect.READS_FROM_KB,
        SideEffect.SPAWNS_SUBPROCESS,
        SideEffect.NETWORK_IO_EXTERNAL,
    ]

    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    executor: ClassVar[str] = "codex_cli"
    max_duration_seconds: ClassVar[float] = 600.0

    def __init__(
        self,
        *,
        tasks_dir: Path,
        project_root: Path,
        admin_user_id: int,
        self_coding_enabled: bool,
        self_coding_max_tier: int,
        caps_enforcer: CapsEnforcer,
        branch_manager: BranchManager,
        commit_runner: CommitRunner,
        sdk_runner: _EditorSdkProto,
        clock: Callable[[], datetime] | None = None,
        block_publisher: BlockPublisher | None = None,
        cycle_analyzer: _CycleAnalyzerProto | None = None,
        formal_gate: _FormalGateProto | None = None,
        reviewer: _ReviewerProto | None = None,
        adversarial_provider: _AdversarialProviderProto | None = None,
        env_guard: _EnvGuardProto | None = None,
        live_env_activator: LiveEnvActivator | None = None,
        workspace_manager: _WorkspaceManagerProto | None = None,
        commit_runner_factory: Callable[[Path], CommitRunner] | None = None,
    ) -> None:
        self._tasks_dir = tasks_dir
        self._project_root = project_root
        self._admin_user_id = admin_user_id
        self._self_coding_enabled = self_coding_enabled
        self._self_coding_max_tier = self_coding_max_tier
        self._caps = caps_enforcer
        # Backwards-compatible constructor parameter; self-coding now runs in a
        # temporary detached worktree instead of checking out ``zhvusha/*`` here.
        del branch_manager
        self._commits = commit_runner
        self._sdk = sdk_runner
        self._clock = clock or (lambda: datetime.now(tz=UTC))
        # Phase 40 — emit per-stage block events so chat-mode can show
        # the four block messages (preparation / implementation / done /
        # error). Default Noop preserves slash-only behaviour for callers
        # that don't wire chat mode.
        self._block_publisher = block_publisher or NoopBlockPublisher()
        self._cycle_analyzer = cycle_analyzer
        self._formal_gate = formal_gate
        self._reviewer = reviewer
        self._adversarial_provider = adversarial_provider
        self._env_guard = env_guard or EnvGuard.from_env_file(project_root / ".env")
        self._live_env_activator = live_env_activator or LiveEnvActivator(
            live_env_path=project_root / ".env"
        )
        self._workspaces = workspace_manager or IsolatedWorkspaceManager(
            repo_root=project_root
        )
        self._commit_runner_factory = commit_runner_factory or (
            lambda repo_root: CommitRunner(repo_root=repo_root)
        )

    # =================================================================
    # routing
    # =================================================================

    async def can_handle(self, message: str, context: AgentContext) -> float:
        if context.user_id != self._admin_user_id or context.mode != "personal":
            return 0.0
        text = message.strip()
        if any(text.startswith(t) for t in _TRIGGERS):
            return 1.0
        if _extract_natural_spec_slug(message) is not None:
            return 0.93
        return 0.0

    async def prepare(self, message: str, context: AgentContext) -> ExecutionPlan:
        del context
        slug = self._extract_slug(message)
        missing_fields = ["spec_slug"] if not slug else []
        return ExecutionPlan(
            skill_name=self.name,
            skill_type="delegated",
            human_summary=(
                f"Editor → реализовать spec `{slug}`"
                if slug
                else "Нужен slug approved spec для запуска."
            ),
            estimated_tokens=80000,
            estimated_cost_usd=Decimal("1.00"),
            estimated_duration_seconds=self.max_duration_seconds,
            files_to_modify=[self._tasks_dir],
            side_effects_invoked=list(self.side_effects),
            llm_calls_planned=1,
            delegated_to=self.executor,
            metadata={
                **(
                    {
                        "requires_user_input": True,
                        "missing_fields": missing_fields,
                    }
                    if missing_fields
                    else {}
                )
            },
        )

    # =================================================================
    # execute
    # =================================================================

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        slug = self._extract_slug(message)
        if not slug:
            return SkillResult(
                success=False,
                response=("Укажи slug: `/spec_run <slug>`. Список — `/spec list`."),
            )
        return await self.run_slug(slug, context)

    async def run_slug(self, slug: str, context: AgentContext) -> SkillResult:
        """Run one approved spec by slug.

        This is the reusable implementation-engine entrypoint. Slash command
        handling remains a thin wrapper in ``execute()``, while Agent Runtime's
        native self-coding worker calls this method directly.
        """
        spec_path = find_spec_path(self._tasks_dir, slug)
        if spec_path is None:
            return SkillResult(
                success=False,
                response=f"Spec `{slug}` не найден в tasks/.",
            )

        loaded = self._load_and_validate(spec_path, slug)
        if isinstance(loaded, SkillResult):
            return loaded
        spec, raw = loaded

        gate = self._gate_status_and_tier(spec)
        if gate is not None:
            return gate

        caps_check = await self._caps.check()
        if not caps_check.allowed:
            return SkillResult(
                success=False, response=f"Caps gate: {caps_check.reason}"
            )

        if not self._self_coding_enabled or context.is_dry_run:
            sim = simulate(spec=spec)
            return SkillResult(success=True, response=sim.would_produce)

        spec_checkpoint = self._checkpoint_active_spec_if_needed(spec, spec_path)
        if spec_checkpoint is not None:
            return spec_checkpoint

        return await self._run_live_cycle(
            spec=spec,
            spec_path=spec_path,
            raw=raw,
            user_id=context.user_id,
            runtime_chat_context=_runtime_chat_context(context),
            code_task_id=_runtime_code_task_id(context),
            editor_resume=_editor_resume_context(context),
        )

    # ------------------------------------------------------------- guards

    def _load_and_validate(
        self, spec_path: Path, slug: str
    ) -> tuple[SpecModel, dict[str, Any]] | SkillResult:
        try:
            raw = load_spec_raw(spec_path)
            spec = SpecModel.model_validate(raw)
        except (ValueError, KeyError) as exc:
            logger.warning(
                "implement_spec_validation_failed", slug=slug, error=str(exc)
            )
            return SkillResult(
                success=False,
                response=f"Spec `{slug}` не валидируется: {exc}",
            )
        return spec, raw

    def _gate_status_and_tier(self, spec: SpecModel) -> SkillResult | None:
        if spec.status != SpecStatus.APPROVED:
            return SkillResult(
                success=False,
                response=(
                    f"Spec `{spec.slug}` в статусе `{spec.status.value}`. "
                    f"Сначала одобри: `/spec approve {spec.slug}`."
                ),
            )
        if spec.tier > self._self_coding_max_tier:
            return SkillResult(
                success=False,
                response=(
                    f"Spec `{spec.slug}` — Tier {spec.tier}, "
                    f"а self_coding_max_tier = {self._self_coding_max_tier}. "
                    f"Эскалирую к Никите."
                ),
            )
        return None

    def _checkpoint_active_spec_if_needed(
        self, spec: SpecModel, spec_path: Path
    ) -> SkillResult | None:
        """Commit Жвуша's just-approved spec artifact before worktree isolation.

        Architect writes ``tasks/<spec>.yaml`` in the live repo. Editor then
        requires a clean repo before creating the detached worktree. If the
        active spec is the only dirty path, checkpoint it automatically; if
        anything else is dirty, keep the existing safety guard intact.
        """
        dirty_paths = _dirty_git_paths(self._project_root)
        if not dirty_paths:
            return None

        try:
            spec_rel = spec_path.relative_to(self._project_root).as_posix()
        except ValueError:
            return None

        if dirty_paths != [spec_rel]:
            return None

        try:
            self._commits.commit_yaml_update(
                spec_slug=spec.slug,
                spec_path=spec_path,
                subject="checkpoint approved spec",
            )
        except CommitRunnerError as exc:
            return SkillResult(
                success=False,
                response=f"Не смогла сохранить spec checkpoint `{spec.slug}`: {exc}",
            )
        logger.info("implement_spec_checkpointed_active_spec", slug=spec.slug)
        return None

    # --------------------------------------------------------- live cycle

    async def _run_live_cycle(
        self,
        *,
        spec: SpecModel,
        spec_path: Path,
        raw: dict[str, Any],
        user_id: int,
        runtime_chat_context: tuple[str, ...],
        code_task_id: str,
        editor_resume: _EditorResumeContext | None = None,
    ) -> SkillResult:
        slug = spec.slug
        workspace: IsolatedWorkspace | None = None
        resumed_workspace = False
        initial_editor_session_id = ""
        if editor_resume is not None:
            try:
                workspace = self._workspaces.reopen_preserved_workspace(
                    path=editor_resume.worktree_path,
                    label=editor_resume.worktree_label,
                    base_branch=editor_resume.base_branch,
                    base_sha=editor_resume.base_sha,
                )
                _reset_workspace_to_base(workspace)
                resumed_workspace = True
                initial_editor_session_id = editor_resume.session_id
            except (IsolatedWorkspaceError, CommitRunnerError):
                logger.warning(
                    "implement_spec_editor_resume_workspace_unavailable",
                    slug=slug,
                    path=str(editor_resume.worktree_path),
                    exc_info=True,
                )
        try:
            if workspace is None:
                workspace = self._workspaces.create_workspace(slug)
        except IsolatedWorkspaceError as exc:
            await self._record_failure(
                spec=spec,
                spec_path=spec_path,
                branch_name=None,
                reason=f"isolated_workspace: {exc}",
            )
            await self._publish_error(
                user_id=user_id,
                slug=slug,
                code_task_id=code_task_id,
                reason=f"Не смогла создать временную рабочую копию: {exc}",
                next_step=(
                    "Проверь, что основной репозиторий чистый и не находится "
                    "на ветке `zhvusha/*`."
                ),
            )
            return SkillResult(
                success=False,
                response=f"Временная рабочая копия не создана: {exc}",
                metadata={
                    "needs_user_decision": "true",
                    "failure_category": "needs_user_decision",
                    "decision_question": (
                        "Что сделать с текущим dirty git state: сохранить, "
                        "убрать или перенести изменения перед новым запуском?"
                    ),
                },
            )

        assert workspace is not None
        cleanup_workspace = True
        try:
            result = await self._run_isolated_workspace_cycle(
                spec=spec,
                spec_path=spec_path,
                raw=raw,
                user_id=user_id,
                workspace=workspace,
                runtime_chat_context=runtime_chat_context,
                code_task_id=code_task_id,
                resumed_workspace=resumed_workspace,
                initial_editor_session_id=initial_editor_session_id,
            )
            if not result.success:
                cleanup_workspace = False
                return self._preserve_failed_workspace_result(
                    result=result,
                    spec=spec,
                    spec_path=spec_path,
                    raw=raw,
                    workspace=workspace,
                )
            return result
        finally:
            if cleanup_workspace:
                self._workspaces.cleanup(workspace)

    async def _run_isolated_workspace_cycle(
        self,
        *,
        spec: SpecModel,
        spec_path: Path,
        raw: dict[str, Any],
        user_id: int,
        workspace: IsolatedWorkspace,
        runtime_chat_context: tuple[str, ...],
        code_task_id: str,
        resumed_workspace: bool = False,
        initial_editor_session_id: str = "",
    ) -> SkillResult:
        slug = spec.slug
        cycle_started_at = time.monotonic()
        worktree_commits = self._commit_runner_factory(workspace.path)
        worktree_spec_path = workspace.path / spec_path.relative_to(self._project_root)

        await self._publish_block(
            user_id=user_id,
            slug=slug,
            code_task_id=code_task_id,
            event_type=BlockEventType.PREPARATION,
            percent=15,
            detail=(
                "Открыла сохранённую failed worktree и откатила её к базе; "
                "продолжаю прежнюю Codex Editor session."
                if resumed_workspace
                else (
                    "Создала временную рабочую копию; основной репозиторий не "
                    "переключаю. Сейчас считаю baseline и собираю контекст для Codex."
                )
            ),
            stage="рабочая копия готова",
            elapsed_seconds=int(time.monotonic() - cycle_started_at),
            facts=[
                f"base: {workspace.base_branch}",
                f"worktree: {workspace.label}",
                "isolation: worktree",
                f"resume: {str(resumed_workspace).lower()}",
                f"tier: {spec.tier}",
            ],
        )

        await self._caps.record()
        baseline_test_count = _count_test_functions(workspace.path)
        adversarial_context = await self._build_adversarial_context(spec)

        await self._publish_block(
            user_id=user_id,
            slug=slug,
            code_task_id=code_task_id,
            event_type=BlockEventType.IMPLEMENTATION,
            percent=40,
            detail=(
                "Запускаю Codex Editor с утверждённым spec, whitelist и "
                "guard-правилами."
            ),
            stage="запуск агента",
            elapsed_seconds=int(time.monotonic() - cycle_started_at),
            facts=[
                f"baseline tests: {baseline_test_count}",
                f"archive context: {'есть' if adversarial_context else 'нет'}",
                f"whitelist paths: {len(spec.whitelist_paths)}",
            ],
        )

        repair_result = await self._run_editor_repair_loop(
            spec=spec,
            spec_path=spec_path,
            raw=raw,
            workspace=workspace,
            worktree_spec_path=worktree_spec_path,
            worktree_commits=worktree_commits,
            adversarial_context=adversarial_context,
            baseline_test_count=baseline_test_count,
            cycle_started_at=cycle_started_at,
            user_id=user_id,
            runtime_chat_context=runtime_chat_context,
            code_task_id=code_task_id,
            initial_editor_session_id=initial_editor_session_id,
        )
        if isinstance(repair_result, SkillResult):
            return repair_result

        commit = repair_result.commit
        sdk_outcome = repair_result.sdk_outcome
        current_test_count = repair_result.current_test_count
        changed_files = list(repair_result.changed_files)
        assert commit is not None
        assert sdk_outcome is not None

        try:
            applied = self._workspaces.apply_commit(commit.sha)
        except IsolatedWorkspaceError as exc:
            reason = str(exc)
            await self._fail_cycle(
                spec=spec,
                spec_path=spec_path,
                raw=raw,
                branch_name=workspace.label,
                reason=reason,
                user_id=user_id,
                code_task_id=code_task_id,
                error_reason=f"Не смогла применить готовый diff: {reason}",
                next_step="Проверь основной репозиторий и попробуй ещё раз.",
            )
            return SkillResult(
                success=False,
                response=f"Временная рабочая копия не создана: {reason}",
                metadata={
                    "needs_user_decision": "true",
                    "failure_category": "needs_user_decision",
                    "decision_question": (
                        "Нужно решить, что делать с временной worktree/git state: "
                        "повторять с чистой копией или разбирать preserved worktree?"
                    ),
                },
            )

        env_activation_result: LiveEnvActivationResult | None = None
        if spec.live_env_activation:
            try:
                env_activation_result = self._live_env_activator.apply_from_workspace(
                    workspace_root=workspace.path,
                    allowed_keys=tuple(spec.live_env_keys),
                    spec_slug=slug,
                )
            except ValueError as exc:
                reason = str(exc)
                await self._fail_cycle(
                    spec=spec,
                    spec_path=spec_path,
                    raw=raw,
                    branch_name=workspace.label,
                    reason=reason,
                    user_id=user_id,
                    code_task_id=code_task_id,
                    error_reason=f"Live `.env` activation blocked: {reason}",
                    next_step=(
                        "Проверь live_env_keys в spec: protected ключи "
                        "нельзя активировать даже явно."
                    ),
                )
                return SkillResult(
                    success=False,
                    response=reason,
                    metadata={
                        "needs_user_decision": "true",
                        "failure_category": "needs_host_ops",
                        "decision_question": (
                            "Нужно решить, какие live_env_keys можно активировать "
                            "и не требует ли это отдельного host-ops решения."
                        ),
                    },
                )

        raw["status"] = SpecStatus.DONE.value
        raw["branch"] = workspace.base_branch
        raw["commit_sha"] = applied.sha
        raw["iterations"] = int(raw.get("iterations", 0)) + 1
        SpecModel.model_validate(raw)
        save_spec_raw(spec_path, raw)
        self._commits.commit_yaml_update(
            spec_slug=slug,
            spec_path=spec_path,
            subject=f"mark {slug} done",
        )
        logger.info(
            "implement_spec_done",
            slug=slug,
            branch=workspace.base_branch,
            commit=applied.sha,
        )
        await self._record_success(
            spec=spec,
            spec_path=spec_path,
            branch_name=workspace.base_branch,
            commit_sha=applied.sha,
            sdk_summary=sdk_outcome.text,
            backend=sdk_outcome.backend,
        )
        await self._publish_done(
            user_id=user_id,
            slug=slug,
            code_task_id=code_task_id,
            sdk_summary=sdk_outcome.text,
            files=list(spec.whitelist_paths),
            branch=workspace.base_branch,
            commit_sha=applied.sha,
            backend=sdk_outcome.backend,
            test_count_delta=current_test_count - baseline_test_count,
            allowed_simplifications=list(spec.allowed_simplifications),
            env_activation=env_activation_result,
        )
        metadata = {
            "branch": workspace.base_branch,
            "commit_sha": applied.sha,
            "spec_path": spec_path.relative_to(self._project_root).as_posix(),
            "backend": sdk_outcome.backend,
            "changed_files": changed_files,
            "reasoning_summary": sdk_outcome.text,
            "tests": [
                "commit gate: tests, style and types passed",
                f"test_count_delta: {current_test_count - baseline_test_count:+d}",
            ],
            "failed_repairs": list(repair_result.repair_notes),
            "risk_review": (
                "Tier, whitelist, commit gate, optional formal gate and reviewer "
                "gate stayed on the existing ImplementSpec path; summary artifact "
                "does not grant approvals."
            ),
            "memory_candidates": [
                (
                    f"Self-coding `{slug}` completed via {sdk_outcome.backend}: "
                    f"{len(changed_files)} changed files, tests delta "
                    f"{current_test_count - baseline_test_count:+d}."
                )
            ],
            "archive_candidates": [
                (
                    f"self-coding `{slug}` -> {applied.sha[:12]}, "
                    f"changed_files={len(changed_files)}, "
                    f"repair_notes={len(repair_result.repair_notes)}"
                )
            ],
        }
        if (
            env_activation_result is not None
            and env_activation_result.audit_path is not None
        ):
            metadata["env_audit_path"] = str(env_activation_result.audit_path)
        return SkillResult(
            success=True,
            response=(
                f"Готово: `{workspace.base_branch}` @ `{applied.sha[:12]}`. "
                "Изменения уже применены в основную рабочую ветку.\n"
                f"Backend `{sdk_outcome.backend}`: {sdk_outcome.text[:120]}\n"
                f"`/spec show {slug}` для деталей."
            ),
            metadata=metadata,
        )

    async def _run_editor_repair_loop(
        self,
        *,
        spec: SpecModel,
        spec_path: Path,
        raw: dict[str, Any],
        workspace: IsolatedWorkspace,
        worktree_spec_path: Path,
        worktree_commits: CommitRunner,
        adversarial_context: str,
        baseline_test_count: int,
        cycle_started_at: float,
        user_id: int,
        runtime_chat_context: tuple[str, ...],
        code_task_id: str,
        initial_editor_session_id: str = "",
    ) -> _EditorAttemptOutcome | SkillResult:
        repair_notes: list[str] = []
        max_attempts = _REPAIR_RETRY_LIMIT + 1
        editor_session_id = initial_editor_session_id

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                reset_result = await self._prepare_repair_attempt(
                    spec=spec,
                    spec_path=spec_path,
                    raw=raw,
                    workspace=workspace,
                    repair_notes=repair_notes,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    cycle_started_at=cycle_started_at,
                    user_id=user_id,
                    code_task_id=code_task_id,
                )
                if isinstance(reset_result, SkillResult):
                    return reset_result
                worktree_commits = reset_result

            outcome = await self._run_editor_attempt(
                spec=spec,
                spec_path=spec_path,
                workspace=workspace,
                worktree_spec_path=worktree_spec_path,
                worktree_commits=worktree_commits,
                adversarial_context=adversarial_context,
                repair_context=_format_repair_context(repair_notes),
                runtime_chat_context=runtime_chat_context,
                editor_session_id=editor_session_id,
                baseline_test_count=baseline_test_count,
                attempt=attempt,
                max_attempts=max_attempts,
                cycle_started_at=cycle_started_at,
                user_id=user_id,
                code_task_id=code_task_id,
            )
            if outcome.sdk_outcome is not None and outcome.sdk_outcome.session_id:
                editor_session_id = outcome.sdk_outcome.session_id
            if outcome.success:
                return replace(outcome, repair_notes=tuple(repair_notes))
            if not outcome.retryable:
                await self._fail_cycle(
                    spec=spec,
                    spec_path=spec_path,
                    raw=raw,
                    branch_name=workspace.label,
                    reason=outcome.reason,
                    user_id=user_id,
                    code_task_id=code_task_id,
                    error_reason=outcome.error_reason,
                    next_step=outcome.next_step,
                )
                return SkillResult(
                    success=False,
                    response=outcome.response,
                    metadata=_failure_policy_metadata(outcome),
                )
            if attempt == max_attempts:
                final_repair_notes = repair_notes
                if outcome.repair_note:
                    final_repair_notes = [*repair_notes, outcome.repair_note]
                reason = _repair_exhausted_reason(
                    slug=spec.slug,
                    gate=outcome.gate,
                    reason=outcome.reason,
                    repair_notes=final_repair_notes,
                )
                next_step = (
                    outcome.next_step
                    if outcome.needs_user_decision
                    else "Сохранила blocker и failed worktree для проверки."
                )
                await self._fail_cycle(
                    spec=spec,
                    spec_path=spec_path,
                    raw=raw,
                    branch_name=workspace.label,
                    reason=reason,
                    user_id=user_id,
                    code_task_id=code_task_id,
                    error_reason=outcome.error_reason,
                    next_step=next_step,
                )
                return SkillResult(
                    success=False,
                    response=reason,
                    metadata={
                        "needs_user_decision": (
                            "true" if outcome.needs_user_decision else "false"
                        ),
                        "auto_retryable": "false",
                        "failure_gate": outcome.gate,
                        "failure_category": (
                            "needs_user_decision"
                            if outcome.needs_user_decision
                            else "technical_blocker"
                        ),
                        **(
                            {
                                "decision_question": _decision_question_for_gate(
                                    outcome.gate, reason
                                )
                            }
                            if outcome.needs_user_decision
                            else {}
                        ),
                        **_editor_session_metadata(outcome),
                    },
                )
            repair_notes.append(outcome.repair_note)

        reason = f"Repair loop ended without a commit for `{spec.slug}`."
        await self._fail_cycle(
            spec=spec,
            spec_path=spec_path,
            raw=raw,
            branch_name=workspace.label,
            reason=reason,
            user_id=user_id,
            code_task_id=code_task_id,
            error_reason=reason,
            next_step="Сохранила blocker и failed worktree для проверки.",
        )
        return SkillResult(
            success=False,
            response=reason,
            metadata={
                "needs_user_decision": "false",
                "auto_retryable": "false",
                "failure_category": "technical_blocker",
            },
        )

    async def _prepare_repair_attempt(
        self,
        *,
        spec: SpecModel,
        spec_path: Path,
        raw: dict[str, Any],
        workspace: IsolatedWorkspace,
        repair_notes: list[str],
        attempt: int,
        max_attempts: int,
        cycle_started_at: float,
        user_id: int,
        code_task_id: str,
    ) -> CommitRunner | SkillResult:
        await self._publish_block(
            user_id=user_id,
            slug=spec.slug,
            code_task_id=code_task_id,
            event_type=BlockEventType.IMPLEMENTATION,
            percent=min(80, 40 + attempt * 10),
            detail=(
                "Предыдущая попытка не прошла gate; откатываю временную "
                "worktree к базе и запускаю repair-попытку "
                f"{attempt}/{max_attempts}."
            ),
            stage="repair loop",
            elapsed_seconds=int(time.monotonic() - cycle_started_at),
            facts=[
                f"attempt: {attempt}/{max_attempts}",
                f"last failure: {_first_line(repair_notes[-1])}",
            ],
        )
        try:
            _reset_workspace_to_base(workspace)
        except CommitRunnerError as exc:
            reason = f"Repair reset failed: {exc}"
            await self._fail_cycle(
                spec=spec,
                spec_path=spec_path,
                raw=raw,
                branch_name=workspace.label,
                reason=reason,
                user_id=user_id,
                code_task_id=code_task_id,
                error_reason=f"Не смогла подготовить repair-попытку: {exc}",
                next_step="Проверь preserved worktree и основной git state.",
            )
            return SkillResult(
                success=False,
                response=reason,
                metadata={
                    "needs_user_decision": "true",
                    "failure_category": "needs_user_decision",
                    "decision_question": (
                        "Нужно решить, что делать с временной worktree/git state: "
                        "повторять с чистой копией или разбирать preserved worktree?"
                    ),
                },
            )
        return self._commit_runner_factory(workspace.path)

    async def _run_editor_attempt(
        self,
        *,
        spec: SpecModel,
        spec_path: Path,
        workspace: IsolatedWorkspace,
        worktree_spec_path: Path,
        worktree_commits: CommitRunner,
        adversarial_context: str,
        repair_context: str,
        runtime_chat_context: tuple[str, ...],
        editor_session_id: str,
        baseline_test_count: int,
        attempt: int,
        max_attempts: int,
        cycle_started_at: float,
        user_id: int,
        code_task_id: str,
    ) -> _EditorAttemptOutcome:
        sdk_result = await self._invoke_sdk(
            spec=spec,
            workspace_path=workspace.path,
            workspace_label=workspace.label,
            adversarial_context=adversarial_context,
            repair_context=repair_context,
            runtime_chat_context=runtime_chat_context,
            editor_session_id=editor_session_id,
            user_id=user_id,
            slug=spec.slug,
            code_task_id=code_task_id,
        )
        if isinstance(sdk_result, SkillResult):
            gate = str(sdk_result.metadata.get("failure_gate") or "SDK")
            decision_question = str(
                sdk_result.metadata.get("decision_question") or ""
            ).strip() or _decision_question_for_gate(gate, sdk_result.response)
            needs_user_decision = _metadata_bool(
                sdk_result.metadata.get("needs_user_decision"),
                default=False,
            ) or _failure_requires_user_decision(gate, sdk_result.response)
            return _EditorAttemptOutcome(
                success=False,
                retryable=False,
                gate=gate,
                reason=sdk_result.response,
                response=sdk_result.response,
                error_reason="Code-agent backend не справился с задачей.",
                next_step=(
                    "Нужно обсудить blocker и только потом запускать новый проход."
                    if needs_user_decision
                    else "Сохранила blocker и failed worktree для проверки."
                ),
                needs_user_decision=needs_user_decision,
                decision_question=decision_question if needs_user_decision else "",
            )
        sdk_outcome = sdk_result

        env_result = self._env_guard.enforce(workspace.path)
        if env_result.triggered:
            return _EditorAttemptOutcome(
                success=False,
                retryable=False,
                gate="Env guard",
                reason=env_result.message,
                response=env_result.message,
                error_reason=env_result.message,
                next_step=(
                    "Убери protected `.env` ключ из задачи или вынеси это "
                    "в отдельное host-ops решение."
                ),
                needs_user_decision=True,
                decision_question=(
                    "Нужно решить: убрать изменение protected `.env` из задачи "
                    "или оформить отдельное host-ops решение с явным доступом?"
                ),
                sdk_outcome=sdk_outcome,
            )

        await self._publish_commit_gate_started(
            user_id=user_id,
            slug=spec.slug,
            code_task_id=code_task_id,
            sdk_outcome=sdk_outcome,
            attempt=attempt,
            max_attempts=max_attempts,
            cycle_started_at=cycle_started_at,
        )

        try:
            commit = worktree_commits.commit(
                spec_slug=spec.slug,
                spec_title=spec.title,
                spec_tier=spec.tier,
                spec_path=worktree_spec_path,
                whitelist_paths=list(spec.whitelist_paths),
                existing_tests_to_update_paths=[
                    entry.path for entry in spec.existing_tests_to_update
                ],
                allowed_simplifications=list(spec.allowed_simplifications),
            )
        except CommitRunnerError as exc:
            return _EditorAttemptOutcome(
                success=False,
                retryable=True,
                gate="Commit gate",
                reason=str(exc),
                response=str(exc),
                error_reason=f"Проверки на коммите не прошли: {exc}",
                next_step="Запускаю repair-проход по сохранённому контексту.",
                sdk_outcome=sdk_outcome,
                repair_note=_build_repair_note(
                    attempt=attempt,
                    gate="Commit gate",
                    reason=str(exc),
                    diff=_workspace_failure_context(
                        workspace.path, base_sha=workspace.base_sha
                    ),
                ),
            )

        changed_files = _changed_files_for_commit(workspace.path, commit.sha)
        current_test_count = _count_test_functions(workspace.path)
        await self._publish_review_gate_started(
            user_id=user_id,
            slug=spec.slug,
            code_task_id=code_task_id,
            commit=commit,
            changed_files=changed_files,
            baseline_test_count=baseline_test_count,
            current_test_count=current_test_count,
            attempt=attempt,
            max_attempts=max_attempts,
            cycle_started_at=cycle_started_at,
        )
        audit_log = _attempt_audit_log(
            spec=spec,
            sdk_outcome=sdk_outcome,
            commit=commit,
            changed_files=changed_files,
            baseline_test_count=baseline_test_count,
            current_test_count=current_test_count,
            adversarial_context=adversarial_context,
            attempt=attempt,
            max_attempts=max_attempts,
        )
        formal_result = self._run_optional_formal_gate(
            spec=spec,
            changed_files=changed_files,
            baseline_test_count=baseline_test_count,
            current_test_count=current_test_count,
            audit_log=audit_log,
        )
        if formal_result is not None and not formal_result.passed:
            return _gate_failure_outcome(
                attempt=attempt,
                gate="Formal gate",
                reason="; ".join(formal_result.issues),
                diff=_commit_diff(workspace.path, commit.sha),
                error_reason=(
                    f"Formal gate blocked `{spec.slug}`: "
                    + "; ".join(formal_result.issues)
                ),
                sdk_outcome=sdk_outcome,
            )

        review_result = await self._run_optional_reviewer(
            spec=spec,
            spec_path=spec_path,
            commit_sha=commit.sha,
            sdk_summary=sdk_outcome.text,
            diff=_commit_diff(workspace.path, commit.sha),
            test_output=_review_test_output(
                sdk_summary=sdk_outcome.text,
                baseline_test_count=baseline_test_count,
                current_test_count=current_test_count,
                formal_result=formal_result,
            ),
            audit_log=audit_log,
        )
        if review_result is not None and review_result.verdict != "approve":
            gate = f"Reviewer verdict `{review_result.verdict}`"
            return _gate_failure_outcome(
                attempt=attempt,
                gate=gate,
                reason=review_result.rationale,
                diff=_commit_diff(workspace.path, commit.sha),
                error_reason=f"{gate} for `{spec.slug}`: {review_result.rationale}",
                sdk_outcome=sdk_outcome,
            )

        return _EditorAttemptOutcome(
            success=True,
            retryable=False,
            gate="done",
            reason="",
            response="",
            error_reason="",
            next_step="",
            commit=commit,
            sdk_outcome=sdk_outcome,
            formal_result=formal_result,
            current_test_count=current_test_count,
            changed_files=tuple(changed_files),
            attempt=attempt,
            max_attempts=max_attempts,
        )

    async def _publish_commit_gate_started(
        self,
        *,
        user_id: int,
        slug: str,
        code_task_id: str,
        sdk_outcome: EditorSdkResult,
        attempt: int,
        max_attempts: int,
        cycle_started_at: float,
    ) -> None:
        await self._publish_block(
            user_id=user_id,
            slug=slug,
            code_task_id=code_task_id,
            event_type=BlockEventType.IMPLEMENTATION,
            percent=70,
            detail=(
                "Codex Editor завершил правки; запускаю commit gate, whitelist "
                "и проверки во временной рабочей копии."
            ),
            stage="проверки worktree",
            elapsed_seconds=int(time.monotonic() - cycle_started_at),
            facts=[
                f"backend: {sdk_outcome.backend}",
                f"attempt: {attempt}/{max_attempts}",
            ],
        )

    async def _publish_review_gate_started(
        self,
        *,
        user_id: int,
        slug: str,
        code_task_id: str,
        commit: CommitResult,
        changed_files: list[str],
        baseline_test_count: int,
        current_test_count: int,
        attempt: int,
        max_attempts: int,
        cycle_started_at: float,
    ) -> None:
        await self._publish_block(
            user_id=user_id,
            slug=slug,
            code_task_id=code_task_id,
            event_type=BlockEventType.IMPLEMENTATION,
            percent=85,
            detail=(
                "Commit gate прошёл; проверяю формальные гарантии и reviewer, "
                "если они подключены."
            ),
            stage="review gate",
            elapsed_seconds=int(time.monotonic() - cycle_started_at),
            facts=[
                f"commit: {commit.sha[:12]}",
                f"changed surfaces: {len(changed_files)}",
                f"tests delta: {current_test_count - baseline_test_count:+d}",
                f"attempt: {attempt}/{max_attempts}",
            ],
        )

    def _run_optional_formal_gate(
        self,
        *,
        spec: SpecModel,
        changed_files: list[str],
        baseline_test_count: int,
        current_test_count: int,
        audit_log: dict[str, Any],
    ) -> FormalGateVerdict | None:
        if self._formal_gate is None:
            return None
        return self._formal_gate(
            FormalGateInputs(
                baseline_test_count=baseline_test_count,
                current_test_count=current_test_count,
                changed_files=changed_files,
                whitelist_paths=list(spec.whitelist_paths),
                existing_tests_to_update_paths=[
                    entry.path for entry in spec.existing_tests_to_update
                ],
                audit_log=audit_log,
            )
        )

    async def _run_optional_reviewer(
        self,
        *,
        spec: SpecModel,
        spec_path: Path,
        commit_sha: str,
        sdk_summary: str,
        diff: str,
        test_output: str,
        audit_log: dict[str, Any],
    ) -> ReviewVerdict | None:
        if self._reviewer is None:
            return None
        return await self._reviewer.review(
            ReviewRequest(
                slug=spec.slug,
                tier=spec.tier,
                spec_yaml=spec_path.read_text(encoding="utf-8"),
                diff=diff or f"Commit: {commit_sha}",
                test_output=test_output,
                audit_log=audit_log,
            )
        )

    async def _record_success(
        self,
        *,
        spec: SpecModel,
        spec_path: Path,
        branch_name: str,
        commit_sha: str,
        sdk_summary: str,
        backend: str,
    ) -> None:
        if self._cycle_analyzer is None:
            return
        try:
            await self._cycle_analyzer.record_success(
                spec=spec,
                spec_path=spec_path,
                branch_name=branch_name,
                commit_sha=commit_sha,
                sdk_summary=sdk_summary,
                backend=backend,
            )
        except Exception:
            logger.warning(
                "cycle_analyzer_success_failed", slug=spec.slug, exc_info=True
            )

    async def _record_failure(
        self,
        *,
        spec: SpecModel,
        spec_path: Path,
        branch_name: str | None,
        reason: str,
    ) -> None:
        if self._cycle_analyzer is None:
            return
        try:
            await self._cycle_analyzer.record_failure(
                spec=spec,
                spec_path=spec_path,
                branch_name=branch_name,
                reason=reason,
                backend="codex_cli",
            )
        except Exception:
            logger.warning(
                "cycle_analyzer_failure_failed", slug=spec.slug, exc_info=True
            )

    # -------------------------------------------------------- block events

    async def _publish_block(
        self,
        *,
        user_id: int,
        slug: str,
        code_task_id: str,
        event_type: BlockEventType,
        percent: int,
        detail: str,
        facts: list[str] | None = None,
        stage: str = "",
        elapsed_seconds: int | None = None,
        message_kind: Literal["status", "note"] = "status",
    ) -> None:
        """Publish a progress block with data from the current Editor cycle."""
        payload: dict[str, Any] = {
            "percent": percent,
            "detail": detail,
            "message_kind": message_kind,
        }
        if facts:
            payload["facts"] = facts
        if stage:
            payload["stage"] = stage
        if elapsed_seconds is not None:
            payload["elapsed_seconds"] = elapsed_seconds
        await self._block_publisher.publish(
            BlockEvent(
                user_id=user_id,
                event_type=event_type,
                slug=slug,
                payload=payload,
                task_id=code_task_id,
            )
        )

    async def _publish_done(
        self,
        *,
        user_id: int,
        slug: str,
        code_task_id: str,
        sdk_summary: str,
        files: list[str],
        branch: str,
        commit_sha: str,
        backend: str,
        test_count_delta: int,
        allowed_simplifications: list[str],
        env_activation: LiveEnvActivationResult | None = None,
    ) -> None:
        checks: list[list[object]] = [
            ["тесты", True],
            ["стиль", True],
            ["типы", True],
        ]
        if env_activation is not None:
            checks.append(["live .env", env_activation.applied])
        await self._block_publisher.publish(
            BlockEvent(
                user_id=user_id,
                event_type=BlockEventType.DONE,
                slug=slug,
                task_id=code_task_id,
                payload={
                    "description": sdk_summary,
                    "files": files,
                    "branch": branch,
                    "commit_sha": commit_sha,
                    "backend": backend,
                    "test_count_delta": test_count_delta,
                    "allowed_simplifications": allowed_simplifications,
                    "env_audit_path": str(env_activation.audit_path)
                    if env_activation is not None and env_activation.audit_path
                    else "",
                    # Commit gate already verified tests / style / types —
                    # CommitRunner re-raised on any failure, so reaching
                    # here means all green.
                    "checks": checks,
                },
            )
        )

    async def _publish_error(
        self,
        *,
        user_id: int,
        slug: str,
        code_task_id: str,
        reason: str,
        next_step: str,
    ) -> None:
        await self._block_publisher.publish(
            BlockEvent(
                user_id=user_id,
                event_type=BlockEventType.ERROR,
                slug=slug,
                task_id=code_task_id,
                payload={"reason": reason, "next_step": next_step},
            )
        )

    async def _fail_cycle(
        self,
        *,
        spec: SpecModel,
        spec_path: Path,
        raw: dict[str, Any],
        branch_name: str,
        reason: str,
        user_id: int,
        code_task_id: str,
        error_reason: str,
        next_step: str,
    ) -> None:
        self._mark_failed(spec_path, raw, branch_name=branch_name, reason=reason)
        with contextlib.suppress(CommitRunnerError):
            self._commits.commit_yaml_update(
                spec_slug=spec.slug,
                spec_path=spec_path,
                subject=f"mark {spec.slug} failed",
            )
        await self._record_failure(
            spec=spec,
            spec_path=spec_path,
            branch_name=branch_name,
            reason=reason,
        )
        await self._publish_error(
            user_id=user_id,
            slug=spec.slug,
            code_task_id=code_task_id,
            reason=error_reason,
            next_step=next_step,
        )

    def _preserve_failed_workspace_result(
        self,
        *,
        result: SkillResult,
        spec: SpecModel,
        spec_path: Path,
        raw: dict[str, Any],
        workspace: IsolatedWorkspace,
    ) -> SkillResult:
        try:
            artifact = self._workspaces.preserve_failure(
                workspace,
                reason=result.response,
            )
        except Exception as exc:
            logger.warning(
                "implement_spec_failed_workspace_preserve_failed",
                slug=spec.slug,
                path=str(workspace.path),
                exc_info=True,
            )
            return replace(
                result,
                response=(
                    result.response.rstrip()
                    + "\n\nНе смогла сохранить failed worktree для проверки: "
                    + str(exc)
                ),
            )

        note = _failure_artifact_note(artifact)
        self._append_failed_attempt_note(spec_path, raw, note)
        with contextlib.suppress(CommitRunnerError):
            self._commits.commit_yaml_update(
                spec_slug=spec.slug,
                spec_path=spec_path,
                subject=f"record {spec.slug} failed worktree",
            )

        metadata = dict(result.metadata)
        metadata.update(
            {
                "failed_worktree_path": str(artifact.path),
                "failed_worktree_label": workspace.label,
                "failed_worktree_base_branch": workspace.base_branch,
                "failed_worktree_base_sha": workspace.base_sha,
                "failed_worktree_status_path": str(artifact.status_path),
                "failed_worktree_diff_path": str(artifact.diff_path),
            }
        )
        response = (
            result.response.rstrip()
            + "\n\nСохранила failed worktree для проверки:"
            + f"\n- path: `{artifact.path}`"
            + f"\n- status: `{artifact.status_path}`"
            + f"\n- diff: `{artifact.diff_path}`"
        )
        return replace(result, response=response, metadata=metadata)

    def _append_failed_attempt_note(
        self,
        spec_path: Path,
        raw: dict[str, Any],
        note: str,
    ) -> None:
        try:
            failed = list(raw.get("failed_attempts") or [])
            if failed:
                failed[-1] = _compact_failure_reason(f"{failed[-1]}\n\n{note}")
            else:
                failed.append(_compact_failure_reason(note))
            raw["failed_attempts"] = failed
            SpecModel.model_validate(raw)
            save_spec_raw(spec_path, raw)
        except Exception:
            logger.exception("implement_spec_append_failed_artifact_error")

    async def _invoke_sdk(
        self,
        *,
        spec: SpecModel,
        workspace_path: Path,
        workspace_label: str,
        adversarial_context: str = "",
        repair_context: str = "",
        runtime_chat_context: tuple[str, ...] = (),
        editor_session_id: str = "",
        user_id: int,
        slug: str,
        code_task_id: str,
    ) -> EditorSdkResult | SkillResult:
        del workspace_label
        user_prompt = self._build_user_prompt(
            spec=spec,
            adversarial_context=adversarial_context,
            repair_context=repair_context,
            runtime_chat_context=runtime_chat_context,
        )
        system_prompt = self._build_system_prompt(
            kind=spec.kind,
            existing_tests_to_update=list(spec.existing_tests_to_update),
            protected_env_instruction=self._env_guard.instruction_text,
        )
        progress_callback = self._make_editor_progress_callback(
            user_id=user_id,
            slug=slug,
            code_task_id=code_task_id,
        )
        try:
            return await self._sdk(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                cwd=workspace_path,
                project_root=workspace_path,
                whitelist_paths=list(spec.whitelist_paths),
                existing_tests_to_update_paths=[
                    entry.path for entry in spec.existing_tests_to_update
                ],
                progress_callback=progress_callback,
                session_id=editor_session_id,
                persist_session=True,
            )
        except SDKUnavailableError as exc:
            return SkillResult(
                success=False,
                response=(
                    f"Codex backend недоступен: {exc}. "
                    f"Spec `{spec.slug}` помечен failed."
                ),
                metadata={
                    "needs_user_decision": "true",
                    "auto_retryable": "false",
                    "failure_gate": "SDK unavailable",
                    "failure_category": "fatal",
                    "decision_question": (
                        "чинить доступность Codex backend/runtime перед новой "
                        "попыткой или остановить эту self-coding задачу?"
                    ),
                },
            )
        except Exception as exc:
            logger.exception("implement_spec_sdk_error", slug=spec.slug)
            needs_user_decision = _failure_requires_user_decision(
                "SDK backend", str(exc)
            )
            return SkillResult(
                success=False,
                response=f"Backend упал: {exc}. Spec `{spec.slug}` помечен failed.",
                metadata={
                    "needs_user_decision": "true" if needs_user_decision else "false",
                    "auto_retryable": "false",
                    "failure_gate": "SDK backend",
                    "failure_category": (
                        "needs_user_decision"
                        if needs_user_decision
                        else "technical_blocker"
                    ),
                    **(
                        {
                            "decision_question": _decision_question_for_gate(
                                "SDK backend", str(exc)
                            )
                        }
                        if needs_user_decision
                        else {}
                    ),
                },
            )

    def _make_editor_progress_callback(
        self, *, user_id: int, slug: str, code_task_id: str
    ) -> Callable[[str], Awaitable[None]]:
        last_sent_at = 0.0
        sent: set[str] = set()
        counter = 0

        async def _callback(detail: str) -> None:
            nonlocal counter, last_sent_at
            normalized = " ".join(detail.strip().split())
            if not normalized or normalized in sent:
                return
            now = time.monotonic()
            if now - last_sent_at < 8.0:
                return
            sent.add(normalized)
            last_sent_at = now
            counter += 1
            await self._publish_block(
                user_id=user_id,
                slug=slug,
                code_task_id=code_task_id,
                event_type=BlockEventType.IMPLEMENTATION,
                percent=min(65, 45 + counter * 4),
                detail=normalized[:240],
                stage="agent работает",
                message_kind="note",
            )

        return _callback

    async def _build_adversarial_context(self, spec: SpecModel) -> str:
        if spec.tier != 2 or self._adversarial_provider is None:
            return ""
        try:
            drafts = await self._adversarial_provider.generate(
                _adversarial_query(spec),
                limit=5,
            )
        except Exception:
            logger.warning("adversarial_context_failed", slug=spec.slug, exc_info=True)
            return ""
        if not drafts:
            return ""
        lines = [
            "# Tier 2 adversarial archive anchors",
            "Convert relevant drafts into real regression tests when they fit "
            "the spec whitelist. Every draft is anchored to a concrete archive "
            "node; keep the anchor if you use it.",
        ]
        for draft in drafts:
            archive_slug = str(getattr(draft, "archive_slug", "unknown"))
            test_file = str(getattr(draft, "test_file", ""))
            test_name = str(getattr(draft, "test_name", ""))
            rationale = str(getattr(draft, "rationale", ""))
            body = str(getattr(draft, "body", ""))
            lines.append(
                f"- `{test_file}::{test_name}` ← `{archive_slug}`\n"
                f"  rationale: {rationale}\n"
                f"  draft:\n{body}"
            )
        return "\n".join(lines)

    # =================================================================
    # helpers
    # =================================================================

    @staticmethod
    def _extract_slug(message: str) -> str:
        text = message.strip()
        for trigger in _TRIGGERS:
            if text.startswith(trigger):
                rest = text[len(trigger) :].strip()
                return rest.split(maxsplit=1)[0] if rest else ""
        natural = _extract_natural_spec_slug(text)
        if natural is not None:
            return natural
        return ""

    @staticmethod
    def _build_user_prompt(
        *,
        spec: SpecModel,
        adversarial_context: str = "",
        repair_context: str = "",
        runtime_chat_context: tuple[str, ...] = (),
    ) -> str:
        whitelist_block = "\n".join(f"  • {p}" for p in spec.whitelist_paths)
        readonly_block = (
            "\n".join(f"  • {p}" for p in spec.read_only_paths) or "  (нет)"
        )
        blast_block = "\n".join(f"  • {b}" for b in spec.blast_radius)
        rollback_block = "\n".join(f"  • {r}" for r in spec.rollback_path)
        preserve_block = (
            "\n".join(f"  • {p}" for p in spec.preserve_behavior)
            or "  • Preserve all existing behaviours, fallbacks, tests, "
            "personality/context nuance and safety gates not explicitly "
            "changed by this spec."
        )
        simplification_block = (
            "\n".join(f"  • {s}" for s in spec.allowed_simplifications)
            or "  (нет — удаления и упрощения не разрешены)"
        )

        sections: list[str] = [
            (
                f"# Spec\nSlug: {spec.slug}\nTitle: {spec.title}\n"
                f"Tier: {spec.tier}\nKind: {spec.kind.value}"
            ),
            f"# Goal\n{spec.goal}",
            (
                f"# Failing test (objective function)\n"
                f"File: {spec.failing_test.file}\n"
                f"Name: {spec.failing_test.name}\n"
                f"Spec: {spec.failing_test.spec}"
            ),
            f"# Whitelist (touchable files only)\n{whitelist_block}",
            f"# Read-only references\n{readonly_block}",
            f"# Preserve behaviour / no-downgrade contract\n{preserve_block}",
            f"# Allowed simplifications\n{simplification_block}",
            f"# Blast radius\n{blast_block}",
            f"# Rollback\n{rollback_block}",
        ]

        if spec.chat_context:
            chat_context_block = "\n".join(f"  • {line}" for line in spec.chat_context)
            sections.append(
                "# /код dialogue context\n"
                "These lines are the short discussion that led to this spec. "
                "Preserve the user's intent and any clarification already "
                "settled here.\n"
                f"{chat_context_block}"
            )

        if spec.existing_tests_to_update:
            update_lines: list[str] = []
            for entry in spec.existing_tests_to_update:
                update_lines.append(f"  • {entry.path}::{entry.test_name}")
                update_lines.append(f"    reason: {entry.reason}")
                update_lines.append(f"    allowed_changes: {entry.allowed_changes}")
            sections.append(
                "# Legitimate existing-test mutations\n"
                "These existing tests MAY be edited within the listed "
                "`allowed_changes` envelope, and only those tests. Any "
                "other existing test remains immutable.\n" + "\n".join(update_lines)
            )

        if spec.research_findings:
            research_lines: list[str] = []
            for finding in spec.research_findings:
                research_lines.append(f"  • [{finding.source}] {finding.excerpt}")
                research_lines.append(f"    relevance: {finding.relevance}")
            sections.append(
                "# Research findings (from Architect — read these before "
                "writing code)\n" + "\n".join(research_lines)
            )

        if spec.failed_attempts:
            attempts_block = "\n".join(f"  • {a}" for a in spec.failed_attempts)
            sections.append(
                "# Previous failed attempts (don't repeat the same path)\n"
                + attempts_block
            )

        if spec.previous_attempts:
            archive_lines: list[str] = []
            for attempt in spec.previous_attempts:
                commit = attempt.commit_sha[:12] if attempt.commit_sha else "no commit"
                archive_lines.append(
                    f"  • `{attempt.archive_slug}` · {attempt.status} · "
                    f"tier {attempt.tier} · {commit}"
                )
                archive_lines.append(f"    insight: {attempt.insight}")
                archive_lines.append(
                    f"    tests: {attempt.tests_summary or 'not recorded'}"
                )
            sections.append(
                "# Archive previous attempts (binding cycle memory)\n"
                "These were retrieved before the spec was written. Reuse "
                "successful patterns and avoid failed traps; do not flatten "
                "this context out of the implementation.\n" + "\n".join(archive_lines)
            )

        if adversarial_context:
            sections.append(adversarial_context)

        sections.extend(
            _optional_prompt_sections(
                repair_context=repair_context,
                runtime_chat_context=runtime_chat_context,
            )
        )

        sections.append(
            "# Steps\n"
            "1. RED: создай failing test, запусти — должен падать.\n"
            "2. GREEN: реализуй код, запусти — должен пройти.\n"
            "3. Прогнать ruff, mypy --strict, lint-imports, "
            "contract+chain pytest.\n"
            "4. Доложи `done` и кратко что сделал."
        )

        return "\n\n".join(sections)

    @staticmethod
    def _build_system_prompt(
        *,
        kind: SpecKind = SpecKind.FEATURE,
        existing_tests_to_update: list[ExistingTestUpdate] | None = None,
        protected_env_instruction: str | None = None,
    ) -> str:
        """System prompt for the Editor backend call.

        Branches the test-mutation rule by ``kind``:

        * ``REFACTOR`` — structural updates to existing tests are allowed
          (renamed imports, renamed symbol references, updated call
          signatures) because the spec's whole point *is* to rename
          things; the rule only forbids changing assertion logic /
          expected values.
        * Everything else (FEATURE / FIX / DOCS) — existing tests are
          immutable by default. Adding new tests inside whitelist is
          fine; rewriting an existing test's assertions to flip
          red→green is a workflow violation.

        ``existing_tests_to_update`` is the legitimate-mutation channel
        (Phase 16). When non-empty, the prompt enumerates each entry
        with its ``allowed_changes`` envelope; the Editor may edit
        exactly those tests within their declared envelope, and any
        existing test outside the list remains immutable per the
        kind-rule.

        Default ``kind=FEATURE`` and empty list preserve pre-Phase-16
        callers verbatim.
        """
        listed = list(existing_tests_to_update or [])

        if kind is SpecKind.REFACTOR:
            tdd_rule = (
                "- Failing tests are the spec. Because this is a REFACTOR, "
                "you MAY update existing tests *structurally* — renamed "
                "imports, renamed symbol references, updated call "
                "signatures — to follow the rename described in the spec. "
                "You must NOT change assertion logic or expected values to "
                "flip red→green; the whole point of an assertion is to be "
                "preserved across the refactor.\n"
            )
        else:
            tdd_rule = (
                "- Failing tests are the spec — never modify existing "
                "tests to make them pass. Fix the code, not the test. "
                "Adding new test files inside the whitelist is fine; "
                "rewriting an existing test's assertions to flip red→green "
                "is a workflow violation.\n"
            )

        if listed:
            update_block = (
                "- EXCEPTION: `spec.existing_tests_to_update` declares "
                "specific existing tests that you ARE permitted to edit, "
                "each within its `allowed_changes` envelope. Any existing "
                "test not in this list remains immutable per the rule "
                "above. Listed tests below — edit only what each entry's "
                "`allowed_changes` permits:\n"
            )
            for entry in listed:
                update_block += (
                    f"  • `{entry.path}::{entry.test_name}` — "
                    f"reason: {entry.reason} | "
                    f"allowed_changes: {entry.allowed_changes}\n"
                )
        else:
            update_block = ""
        env_block = protected_env_instruction or format_protected_env_prompt()

        return (
            "You are Жвуша's Editor. Your objective is to implement the full "
            "approved spec, satisfy its failing test, preserve every listed "
            "behavior, and leave no obvious reviewer blocker in one Editor "
            "session. Edit only files in `whitelist_paths`. Hard rules:\n"
            "- Edit/Write/MultiEdit/NotebookEdit only inside whitelist_paths "
            "(PreToolUse hook will deny otherwise).\n"
            "- Bash only for pytest/ruff/mypy/lint-imports/git-readonly/"
            'python -c "..." smoke imports. No webfetch, no curl, no rm.\n'
            "- Per AGENTS.md rule 0: tests must use Haiku, never Sonnet/Opus.\n"
            "- No-downgrade principle: Жвуша grows by enriching capability, "
            "context, nuance, memory, checks and controls. Do not flatten "
            "personality, remove fallbacks, weaken prompt/context rules, drop "
            "tests, or narrow existing user flows to make the local task "
            "easier.\n"
            "- `preserve_behavior` is binding. Preserve every listed behaviour "
            "and every unlisted existing behaviour unless the spec explicitly "
            "changes it.\n"
            "- `allowed_simplifications` is the only permission to delete or "
            "simplify existing behaviour. If it is empty, do not remove "
            "behaviour/rules/fallbacks/context; append, adapt, or layer the "
            "new behaviour instead. If implementation requires an unlisted "
            "simplification, STOP and report the blocker.\n"
            "- Telegram progress: during longer work, emit occasional short "
            "human progress notes prefixed exactly `TG_STATUS:`. These notes "
            "are for Никита in Telegram; do not mention raw tool calls, shell "
            "commands, command output, file lists, or internal trace bullets. "
            "Good style: `TG_STATUS: Сейчас дочитаю затронутые файлы целиком "
            "и добавлю RED-тесты.`\n"
            + env_block
            + "\n"
            + tdd_rule
            + update_block
            + "- One commit at the end is handled by the framework (commit_runner) "
            "— don't run `git commit` yourself.\n"
            "- If the test cannot be made green within the whitelist, STOP and "
            "explain what's blocking. Do not edit outside the whitelist."
        )

    def _mark_failed(
        self,
        spec_path: Path,
        raw: dict[str, Any],
        *,
        branch_name: str,
        reason: str,
    ) -> None:
        try:
            raw["status"] = SpecStatus.FAILED.value
            raw["branch"] = branch_name
            failed = list(raw.get("failed_attempts") or [])
            failed.append(_compact_failure_reason(reason))
            raw["failed_attempts"] = failed
            raw["iterations"] = int(raw.get("iterations", 0)) + 1
            SpecModel.model_validate(raw)
            save_spec_raw(spec_path, raw)
        except Exception:
            logger.exception("implement_spec_mark_failed_error", path=str(spec_path))


def _count_test_functions(project_root: Path) -> int:
    """Count Python test functions under tests/ for monotonic gate input."""
    tests_dir = project_root / "tests"
    if not tests_dir.exists():
        return 0
    count = 0
    for path in tests_dir.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("def test_") or stripped.startswith(
                "async def test_"
            ):
                count += 1
    return count


def _changed_files_for_commit(project_root: Path, commit_sha: str) -> list[str]:
    out = _git_output(
        project_root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit_sha,
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def _commit_diff(project_root: Path, commit_sha: str) -> str:
    return _git_output(
        project_root,
        "show",
        "--format=",
        "--no-ext-diff",
        "--unified=80",
        commit_sha,
    ).strip()


def _dirty_git_paths(project_root: Path) -> list[str]:
    status = _git_output(project_root, "status", "--porcelain", "--untracked-files=all")
    paths: list[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        paths.append(line[3:].strip())
    return paths


_FAILED_ATTEMPT_MAX_CHARS = 2000
_FAILED_ATTEMPT_EDGE_CHARS = 900


def _compact_failure_reason(reason: str) -> str:
    cleaned = reason.strip()
    if len(cleaned) <= _FAILED_ATTEMPT_MAX_CHARS:
        return cleaned
    head = cleaned[:_FAILED_ATTEMPT_EDGE_CHARS].rstrip()
    tail = cleaned[-_FAILED_ATTEMPT_EDGE_CHARS:].lstrip()
    return f"{head}\n... [truncated] ...\n{tail}"


def _failure_artifact_note(artifact: PreservedFailureWorkspace) -> str:
    return (
        "failed worktree preserved for inspection:\n"
        f"path: {artifact.path}\n"
        f"status: {artifact.status_path}\n"
        f"diff: {artifact.diff_path}"
    )


def _optional_prompt_sections(
    *,
    repair_context: str,
    runtime_chat_context: tuple[str, ...],
) -> list[str]:
    sections: list[str] = []
    if runtime_chat_context:
        chat_lines = "\n".join(f"  • {line}" for line in runtime_chat_context)
        sections.append(
            "# /код recovery discussion before this run\n"
            "These are the newest chat-mode messages before the current Editor "
            "run. Treat them as binding clarification when they refine the "
            "approved spec or the previous failure.\n"
            f"{chat_lines}"
        )
    if repair_context:
        sections.append(repair_context)
    return sections


def _attempt_audit_log(
    *,
    spec: SpecModel,
    sdk_outcome: EditorSdkResult,
    commit: CommitResult,
    changed_files: list[str],
    baseline_test_count: int,
    current_test_count: int,
    adversarial_context: str,
    attempt: int,
    max_attempts: int,
) -> dict[str, Any]:
    return {
        "slug": spec.slug,
        "backend": sdk_outcome.backend,
        "commit_sha": commit.sha,
        "changed_files": changed_files,
        "baseline_test_count": baseline_test_count,
        "current_test_count": current_test_count,
        "adversarial_context_present": bool(adversarial_context),
        "repair_attempt": attempt,
        "repair_max_attempts": max_attempts,
    }


def _gate_failure_outcome(
    *,
    attempt: int,
    gate: str,
    reason: str,
    diff: str,
    error_reason: str,
    sdk_outcome: EditorSdkResult | None = None,
) -> _EditorAttemptOutcome:
    needs_user_decision = _failure_requires_user_decision(gate, reason)
    decision_question = _decision_question_for_gate(gate, reason)
    return _EditorAttemptOutcome(
        success=False,
        retryable=not needs_user_decision,
        gate=gate,
        reason=reason,
        response=reason,
        error_reason=error_reason,
        next_step=(
            "Нужно обсудить blocker и только потом запускать новый проход."
            if needs_user_decision
            else "Запускаю repair-проход по сохранённому контексту."
        ),
        needs_user_decision=needs_user_decision,
        decision_question=decision_question if needs_user_decision else "",
        repair_note=_build_repair_note(
            attempt=attempt,
            gate=gate,
            reason=reason,
            diff=diff,
        ),
        sdk_outcome=sdk_outcome,
    )


_USER_DECISION_GATES = frozenset(
    {
        "Env guard",
        "SDK unavailable",
    }
)

_USER_DECISION_MARKERS = (
    "нужно твое решение",
    "нужно твоё решение",
    "нужен выбор",
    "нужно выбрать",
    "выбери",
    "какой вариант",
    "needs user decision",
    "requires user decision",
    "user decision",
    "needs decision",
    "requires decision",
    "choose",
    "какую модель",
    "какой провайдер",
    "какой сервис",
    "как будет происходить",
    "как должна происходить",
    "непонятно что делать",
    "не ясно что делать",
    "неясно что делать",
    "нет доступа",
    "нужен доступ",
    "permission",
    "credentials",
    "account",
    "подписк",
    "subscription",
    "billing",
    "quota",
    "нельзя через api",
    "api не позволяет",
    "не позволяет сделать",
    "policy",
    "terms",
    "ban risk",
    "риск бана",
    "host-ops",
    "live .env",
    "external side effect",
    "purchase",
    "login",
)


def _failure_requires_user_decision(gate: str, reason: str) -> bool:
    if gate in _USER_DECISION_GATES:
        return True
    normalized = " ".join(reason.lower().split())
    return any(marker in normalized for marker in _USER_DECISION_MARKERS)


def _decision_question_for_gate(gate: str, reason: str) -> str:
    first_line = _first_line(reason).strip()
    if gate == "Env guard":
        return (
            "убираем изменение protected `.env` из задачи или оформляем "
            "отдельное host-ops решение с явным доступом?"
        )
    if gate == "Commit gate":
        return (
            "чинить commit/pre-commit blocker в текущем подходе или менять "
            "границы spec перед новой попыткой?"
        )
    if gate == "SDK backend":
        return (
            "чинить backend/runtime перед новой попыткой или менять задачу так, "
            "чтобы Codex не падал на этом шаге?"
        )
    if gate == "SDK unavailable":
        return (
            "чинить доступность Codex backend/runtime перед новой попыткой или "
            "остановить эту self-coding задачу?"
        )
    if "Reviewer" in gate or "Formal" in gate:
        if first_line:
            return (
                "какое архитектурное или safety-решение принять по blocker: "
                f"{first_line}?"
            )
        return (
            "какое архитектурное или safety-решение принять перед новой "
            "попыткой реализации?"
        )
    return "что менять в подходе перед следующей попыткой реализации?"


def _metadata_bool(value: object, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return default


def _failure_policy_metadata(outcome: _EditorAttemptOutcome) -> dict[str, str]:
    metadata = {
        "needs_user_decision": "true" if outcome.needs_user_decision else "false",
        "auto_retryable": "true" if outcome.retryable else "false",
        "failure_gate": outcome.gate,
        "failure_category": _failure_category_for_outcome(outcome),
        **_editor_session_metadata(outcome),
    }
    if outcome.decision_question:
        metadata["decision_question"] = outcome.decision_question
    return metadata


def _editor_session_metadata(outcome: _EditorAttemptOutcome) -> dict[str, str]:
    if outcome.sdk_outcome is None or not outcome.sdk_outcome.session_id:
        return {}
    return {"editor_codex_session_id": outcome.sdk_outcome.session_id}


def _failure_category_for_outcome(outcome: _EditorAttemptOutcome) -> str:
    if outcome.gate == "Env guard":
        return "needs_host_ops"
    if outcome.retryable and not outcome.needs_user_decision:
        return "auto_repairable"
    if outcome.needs_user_decision:
        return "needs_user_decision"
    return "technical_blocker"


def _runtime_chat_context(context: AgentContext) -> tuple[str, ...]:
    raw = context.metadata.get("chat_self_coding_recent_messages", ())
    if isinstance(raw, tuple):
        return tuple(str(item).strip() for item in raw if str(item).strip())
    if isinstance(raw, list):
        return tuple(str(item).strip() for item in raw if str(item).strip())
    if isinstance(raw, str) and raw.strip():
        return (raw.strip(),)
    return ()


def _runtime_code_task_id(context: AgentContext) -> str:
    raw = context.metadata.get("chat_self_coding_code_task_id", "")
    return str(raw).strip() if raw else ""


def _editor_resume_context(context: AgentContext) -> _EditorResumeContext | None:
    session_id = _context_metadata_value(
        context, "chat_self_coding_editor_codex_session_id"
    )
    worktree_path = _context_metadata_value(
        context, "chat_self_coding_failed_worktree_path"
    )
    worktree_label = _context_metadata_value(
        context, "chat_self_coding_failed_worktree_label"
    )
    base_branch = _context_metadata_value(
        context, "chat_self_coding_failed_worktree_base_branch"
    )
    base_sha = _context_metadata_value(
        context, "chat_self_coding_failed_worktree_base_sha"
    )
    if not (
        session_id and worktree_path and worktree_label and base_branch and base_sha
    ):
        return None
    return _EditorResumeContext(
        session_id=session_id,
        worktree_path=Path(worktree_path),
        worktree_label=worktree_label,
        base_branch=base_branch,
        base_sha=base_sha,
    )


def _context_metadata_value(context: AgentContext, key: str) -> str:
    raw = context.metadata.get(key, "")
    return str(raw).strip() if raw else ""


def _reset_workspace_to_base(workspace: IsolatedWorkspace) -> None:
    """Reset a disposable isolated worktree before another Editor attempt."""
    _git_output(workspace.path, "reset", "--hard", workspace.base_sha)
    _git_output(workspace.path, "clean", "-fd")


def _format_repair_context(repair_notes: list[str]) -> str:
    if not repair_notes:
        return ""
    return (
        "# Repair context\n"
        "Previous Editor attempt(s) failed after code was written. Do not stop "
        "for discussion yet: repair the implementation inside the same spec, "
        "preserve the original intent, and specifically address every blocker "
        "below.\n\n" + "\n\n---\n\n".join(repair_notes)
    )


def _build_repair_note(
    *,
    attempt: int,
    gate: str,
    reason: str,
    diff: str,
) -> str:
    parts = [
        f"Attempt {attempt} failed at {gate}.",
        "Reason:\n" + (reason.strip() or "(no reason recorded)"),
    ]
    if diff.strip():
        parts.append("Rejected diff / workspace context:\n" + diff.strip())
    return _compact_repair_context("\n\n".join(parts))


def _repair_exhausted_reason(
    *,
    slug: str,
    gate: str,
    reason: str,
    repair_notes: list[str],
) -> str:
    message = (
        f"Repair attempts exhausted for `{slug}` at {gate}: "
        f"{reason.strip() or '(no reason recorded)'}"
    )
    if repair_notes:
        message += "\n\nRepair history:\n" + "\n\n---\n\n".join(repair_notes)
    return _compact_failure_reason(message)


def _workspace_failure_context(
    project_root: Path,
    *,
    base_sha: str,
    include_untracked_paths: list[str] | None = None,
) -> str:
    parts: list[str] = []
    status = _git_output(
        project_root,
        "status",
        "--short",
        "--untracked-files=all",
    ).strip()
    if status:
        parts.append("Git status:\n" + status)
    committed = _git_output(
        project_root,
        "diff",
        "--binary",
        base_sha,
        "HEAD",
    ).strip()
    if committed:
        parts.append("Committed diff since base:\n" + committed)
    staged = _git_output(
        project_root,
        "diff",
        "--cached",
        "--binary",
        "HEAD",
    ).strip()
    if staged:
        parts.append("Staged diff:\n" + staged)
    unstaged = _git_output(project_root, "diff", "--binary").strip()
    if unstaged:
        parts.append("Unstaged diff:\n" + unstaged)
    untracked = _untracked_whitelist_context(
        project_root,
        status=status,
        whitelist_paths=include_untracked_paths or [],
    )
    if untracked:
        parts.append("Untracked whitelist files:\n" + untracked)
    return "\n\n".join(parts)


def _untracked_whitelist_context(
    project_root: Path,
    *,
    status: str,
    whitelist_paths: list[str],
) -> str:
    if not status or not whitelist_paths:
        return ""
    allowed = set(whitelist_paths)
    snippets: list[str] = []
    root = project_root.resolve()
    for line in status.splitlines():
        if not line.startswith("?? "):
            continue
        rel_path = line[3:].strip()
        if rel_path not in allowed:
            continue
        path = (root / rel_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            snippets.append(f"# {rel_path}\n(binary or non-utf8 file)")
            continue
        snippets.append(f"# {rel_path}\n{text[:2000]}")
    return "\n\n".join(snippets)


def _compact_repair_context(text: str) -> str:
    cleaned = text.strip()
    if len(cleaned) <= _REPAIR_CONTEXT_MAX_CHARS:
        return cleaned
    head = cleaned[:_REPAIR_CONTEXT_EDGE_CHARS].rstrip()
    tail = cleaned[-_REPAIR_CONTEXT_EDGE_CHARS:].lstrip()
    return f"{head}\n... [repair context truncated] ...\n{tail}"


def _first_line(text: str) -> str:
    return (text.strip().splitlines() or [""])[0][:160]


def _git_output(project_root: Path, *args: str) -> str:
    try:
        return subprocess.run(  # noqa: S603 — fixed git subcommand wrapper
            ["git", *args],  # noqa: S607 — git on PATH
            cwd=project_root,
            check=True,
            capture_output=True,
            env=clean_env_for_git_subprocess(),
            text=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise CommitRunnerError(
            f"git {' '.join(args)} failed: {exc.stderr or exc.stdout}"
        ) from exc


def _review_test_output(
    *,
    sdk_summary: str,
    baseline_test_count: int,
    current_test_count: int,
    formal_result: FormalGateVerdict | None,
) -> str:
    gate_summary = "formal gate not configured"
    if formal_result is not None:
        gate_summary = (
            "formal gate passed"
            if formal_result.passed
            else "formal gate failed: " + "; ".join(formal_result.issues)
        )
    return (
        f"SDK summary:\n{sdk_summary.strip() or 'no sdk summary'}\n\n"
        f"Test count: {baseline_test_count} -> {current_test_count}\n"
        f"{gate_summary}\n"
        "Commit gate completed before this reviewer step."
    )


def _adversarial_query(spec: SpecModel) -> str:
    parts = [
        spec.slug,
        spec.title,
        spec.goal,
        *spec.preserve_behavior,
        *[attempt.insight for attempt in spec.previous_attempts],
    ]
    return "\n".join(part for part in parts if part)

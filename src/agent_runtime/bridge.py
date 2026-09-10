"""Compatibility adapters from existing call sites to Agent Runtime jobs."""

from __future__ import annotations

import asyncio
from hashlib import sha256
from typing import TYPE_CHECKING

import structlog

from src.agent_runtime.context import ContextPackBuilder
from src.agent_runtime.models import AgentJobStatus
from src.agent_runtime.rendering import (
    AgentResultRendererRegistry,
    build_builtin_result_renderer_registry,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.agent_runtime.models import (
        AgentJob,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.runtime import AgentRuntime

logger = structlog.get_logger()


class AgentRuntimeExplorerRunner:
    """Expose Agent Runtime through the legacy ExplorerRunner signature."""

    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        profile: InvocationProfile,
        owner_user_id: int,
        chat_id: int,
        kind: str,
        context_builder: ContextPackBuilder | None = None,
        result_renderer: AgentResultRendererRegistry | None = None,
    ) -> None:
        self._runtime = runtime
        self._profile = profile
        self._owner_user_id = owner_user_id
        self._chat_id = chat_id
        self._kind = kind
        self._context_builder = context_builder or ContextPackBuilder()
        self._result_renderer = (
            result_renderer or build_builtin_result_renderer_registry()
        )
        self._delivery_tasks: set[asyncio.Task[None]] = set()

    async def __call__(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        progress_callback: Callable[[str], Awaitable[None]] | None = None,
        session_id: str = "",
        persist_session: bool = False,
        session_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Create/start a durable read-only job and return its human report."""
        job = await self._create_job(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            session_id=session_id,
            persist_session=persist_session,
        )
        if progress_callback is not None:
            await progress_callback("Открыла read-only agent job и собираю контекст.")

        if job.status is AgentJobStatus.DONE and job.result is not None:
            await _notify_codex_session(job.result, session_callback)
            return self._result_renderer.render(job, job.result)

        completed = await self._runtime.start(job.id)
        if completed.result is None:
            raise RuntimeError(completed.error or "agent job did not return result")
        await _notify_codex_session(completed.result, session_callback)
        return self._result_renderer.render(completed, completed.result)

    async def start_background(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        completion_callback: Callable[[str], Awaitable[None]],
        progress_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> AgentJob:
        """Create/start a durable read-only job and deliver its result later."""
        job = await self._create_job(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        if progress_callback is not None:
            await progress_callback("Открыла read-only agent job в фоне.")

        if job.status is AgentJobStatus.DONE and job.result is not None:
            await completion_callback(self._result_renderer.render(job, job.result))
            return job

        running = await self._runtime.start_background(job.id)
        delivery_task = asyncio.create_task(
            self._deliver_background_result(
                job_id=job.id,
                completion_callback=completion_callback,
            )
        )
        self._delivery_tasks.add(delivery_task)
        delivery_task.add_done_callback(self._delivery_tasks.discard)
        return running

    async def create_awaiting_input(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> AgentJob:
        """Create a durable job that waits for the promised source/material."""
        return await self._create_job(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            status=AgentJobStatus.AWAITING_INPUT,
        )

    async def start_existing_background(
        self,
        *,
        job_id: str,
        completion_callback: Callable[[str], Awaitable[None]],
        progress_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> AgentJob:
        """Start an existing job, usually after awaited input arrived."""
        if progress_callback is not None:
            await progress_callback("Материал получен, запускаю agent job.")
        running = await self._runtime.start_background(job_id)
        delivery_task = asyncio.create_task(
            self._deliver_background_result(
                job_id=job_id,
                completion_callback=completion_callback,
            )
        )
        self._delivery_tasks.add(delivery_task)
        delivery_task.add_done_callback(self._delivery_tasks.discard)
        return running

    async def _create_job(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        status: AgentJobStatus = AgentJobStatus.QUEUED,
        session_id: str = "",
        persist_session: bool = False,
    ) -> AgentJob:
        active_state_lines: list[str] = []
        if session_id:
            active_state_lines.append(f"codex_session_id: {session_id}")
        if persist_session:
            active_state_lines.append("codex_persist_session: true")
        pack = self._context_builder.build(
            user_request=user_prompt,
            chat_context=(system_prompt,),
            active_code_state="\n".join(active_state_lines),
            constraints=("read-only",),
        )
        source_message_id = _synthetic_source_message_id(system_prompt, user_prompt)
        fingerprint = self._context_builder.fingerprint(
            owner_user_id=self._owner_user_id,
            chat_id=self._chat_id,
            source_message_id=source_message_id,
            kind=self._kind,
            context_pack=pack,
        )
        existing = await self._runtime.store.find_by_fingerprint(fingerprint)
        if existing is not None and _explorer_job_is_reusable(existing):
            return existing
        job = await self._runtime.create_job(
            owner_user_id=self._owner_user_id,
            chat_id=self._chat_id,
            source_message_id=source_message_id,
            fingerprint=fingerprint,
            kind=self._kind,
            profile=self._profile,
            context_pack=pack,
            status=status,
        )
        return job

    async def _deliver_background_result(
        self,
        *,
        job_id: str,
        completion_callback: Callable[[str], Awaitable[None]],
    ) -> None:
        try:
            completed = await self._runtime.wait_background(job_id)
            if completed is None:
                completed = await self._runtime.status(job_id)
            if completed.result is not None:
                text = self._result_renderer.render(completed, completed.result)
            else:
                reason = completed.error or "agent job did not return result"
                text = f"Read-only agent job завершилась без результата: {reason}"
            await completion_callback(text)
        except Exception:
            logger.warning(
                "agent_runtime_background_delivery_failed",
                job_id=job_id,
                exc_info=True,
            )


def _synthetic_source_message_id(system_prompt: str, user_prompt: str) -> str:
    digest = sha256(f"{system_prompt}\n{user_prompt}".encode()).hexdigest()
    return f"explorer:{digest[:24]}"


def _explorer_job_is_reusable(job: AgentJob) -> bool:
    if job.status is AgentJobStatus.DONE:
        return job.result is not None
    return job.status not in {
        AgentJobStatus.FAILED,
        AgentJobStatus.CANCELED,
        AgentJobStatus.NEEDS_REVIEW,
    }


async def _notify_codex_session(
    capsule: ContextPack | object,
    callback: Callable[[str], Awaitable[None]] | None,
) -> None:
    if callback is None:
        return
    artifacts = getattr(capsule, "artifacts", ())
    if not isinstance(artifacts, tuple):
        return
    for artifact in artifacts:
        if not isinstance(artifact, str):
            continue
        key, sep, value = artifact.partition(":")
        if sep and key.strip() == "codex_session_id" and value.strip():
            await callback(value.strip())
            return


class SelfCodingAgentRuntimeRunner:
    """Run `/код` implementation turns through Agent Runtime."""

    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        profile: InvocationProfile,
        owner_user_id: int,
        context_builder: ContextPackBuilder | None = None,
        result_renderer: AgentResultRendererRegistry | None = None,
    ) -> None:
        self._runtime = runtime
        self._profile = profile
        self._owner_user_id = owner_user_id
        self._context_builder = context_builder or ContextPackBuilder()
        self._result_renderer = (
            result_renderer or build_builtin_result_renderer_registry()
        )
        self._delivery_tasks: set[asyncio.Task[None]] = set()

    async def __call__(
        self,
        *,
        slug: str,
        context: AgentContextLike,
        recent_messages: tuple[str, ...] = (),
    ) -> SkillResultLike:
        job = await self._create_job(
            slug=slug,
            context=context,
            recent_messages=recent_messages,
        )
        if job.status is not AgentJobStatus.QUEUED:
            return self._self_coding_result_from_job(job)
        completed = await self._runtime.start(job.id)
        return self._self_coding_result_from_job(completed)

    async def start_background(
        self,
        *,
        slug: str,
        context: AgentContextLike,
        recent_messages: tuple[str, ...] = (),
        completion_callback: Callable[[SkillResultLike], Awaitable[None]],
    ) -> AgentJob:
        """Start `/код` implementation in the background and notify later."""
        job = await self._create_job(
            slug=slug,
            context=context,
            recent_messages=recent_messages,
        )
        if job.status is AgentJobStatus.DONE:
            await completion_callback(self._self_coding_result_from_job(job))
            return job
        if job.status in {
            AgentJobStatus.FAILED,
            AgentJobStatus.CANCELED,
            AgentJobStatus.NEEDS_REVIEW,
        }:
            await completion_callback(self._self_coding_result_from_job(job))
            return job
        running = await self._runtime.start_background(job.id)
        delivery_task = asyncio.create_task(
            self._deliver_background_result(
                job_id=job.id,
                completion_callback=completion_callback,
            )
        )
        self._delivery_tasks.add(delivery_task)
        delivery_task.add_done_callback(self._delivery_tasks.discard)
        return running

    async def _create_job(
        self,
        *,
        slug: str,
        context: AgentContextLike,
        recent_messages: tuple[str, ...],
    ) -> AgentJob:
        chat_id = int(context.chat_id or context.user_id)
        source_message_id = _self_coding_source_message_id(slug, context)
        code_task_id = _self_coding_code_task_id(context)
        pack = _self_coding_context_pack(
            context_builder=self._context_builder,
            slug=slug,
            recent_messages=recent_messages,
            code_task_id=code_task_id,
            editor_resume=_self_coding_editor_resume(context),
        )
        fingerprint = self._context_builder.fingerprint(
            owner_user_id=self._owner_user_id,
            chat_id=chat_id,
            source_message_id=source_message_id,
            kind="self_coding",
            context_pack=pack,
        )
        existing = await self._runtime.store.find_by_fingerprint(fingerprint)
        if existing is not None:
            return existing
        return await self._runtime.create_job(
            owner_user_id=self._owner_user_id,
            chat_id=chat_id,
            source_message_id=source_message_id,
            fingerprint=fingerprint,
            kind="self_coding",
            profile=self._profile,
            context_pack=pack,
        )

    async def _deliver_background_result(
        self,
        *,
        job_id: str,
        completion_callback: Callable[[SkillResultLike], Awaitable[None]],
    ) -> None:
        try:
            completed = await self._runtime.wait_background(job_id)
            if completed is None:
                completed = await self._runtime.status(job_id)
            await completion_callback(self._self_coding_result_from_job(completed))
        except Exception:
            logger.warning(
                "self_coding_background_delivery_failed",
                job_id=job_id,
                exc_info=True,
            )

    def _self_coding_result_from_job(self, job: AgentJob) -> SkillResultLike:
        metadata = {"agent_job_id": job.id}
        if job.result is not None:
            return _skill_result(
                success=True,
                response=self._result_renderer.render(job, job.result),
                metadata=metadata,
            )
        if job.status in {
            AgentJobStatus.RUNNING,
            AgentJobStatus.WAITING_USER,
            AgentJobStatus.QUEUED,
            AgentJobStatus.AWAITING_INPUT,
        }:
            return _skill_result(
                success=True,
                response="Self-coding agent job уже выполняется.",
                metadata=metadata,
            )
        metadata.update(_self_coding_failure_metadata(job.observability))
        return _skill_result(
            success=False,
            response=job.error or "self-coding job did not return result",
            metadata=metadata,
        )


if TYPE_CHECKING:
    from src.skills.base import AgentContext as AgentContextLike
    from src.skills.base import SkillResult as SkillResultLike
else:
    AgentContextLike = object
    SkillResultLike = object


def _self_coding_source_message_id(slug: str, context: AgentContextLike) -> str:
    message_id = getattr(context, "message_id", None)
    chat_id = getattr(context, "chat_id", None)
    metadata = getattr(context, "metadata", {}) or {}
    attempt = metadata.get("chat_self_coding_goal_attempt")
    attempt_suffix = f":attempt:{attempt}" if attempt is not None else ""
    task_id = _self_coding_code_task_id(context)
    task_suffix = f":task:{task_id}" if task_id else ""
    if message_id is not None and chat_id is not None:
        return (
            f"tg:{chat_id}:{message_id}:self-coding:{slug}{task_suffix}{attempt_suffix}"
        )
    digest_source = (
        f"{slug}:{getattr(context, 'user_id', '')}:{task_suffix}:{attempt_suffix}"
    )
    digest = sha256(digest_source.encode()).hexdigest()
    return f"self-coding:{slug}:{digest[:12]}"


def _self_coding_code_task_id(context: AgentContextLike) -> str:
    metadata = getattr(context, "metadata", {}) or {}
    task_id = metadata.get("chat_self_coding_code_task_id")
    return str(task_id).strip() if task_id else ""


def _self_coding_editor_resume(context: AgentContextLike) -> dict[str, str]:
    metadata = getattr(context, "metadata", {}) or {}
    mapping = {
        "editor_codex_session_id": "chat_self_coding_editor_codex_session_id",
        "failed_worktree_path": "chat_self_coding_failed_worktree_path",
        "failed_worktree_label": "chat_self_coding_failed_worktree_label",
        "failed_worktree_base_branch": "chat_self_coding_failed_worktree_base_branch",
        "failed_worktree_base_sha": "chat_self_coding_failed_worktree_base_sha",
    }
    resume: dict[str, str] = {}
    for target_key, source_key in mapping.items():
        value = metadata.get(source_key)
        clean = str(value).strip() if value else ""
        if clean:
            resume[target_key] = clean
    return resume


def _self_coding_failure_metadata(observability: dict[str, str]) -> dict[str, str]:
    allowed = {
        "needs_user_decision",
        "auto_retryable",
        "failure_gate",
        "decision_question",
        "failure_category",
        "editor_codex_session_id",
        "failed_worktree_path",
        "failed_worktree_label",
        "failed_worktree_base_branch",
        "failed_worktree_base_sha",
    }
    metadata = {
        key: value for key, value in observability.items() if key in allowed and value
    }
    if metadata.get("auto_retryable", "").lower() == "true":
        metadata["auto_retryable"] = "false"
        metadata.setdefault("needs_user_decision", "false")
        if metadata.get("failure_category") in {None, "", "auto_repairable"}:
            metadata["failure_category"] = "technical_blocker"
    return metadata


def _skill_result(
    *,
    success: bool,
    response: str,
    metadata: dict[str, str],
) -> SkillResultLike:
    from src.skills.base import SkillResult

    return SkillResult(success=success, response=response, metadata=metadata)


def _self_coding_context_pack(
    *,
    context_builder: ContextPackBuilder,
    slug: str,
    recent_messages: tuple[str, ...],
    code_task_id: str = "",
    editor_resume: dict[str, str] | None = None,
) -> ContextPack:
    active_state_lines = [f"active_spec_slug: {slug}"]
    if code_task_id:
        active_state_lines.append(f"code_task_id: {code_task_id}")
    for key, value in (editor_resume or {}).items():
        active_state_lines.append(f"{key}: {value}")
    return context_builder.build(
        user_request=f"/spec_run {slug}",
        chat_context=recent_messages,
        active_code_state="\n".join(active_state_lines),
        constraints=(
            "spec-first",
            "approval already granted via /spec approve",
            "preserve whitelist/env/test/no-downgrade gates",
        ),
    )

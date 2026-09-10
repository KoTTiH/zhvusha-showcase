"""Composite source-compare worker for web evidence plus code analysis."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Protocol, cast

from src.agent_runtime.models import ContextCapsule, Finding, FindingStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.agent_runtime.models import AgentJob, ContextPack

_URL_RE = re.compile(r"https?://[^\s<>()\"']+", re.IGNORECASE)


class ProgressWorker(Protocol):
    name: str

    async def run(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
    ) -> ContextCapsule: ...

    async def cancel(self, job_id: str) -> bool: ...


class SourceComparePartialFailureError(RuntimeError):
    """Raised with preserved source context when code comparison cannot finish."""

    def __init__(
        self,
        message: str,
        *,
        partial_result: ContextCapsule,
        metadata: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.partial_result = partial_result
        self.metadata = metadata or {}


class SourceCompareWorkerBackend:
    """Run source/link extraction before read-only Codex code comparison."""

    name = "source_compare"

    def __init__(
        self,
        *,
        code_worker: ProgressWorker,
        web_worker: ProgressWorker,
    ) -> None:
        self._code_worker = code_worker
        self._web_worker = web_worker

    async def run(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
    ) -> ContextCapsule:
        return await self._run(
            job=job,
            context_pack=context_pack,
            progress_callback=None,
        )

    async def run_with_progress(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
        progress_callback: Callable[[str], Awaitable[None]],
    ) -> ContextCapsule:
        return await self._run(
            job=job,
            context_pack=context_pack,
            progress_callback=progress_callback,
        )

    async def cancel(self, job_id: str) -> bool:
        code_cancelled = await self._code_worker.cancel(job_id)
        web_cancelled = await self._web_worker.cancel(job_id)
        return code_cancelled or web_cancelled

    async def _run(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
        progress_callback: Callable[[str], Awaitable[None]] | None,
    ) -> ContextCapsule:
        web_capsule = await self._maybe_read_web_context(
            job=job,
            context_pack=context_pack,
            progress_callback=progress_callback,
        )
        code_pack = _augment_context_pack(context_pack, web_capsule)
        try:
            code_capsule = await _run_worker(
                self._code_worker,
                job=job,
                context_pack=code_pack,
                progress_callback=progress_callback,
            )
        except SourceComparePartialFailureError:
            raise
        except asyncio.CancelledError as exc:
            if web_capsule is None:
                raise
            raise _partial_failure_error(
                "source_compare code analysis canceled after source context read",
                partial_result=web_capsule,
                reason="runtime_timeout_or_cancellation",
            ) from exc
        except Exception as exc:
            if web_capsule is None:
                raise
            raise _partial_failure_error(
                f"source_compare code analysis failed after source context read: {exc}",
                partial_result=web_capsule,
                reason="code_analysis_failed",
            ) from exc
        return _merge_capsules(web_capsule, code_capsule)

    async def _maybe_read_web_context(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
        progress_callback: Callable[[str], Awaitable[None]] | None,
    ) -> ContextCapsule | None:
        if not _has_url(job=job, context_pack=context_pack):
            return None
        if progress_callback is not None:
            await progress_callback("Читаю ссылку read-only перед сравнением с кодом.")
        try:
            return await _run_worker(
                self._web_worker,
                job=job,
                context_pack=context_pack,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            return ContextCapsule(
                summary="web source read failed",
                findings=(
                    Finding(
                        claim=f"web source read failed: {exc}",
                        status=FindingStatus.UNCONFIRMED,
                        confidence=0.8,
                    ),
                ),
                next_actions=("Продолжить сравнение по коду без внешнего источника.",),
            )


async def _run_worker(
    worker: ProgressWorker,
    *,
    job: AgentJob,
    context_pack: ContextPack,
    progress_callback: Callable[[str], Awaitable[None]] | None,
) -> ContextCapsule:
    run_with_progress = getattr(worker, "run_with_progress", None)
    if progress_callback is not None and callable(run_with_progress):
        return cast(
            "ContextCapsule",
            await run_with_progress(
                job=job,
                context_pack=context_pack,
                progress_callback=progress_callback,
            ),
        )
    return await worker.run(job=job, context_pack=context_pack)


def _partial_failure_error(
    message: str,
    *,
    partial_result: ContextCapsule,
    reason: str,
) -> SourceComparePartialFailureError:
    return SourceComparePartialFailureError(
        message,
        partial_result=partial_result,
        metadata={
            "stage": "code_analysis",
            "reason": reason,
            "partial_result": partial_result.summary,
        },
    )


def _augment_context_pack(
    context_pack: ContextPack,
    web_capsule: ContextCapsule | None,
) -> ContextPack:
    if web_capsule is None:
        return context_pack
    chat_context = list(context_pack.chat_context)
    constraints = list(context_pack.constraints)
    if web_capsule.processed_context:
        chat_context.append(
            "# Read-only web/source context\n" + web_capsule.processed_context
        )
        constraints.append(
            "Use read-only web/source context as evidence; separate it from code evidence."
        )
    if web_capsule.findings and not web_capsule.processed_context:
        constraints.append(web_capsule.summary)
        constraints.extend(finding.claim for finding in web_capsule.findings)
    return context_pack.model_copy(
        update={
            "chat_context": tuple(chat_context),
            "constraints": tuple(dict.fromkeys(constraints)),
        }
    )


def _merge_capsules(
    web_capsule: ContextCapsule | None,
    code_capsule: ContextCapsule,
) -> ContextCapsule:
    if web_capsule is None:
        return code_capsule
    processed_parts = tuple(
        part
        for part in (web_capsule.processed_context, code_capsule.processed_context)
        if part
    )
    report_parts = tuple(
        part
        for part in (web_capsule.markdown_report, code_capsule.markdown_report)
        if part
    )
    return ContextCapsule(
        summary=code_capsule.summary,
        processed_context="\n\n".join(processed_parts),
        findings=(*web_capsule.findings, *code_capsule.findings),
        sources=(*web_capsule.sources, *code_capsule.sources),
        artifacts=(*web_capsule.artifacts, *code_capsule.artifacts),
        memory_candidates=(
            *web_capsule.memory_candidates,
            *code_capsule.memory_candidates,
        ),
        next_actions=(*web_capsule.next_actions, *code_capsule.next_actions),
        markdown_report="\n\n".join(report_parts),
    )


def _has_url(*, job: AgentJob, context_pack: ContextPack) -> bool:
    return bool(
        _extract_urls(
            (
                context_pack.user_request,
                *context_pack.chat_context,
                *context_pack.attachments,
                *job.followups,
            )
        )
    )


def _extract_urls(parts: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    urls: list[str] = []
    for part in parts:
        for match in _URL_RE.findall(part):
            url = match.rstrip(".,;:!?)]}")
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)
    return tuple(urls)

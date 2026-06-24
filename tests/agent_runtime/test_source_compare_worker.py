"""Composite source-compare worker tests."""

from __future__ import annotations


async def test_source_compare_worker_prefetches_url_context_before_code_analysis(
    tmp_path,
) -> None:
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import AgentJob, ContextCapsule, ContextPack
    from src.agent_runtime.profiles import SOURCE_COMPARE_READONLY
    from src.agent_runtime.workers.source_compare import SourceCompareWorkerBackend
    from src.agent_runtime.workers.web import WebResearchWorkerBackend

    class CodeWorker:
        name = "codex_cli"

        def __init__(self) -> None:
            self.contexts: list[ContextPack] = []

        async def run(
            self, *, job: AgentJob, context_pack: ContextPack
        ) -> ContextCapsule:
            del job
            self.contexts.append(context_pack)
            assert any("source text" in item for item in context_pack.chat_context)
            return ContextCapsule(
                summary="код сверил",
                processed_context="code context",
                sources=("src/agent_runtime/runtime.py",),
                markdown_report="SUMMARY: код сверил",
            )

        async def cancel(self, job_id: str) -> bool:
            del job_id
            return False

    async def fetch(url: str) -> str:
        return f"source text for {url}"

    gateway = build_builtin_tool_gateway(workspace_root=tmp_path, web_fetcher=fetch)
    code_worker = CodeWorker()
    worker = SourceCompareWorkerBackend(
        code_worker=code_worker,
        web_worker=WebResearchWorkerBackend(tool_gateway=gateway),
    )
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:source",
        fingerprint="source",
        kind="source_compare",
        profile=SOURCE_COMPARE_READONLY,
        context_pack=ContextPack(
            user_request="Сравни проект с https://example.com/post"
        ),
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    assert len(code_worker.contexts) == 1
    assert capsule.summary == "код сверил"
    assert capsule.sources == (
        "https://example.com/post",
        "src/agent_runtime/runtime.py",
    )
    assert "source text for https://example.com/post" in capsule.processed_context


async def test_source_compare_worker_keeps_code_analysis_when_web_read_fails() -> None:
    from src.agent_runtime.models import AgentJob, ContextCapsule, ContextPack
    from src.agent_runtime.profiles import SOURCE_COMPARE_READONLY
    from src.agent_runtime.workers.source_compare import SourceCompareWorkerBackend

    class CodeWorker:
        name = "codex_cli"

        async def run(
            self, *, job: AgentJob, context_pack: ContextPack
        ) -> ContextCapsule:
            del job
            assert any(
                "web source read failed" in item for item in context_pack.constraints
            )
            return ContextCapsule(summary="код всё равно сверил")

        async def cancel(self, job_id: str) -> bool:
            del job_id
            return False

    class BrokenWebWorker:
        name = "web_research"

        async def run(
            self, *, job: AgentJob, context_pack: ContextPack
        ) -> ContextCapsule:
            del job, context_pack
            raise RuntimeError("network down")

        async def cancel(self, job_id: str) -> bool:
            del job_id
            return False

    worker = SourceCompareWorkerBackend(
        code_worker=CodeWorker(),
        web_worker=BrokenWebWorker(),
    )
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:source-fail",
        fingerprint="source-fail",
        kind="source_compare",
        profile=SOURCE_COMPARE_READONLY,
        context_pack=ContextPack(
            user_request="Сравни проект с https://example.com/post"
        ),
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    assert capsule.summary == "код всё равно сверил"
    assert capsule.findings[0].status.value == "unconfirmed"
    assert "network down" in capsule.findings[0].claim


async def test_source_compare_worker_preserves_source_context_when_code_fails() -> None:
    from src.agent_runtime.models import AgentJob, ContextCapsule, ContextPack
    from src.agent_runtime.profiles import SOURCE_COMPARE_READONLY
    from src.agent_runtime.workers.source_compare import (
        SourceComparePartialFailureError,
        SourceCompareWorkerBackend,
    )

    class CodeWorker:
        name = "codex_cli"

        async def run(
            self, *, job: AgentJob, context_pack: ContextPack
        ) -> ContextCapsule:
            del job
            assert any("source text" in item for item in context_pack.chat_context)
            raise RuntimeError("code boom")

        async def cancel(self, job_id: str) -> bool:
            del job_id
            return False

    class SourceWorker:
        name = "web_research"

        async def run(
            self, *, job: AgentJob, context_pack: ContextPack
        ) -> ContextCapsule:
            del job, context_pack
            return ContextCapsule(
                summary="source context preserved",
                processed_context="source text",
                sources=("https://example.com/post",),
            )

        async def cancel(self, job_id: str) -> bool:
            del job_id
            return False

    worker = SourceCompareWorkerBackend(
        code_worker=CodeWorker(),
        web_worker=SourceWorker(),
    )
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:source-code-fail",
        fingerprint="source-code-fail",
        kind="source_compare",
        profile=SOURCE_COMPARE_READONLY,
        context_pack=ContextPack(
            user_request="Сравни проект с https://example.com/post"
        ),
    )

    try:
        await worker.run(job=job, context_pack=job.context_pack)
    except SourceComparePartialFailureError as exc:
        error = exc
    else:
        raise AssertionError("source_compare should preserve partial source context")

    assert (
        str(error)
        == "source_compare code analysis failed after source context read: code boom"
    )
    assert error.partial_result.summary == "source context preserved"
    assert error.partial_result.processed_context == "source text"
    assert error.partial_result.sources == ("https://example.com/post",)
    assert error.metadata == {
        "stage": "code_analysis",
        "reason": "code_analysis_failed",
        "partial_result": "source context preserved",
    }

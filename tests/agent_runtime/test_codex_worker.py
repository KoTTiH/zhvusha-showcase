"""Codex worker adapter tests for Agent Runtime."""

from __future__ import annotations

from pathlib import Path


async def test_codex_worker_runs_readonly_context_pack_through_explorer() -> None:
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.workers.codex import CodexWorkerBackend
    from src.skills.code_agent.protocols import CodeAgentResult, ExplorerRequest

    captured: dict[str, ExplorerRequest] = {}

    class FakeCodeBackend:
        name = "fake_codex"

        async def run_explorer(self, request: ExplorerRequest) -> CodeAgentResult:
            captured["request"] = request
            return CodeAgentResult(
                text=(
                    "SUMMARY: Проверил.\n"
                    "FINDING: source_compare использует read-only контекст."
                ),
                backend="fake_codex",
            )

    worker = CodexWorkerBackend(
        code_backend=FakeCodeBackend(),
        cwd=Path("/repo"),
    )
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:14",
        fingerprint="codex",
        kind="source_compare",
        profile=InvocationProfile(
            id="source_compare.readonly",
            allowed_capabilities=("read_code", "read_attachments"),
        ),
        context_pack=ContextPack(
            user_request="сравни пост",
            attachments=("workspace/post.txt",),
        ),
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    assert "source_compare" in captured["request"].user_prompt
    assert "workspace/post.txt" in captured["request"].user_prompt
    assert capsule.summary == "Проверил."
    assert capsule.markdown_report


async def test_codex_worker_uses_zhvusha_chat_voice_for_code_discussion() -> None:
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.workers.codex import CodexWorkerBackend
    from src.skills.code_agent.protocols import CodeAgentResult, ExplorerRequest

    captured: dict[str, ExplorerRequest] = {}

    class FakeCodeBackend:
        async def run_explorer(self, request: ExplorerRequest) -> CodeAgentResult:
            captured["request"] = request
            return CodeAgentResult(text="Я посмотрела: тут лучше сначала обсудить.")

    worker = CodexWorkerBackend(
        code_backend=FakeCodeBackend(),
        cwd=Path("/repo"),
    )
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:discussion",
        fingerprint="discussion",
        kind="self_coding",
        profile=InvocationProfile(id="self_coding.readonly_discussion"),
        context_pack=ContextPack(user_request="что думаешь?"),
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    request = captured["request"]
    assert "женском роде" in request.system_prompt
    assert "Не возвращай machine-capsule" in request.system_prompt
    assert "Do not use labeled capsule lines" in request.user_prompt
    assert "SUMMARY:" not in capsule.markdown_report
    assert capsule.summary == "Я посмотрела: тут лучше сначала обсудить."


async def test_codex_worker_passes_persistent_session_controls() -> None:
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.workers.codex import CodexWorkerBackend
    from src.skills.code_agent.protocols import CodeAgentResult, ExplorerRequest

    captured: dict[str, ExplorerRequest] = {}

    class FakeCodeBackend:
        async def run_explorer(self, request: ExplorerRequest) -> CodeAgentResult:
            captured["request"] = request
            return CodeAgentResult(
                text="Проверила в persistent session.",
                session_id="019e1cf5-a63c-7ca1-a44e-44e555239799",
            )

    worker = CodexWorkerBackend(code_backend=FakeCodeBackend(), cwd=Path("/repo"))
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:discussion",
        fingerprint="discussion",
        kind="self_coding",
        profile=InvocationProfile(id="self_coding.readonly_discussion"),
        context_pack=ContextPack(
            user_request="изучи код",
            active_code_state=(
                "codex_session_id: 019e1cf5-a63c-7ca1-a44e-44e555239799\n"
                "codex_persist_session: true"
            ),
        ),
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    request = captured["request"]
    assert request.persist_session is True
    assert request.session_id == "019e1cf5-a63c-7ca1-a44e-44e555239799"
    assert "codex_session_id:019e1cf5-a63c-7ca1-a44e-44e555239799" in capsule.artifacts


async def test_codex_worker_forwards_progress_callback_to_explorer() -> None:
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.workers.codex import CodexWorkerBackend
    from src.skills.code_agent.protocols import CodeAgentResult, ExplorerRequest

    progress: list[str] = []

    async def progress_callback(message: str) -> None:
        progress.append(message)

    class FakeCodeBackend:
        async def run_explorer(self, request: ExplorerRequest) -> CodeAgentResult:
            assert request.progress_callback is not None
            await request.progress_callback("Сверяю пост с файлами.")
            return CodeAgentResult(text="SUMMARY: done")

    worker = CodexWorkerBackend(
        code_backend=FakeCodeBackend(),
        cwd=Path("/repo"),
    )
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:progress",
        fingerprint="progress",
        kind="source_compare",
        profile=InvocationProfile(id="source_compare.readonly"),
        context_pack=ContextPack(user_request="сравни"),
    )

    await worker.run_with_progress(
        job=job,
        context_pack=job.context_pack,
        progress_callback=progress_callback,
    )

    assert progress == ["Сверяю пост с файлами."]


async def test_codex_worker_extracts_capsule_fields_from_labeled_lines() -> None:
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.workers.codex import CodexWorkerBackend
    from src.skills.code_agent.protocols import CodeAgentResult, ExplorerRequest

    class FakeCodeBackend:
        async def run_explorer(self, request: ExplorerRequest) -> CodeAgentResult:
            return CodeAgentResult(
                text=(
                    "SUMMARY: проверила\n"
                    "FINDING: Agent Runtime хранит jobs.\n"
                    "SOURCE: src/agent_runtime/runtime.py\n"
                    "ARTIFACT: workspace/agent_runtime/job/report.md\n"
                    "MEMORY: source_compare теперь идёт через Agent Runtime.\n"
                    "NEXT: подключить Telegram status renderer.\n"
                )
            )

    worker = CodexWorkerBackend(
        code_backend=FakeCodeBackend(),
        cwd=Path("/repo"),
    )
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:2",
        fingerprint="fp2",
        kind="source_compare",
        profile=InvocationProfile(id="source_compare.readonly"),
        context_pack=ContextPack(user_request="сравни"),
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    assert capsule.sources == ("src/agent_runtime/runtime.py",)
    assert capsule.artifacts == ("workspace/agent_runtime/job/report.md",)
    assert capsule.memory_candidates == (
        "source_compare теперь идёт через Agent Runtime.",
    )
    assert capsule.next_actions == ("подключить Telegram status renderer.",)

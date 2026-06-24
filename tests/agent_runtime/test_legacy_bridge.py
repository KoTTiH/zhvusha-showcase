"""Compatibility bridge tests for existing ExplorerRunner call sites."""

from __future__ import annotations


async def test_explorer_bridge_runs_job_and_returns_context_capsule_report() -> None:
    from src.agent_runtime.bridge import AgentRuntimeExplorerRunner
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import (
        AgentJob,
        ContextCapsule,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.runtime import AgentRuntime, AgentWorkerBackend
    from src.agent_runtime.storage import InMemoryAgentJobStore

    class Worker(AgentWorkerBackend):
        name = "worker"

        def __init__(self) -> None:
            self.calls = 0

        async def run(
            self,
            *,
            job: AgentJob,
            context_pack: ContextPack,
        ) -> ContextCapsule:
            self.calls += 1
            assert context_pack.constraints == ("read-only",)
            return ContextCapsule(
                summary="нашла",
                processed_context=context_pack.user_request,
                markdown_report="## нашла\nПроверила проект.",
            )

        async def cancel(self, job_id: str) -> bool:
            return False

    worker = Worker()
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={"worker": worker},
    )
    runner = AgentRuntimeExplorerRunner(
        runtime=runtime,
        profile=InvocationProfile(id="source_compare", worker="worker"),
        owner_user_id=1,
        chat_id=2,
        kind="source_compare",
    )

    first = await runner(system_prompt="system", user_prompt="изучи проект")
    second = await runner(system_prompt="system", user_prompt="изучи проект")

    assert first == "## нашла\nПроверила проект."
    assert second == first
    assert worker.calls == 1


async def test_explorer_bridge_carries_codex_session_state_and_callback() -> None:
    from src.agent_runtime.bridge import AgentRuntimeExplorerRunner
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import (
        AgentJob,
        ContextCapsule,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.runtime import AgentRuntime, AgentWorkerBackend
    from src.agent_runtime.storage import InMemoryAgentJobStore

    seen_active_state: list[str] = []

    class Worker(AgentWorkerBackend):
        name = "worker"

        async def run(
            self,
            *,
            job: AgentJob,
            context_pack: ContextPack,
        ) -> ContextCapsule:
            seen_active_state.append(context_pack.active_code_state)
            return ContextCapsule(
                summary="нашла",
                markdown_report="нашла",
                artifacts=("codex_session_id:019e1cf5-a63c-7ca1-a44e-44e555239799",),
            )

        async def cancel(self, job_id: str) -> bool:
            return False

    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={"worker": Worker()},
    )
    runner = AgentRuntimeExplorerRunner(
        runtime=runtime,
        profile=InvocationProfile(
            id="self_coding.readonly_discussion", worker="worker"
        ),
        owner_user_id=1,
        chat_id=2,
        kind="self_coding",
    )
    sessions: list[str] = []

    async def remember(session_id: str) -> None:
        sessions.append(session_id)

    result = await runner(
        system_prompt="system",
        user_prompt="изучи код",
        session_id="019e1cf5-a63c-7ca1-a44e-44e555239799",
        persist_session=True,
        session_callback=remember,
    )

    assert result == "нашла"
    assert seen_active_state == [
        "codex_session_id: 019e1cf5-a63c-7ca1-a44e-44e555239799\n"
        "codex_persist_session: true"
    ]
    assert sessions == ["019e1cf5-a63c-7ca1-a44e-44e555239799"]


async def test_explorer_bridge_can_submit_background_and_notify_completion() -> None:
    import asyncio

    from src.agent_runtime.bridge import AgentRuntimeExplorerRunner
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import (
        AgentJob,
        AgentJobStatus,
        ContextCapsule,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.runtime import AgentRuntime, AgentWorkerBackend
    from src.agent_runtime.storage import InMemoryAgentJobStore

    class Worker(AgentWorkerBackend):
        name = "worker"

        def __init__(self) -> None:
            self.release = asyncio.Event()

        async def run(
            self,
            *,
            job: AgentJob,
            context_pack: ContextPack,
        ) -> ContextCapsule:
            await self.release.wait()
            return ContextCapsule(summary="готово", markdown_report="## готово")

        async def cancel(self, job_id: str) -> bool:
            return False

    worker = Worker()
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={"worker": worker},
    )
    runner = AgentRuntimeExplorerRunner(
        runtime=runtime,
        profile=InvocationProfile(id="source_compare", worker="worker"),
        owner_user_id=1,
        chat_id=2,
        kind="source_compare",
    )
    delivered: list[str] = []

    async def on_complete(text: str) -> None:
        delivered.append(text)

    running = await runner.start_background(
        system_prompt="system",
        user_prompt="изучи проект",
        completion_callback=on_complete,
    )

    assert running.status is AgentJobStatus.RUNNING
    worker.release.set()
    for _ in range(20):
        if delivered:
            break
        await asyncio.sleep(0)

    assert delivered == ["## готово"]


async def test_self_coding_bridge_runs_implementation_job() -> None:
    from src.agent_runtime.bridge import SelfCodingAgentRuntimeRunner
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import (
        AgentJob,
        ContextCapsule,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.runtime import AgentRuntime, AgentWorkerBackend
    from src.agent_runtime.storage import InMemoryAgentJobStore
    from src.skills.base import AgentContext

    class Worker(AgentWorkerBackend):
        name = "self_coding_legacy"

        def __init__(self) -> None:
            self.calls = 0

        async def run(
            self,
            *,
            job: AgentJob,
            context_pack: ContextPack,
        ) -> ContextCapsule:
            self.calls += 1
            assert job.kind == "self_coding"
            assert context_pack.user_request == "/spec_run my-spec"
            assert "spec-first" in context_pack.constraints
            return ContextCapsule(
                summary="готово",
                markdown_report="✅ self-coding done",
            )

        async def cancel(self, job_id: str) -> bool:
            return False

    worker = Worker()
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={"self_coding_legacy": worker},
    )
    runner = SelfCodingAgentRuntimeRunner(
        runtime=runtime,
        profile=InvocationProfile(
            id="self_coding.implementation",
            worker="self_coding_legacy",
        ),
        owner_user_id=1,
    )

    context = AgentContext(user_id=1, chat_id=2, mode="personal", message_id=10)
    result = await runner(
        slug="my-spec",
        context=context,
        recent_messages=("Никита: делай",),
    )
    repeated = await runner(
        slug="my-spec",
        context=context,
        recent_messages=("Никита: делай",),
    )

    assert result.success is True
    assert result.response == "✅ self-coding done"
    assert repeated.response == "✅ self-coding done"
    assert worker.calls == 1
    assert result.metadata["agent_job_id"]


async def test_self_coding_bridge_uses_attempt_in_job_fingerprint() -> None:
    from src.agent_runtime.bridge import SelfCodingAgentRuntimeRunner
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import (
        AgentJob,
        ContextCapsule,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.runtime import AgentRuntime, AgentWorkerBackend
    from src.agent_runtime.storage import InMemoryAgentJobStore
    from src.skills.base import AgentContext

    class Worker(AgentWorkerBackend):
        name = "self_coding_legacy"

        def __init__(self) -> None:
            self.calls = 0

        async def run(
            self,
            *,
            job: AgentJob,
            context_pack: ContextPack,
        ) -> ContextCapsule:
            del job, context_pack
            self.calls += 1
            return ContextCapsule(
                summary=f"готово {self.calls}",
                markdown_report=f"done {self.calls}",
            )

        async def cancel(self, job_id: str) -> bool:
            return False

    worker = Worker()
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={"self_coding_legacy": worker},
    )
    runner = SelfCodingAgentRuntimeRunner(
        runtime=runtime,
        profile=InvocationProfile(
            id="self_coding.implementation",
            worker="self_coding_legacy",
        ),
        owner_user_id=1,
    )

    base_context = AgentContext(user_id=1, chat_id=2, mode="personal", message_id=10)
    retry_context = AgentContext(
        user_id=1,
        chat_id=2,
        mode="personal",
        message_id=10,
        metadata={"chat_self_coding_goal_attempt": 1},
    )

    first = await runner(
        slug="my-spec",
        context=base_context,
        recent_messages=("Никита: делай",),
    )
    second = await runner(
        slug="my-spec",
        context=retry_context,
        recent_messages=("Никита: делай",),
    )

    assert first.response == "done 1"
    assert second.response == "done 2"
    assert worker.calls == 2


async def test_self_coding_bridge_carries_code_task_id_in_context_pack() -> None:
    from src.agent_runtime.bridge import SelfCodingAgentRuntimeRunner
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import (
        AgentJob,
        ContextCapsule,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.runtime import AgentRuntime, AgentWorkerBackend
    from src.agent_runtime.storage import InMemoryAgentJobStore
    from src.skills.base import AgentContext

    seen_packs: list[ContextPack] = []

    class Worker(AgentWorkerBackend):
        name = "self_coding_legacy"

        async def run(
            self,
            *,
            job: AgentJob,
            context_pack: ContextPack,
        ) -> ContextCapsule:
            del job
            seen_packs.append(context_pack)
            return ContextCapsule(summary="готово", markdown_report="done")

        async def cancel(self, job_id: str) -> bool:
            return False

    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={"self_coding_legacy": Worker()},
    )
    runner = SelfCodingAgentRuntimeRunner(
        runtime=runtime,
        profile=InvocationProfile(
            id="self_coding.implementation",
            worker="self_coding_legacy",
        ),
        owner_user_id=1,
    )

    await runner(
        slug="my-spec",
        context=AgentContext(
            user_id=1,
            chat_id=2,
            mode="personal",
            message_id=10,
            metadata={"chat_self_coding_code_task_id": "code-task-fixed"},
        ),
    )

    assert seen_packs
    assert "code_task_id: code-task-fixed" in seen_packs[0].active_code_state


async def test_self_coding_bridge_carries_editor_resume_state() -> None:
    from src.agent_runtime.bridge import SelfCodingAgentRuntimeRunner
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import (
        AgentJob,
        ContextCapsule,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.runtime import AgentRuntime, AgentWorkerBackend
    from src.agent_runtime.storage import InMemoryAgentJobStore
    from src.skills.base import AgentContext

    seen_packs: list[ContextPack] = []

    class Worker(AgentWorkerBackend):
        name = "self_coding_legacy"

        async def run(
            self,
            *,
            job: AgentJob,
            context_pack: ContextPack,
        ) -> ContextCapsule:
            del job
            seen_packs.append(context_pack)
            return ContextCapsule(summary="готово", markdown_report="done")

        async def cancel(self, job_id: str) -> bool:
            return False

    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={"self_coding_legacy": Worker()},
    )
    runner = SelfCodingAgentRuntimeRunner(
        runtime=runtime,
        profile=InvocationProfile(
            id="self_coding.implementation",
            worker="self_coding_legacy",
        ),
        owner_user_id=1,
    )

    await runner(
        slug="my-spec",
        context=AgentContext(
            user_id=1,
            chat_id=2,
            mode="personal",
            message_id=10,
            metadata={
                "chat_self_coding_editor_codex_session_id": "codex-editor-thread-1",
                "chat_self_coding_failed_worktree_path": "/repo-worktrees/failed",
                "chat_self_coding_failed_worktree_label": "isolated:spec:1:1",
                "chat_self_coding_failed_worktree_base_branch": "main",
                "chat_self_coding_failed_worktree_base_sha": "abc123",
            },
        ),
    )

    assert seen_packs
    active_state = seen_packs[0].active_code_state
    assert "editor_codex_session_id: codex-editor-thread-1" in active_state
    assert "failed_worktree_path: /repo-worktrees/failed" in active_state
    assert "failed_worktree_base_sha: abc123" in active_state


async def test_self_coding_bridge_blocks_retryable_failure_metadata() -> None:
    from src.agent_runtime.bridge import SelfCodingAgentRuntimeRunner
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import InvocationProfile
    from src.agent_runtime.runtime import AgentRuntime
    from src.agent_runtime.storage import InMemoryAgentJobStore
    from src.agent_runtime.workers.self_coding import SelfCodingLegacyWorkerBackend
    from src.skills.base import AgentContext, SkillResult

    async def legacy_execute(message: str, context: AgentContext) -> SkillResult:
        assert message == "/spec_run my-spec"
        assert context.metadata["agent_job_id"]
        return SkillResult(
            success=False,
            response="Repair attempts exhausted after reviewer reject.",
            metadata={
                "needs_user_decision": "false",
                "auto_retryable": "true",
                "failure_gate": "Reviewer verdict `reject`",
                "editor_codex_session_id": "codex-editor-thread-1",
                "failed_worktree_path": "/repo-worktrees/failed",
            },
        )

    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={
            "self_coding_legacy": SelfCodingLegacyWorkerBackend(
                legacy_execute=legacy_execute
            )
        },
    )
    runner = SelfCodingAgentRuntimeRunner(
        runtime=runtime,
        profile=InvocationProfile(
            id="self_coding.implementation",
            worker="self_coding_legacy",
        ),
        owner_user_id=1,
    )

    result = await runner(
        slug="my-spec",
        context=AgentContext(user_id=1, chat_id=2, mode="personal", message_id=10),
    )

    assert result.success is False
    assert "Repair attempts exhausted" in result.response
    assert result.metadata["needs_user_decision"] == "false"
    assert result.metadata["auto_retryable"] == "false"
    assert result.metadata["failure_gate"] == "Reviewer verdict `reject`"
    assert result.metadata["failure_category"] == "technical_blocker"
    assert result.metadata["editor_codex_session_id"] == "codex-editor-thread-1"
    assert result.metadata["failed_worktree_path"] == "/repo-worktrees/failed"


async def test_self_coding_bridge_can_start_implementation_in_background() -> None:
    import asyncio

    from src.agent_runtime.bridge import SelfCodingAgentRuntimeRunner
    from src.agent_runtime.events import InMemoryAgentEventStream
    from src.agent_runtime.models import (
        AgentJob,
        AgentJobStatus,
        ContextCapsule,
        ContextPack,
        InvocationProfile,
    )
    from src.agent_runtime.runtime import AgentRuntime, AgentWorkerBackend
    from src.agent_runtime.storage import InMemoryAgentJobStore
    from src.skills.base import AgentContext, SkillResult

    class Worker(AgentWorkerBackend):
        name = "self_coding_legacy"

        def __init__(self) -> None:
            self.release = asyncio.Event()

        async def run(
            self,
            *,
            job: AgentJob,
            context_pack: ContextPack,
        ) -> ContextCapsule:
            await self.release.wait()
            return ContextCapsule(summary="готово", markdown_report="готово")

        async def cancel(self, job_id: str) -> bool:
            return False

    worker = Worker()
    runtime = AgentRuntime(
        store=InMemoryAgentJobStore(),
        events=InMemoryAgentEventStream(),
        workers={"self_coding_legacy": worker},
    )
    runner = SelfCodingAgentRuntimeRunner(
        runtime=runtime,
        profile=InvocationProfile(
            id="self_coding.implementation",
            worker="self_coding_legacy",
        ),
        owner_user_id=1,
    )
    delivered: list[SkillResult] = []

    async def on_complete(result: SkillResult) -> None:
        delivered.append(result)

    running = await runner.start_background(
        slug="my-spec",
        context=AgentContext(user_id=1, chat_id=2, mode="personal", message_id=10),
        recent_messages=("Никита: делай",),
        completion_callback=on_complete,
    )

    assert running.status is AgentJobStatus.RUNNING
    worker.release.set()
    for _ in range(20):
        if delivered:
            break
        await asyncio.sleep(0)

    assert delivered[0].success is True
    assert delivered[0].response == "готово"

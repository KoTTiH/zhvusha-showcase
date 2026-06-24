"""Self-coding worker bridge tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest


def test_self_coding_implementation_profile_uses_native_worker() -> None:
    from src.agent_runtime.profiles import SELF_CODING_IMPLEMENTATION

    assert SELF_CODING_IMPLEMENTATION.worker == "self_coding_native"


async def test_native_self_coding_worker_runs_engine_without_spec_run() -> None:
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.workers.self_coding import SelfCodingNativeWorkerBackend
    from src.skills.base import AgentContext, SkillResult

    calls: list[tuple[str, AgentContext]] = []

    class Engine:
        async def run_slug(self, slug: str, context: AgentContext) -> SkillResult:
            calls.append((slug, context))
            return SkillResult(
                success=True,
                response="Готово: `main` @ `abc123`",
                metadata={
                    "commit_sha": "abc123",
                    "branch": "main",
                    "spec_path": "tasks/my-spec.yaml",
                },
            )

    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:1",
        fingerprint="fp",
        kind="self_coding",
        profile=InvocationProfile(
            id="self_coding.implementation",
            worker="self_coding_native",
        ),
        context_pack=ContextPack(user_request="/spec_run my-spec"),
    )
    worker = SelfCodingNativeWorkerBackend(implementation_engine=Engine())

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    assert calls[0][0] == "my-spec"
    assert calls[0][1].metadata["agent_job_id"] == job.id
    assert capsule.summary == "Готово: `main` @ `abc123`"
    assert "commit_sha: abc123" in capsule.artifacts
    assert capsule.sources == ("tasks/my-spec.yaml",)


async def test_self_coding_worker_returns_structured_run_summary() -> None:
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.workers.self_coding import SelfCodingNativeWorkerBackend
    from src.skills.base import AgentContext, SkillResult

    class Engine:
        async def run_slug(self, slug: str, context: AgentContext) -> SkillResult:
            del context
            assert slug == "my-spec"
            return SkillResult(
                success=True,
                response="Готово: `main` @ `abc123`",
                metadata={
                    "commit_sha": "abc123",
                    "branch": "main",
                    "spec_path": "tasks/my-spec.yaml",
                    "changed_files": [
                        "src/agent_runtime/workers/self_coding.py",
                        "tests/agent_runtime/test_self_coding_worker.py",
                    ],
                    "reasoning_summary": "Добавлен audit summary без изменения gates.",
                    "tests": [
                        "uv run pytest -q tests/agent_runtime/test_self_coding_worker.py"
                    ],
                    "failed_repairs": ["Commit gate failed: ruff, fixed formatting."],
                    "risk_review": "Tier 3 approval and commit gate stayed enforced.",
                    "memory_candidates": [
                        "Self-coding summaries must include risk review."
                    ],
                    "archive_candidates": [
                        "self-coding my-spec changed 2 files and passed worker tests."
                    ],
                    "next_actions": ["Run parity benchmark on representative tasks."],
                },
            )

    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:1",
        fingerprint="fp",
        kind="self_coding",
        profile=InvocationProfile(
            id="self_coding.implementation",
            worker="self_coding_native",
        ),
        context_pack=ContextPack(user_request="/spec_run my-spec"),
    )
    worker = SelfCodingNativeWorkerBackend(implementation_engine=Engine())

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    artifact = next(
        item
        for item in capsule.artifacts
        if item.startswith("self_coding_run_summary: ")
    )
    payload = json.loads(artifact.removeprefix("self_coding_run_summary: "))
    assert payload["slug"] == "my-spec"
    assert payload["changed_files"] == [
        "src/agent_runtime/workers/self_coding.py",
        "tests/agent_runtime/test_self_coding_worker.py",
    ]
    assert payload["tests"] == [
        "uv run pytest -q tests/agent_runtime/test_self_coding_worker.py"
    ]
    assert payload["failed_repairs"] == ["Commit gate failed: ruff, fixed formatting."]
    assert payload["risk_review"] == "Tier 3 approval and commit gate stayed enforced."
    assert capsule.memory_candidates == (
        "Self-coding summaries must include risk review.",
    )
    assert capsule.next_actions == ("Run parity benchmark on representative tasks.",)
    assert "## Сводка self-coding" in capsule.markdown_report
    assert "src/agent_runtime/workers/self_coding.py" in capsule.markdown_report
    assert "uv run pytest -q tests/agent_runtime/test_self_coding_worker.py" in (
        capsule.markdown_report
    )
    assert "Tier 3 approval and commit gate stayed enforced." in capsule.markdown_report


async def test_self_coding_worker_merges_decision_and_research_transcript() -> None:
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.workers.self_coding import SelfCodingNativeWorkerBackend
    from src.skills.base import AgentContext, SkillResult

    class Engine:
        async def run_slug(self, slug: str, context: AgentContext) -> SkillResult:
            del context
            assert slug == "transcript-spec"
            return SkillResult(
                success=True,
                response="Готово: transcript-spec.",
                metadata={
                    "changed_files": ["src/agent_runtime/workers/self_coding.py"],
                    "tests": [
                        "uv run pytest -q tests/agent_runtime/test_self_coding_worker.py"
                    ],
                    "risk_review": "Transcript is audit-only and does not grant approvals.",
                    "decision_transcript": [
                        "Implement smallest audit-summary extension.",
                    ],
                    "research_sources": ["docs/agent-runtime-principles.md"],
                },
            )

    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:transcript",
        fingerprint="transcript",
        kind="self_coding",
        profile=InvocationProfile(
            id="self_coding.implementation",
            worker="self_coding_native",
        ),
        context_pack=ContextPack(
            user_request="/spec_run transcript-spec",
            metadata={
                "approval_summary": "Tier 3 spec was approved before implementation.",
                "decision_transcript": "Никита chose auditability over prompt-only summary.",
                "research_sources": "docs/recent-unimplemented-backlog.md",
            },
        ),
    )
    worker = SelfCodingNativeWorkerBackend(implementation_engine=Engine())

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    artifact = next(
        item
        for item in capsule.artifacts
        if item.startswith("self_coding_run_summary: ")
    )
    payload = json.loads(artifact.removeprefix("self_coding_run_summary: "))
    assert payload["approval_summary"] == (
        "Tier 3 spec was approved before implementation."
    )
    assert payload["decision_transcript"] == [
        "Implement smallest audit-summary extension.",
        "Никита chose auditability over prompt-only summary.",
    ]
    assert payload["research_sources"] == [
        "docs/agent-runtime-principles.md",
        "docs/recent-unimplemented-backlog.md",
    ]
    assert "Decision transcript" in capsule.markdown_report
    assert "Research sources" in capsule.markdown_report
    assert "Tier 3 spec was approved" in capsule.markdown_report


async def test_self_coding_worker_persists_run_summary_archive_artifacts(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.workers.self_coding import (
        FileSelfCodingRunSummaryArchive,
        SelfCodingNativeWorkerBackend,
    )
    from src.skills.base import AgentContext, SkillResult

    class Engine:
        async def run_slug(self, slug: str, context: AgentContext) -> SkillResult:
            del context
            assert slug == "my-spec"
            return SkillResult(
                success=True,
                response="Готово: `main` @ `abc123`",
                metadata={
                    "commit_sha": "abc123",
                    "branch": "main",
                    "spec_path": "tasks/my-spec.yaml",
                    "changed_files": ["src/agent_runtime/workers/self_coding.py"],
                    "tests": [
                        "uv run pytest -q tests/agent_runtime/test_self_coding_worker.py"
                    ],
                    "risk_review": "Summary archive does not grant approvals.",
                },
            )

    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:1",
        fingerprint="fp",
        kind="self_coding",
        profile=InvocationProfile(
            id="self_coding.implementation",
            worker="self_coding_native",
        ),
        context_pack=ContextPack(user_request="/spec_run my-spec"),
    )
    archive = FileSelfCodingRunSummaryArchive(
        workspace_root=tmp_path,
        clock=lambda: datetime(2026, 5, 14, 12, 30, tzinfo=UTC),
    )
    worker = SelfCodingNativeWorkerBackend(
        implementation_engine=Engine(),
        summary_archive=archive,
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    markdown_artifact = next(
        item
        for item in capsule.artifacts
        if item.startswith("self_coding_run_summary_markdown_path: ")
    )
    json_artifact = next(
        item
        for item in capsule.artifacts
        if item.startswith("self_coding_run_summary_json_path: ")
    )
    markdown_path = tmp_path / markdown_artifact.split(": ", 1)[1]
    json_path = tmp_path / json_artifact.split(": ", 1)[1]
    assert markdown_path.exists()
    assert json_path.exists()
    assert markdown_path.parent == tmp_path / "self_coding_summaries" / "agent_runtime"
    assert "## Сводка self-coding" in markdown_path.read_text(encoding="utf-8")
    assert "Summary archive does not grant approvals." in markdown_path.read_text(
        encoding="utf-8"
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["slug"] == "my-spec"
    assert payload["commit_sha"] == "abc123"
    assert payload["tests"] == [
        "uv run pytest -q tests/agent_runtime/test_self_coding_worker.py"
    ]


async def test_self_coding_worker_marks_missing_audit_fields_as_quality_warnings() -> (
    None
):
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.workers.self_coding import SelfCodingNativeWorkerBackend
    from src.skills.base import AgentContext, SkillResult

    class Engine:
        async def run_slug(self, slug: str, context: AgentContext) -> SkillResult:
            del context
            assert slug == "thin-summary"
            return SkillResult(
                success=True,
                response="Готово без подробной сводки.",
                metadata={"commit_sha": "abc123"},
            )

    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:thin",
        fingerprint="thin",
        kind="self_coding",
        profile=InvocationProfile(
            id="self_coding.implementation",
            worker="self_coding_native",
        ),
        context_pack=ContextPack(user_request="/spec_run thin-summary"),
    )
    worker = SelfCodingNativeWorkerBackend(implementation_engine=Engine())

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    artifact = next(
        item
        for item in capsule.artifacts
        if item.startswith("self_coding_run_summary: ")
    )
    payload = json.loads(artifact.removeprefix("self_coding_run_summary: "))
    assert payload["quality_warnings"] == [
        "missing_changed_files",
        "missing_tests",
        "missing_risk_review",
    ]
    assert any("missing audit fields" in action for action in capsule.next_actions)
    assert "Summary quality warnings" in capsule.markdown_report


async def test_native_self_coding_worker_forwards_code_task_id() -> None:
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.workers.self_coding import SelfCodingNativeWorkerBackend
    from src.skills.base import AgentContext, SkillResult

    calls: list[AgentContext] = []

    class Engine:
        async def run_slug(self, slug: str, context: AgentContext) -> SkillResult:
            assert slug == "my-spec"
            calls.append(context)
            return SkillResult(success=True, response="done")

    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:1",
        fingerprint="fp",
        kind="self_coding",
        profile=InvocationProfile(
            id="self_coding.implementation",
            worker="self_coding_native",
        ),
        context_pack=ContextPack(
            user_request="/spec_run my-spec",
            active_code_state="active_spec_slug: my-spec\ncode_task_id: code-task-fixed",
        ),
    )
    worker = SelfCodingNativeWorkerBackend(implementation_engine=Engine())

    await worker.run(job=job, context_pack=job.context_pack)

    assert calls[0].metadata["chat_self_coding_code_task_id"] == "code-task-fixed"


async def test_native_self_coding_worker_forwards_editor_resume_metadata() -> None:
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.workers.self_coding import SelfCodingNativeWorkerBackend
    from src.skills.base import AgentContext, SkillResult

    calls: list[AgentContext] = []

    class Engine:
        async def run_slug(self, slug: str, context: AgentContext) -> SkillResult:
            assert slug == "my-spec"
            calls.append(context)
            return SkillResult(success=True, response="done")

    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:1",
        fingerprint="fp",
        kind="self_coding",
        profile=InvocationProfile(
            id="self_coding.implementation",
            worker="self_coding_native",
        ),
        context_pack=ContextPack(
            user_request="/spec_run my-spec",
            active_code_state=(
                "active_spec_slug: my-spec\n"
                "editor_codex_session_id: codex-editor-thread-1\n"
                "failed_worktree_path: /repo-worktrees/failed\n"
                "failed_worktree_label: isolated:spec:1:1\n"
                "failed_worktree_base_branch: main\n"
                "failed_worktree_base_sha: abc123"
            ),
        ),
    )
    worker = SelfCodingNativeWorkerBackend(implementation_engine=Engine())

    await worker.run(job=job, context_pack=job.context_pack)

    metadata = calls[0].metadata
    assert metadata["chat_self_coding_editor_codex_session_id"] == (
        "codex-editor-thread-1"
    )
    assert metadata["chat_self_coding_failed_worktree_path"] == "/repo-worktrees/failed"
    assert metadata["chat_self_coding_failed_worktree_base_sha"] == "abc123"


async def test_self_coding_worker_runs_legacy_spec_skill() -> None:
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.workers.self_coding import SelfCodingLegacyWorkerBackend
    from src.skills.base import AgentContext, SkillResult

    calls: list[tuple[str, AgentContext]] = []

    async def legacy_execute(message: str, context: AgentContext) -> SkillResult:
        calls.append((message, context))
        return SkillResult(
            success=True,
            response="✅ Готово: commit abc123",
            metadata={"commit_sha": "abc123"},
        )

    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:1",
        fingerprint="fp",
        kind="self_coding",
        profile=InvocationProfile(id="self_coding.implementation"),
        context_pack=ContextPack(user_request="/spec_run my-spec"),
    )
    worker = SelfCodingLegacyWorkerBackend(legacy_execute=legacy_execute)

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    assert calls[0][0] == "/spec_run my-spec"
    assert calls[0][1].metadata["agent_job_id"] == job.id
    assert capsule.summary == "✅ Готово: commit abc123"
    assert "commit_sha: abc123" in capsule.artifacts
    assert capsule.sources == ("tasks/my-spec.yaml",)


async def test_self_coding_worker_marks_legacy_failure_as_runtime_failure() -> None:
    from src.agent_runtime.models import AgentJob, ContextPack, InvocationProfile
    from src.agent_runtime.workers.self_coding import SelfCodingLegacyWorkerBackend
    from src.skills.base import AgentContext, SkillResult

    async def legacy_execute(message: str, context: AgentContext) -> SkillResult:
        del message, context
        return SkillResult(success=False, response="env guard blocked")

    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:1",
        fingerprint="fp",
        kind="self_coding",
        profile=InvocationProfile(id="self_coding.implementation"),
        context_pack=ContextPack(user_request="/spec_run my-spec"),
    )
    worker = SelfCodingLegacyWorkerBackend(legacy_execute=legacy_execute)

    with pytest.raises(RuntimeError, match="env guard blocked"):
        await worker.run(job=job, context_pack=job.context_pack)

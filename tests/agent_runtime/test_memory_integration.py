"""Agent Runtime memory staging integration tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory import LearningSignal


async def test_agent_memory_candidate_sink_writes_pending_staging(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.memory import AgentMemoryCandidateSink
    from src.agent_runtime.models import (
        AgentJob,
        ContextCapsule,
        ContextPack,
        InvocationProfile,
    )
    from src.memory import get_staging_writer

    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:1",
        fingerprint="memory",
        kind="source_compare",
        profile=InvocationProfile(id="source_compare"),
        context_pack=ContextPack(user_request="проверь проект"),
    )
    sink = AgentMemoryCandidateSink(get_staging_writer(tmp_path / ".staging"))

    staged = await sink.stage_candidates(
        job=job,
        capsule=ContextCapsule(
            summary="готово",
            memory_candidates=("read-only jobs не должны создавать git branches",),
        ),
    )

    pending = tmp_path / ".staging" / "learnings_pending.md"
    assert staged == 1
    assert pending.exists()
    content = pending.read_text(encoding="utf-8")
    assert "[fact] work" in content
    assert "Agent source_compare: read-only jobs" in content
    assert "**Chat:** 2" in content


async def test_agent_memory_candidate_sink_routes_candidate_types_and_sources(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.memory import AgentMemoryCandidateSink
    from src.agent_runtime.models import (
        AgentJob,
        ContextCapsule,
        ContextPack,
        InvocationProfile,
    )

    class FakeStagingWriter:
        def __init__(self) -> None:
            self.signals: list[LearningSignal] = []

        def append(
            self,
            signal: LearningSignal,
            episode_id: int,
            chat_id: int | None = None,
        ) -> Path:
            del episode_id, chat_id
            self.signals.append(signal)
            return tmp_path / "staged.md"

    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:external",
        fingerprint="memory-routing",
        kind="external_skill.execution",
        profile=InvocationProfile(id="external_skill.execution.browser_read"),
        context_pack=ContextPack(user_request="используй внешний skill"),
    )
    writer = FakeStagingWriter()
    sink = AgentMemoryCandidateSink(writer)

    staged = await sink.stage_candidates(
        job=job,
        capsule=ContextCapsule(
            summary="готово",
            memory_candidates=(
                "preference: для импортированных навыков сначала показывать audit",
                "boundary: external skills cannot write core memory directly",
                "correction: external skill был доверенным runtime | external skill is untrusted procedural input",
                "external_skill_use:kube-debug:execution:use_count=2",
                "native_skill_conversion_candidate:kube-debug:3",
            ),
        ),
    )

    assert staged == 5
    signals = writer.signals
    assert [signal.type for signal in signals] == [
        "preference",
        "boundary",
        "correction",
        "fact",
        "fact",
    ]
    assert [signal.scope for signal in signals] == [
        "preferences",
        "boundaries",
        "work",
        "work",
        "work",
    ]
    assert (
        signals[0].statement == "для импортированных навыков сначала показывать audit"
    )
    assert signals[2].original_claim == "external skill был доверенным runtime"
    assert (
        signals[3].statement
        == "source=external_skill skill_id=kube-debug mode=execution use_count=2"
    )
    assert (
        signals[4].statement
        == "source=external_skill skill_id=kube-debug conversion_candidate uses=3"
    )


def test_source_aware_recall_filters_sensitive_items_and_keeps_evidence() -> None:
    from src.agent_runtime.retrieval import (
        MemorySourceKind,
        SourceAwareMemoryRecall,
        SourceAwareMemoryRecord,
    )

    recall = SourceAwareMemoryRecall(
        records=(
            SourceAwareMemoryRecord(
                source_kind=MemorySourceKind.EXTERNAL_SKILL,
                text="kube-debug audit approved read-only before execution",
                evidence=("external_skill.kube-debug:audit",),
                confidence=0.9,
            ),
            SourceAwareMemoryRecord(
                source_kind=MemorySourceKind.TELEGRAM_MCP,
                text="private message says kube credentials are in chat",
                evidence=("telegram_mcp:chat:42",),
                confidence=0.8,
                sensitive=True,
            ),
            SourceAwareMemoryRecord(
                source_kind=MemorySourceKind.SELF_CODING_ARCHIVE,
                text="self-coding fixed ingress checker with browser draft artifact",
                evidence=("archive:kube-ingress-checker",),
                confidence=0.7,
                stale=True,
            ),
        )
    )

    hits = recall.recall("kube ingress audit credentials")
    rendered = recall.render_for_context(hits)

    assert [hit.record.source_kind for hit in hits] == [
        MemorySourceKind.EXTERNAL_SKILL,
        MemorySourceKind.SELF_CODING_ARCHIVE,
    ]
    assert "telegram" not in rendered.lower()
    assert "private message" not in rendered
    assert "source=external_skill" in rendered
    assert "evidence=external_skill.kube-debug:audit" in rendered
    assert "stale=true" in rendered


def test_source_aware_recall_loads_staging_records_with_source_markers(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.retrieval import (
        MemorySourceKind,
        load_source_aware_staging_records,
    )

    staging = tmp_path / ".staging"
    staging.mkdir()
    (staging / "learnings_pending.md").write_text(
        "\n".join(
            [
                "## [fact] work — 2026-05-20 10:00",
                "**Statement:** source=external_skill skill_id=kube-debug mode=execution use_count=2",
                "**Confidence:** 0.6",
                "**Chat:** 2",
                "**Trigger episode:** 0",
                "",
                "## [fact] work — 2026-05-20 10:01",
                "**Statement:** source=life_runtime focus=open loops attention=parity",
                "**Confidence:** 0.7",
                "**Trigger episode:** 0",
            ]
        ),
        encoding="utf-8",
    )

    records = load_source_aware_staging_records(staging)

    assert [record.source_kind for record in records] == [
        MemorySourceKind.EXTERNAL_SKILL,
        MemorySourceKind.LIFE_RUNTIME,
    ]
    assert records[0].evidence == ("learnings_pending.md:1",)
    assert records[0].confidence == 0.6


def test_file_source_aware_recall_refreshes_staging_records(tmp_path: Path) -> None:
    from src.agent_runtime.retrieval import (
        FileSourceAwareMemoryRecall,
        MemorySourceKind,
    )

    staging = tmp_path / ".staging"
    staging.mkdir()
    pending = staging / "learnings_pending.md"
    pending.write_text(
        "\n".join(
            [
                "## [fact] work — 2026-05-20 10:00",
                "**Statement:** source=external_skill skill_id=kube-debug mode=readonly use_count=1",
                "**Confidence:** 0.6",
            ]
        ),
        encoding="utf-8",
    )
    recall = FileSourceAwareMemoryRecall(staging_dir=staging)

    first = recall.recall("kube-debug readonly")
    pending.write_text(
        pending.read_text(encoding="utf-8")
        + "\n".join(
            [
                "",
                "## [fact] work — 2026-05-20 10:05",
                "**Statement:** source=life_runtime open_loop=kube-debug parity",
                "**Confidence:** 0.7",
            ]
        ),
        encoding="utf-8",
    )
    second = recall.recall("life_runtime kube-debug parity")

    assert first[0].record.source_kind is MemorySourceKind.EXTERNAL_SKILL
    assert any(
        hit.record.source_kind is MemorySourceKind.LIFE_RUNTIME for hit in second
    )


def test_context_pack_builder_attaches_source_aware_recall_context() -> None:
    from src.agent_runtime.context import ContextPackBuilder
    from src.agent_runtime.retrieval import (
        MemorySourceKind,
        SourceAwareMemoryRecall,
        SourceAwareMemoryRecord,
    )

    recall = SourceAwareMemoryRecall(
        records=(
            SourceAwareMemoryRecord(
                source_kind=MemorySourceKind.EXTERNAL_SKILL,
                text="external skill kube-debug execution requires scoped approval",
                evidence=("external_skill.kube-debug:approval",),
                confidence=0.9,
            ),
            SourceAwareMemoryRecord(
                source_kind=MemorySourceKind.TELEGRAM_MCP,
                text="private Telegram note about kube-debug",
                evidence=("telegram_mcp:private",),
                sensitive=True,
            ),
        )
    )
    builder = ContextPackBuilder(memory_recall=recall, max_memory_recall=2)

    pack = builder.build(user_request="что мы помним про kube-debug approval?")

    assert pack.chat_context[0].startswith("## Source-aware memory recall")
    assert "source=external_skill" in pack.chat_context[0]
    assert "private Telegram" not in pack.chat_context[0]
    assert pack.metadata["source_aware_recall"] == "true"
    assert pack.metadata["source_aware_recall_sources"] == "external_skill"
    assert (
        "external_skill.kube-debug:approval"
        in (pack.metadata["source_aware_recall_evidence"])
    )

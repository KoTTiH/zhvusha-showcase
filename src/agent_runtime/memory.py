"""Memory staging integration for Agent Runtime results."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from src.memory import LearningSignal, StagingWriterProtocol, get_staging_writer

if TYPE_CHECKING:
    from pathlib import Path

    from src.agent_runtime.models import AgentJob, ContextCapsule

MemorySignalType = Literal["rule", "preference", "correction", "fact", "boundary"]
MemorySignalScope = Literal[
    "tone",
    "work",
    "personal_facts",
    "boundaries",
    "preferences",
]


@dataclass(frozen=True)
class RoutedMemoryCandidate:
    """A memory candidate converted to the public LearningSignal shape."""

    signal_type: MemorySignalType
    scope: MemorySignalScope
    statement: str
    original_claim: str | None = None


class AgentMemoryCandidateSink:
    """Stage Context Capsule memory candidates through the public memory API."""

    def __init__(
        self,
        staging_writer: StagingWriterProtocol,
        *,
        confidence: float = 0.6,
    ) -> None:
        self._staging_writer = staging_writer
        self._confidence = confidence

    async def stage_candidates(
        self,
        *,
        job: AgentJob,
        capsule: ContextCapsule,
    ) -> int:
        staged = 0
        for candidate in capsule.memory_candidates:
            signal = _candidate_to_signal(
                kind=job.kind,
                candidate=candidate,
                confidence=self._confidence,
            )
            if signal is None:
                continue
            target = await asyncio.to_thread(
                self._staging_writer.append,
                signal,
                0,
                job.chat_id,
            )
            if target is not None:
                staged += 1
        return staged


def build_agent_memory_candidate_sink(workspace_root: Path) -> AgentMemoryCandidateSink:
    """Build the default sink that writes to personality/.staging."""
    return AgentMemoryCandidateSink(
        get_staging_writer(workspace_root / "personality" / ".staging")
    )


def _candidate_to_signal(
    *,
    kind: str,
    candidate: str,
    confidence: float,
) -> LearningSignal | None:
    clean = " ".join(candidate.strip().split())
    if not clean:
        return None
    routed = _route_memory_candidate(kind=kind, candidate=clean)
    return LearningSignal(
        type=routed.signal_type,
        statement=routed.statement,
        scope=routed.scope,
        confidence=confidence,
        apply_immediately=False,
        original_claim=routed.original_claim,
    )


def _route_memory_candidate(*, kind: str, candidate: str) -> RoutedMemoryCandidate:
    prefix, separator, body = candidate.partition(":")
    normalized_prefix = prefix.strip().lower()
    clean_body = body.strip()

    if separator and normalized_prefix == "correction":
        original, correction_separator, replacement = clean_body.partition("|")
        if correction_separator and original.strip() and replacement.strip():
            return RoutedMemoryCandidate(
                signal_type="correction",
                scope="work",
                statement=_truncate_statement(replacement.strip()),
                original_claim=_truncate_original_claim(original.strip()),
            )

    simple_route = _SIMPLE_CANDIDATE_ROUTES.get(normalized_prefix)
    if separator and simple_route is not None and clean_body:
        signal_type, scope = simple_route
        return RoutedMemoryCandidate(
            signal_type=signal_type,
            scope=scope,
            statement=_truncate_statement(clean_body),
        )

    external_skill = _route_external_skill_candidate(candidate)
    if external_skill is not None:
        return external_skill

    desktop_control = _route_desktop_control_candidate(candidate)
    if desktop_control is not None:
        return desktop_control

    return RoutedMemoryCandidate(
        signal_type="fact",
        scope="work",
        statement=_truncate_statement(f"Agent {kind}: {candidate}"),
    )


def _route_external_skill_candidate(candidate: str) -> RoutedMemoryCandidate | None:
    parts = candidate.split(":")
    if len(parts) >= 4 and parts[0] == "external_skill_use":
        skill_id = parts[1].strip()
        mode = parts[2].strip()
        details = ":".join(parts[3:]).strip()
        if skill_id and mode:
            return RoutedMemoryCandidate(
                signal_type="fact",
                scope="work",
                statement=_truncate_statement(
                    "source=external_skill "
                    f"skill_id={skill_id} mode={mode} {details}".strip()
                ),
            )
    if len(parts) >= 3 and parts[0] == "native_skill_conversion_candidate":
        skill_id = parts[1].strip()
        uses = ":".join(parts[2:]).strip()
        if skill_id and uses:
            return RoutedMemoryCandidate(
                signal_type="fact",
                scope="work",
                statement=_truncate_statement(
                    "source=external_skill "
                    f"skill_id={skill_id} conversion_candidate uses={uses}"
                ),
            )
    return None


def _route_desktop_control_candidate(candidate: str) -> RoutedMemoryCandidate | None:
    parts = candidate.split(":")
    if len(parts) >= 3 and parts[0] == "desktop_control_use":
        action = parts[1].strip()
        operation = ":".join(parts[2:]).strip()
        if action and operation:
            return RoutedMemoryCandidate(
                signal_type="fact",
                scope="work",
                statement=_truncate_statement(
                    f"source=desktop_control action={action} operation={operation}"
                ),
            )
    return None


def _truncate_statement(statement: str) -> str:
    if len(statement) <= 300:
        return statement
    return f"{statement[:297]}..."


def _truncate_original_claim(statement: str) -> str:
    if len(statement) <= 500:
        return statement
    return f"{statement[:497]}..."


_SIMPLE_CANDIDATE_ROUTES: dict[
    str,
    tuple[MemorySignalType, MemorySignalScope],
] = {
    "fact": ("fact", "work"),
    "work_fact": ("fact", "work"),
    "personal_fact": ("fact", "personal_facts"),
    "preference": ("preference", "preferences"),
    "rule": ("rule", "work"),
    "boundary": ("boundary", "boundaries"),
}

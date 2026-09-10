"""Context pack construction helpers for Agent Runtime jobs."""

from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING

from src.agent_runtime.models import ContextPack

if TYPE_CHECKING:
    from src.agent_runtime.retrieval import (
        RelevantFileFinder,
        SourceAwareMemoryRecallProvider,
        SourceAwareRecallHit,
    )


class ContextPackBuilder:
    """Build normalized ContextPack objects and idempotency fingerprints."""

    def __init__(
        self,
        *,
        max_chat_messages: int = 12,
        relevant_file_finder: RelevantFileFinder | None = None,
        max_relevant_files: int = 12,
        memory_recall: SourceAwareMemoryRecallProvider | None = None,
        max_memory_recall: int = 8,
    ) -> None:
        self._max_chat_messages = max_chat_messages
        self._relevant_file_finder = relevant_file_finder
        self._max_relevant_files = max_relevant_files
        self._memory_recall = memory_recall
        self._max_memory_recall = max_memory_recall

    def build(
        self,
        *,
        user_request: str,
        chat_context: tuple[str, ...] = (),
        active_code_state: str = "",
        attachments: tuple[str, ...] = (),
        relevant_files: tuple[str, ...] = (),
        constraints: tuple[str, ...] = (),
        metadata: dict[str, str] | None = None,
    ) -> ContextPack:
        """Normalize raw chat/runtime inputs into one worker context pack."""
        normalized_relevant_files = self._relevant_files(
            user_request=user_request,
            chat_context=chat_context,
            active_code_state=active_code_state,
            attachments=attachments,
            relevant_files=relevant_files,
        )
        recall_block, recall_metadata = self._source_aware_recall(
            user_request=user_request,
            chat_context=chat_context,
            active_code_state=active_code_state,
            attachments=attachments,
        )
        normalized_chat_context = tuple(
            _clean(chat_context)[-self._max_chat_messages :]
        )
        if recall_block:
            normalized_chat_context = (recall_block, *normalized_chat_context)
        return ContextPack(
            user_request=user_request.strip(),
            chat_context=normalized_chat_context,
            active_code_state=active_code_state.strip(),
            attachments=tuple(_clean(attachments)),
            relevant_files=normalized_relevant_files,
            constraints=tuple(_dedupe(_clean(constraints))),
            metadata={**(metadata or {}), **recall_metadata},
        )

    def fingerprint(
        self,
        *,
        owner_user_id: int,
        chat_id: int,
        source_message_id: str,
        kind: str,
        context_pack: ContextPack,
    ) -> str:
        """Return a deterministic idempotency key for one incoming source."""
        payload = "|".join(
            (
                str(owner_user_id),
                str(chat_id),
                source_message_id,
                kind,
                context_pack.model_dump_json(),
            )
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def _relevant_files(
        self,
        *,
        user_request: str,
        chat_context: tuple[str, ...],
        active_code_state: str,
        attachments: tuple[str, ...],
        relevant_files: tuple[str, ...],
    ) -> tuple[str, ...]:
        explicit = tuple(_dedupe(_clean(relevant_files)))
        if self._relevant_file_finder is None:
            return explicit
        return self._relevant_file_finder.find(
            query_parts=(
                user_request,
                *chat_context,
                active_code_state,
                *attachments,
            ),
            explicit_files=explicit,
            max_files=self._max_relevant_files,
        )

    def _source_aware_recall(
        self,
        *,
        user_request: str,
        chat_context: tuple[str, ...],
        active_code_state: str,
        attachments: tuple[str, ...],
    ) -> tuple[str, dict[str, str]]:
        if self._memory_recall is None:
            return "", {}
        query = "\n".join(
            part
            for part in (user_request, *chat_context, active_code_state, *attachments)
            if part.strip()
        )
        hits = self._memory_recall.recall(
            query,
            max_results=self._max_memory_recall,
        )
        rendered = self._memory_recall.render_for_context(hits)
        if not rendered:
            return "", {}
        return rendered, _recall_metadata(hits)


def _clean(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(value.strip() for value in values if value.strip())


def _dedupe(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _recall_metadata(hits: tuple[SourceAwareRecallHit, ...]) -> dict[str, str]:
    sources = tuple(_dedupe(tuple(hit.record.source_kind.value for hit in hits)))
    evidence = tuple(
        _dedupe(tuple(item for hit in hits for item in hit.record.evidence))
    )
    return {
        "source_aware_recall": "true",
        "source_aware_recall_sources": ",".join(sources),
        "source_aware_recall_evidence": ",".join(evidence),
    }

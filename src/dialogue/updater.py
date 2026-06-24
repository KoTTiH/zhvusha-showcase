"""Update dialogue state from dispatcher-level events."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.dialogue.people import FilePeopleAliasStore, extract_people_alias_candidates
from src.dialogue.state import (
    DialogueStatePatch,
    FileDialogueStateStore,
    dialogue_state_patch_from_metadata,
)

if TYPE_CHECKING:
    from src.skills.base import SkillResult


class DialogueStateUpdater:
    """Small synchronous updater used by the bot dispatcher."""

    def __init__(
        self,
        store: FileDialogueStateStore,
        *,
        people_alias_store: FilePeopleAliasStore | None = None,
    ) -> None:
        self._store = store
        self._people_alias_store = people_alias_store

    def record_user_message(
        self,
        *,
        chat_id: int | str | None,
        text: str,
        mode: str,
        source_message_id: str = "",
    ) -> None:
        if chat_id is None:
            return
        self._store.apply_patch(
            chat_id,
            DialogueStatePatch(
                mode=mode,
                last_user_message=text,
                increment_turn=True,
                source="bot.user_message",
            ),
        )
        self._record_people_alias_candidates(
            chat_id=chat_id,
            text=text,
            mode=mode,
            source_message_id=source_message_id,
        )

    def record_skill_result(
        self,
        *,
        chat_id: int | str | None,
        result: SkillResult,
    ) -> None:
        if chat_id is None:
            return
        patch = dialogue_state_patch_from_metadata(
            result.metadata.get("dialogue_state_patch")
        )
        if patch is not None:
            self._store.apply_patch(chat_id, patch)

        skill_name = result.metadata.get("skill_name")
        assistant_response = (
            None
            if result.metadata.get("skip_dialogue_assistant_response") is True
            else result.response
        )
        self._store.apply_patch(
            chat_id,
            DialogueStatePatch(
                selected_skill=skill_name if isinstance(skill_name, str) else None,
                last_assistant_response=assistant_response,
                last_result="success" if result.success else "failure",
                source="bot.skill_result",
            ),
        )

    def record_observation(
        self,
        *,
        chat_id: int | str | None,
        mode: str,
        kind: str,
        summary: str,
        source: str,
    ) -> None:
        """Record a non-text body observation as structured dialogue state."""
        if chat_id is None:
            return
        self._store.apply_patch(
            chat_id,
            DialogueStatePatch(
                mode=mode,
                last_intent=kind,
                last_user_message=summary,
                last_tool=source,
                last_result="success",
                increment_turn=True,
                source=f"bot.{kind}",
            ),
        )

    def _record_people_alias_candidates(
        self,
        *,
        chat_id: int | str,
        text: str,
        mode: str,
        source_message_id: str,
    ) -> None:
        if self._people_alias_store is None or mode != "personal":
            return
        for candidate in extract_people_alias_candidates(
            text,
            source_message_id=source_message_id,
            scope=mode,
        ):
            self._people_alias_store.append(chat_id=chat_id, candidate=candidate)

"""Shared per-chat dialogue state."""

from src.dialogue.decisions import (
    DecisionOutcome,
    DecisionResolution,
    DecisionSignal,
    FilePendingDecisionStore,
    PendingDecision,
    resolution_from_approval_signal,
    should_defer_to_cognitive_loop,
)
from src.dialogue.people import (
    FilePeopleAliasStore,
    PeopleAliasCandidate,
    PeopleAliasLookupResult,
    extract_people_alias_candidates,
    render_people_alias_lookup_status,
)
from src.dialogue.state import (
    DialogueState,
    DialogueStatePatch,
    FileDialogueStateStore,
    render_dialogue_context,
    render_dialogue_status,
)
from src.dialogue.updater import DialogueStateUpdater

__all__ = [
    "DecisionOutcome",
    "DecisionResolution",
    "DecisionSignal",
    "DialogueState",
    "DialogueStatePatch",
    "DialogueStateUpdater",
    "FileDialogueStateStore",
    "FilePendingDecisionStore",
    "FilePeopleAliasStore",
    "PendingDecision",
    "PeopleAliasCandidate",
    "PeopleAliasLookupResult",
    "extract_people_alias_candidates",
    "render_dialogue_context",
    "render_dialogue_status",
    "render_people_alias_lookup_status",
    "resolution_from_approval_signal",
    "should_defer_to_cognitive_loop",
]

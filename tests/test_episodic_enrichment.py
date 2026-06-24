"""Tests for EpisodicMemory.update_enrichment() method and metadata merge helper."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from sqlalchemy import Select, Update
from src.memory.episodic import EpisodicMemory, _build_merged_metadata_json
from src.memory.sonnet_enricher import EnrichmentResult


def _sample_result(**overrides: object) -> EnrichmentResult:
    base: dict[str, object] = {
        "importance": 0.85,
        "valence": "negative",
        "intent": "correction",
        "emotion": "frustrated",
        "confidence": 0.9,
        "is_feedback": True,
        "feedback_strength": -0.7,
        "reasoning": "Nikita corrects a factual mistake with frustration.",
    }
    base.update(overrides)
    return EnrichmentResult(**base)  # type: ignore[arg-type]


def _captured_update_values(session: MagicMock) -> dict[str, object]:
    """Extract `.values(...)` dict from the last UPDATE call.

    The session may have both SELECT (for metadata fetch) and UPDATE
    statements; filter to the UPDATE(s) so we capture the actual write.
    """
    assert session.execute.await_count >= 1
    update_calls = [
        call
        for call in session.execute.await_args_list
        if call.args and isinstance(call.args[0], Update)
    ]
    assert update_calls, "expected at least one UPDATE execute() call"
    stmt = update_calls[-1].args[0]
    values: dict[str, object] = {
        col.key: value.value
        for col, value in stmt._values.items()  # type: ignore[attr-defined,union-attr]
    }
    return values


def _configure_select_result(session: MagicMock, value: str | None = None) -> None:
    """Configure `session.execute(...).scalar_one_or_none()` to return `value`.

    Explicitly sets a fresh `MagicMock` as `session.execute.return_value` so
    child attributes (`scalar_one_or_none`) are plain mocks rather than
    AsyncMocks inherited from the parent — AsyncMock auto-propagates
    async-ness to children by default, which would turn
    `row.scalar_one_or_none()` into a coroutine.
    """
    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none = MagicMock(return_value=value)
    session.execute.return_value = scalar_result


# --- Pure helper tests (no mocking needed) ---


def test_build_merged_metadata_json_none_input_produces_enrichment_key() -> None:
    merged = _build_merged_metadata_json(
        None,
        {"feedback_strength": 0.8, "is_feedback": True},
    )
    parsed = json.loads(merged)
    assert parsed == {"enrichment": {"feedback_strength": 0.8, "is_feedback": True}}


def test_build_merged_metadata_json_preserves_top_level_keys() -> None:
    """Existing top-level keys (e.g. 'file_path' used by decision.py) survive."""
    current = json.dumps({"file_path": "knowledge/kwork.md"})
    merged = _build_merged_metadata_json(
        current,
        {"feedback_strength": -0.5, "is_feedback": True},
    )
    parsed = json.loads(merged)
    assert parsed["file_path"] == "knowledge/kwork.md"
    assert parsed["enrichment"]["feedback_strength"] == -0.5
    assert parsed["enrichment"]["is_feedback"] is True


def test_build_merged_metadata_json_merges_into_existing_enrichment_subkey() -> None:
    """Update semantics: existing enrichment sub-keys are preserved unless replaced."""
    current = json.dumps({"enrichment": {"old_field": 1, "feedback_strength": 0.1}})
    merged = _build_merged_metadata_json(
        current,
        {"feedback_strength": 0.9, "is_feedback": True},
    )
    parsed = json.loads(merged)
    assert parsed["enrichment"]["old_field"] == 1  # preserved
    assert parsed["enrichment"]["feedback_strength"] == 0.9  # overwritten
    assert parsed["enrichment"]["is_feedback"] is True  # added


def test_build_merged_metadata_json_handles_corrupted_json() -> None:
    """Corrupted input is discarded (row is about to be overwritten anyway)."""
    merged = _build_merged_metadata_json(
        "{invalid json{{{",
        {"feedback_strength": 0.3},
    )
    parsed = json.loads(merged)
    assert parsed == {"enrichment": {"feedback_strength": 0.3}}


def test_build_merged_metadata_json_handles_non_dict_json_list() -> None:
    """If stored JSON is a list (unexpected), start fresh rather than crash."""
    merged = _build_merged_metadata_json(
        "[1, 2, 3]",
        {"feedback_strength": 0.5},
    )
    parsed = json.loads(merged)
    assert parsed == {"enrichment": {"feedback_strength": 0.5}}


def test_build_merged_metadata_json_handles_non_dict_enrichment_subkey() -> None:
    """If existing `enrichment` is a string/list, replace it with a fresh dict."""
    current = json.dumps({"enrichment": "not a dict"})
    merged = _build_merged_metadata_json(current, {"feedback_strength": 0.7})
    parsed = json.loads(merged)
    assert parsed["enrichment"] == {"feedback_strength": 0.7}


# --- update_enrichment integration (with mocked session) ---


async def test_update_enrichment_writes_all_enrichment_fields(
    mock_session_maker: MagicMock,
) -> None:
    memory = EpisodicMemory(session_maker=mock_session_maker, admin_user_id=12345)
    result = _sample_result()

    session = mock_session_maker._mock_session
    _configure_select_result(session, value=None)

    await memory.update_enrichment(episode_id=42, result=result)

    values = _captured_update_values(session)
    assert values["importance"] == 0.85
    assert values["valence"] == "negative"
    assert values["confidence"] == 0.9
    assert values["intent"] == "correction"
    assert values["emotion"] == "frustrated"
    assert values["enrichment_status"] == "complete"
    assert "metadata_json" in values
    session.commit.assert_awaited_once()


async def test_update_enrichment_nonexistent_episode_is_noop(
    mock_session_maker: MagicMock,
) -> None:
    """If UPDATE touches zero rows SQL-wise, method must not raise."""
    memory = EpisodicMemory(session_maker=mock_session_maker, admin_user_id=12345)
    result = _sample_result()

    session = mock_session_maker._mock_session
    _configure_select_result(session, value=None)
    session.execute.return_value.rowcount = 0

    # Should not raise even though "no rows affected"
    await memory.update_enrichment(episode_id=999999, result=result)
    session.commit.assert_awaited_once()


async def test_update_enrichment_preserves_content_and_embedding(
    mock_session_maker: MagicMock,
) -> None:
    """UPDATE must only touch enrichment fields, never content/embedding."""
    memory = EpisodicMemory(session_maker=mock_session_maker, admin_user_id=12345)
    result = _sample_result()

    session = mock_session_maker._mock_session
    _configure_select_result(session, value=None)

    await memory.update_enrichment(episode_id=7, result=result)

    values = _captured_update_values(session)
    forbidden = {"content", "embedding", "summary", "timestamp", "user_id"}
    assert not (forbidden & values.keys()), (
        f"update_enrichment must not touch columns: {forbidden & values.keys()}"
    )
    assert set(values.keys()) == {
        "importance",
        "valence",
        "confidence",
        "intent",
        "emotion",
        "enrichment_status",
        "metadata_json",
    }


async def test_update_enrichment_writes_metadata_json_with_feedback_strength(
    mock_session_maker: MagicMock,
) -> None:
    """metadata_json column receives enrichment sub-dict with strength + bool."""
    memory = EpisodicMemory(session_maker=mock_session_maker, admin_user_id=12345)
    result = _sample_result(
        feedback_strength=0.85,
        is_feedback=True,
        reasoning="Strong positive reaction.",
    )

    session = mock_session_maker._mock_session
    _configure_select_result(session, value=None)

    await memory.update_enrichment(episode_id=42, result=result)

    values = _captured_update_values(session)
    meta = json.loads(str(values["metadata_json"]))
    assert meta["enrichment"]["feedback_strength"] == 0.85
    assert meta["enrichment"]["is_feedback"] is True
    assert meta["enrichment"]["reasoning"] == "Strong positive reaction."


async def test_update_enrichment_preserves_existing_file_path_metadata(
    mock_session_maker: MagicMock,
) -> None:
    """Regression: pre-existing top-level metadata keys (e.g. file_path used
    by decision.py:381) must survive the merge."""
    memory = EpisodicMemory(session_maker=mock_session_maker, admin_user_id=12345)
    result = _sample_result()

    session = mock_session_maker._mock_session
    _configure_select_result(
        session,
        value=json.dumps({"file_path": "knowledge/project.md"}),
    )

    await memory.update_enrichment(episode_id=42, result=result)

    values = _captured_update_values(session)
    meta = json.loads(str(values["metadata_json"]))
    assert meta["file_path"] == "knowledge/project.md"
    assert "enrichment" in meta
    assert meta["enrichment"]["feedback_strength"] == -0.7


async def test_update_enrichment_issues_select_before_update(
    mock_session_maker: MagicMock,
) -> None:
    """Verify SELECT-then-UPDATE ordering (metadata must be read first)."""
    memory = EpisodicMemory(session_maker=mock_session_maker, admin_user_id=12345)
    result = _sample_result()

    session = mock_session_maker._mock_session
    _configure_select_result(session, value=None)

    await memory.update_enrichment(episode_id=42, result=result)

    statements = [call.args[0] for call in session.execute.await_args_list]
    select_calls = [s for s in statements if isinstance(s, Select)]
    update_calls = [s for s in statements if isinstance(s, Update)]
    assert len(select_calls) == 1, "expected exactly one SELECT"
    assert len(update_calls) == 1, "expected exactly one UPDATE"
    # Ordering: SELECT must come before UPDATE
    first_select_idx = statements.index(select_calls[0])
    first_update_idx = statements.index(update_calls[0])
    assert first_select_idx < first_update_idx

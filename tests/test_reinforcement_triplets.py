"""Tests for Phase 5 — reinforcement triplets from enriched metadata.

Covers:
- Tier B: triplet extraction via `_find_reinforcement_triplets` + helper
  `_extract_feedback_strength`.
- Tier C: parser/renderer/writer for `personality/reinforcements.md`.
- Tier D: integration with `_phase_consolidate`, `_phase_orient` auto-managed
  files exclusion, `_phase_prune_index` pointer skip.
- Tier E: Phase 4 staging review regression (reinforcements.md not in Opus
  target list).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

from src.memory.consolidation import (
    ConsolidationEngine,
    _extract_feedback_strength,
)

if TYPE_CHECKING:
    from pathlib import Path


ADMIN = 12345
STRANGER = 999


def _make_episode(
    id: int = 1,
    content: str = "test",
    user_id: int = ADMIN,
    importance: float = 0.5,
    valence: str = "neutral",
    role: str = "user",
    chat_type: str = "personal",
    intent: str | None = None,
    metadata_json: str | None = None,
    timestamp: datetime | None = None,
    **kwargs: object,
) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "id": id,
        "content": content,
        "user_id": user_id,
        "importance": importance,
        "valence": valence,
        "embedding": [0.1] * 384,
        "timestamp": timestamp or datetime.now(tz=UTC),
        "role": role,
        "chat_type": chat_type,
        "intent": intent,
        "metadata_json": metadata_json,
        "consolidated": False,
        "source": "chat",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _feedback_episode(
    id: int,
    content: str,
    feedback_strength: float,
    timestamp: datetime,
    *,
    user_id: int = ADMIN,
    chat_type: str = "personal",
) -> SimpleNamespace:
    meta = json.dumps(
        {
            "enrichment": {
                "feedback_strength": feedback_strength,
                "is_feedback": True,
                "reasoning": "mock",
            }
        }
    )
    return _make_episode(
        id=id,
        role="user",
        user_id=user_id,
        content=content,
        timestamp=timestamp,
        intent="feedback",
        metadata_json=meta,
        chat_type=chat_type,
    )


def _assistant_episode(
    id: int,
    content: str,
    timestamp: datetime,
    *,
    user_id: int = ADMIN,
    chat_type: str = "personal",
) -> SimpleNamespace:
    return _make_episode(
        id=id,
        role="assistant",
        user_id=user_id,
        content=content,
        timestamp=timestamp,
        chat_type=chat_type,
    )


def _setup_engine(tmp_path: Path) -> ConsolidationEngine:
    ws = tmp_path / "workspace"
    personality = ws / "personality"
    personality.mkdir(parents=True)
    (personality / "core.md").write_text("# Core\nI am Zhvusha.\n", encoding="utf-8")
    (personality / "genes.md").write_text(
        "# Genes\n| Gene | Value |\n", encoding="utf-8"
    )
    (personality / "MEMORY.md").write_text(
        "# Memory Index\n- [core.md](core.md) — who I am\n", encoding="utf-8"
    )

    episodic = AsyncMock()
    episodic.get_unconsolidated = AsyncMock(return_value=[])
    episodic.mark_consolidated = AsyncMock()

    people = SimpleNamespace(record_interaction=lambda _uid: None)
    return ConsolidationEngine(episodic, ws, people)  # type: ignore[arg-type]


def _patch_embed() -> object:
    return patch(
        "src.memory.consolidation.EmbeddingService.embed",
        return_value=[0.5] * 384,
    )


def _patch_cosine(val: float = 0.5) -> object:
    return patch(
        "src.memory.consolidation.EmbeddingService.cosine_similarity",
        return_value=val,
    )


# ============================================================================
# Tier B — _extract_feedback_strength helper
# ============================================================================


def test_extract_feedback_strength_reads_from_metadata_json() -> None:
    ep = _make_episode(metadata_json='{"enrichment":{"feedback_strength":0.7}}')
    assert _extract_feedback_strength(ep) == 0.7


def test_extract_feedback_strength_returns_zero_on_missing_metadata() -> None:
    ep = _make_episode(metadata_json=None)
    assert _extract_feedback_strength(ep) == 0.0


def test_extract_feedback_strength_returns_zero_on_corrupted_json() -> None:
    ep = _make_episode(metadata_json="{invalid json{{")
    assert _extract_feedback_strength(ep) == 0.0


def test_extract_feedback_strength_returns_zero_when_no_enrichment_subkey() -> None:
    ep = _make_episode(metadata_json='{"file_path":"x.md"}')
    assert _extract_feedback_strength(ep) == 0.0


def test_extract_feedback_strength_returns_zero_on_wrong_value_type() -> None:
    ep = _make_episode(metadata_json='{"enrichment":{"feedback_strength":"strong"}}')
    assert _extract_feedback_strength(ep) == 0.0


# ============================================================================
# Tier B — _find_reinforcement_triplets rewrite
# ============================================================================


def test_triplet_detection_uses_intent_feedback_and_strength() -> None:
    t0 = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    episodes = [
        _assistant_episode(1, "короткий kwork-оффер для клиента", t0),
        _feedback_episode(
            2,
            "отлично, именно то что нужно",
            0.8,
            t0 + timedelta(minutes=1),
        ),
    ]

    triplets = ConsolidationEngine._find_reinforcement_triplets(episodes, ADMIN)

    assert len(triplets) == 1
    t = triplets[0]
    assert t["action_id"] == 1
    assert t["feedback_id"] == 2
    assert "короткий kwork" in str(t["action_text"])
    assert "отлично" in str(t["feedback_text"])
    assert t["strength"] == 0.8
    assert t["valence"] == "positive"


def test_triplet_detection_negative_strength_produces_negative_valence() -> None:
    t0 = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    episodes = [
        _assistant_episode(1, "длинный формальный ответ", t0),
        _feedback_episode(2, "не надо так", -0.7, t0 + timedelta(minutes=1)),
    ]
    triplets = ConsolidationEngine._find_reinforcement_triplets(episodes, ADMIN)
    assert len(triplets) == 1
    assert triplets[0]["valence"] == "negative"
    assert triplets[0]["strength"] == -0.7


def test_triplet_detection_skips_weak_strength_below_threshold() -> None:
    t0 = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    episodes = [
        _assistant_episode(1, "ответ", t0),
        _feedback_episode(2, "норм", 0.2, t0 + timedelta(minutes=1)),
    ]
    triplets = ConsolidationEngine._find_reinforcement_triplets(episodes, ADMIN)
    assert triplets == []


def test_triplet_detection_sorts_unsorted_input_by_timestamp() -> None:
    """Reverse-chronological input must still pair correctly."""
    t0 = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    feedback = _feedback_episode(2, "отлично", 0.8, t0 + timedelta(minutes=1))
    action = _assistant_episode(1, "ответ", t0)
    # Pass in reverse order (feedback first, then action)
    episodes = [feedback, action]
    triplets = ConsolidationEngine._find_reinforcement_triplets(episodes, ADMIN)
    assert len(triplets) == 1
    assert triplets[0]["action_id"] == 1
    assert triplets[0]["feedback_id"] == 2


def test_triplet_detection_chains_last_assistant_across_non_feedback_user() -> None:
    """Non-feedback user messages don't reset last_assistant."""
    t0 = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    episodes = [
        _assistant_episode(1, "ответ первый", t0),
        _make_episode(
            id=2,
            role="user",
            content="а что по kwork?",
            intent="question",
            timestamp=t0 + timedelta(minutes=1),
        ),
        _feedback_episode(3, "отлично", 0.9, t0 + timedelta(minutes=2)),
    ]
    triplets = ConsolidationEngine._find_reinforcement_triplets(episodes, ADMIN)
    assert len(triplets) == 1
    assert triplets[0]["action_id"] == 1
    assert triplets[0]["feedback_id"] == 3


def test_triplet_detection_later_assistant_replaces_earlier() -> None:
    """When two assistant messages precede a feedback, the LATER one pairs."""
    t0 = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    episodes = [
        _assistant_episode(1, "first answer", t0),
        _assistant_episode(2, "second answer", t0 + timedelta(minutes=1)),
        _feedback_episode(3, "отлично", 0.9, t0 + timedelta(minutes=2)),
    ]
    triplets = ConsolidationEngine._find_reinforcement_triplets(episodes, ADMIN)
    assert len(triplets) == 1
    assert triplets[0]["action_id"] == 2


def test_triplet_detection_weak_feedback_does_not_consume_last_assistant() -> None:
    """Weak feedback skipped, but next strong feedback still matches the action."""
    t0 = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    episodes = [
        _assistant_episode(1, "ответ", t0),
        _feedback_episode(2, "норм", 0.2, t0 + timedelta(minutes=1)),  # weak
        _feedback_episode(3, "даже хорошо", 0.7, t0 + timedelta(minutes=2)),
    ]
    triplets = ConsolidationEngine._find_reinforcement_triplets(episodes, ADMIN)
    assert len(triplets) == 1
    assert triplets[0]["action_id"] == 1
    assert triplets[0]["feedback_id"] == 3


def test_triplet_detection_one_feedback_per_action() -> None:
    """After emission, last_assistant is consumed — second strong feedback has no partner."""
    t0 = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    episodes = [
        _assistant_episode(1, "ответ", t0),
        _feedback_episode(2, "отлично", 0.9, t0 + timedelta(minutes=1)),
        _feedback_episode(3, "заебись", 0.9, t0 + timedelta(minutes=2)),
    ]
    triplets = ConsolidationEngine._find_reinforcement_triplets(episodes, ADMIN)
    assert len(triplets) == 1
    assert triplets[0]["feedback_id"] == 2


def test_triplet_detection_filters_social_chat_type() -> None:
    t0 = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    episodes = [
        _assistant_episode(1, "ответ", t0, chat_type="social"),
        _feedback_episode(
            2, "отлично", 0.8, t0 + timedelta(minutes=1), chat_type="social"
        ),
    ]
    triplets = ConsolidationEngine._find_reinforcement_triplets(episodes, ADMIN)
    assert triplets == []


def test_triplet_detection_allows_assistant_chat_type() -> None:
    t0 = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    episodes = [
        _assistant_episode(1, "ответ", t0, chat_type="assistant"),
        _feedback_episode(
            2, "отлично", 0.8, t0 + timedelta(minutes=1), chat_type="assistant"
        ),
    ]
    triplets = ConsolidationEngine._find_reinforcement_triplets(episodes, ADMIN)
    assert len(triplets) == 1


def test_triplet_detection_filters_sensitive_action_text() -> None:
    """Action content with 'кредит ... рублей' pattern blocks the triplet."""
    t0 = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    episodes = [
        _assistant_episode(1, "взял кредит 500000 рублей для kwork", t0),
        _feedback_episode(2, "отлично, так и делай", 0.9, t0 + timedelta(minutes=1)),
    ]
    triplets = ConsolidationEngine._find_reinforcement_triplets(episodes, ADMIN)
    assert triplets == []


def test_triplet_detection_filters_sensitive_feedback_text() -> None:
    t0 = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    episodes = [
        _assistant_episode(1, "короткий оффер", t0),
        _feedback_episode(
            2,
            "отлично, моя зарплата 200000 рублей в месяц",
            0.9,
            t0 + timedelta(minutes=1),
        ),
    ]
    triplets = ConsolidationEngine._find_reinforcement_triplets(episodes, ADMIN)
    assert triplets == []


def test_triplet_detection_ignores_non_admin_assistant_episodes() -> None:
    """Assistant episode with user_id != admin is not tracked as last_assistant."""
    t0 = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    episodes = [
        _assistant_episode(1, "answer to stranger", t0, user_id=STRANGER),
        _feedback_episode(2, "отлично", 0.9, t0 + timedelta(minutes=1)),
    ]
    triplets = ConsolidationEngine._find_reinforcement_triplets(episodes, ADMIN)
    assert triplets == []


def test_triplet_detection_ignores_non_admin_feedback() -> None:
    """Feedback from non-admin user doesn't count."""
    t0 = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    episodes = [
        _assistant_episode(1, "ответ", t0),
        _feedback_episode(
            2, "отлично", 0.9, t0 + timedelta(minutes=1), user_id=STRANGER
        ),
    ]
    triplets = ConsolidationEngine._find_reinforcement_triplets(episodes, ADMIN)
    assert triplets == []


def test_triplet_detection_empty_episodes_returns_empty() -> None:
    assert ConsolidationEngine._find_reinforcement_triplets([], ADMIN) == []


# ============================================================================
# Tier C — parser / renderer / writer
# ============================================================================


def _sample_triplet(
    action_id: int = 1,
    feedback_id: int = 2,
    strength: float = 0.8,
    action_text: str = "короткий kwork-оффер",
    feedback_text: str = "отлично",
    ts: str = "2026-04-01T10:00:00+00:00",
) -> dict[str, object]:
    return {
        "action_id": action_id,
        "feedback_id": feedback_id,
        "action_text": action_text,
        "feedback_text": feedback_text,
        "strength": strength,
        "valence": "positive" if strength > 0 else "negative",
        "ts": ts,
    }


def test_parse_reinforcements_empty_text_returns_empty() -> None:
    assert ConsolidationEngine._parse_reinforcements_file("") == []


def test_parse_reinforcements_no_blob_returns_empty() -> None:
    text = "# Reinforcement Patterns\n\n## Что работает\n- some line\n"
    assert ConsolidationEngine._parse_reinforcements_file(text) == []


def test_parse_reinforcements_with_blob_returns_list() -> None:
    triplet = _sample_triplet()
    text = (
        "# Reinforcement Patterns\n\n"
        "## Что работает\n- ...\n\n"
        "<!-- reinforcement_data\n"
        f"{json.dumps([triplet])}\n"
        "-->\n"
    )
    parsed = ConsolidationEngine._parse_reinforcements_file(text)
    assert len(parsed) == 1
    assert parsed[0]["action_id"] == 1
    assert parsed[0]["strength"] == 0.8


def test_parse_reinforcements_corrupted_blob_returns_empty() -> None:
    text = "# Reinforcement Patterns\n\n<!-- reinforcement_data\n[{bad json]\n-->\n"
    assert ConsolidationEngine._parse_reinforcements_file(text) == []


def test_parse_reinforcements_blob_with_non_dict_items_filters_them() -> None:
    """List of primitives/strings filters down to empty dict-list."""
    text = "<!-- reinforcement_data\n[1, 2, 3]\n-->\n"
    parsed = ConsolidationEngine._parse_reinforcements_file(text)
    assert parsed == []  # all items filtered out (not dicts)


def test_triplet_strength_coerces_numeric_string() -> None:
    """`_triplet_strength` handles numeric string values via fallback float()."""
    assert ConsolidationEngine._triplet_strength({"strength": "0.5"}) == 0.5


def test_triplet_strength_returns_zero_for_non_numeric_string() -> None:
    assert ConsolidationEngine._triplet_strength({"strength": "not a number"}) == 0.0


def test_triplet_strength_returns_zero_for_none() -> None:
    assert ConsolidationEngine._triplet_strength({"strength": None}) == 0.0


def test_triplet_strength_returns_zero_for_missing_key() -> None:
    assert ConsolidationEngine._triplet_strength({}) == 0.0


def test_parse_reinforcements_non_list_blob_returns_empty() -> None:
    text = '<!-- reinforcement_data\n[{"not_wrapped": true}, "string item"]\n-->\n'
    parsed = ConsolidationEngine._parse_reinforcements_file(text)
    # Only the dict is kept; string is filtered
    assert len(parsed) == 1
    assert parsed[0] == {"not_wrapped": True}


def test_render_reinforcements_splits_positive_and_negative_sections() -> None:
    triplets = [
        _sample_triplet(action_id=1, strength=0.9, feedback_text="отлично"),
        _sample_triplet(action_id=2, strength=-0.7, feedback_text="плохо"),
    ]
    rendered = ConsolidationEngine._render_reinforcements_file(triplets)
    assert "# Reinforcement Patterns" in rendered
    assert "## Что работает" in rendered
    assert "## Что не работает" in rendered
    # Positive section contains the +0.9 triplet before the negative section
    pos_idx = rendered.index("## Что работает")
    neg_idx = rendered.index("## Что не работает")
    assert pos_idx < neg_idx
    # Positive triplet line is under the positive header
    pos_section = rendered[pos_idx:neg_idx]
    assert "отлично" in pos_section
    assert "+0.9" in pos_section
    neg_section = rendered[neg_idx:]
    assert "плохо" in neg_section
    assert "-0.7" in neg_section


def test_render_reinforcements_embeds_json_blob() -> None:
    triplets = [_sample_triplet()]
    rendered = ConsolidationEngine._render_reinforcements_file(triplets)
    assert "<!-- reinforcement_data" in rendered
    assert "-->" in rendered
    # Round-trip: parsing the rendered text recovers the triplet
    parsed = ConsolidationEngine._parse_reinforcements_file(rendered)
    assert len(parsed) == 1
    assert parsed[0]["action_id"] == 1
    assert parsed[0]["strength"] == 0.8


def test_render_json_escapes_dash_arrow_in_action_text() -> None:
    """If action_text contains '-->', it must be escaped to avoid closing the
    HTML comment prematurely."""
    triplets = [_sample_triplet(action_text="reply with --> arrow in it")]
    rendered = ConsolidationEngine._render_reinforcements_file(triplets)
    # Only ONE closing `-->` should exist — the final blob terminator.
    assert rendered.count("-->") == 1
    # Round-trip still works (escape survives json decode via unicode escape)
    parsed = ConsolidationEngine._parse_reinforcements_file(rendered)
    assert len(parsed) == 1


def test_write_reinforcements_returns_none_when_no_triplets(tmp_path: Path) -> None:
    engine = _setup_engine(tmp_path)
    result = engine._write_reinforcements_pending([])
    assert result is None


def test_write_reinforcements_creates_pending_file(tmp_path: Path) -> None:
    engine = _setup_engine(tmp_path)
    engine.pending_dir.mkdir(parents=True, exist_ok=True)
    triplets = [_sample_triplet()]

    action = engine._write_reinforcements_pending(triplets)

    assert action is not None
    assert action.action == "create"  # file didn't exist before
    assert action.file_path == "reinforcements.md"
    pending_file = engine.pending_dir / "reinforcements.md"
    assert pending_file.exists()
    content = pending_file.read_text(encoding="utf-8")
    assert "Reinforcement Patterns" in content
    assert "отлично" in content


def test_write_reinforcements_emits_update_action_when_file_exists(
    tmp_path: Path,
) -> None:
    engine = _setup_engine(tmp_path)
    engine.pending_dir.mkdir(parents=True, exist_ok=True)
    (engine.personality_dir / "reinforcements.md").write_text(
        "# Reinforcement Patterns\n", encoding="utf-8"
    )

    action = engine._write_reinforcements_pending([_sample_triplet()])

    assert action is not None
    assert action.action == "update"


def test_write_reinforcements_dedupes_by_action_feedback_id(tmp_path: Path) -> None:
    engine = _setup_engine(tmp_path)
    engine.pending_dir.mkdir(parents=True, exist_ok=True)

    # Pre-seed personality/reinforcements.md with triplet A
    existing_triplet = _sample_triplet(action_id=1, feedback_id=2, strength=0.9)
    existing_content = engine._render_reinforcements_file([existing_triplet])
    (engine.personality_dir / "reinforcements.md").write_text(
        existing_content, encoding="utf-8"
    )

    # Write new batch containing the same (1, 2) pair + a new (3, 4) pair
    new_triplets = [
        _sample_triplet(action_id=1, feedback_id=2, strength=0.9),
        _sample_triplet(action_id=3, feedback_id=4, strength=0.6),
    ]
    action = engine._write_reinforcements_pending(new_triplets)

    assert action is not None
    pending = engine.pending_dir / "reinforcements.md"
    parsed = ConsolidationEngine._parse_reinforcements_file(
        pending.read_text(encoding="utf-8")
    )
    # Should contain exactly 2 triplets, not 3 (dup (1,2) collapsed)
    assert len(parsed) == 2
    ids = {(int(p["action_id"]), int(p["feedback_id"])) for p in parsed}
    assert ids == {(1, 2), (3, 4)}


def test_write_reinforcements_caps_at_max_entries(tmp_path: Path) -> None:
    engine = _setup_engine(tmp_path)
    engine.pending_dir.mkdir(parents=True, exist_ok=True)

    # Build 50 triplets with strengths increasing so the top 30 by
    # abs(strength) are deterministic.
    triplets: list[dict[str, object]] = []
    for i in range(50):
        strength = 0.3 + (i * 0.01)  # 0.30 → 0.79
        triplets.append(
            _sample_triplet(action_id=i + 100, feedback_id=i + 200, strength=strength)
        )

    action = engine._write_reinforcements_pending(triplets)
    assert action is not None
    pending = engine.pending_dir / "reinforcements.md"
    parsed = ConsolidationEngine._parse_reinforcements_file(
        pending.read_text(encoding="utf-8")
    )
    assert len(parsed) == 30
    # Top 30 by abs(strength): strengths 0.50 → 0.79 (indices 20..49)
    min_kept = min(float(p["strength"]) for p in parsed)
    assert min_kept >= 0.5 - 1e-9


def test_write_reinforcements_sensitive_filter_drops_triplet(tmp_path: Path) -> None:
    engine = _setup_engine(tmp_path)
    engine.pending_dir.mkdir(parents=True, exist_ok=True)

    safe = _sample_triplet(action_id=1, feedback_id=2, strength=0.8)
    dirty = _sample_triplet(
        action_id=3,
        feedback_id=4,
        action_text="взял кредит 500000 рублей для kwork",
        strength=0.9,
    )
    action = engine._write_reinforcements_pending([safe, dirty])

    assert action is not None
    pending = engine.pending_dir / "reinforcements.md"
    parsed = ConsolidationEngine._parse_reinforcements_file(
        pending.read_text(encoding="utf-8")
    )
    assert len(parsed) == 1
    assert int(parsed[0]["action_id"]) == 1


def test_write_reinforcements_all_sensitive_returns_none(tmp_path: Path) -> None:
    """If every triplet is filtered as sensitive, writer returns None."""
    engine = _setup_engine(tmp_path)
    engine.pending_dir.mkdir(parents=True, exist_ok=True)

    dirty_only = [
        _sample_triplet(
            action_id=1, action_text="моя зарплата 200000 рублей", strength=0.8
        ),
    ]
    result = engine._write_reinforcements_pending(dirty_only)
    assert result is None


def test_write_reinforcements_reads_existing_from_pending_first(
    tmp_path: Path,
) -> None:
    """`_read_current_content` checks `.pending/` before `personality/`."""
    engine = _setup_engine(tmp_path)
    engine.pending_dir.mkdir(parents=True, exist_ok=True)

    # Old version in personality/
    old_triplet = _sample_triplet(action_id=1, feedback_id=2, strength=0.3)
    (engine.personality_dir / "reinforcements.md").write_text(
        engine._render_reinforcements_file([old_triplet]), encoding="utf-8"
    )
    # Newer version in .pending/ (pretend Phase 3 or earlier Phase 5 wrote it)
    newer_triplet = _sample_triplet(action_id=5, feedback_id=6, strength=0.7)
    (engine.pending_dir / "reinforcements.md").write_text(
        engine._render_reinforcements_file([newer_triplet]), encoding="utf-8"
    )

    # Writing a third triplet must merge with the .pending/ version, not personality/
    fresh_triplet = _sample_triplet(action_id=7, feedback_id=8, strength=0.9)
    engine._write_reinforcements_pending([fresh_triplet])

    pending = engine.pending_dir / "reinforcements.md"
    parsed = ConsolidationEngine._parse_reinforcements_file(
        pending.read_text(encoding="utf-8")
    )
    ids = {(int(p["action_id"]), int(p["feedback_id"])) for p in parsed}
    # Should contain newer (.pending) triplet + fresh, NOT old (personality)
    assert (5, 6) in ids
    assert (7, 8) in ids
    assert (1, 2) not in ids


# ============================================================================
# Tier D — _AUTO_MANAGED_FILES filter + _phase_consolidate integration
# ============================================================================


async def test_phase_orient_excludes_auto_managed_files_from_topic_to_file(
    tmp_path: Path,
) -> None:
    """reinforcements.md and MEMORY.md must not appear in topic_to_file.
    Regression for the Phase 5 data-loss bug: without this filter, the
    per-episode merge loop would clobber reinforcements.md with arbitrary
    episode content when similarity matched."""
    engine = _setup_engine(tmp_path)
    # Seed reinforcements.md with realistic-looking content
    (engine.personality_dir / "reinforcements.md").write_text(
        "# Reinforcement Patterns\n\nкороткий оффер для kwork работает\n",
        encoding="utf-8",
    )

    state = await engine._phase_orient()

    for topic_text, rel_path in state.topic_to_file.items():
        assert rel_path != "reinforcements.md", (
            f"reinforcements.md must not be in topic_to_file, "
            f"found via topic: {topic_text[:50]}"
        )
        assert rel_path != "MEMORY.md"


async def test_phase_orient_still_includes_regular_files(tmp_path: Path) -> None:
    """Sanity: core.md and genes.md still end up in topic_to_file."""
    engine = _setup_engine(tmp_path)

    state = await engine._phase_orient()

    files = set(state.topic_to_file.values())
    assert "core.md" in files
    assert "genes.md" in files


async def test_phase_prune_index_skips_auto_managed_files_pointer(
    tmp_path: Path,
) -> None:
    """Create actions for reinforcements.md must NOT add a line to MEMORY.md."""
    from src.memory.consolidation import ConsolidationAction

    engine = _setup_engine(tmp_path)
    engine.pending_dir.mkdir(parents=True, exist_ok=True)
    initial_content = engine.memory_index.read_text(encoding="utf-8")
    initial_entries = [ln for ln in initial_content.split("\n") if ln.startswith("- ")]

    actions: list[ConsolidationAction] = [
        ConsolidationAction(
            action="create",
            file_path="reinforcements.md",
            content="...",
            reason="auto-managed",
        ),
        ConsolidationAction(
            action="create",
            file_path="skills/new_skill.md",
            content="...",
            reason="new skill",
        ),
    ]
    await engine._phase_prune_index(actions)

    # Prune now writes to .pending/ (transactional)
    pending_content = (engine.pending_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert "reinforcements.md" not in pending_content
    assert "new_skill" in pending_content
    # Exactly one new entry was added (for new_skill), not two
    final_entries = [ln for ln in pending_content.split("\n") if ln.startswith("- ")]
    assert len(final_entries) == len(initial_entries) + 1


async def test_phase_consolidate_emits_reinforcement_action_when_triplets_found(
    tmp_path: Path,
) -> None:
    """End-to-end: episodes with action+feedback → actions list contains
    a reinforcements.md entry after `_phase_consolidate`."""
    engine = _setup_engine(tmp_path)
    engine.pending_dir.mkdir(parents=True, exist_ok=True)

    t0 = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    episodes = [
        _assistant_episode(1, "короткий kwork-оффер", t0),
        _feedback_episode(2, "отлично, именно так", 0.8, t0 + timedelta(minutes=1)),
    ]

    state = await engine._phase_orient()
    with _patch_embed(), _patch_cosine(0.3):
        actions, _partial = await engine._phase_consolidate(episodes, state, ADMIN)

    reinforcement_actions = [a for a in actions if a.file_path == "reinforcements.md"]
    assert len(reinforcement_actions) == 1
    assert (engine.pending_dir / "reinforcements.md").exists()


async def test_phase_consolidate_no_action_when_no_triplets(
    tmp_path: Path,
) -> None:
    """If triplet detector returns empty, no reinforcements action is emitted."""
    engine = _setup_engine(tmp_path)
    engine.pending_dir.mkdir(parents=True, exist_ok=True)

    # Episodes without feedback
    t0 = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    episodes = [
        _make_episode(
            id=1,
            role="user",
            content="обычный вопрос",
            intent="question",
            timestamp=t0,
            importance=0.3,
        ),
    ]

    state = await engine._phase_orient()
    with _patch_embed(), _patch_cosine(0.3):
        actions, _partial = await engine._phase_consolidate(episodes, state, ADMIN)

    reinforcement_actions = [a for a in actions if a.file_path == "reinforcements.md"]
    assert reinforcement_actions == []
    assert not (engine.pending_dir / "reinforcements.md").exists()


async def test_phase_consolidate_does_not_clobber_existing_reinforcements(
    tmp_path: Path,
) -> None:
    """CRITICAL data-loss regression: pre-seed reinforcements.md with content
    that would semantically match an episode, run consolidate, assert the
    per-episode loop does NOT merge episode content into reinforcements.md.

    The only way reinforcements.md should be written during consolidation is
    via `_write_reinforcements_pending` (which this test's episodes trigger
    since they include a valid action+feedback triplet)."""
    engine = _setup_engine(tmp_path)
    engine.pending_dir.mkdir(parents=True, exist_ok=True)

    # Pre-seed reinforcements.md with content that semantically matches
    # the episode body below.
    (engine.personality_dir / "reinforcements.md").write_text(
        "# Reinforcement Patterns\n\n"
        "Что работает: короткие kwork-офферы для клиентов\n",
        encoding="utf-8",
    )

    t0 = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    episodes = [
        _assistant_episode(1, "короткий kwork-оффер для клиента", t0),
        _feedback_episode(2, "отлично", 0.8, t0 + timedelta(minutes=1)),
    ]

    state = await engine._phase_orient()
    # Verify topic_to_file does NOT contain reinforcements.md (the fix)
    assert "reinforcements.md" not in state.topic_to_file.values()

    # Force high cosine similarity so the generic merge loop WOULD match
    # reinforcements.md if it were in topic_to_file.
    with _patch_embed(), _patch_cosine(0.95):
        _actions, _partial = await engine._phase_consolidate(episodes, state, ADMIN)

    # Phase 5 writes its own version — that's fine
    pending_reinforcements = engine.pending_dir / "reinforcements.md"
    assert pending_reinforcements.exists()
    content = pending_reinforcements.read_text(encoding="utf-8")

    # The Phase 5 write uses structured format with "отлично" (the feedback).
    # The per-episode loop would have appended raw episode content wrapped in
    # a `<!-- Episode N -->` comment — verify that's NOT present.
    assert "<!-- Episode " not in content
    # Phase 5 structured format IS present
    assert "reinforcement_data" in content


async def test_phase_consolidate_reinforcement_failure_does_not_propagate(
    tmp_path: Path,
) -> None:
    """If `_write_reinforcements_pending` raises, consolidation continues."""
    engine = _setup_engine(tmp_path)
    engine.pending_dir.mkdir(parents=True, exist_ok=True)

    t0 = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    episodes = [
        _assistant_episode(1, "ответ", t0),
        _feedback_episode(2, "отлично", 0.8, t0 + timedelta(minutes=1)),
    ]

    state = await engine._phase_orient()
    with (
        _patch_embed(),
        _patch_cosine(0.3),
        patch.object(
            engine,
            "_write_reinforcements_pending",
            side_effect=RuntimeError("oops"),
        ),
    ):
        # Should NOT raise
        actions, _partial = await engine._phase_consolidate(episodes, state, ADMIN)

    # No reinforcements.md action (writer failed)
    assert not any(a.file_path == "reinforcements.md" for a in actions)

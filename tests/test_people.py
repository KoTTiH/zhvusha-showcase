from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from src.memory.people import PeopleManager


def _manager(tmp_path: Path) -> PeopleManager:
    return PeopleManager(workspace_root=tmp_path)


def test_create_new_profile(tmp_path: Path):
    mgr = _manager(tmp_path)
    profile = mgr.get_or_create_profile(12345, username="nikita", first_name="Nikita")

    assert profile["user_id"] == 12345
    assert profile["username"] == "nikita"
    assert profile["significance"] == "stranger"
    assert profile["interaction_count"] == 0

    # File created on disk
    profile_file = tmp_path / "memory" / "people" / "12345" / "profile.md"
    assert profile_file.exists()


def test_get_existing_profile(tmp_path: Path):
    mgr = _manager(tmp_path)
    mgr.get_or_create_profile(12345, username="nikita", first_name="Nikita")

    # Second call should return existing profile, not overwrite
    profile = mgr.get_or_create_profile(12345, username="other", first_name="Other")
    assert profile["username"] == "nikita"


def test_update_profile(tmp_path: Path):
    mgr = _manager(tmp_path)
    mgr.get_or_create_profile(12345)
    mgr.update_profile(12345, {"significance": "known"})

    profile = mgr.get_or_create_profile(12345)
    assert profile["significance"] == "known"


def test_record_interaction_increments(tmp_path: Path):
    mgr = _manager(tmp_path)
    mgr.get_or_create_profile(12345)

    mgr.record_interaction(12345)
    mgr.record_interaction(12345)
    mgr.record_interaction(12345)

    profile = mgr.get_or_create_profile(12345)
    assert profile["interaction_count"] == 3


def test_significance_stranger_default(tmp_path: Path):
    mgr = _manager(tmp_path)
    assert mgr.get_significance_level(99999) == "stranger"


def test_get_profile_for_context_personal(tmp_path: Path):
    mgr = _manager(tmp_path)
    mgr.get_or_create_profile(12345, username="nikita", first_name="Nikita")

    text = mgr.get_profile_for_context(12345, "personal")
    assert "nikita" in text.lower() or "12345" in text


def test_get_profile_for_context_assistant(tmp_path: Path):
    mgr = _manager(tmp_path)
    mgr.get_or_create_profile(12345, username="nikita")

    text = mgr.get_profile_for_context(12345, "assistant")
    assert "nikita" in text.lower() or "12345" in text


def test_get_profile_for_context_social_empty(tmp_path: Path):
    mgr = _manager(tmp_path)
    mgr.get_or_create_profile(12345, username="nikita")

    text = mgr.get_profile_for_context(12345, "social")
    assert text == ""


def test_auto_promote_stranger_to_known(tmp_path: Path):
    mgr = _manager(tmp_path)
    mgr.get_or_create_profile(12345)

    assert mgr.get_significance_level(12345) == "stranger"

    mgr.record_interaction(12345)
    mgr.record_interaction(12345)
    assert mgr.get_significance_level(12345) == "stranger"

    mgr.record_interaction(12345)  # 3rd interaction
    assert mgr.get_significance_level(12345) == "known"


def test_no_promote_if_already_known(tmp_path: Path):
    mgr = _manager(tmp_path)
    mgr.get_or_create_profile(12345)
    mgr.update_profile(12345, {"significance": "inner_circle"})

    for _ in range(5):
        mgr.record_interaction(12345)

    # Should NOT downgrade inner_circle to known
    assert mgr.get_significance_level(12345) == "inner_circle"


def test_parse_write_roundtrip(tmp_path: Path):
    mgr = _manager(tmp_path)
    mgr.get_or_create_profile(12345, username="test_user", first_name="Test")
    mgr.update_profile(12345, {"significance": "known", "interaction_count": 5})

    profile = mgr.get_or_create_profile(12345)
    assert profile["username"] == "test_user"
    assert profile["first_name"] == "Test"
    assert profile["significance"] == "known"
    assert profile["interaction_count"] == 5


def test_get_interaction_count_no_profile(tmp_path: Path):
    mgr = _manager(tmp_path)
    assert mgr.get_interaction_count(99999) == 0


def test_get_interaction_count_tracks(tmp_path: Path):
    mgr = _manager(tmp_path)
    mgr.get_or_create_profile(12345)
    assert mgr.get_interaction_count(12345) == 0

    mgr.record_interaction(12345)
    mgr.record_interaction(12345)
    assert mgr.get_interaction_count(12345) == 2

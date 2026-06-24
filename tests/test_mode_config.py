from src.core.mode_config import is_skill_allowed


def test_personal_allows_all_skills():
    assert is_skill_allowed("kwork_monitor", "personal") is True
    assert is_skill_allowed("channel_writer", "personal") is True
    assert is_skill_allowed("chat_response", "personal") is True


def test_assistant_allows_chat_response():
    assert is_skill_allowed("chat_response", "assistant") is True


def test_assistant_blocks_kwork():
    assert is_skill_allowed("kwork_monitor", "assistant") is False
    assert is_skill_allowed("channel_writer", "assistant") is False
    assert is_skill_allowed("workspace_session", "assistant") is False


def test_social_blocks_admin_skills():
    assert is_skill_allowed("kwork_monitor", "social") is False
    assert is_skill_allowed("channel_writer", "social") is False
    assert is_skill_allowed("workspace_session", "social") is False
    assert is_skill_allowed("chat_response", "social") is True

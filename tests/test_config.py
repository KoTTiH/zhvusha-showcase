import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from src.core.config import Settings

REQUIRED_ENV = {
    "BOT_TOKEN": "test_token",
    "CHANNEL_ID": "@test_channel",
    "ADMIN_USER_ID": "12345",
}


def _settings_no_env() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_settings_loads_from_env():
    with patch.dict(os.environ, REQUIRED_ENV, clear=True):
        s = _settings_no_env()
    assert s.bot_token == "test_token"
    assert s.channel_id == "@test_channel"
    assert s.admin_user_id == 12345
    assert s.telegram_bot_proxy == ""


def test_settings_defaults():
    with patch.dict(os.environ, REQUIRED_ENV, clear=True):
        s = _settings_no_env()
    assert s.google_api_key == ""
    assert s.anthropic_api_key == ""
    assert s.nvidia_nim_api_key == ""
    assert s.huggingface_api_key == ""
    assert s.minimax_api_key == ""
    assert s.nvidia_nim_base_url == "https://integrate.api.nvidia.com/v1"
    assert s.huggingface_base_url == "https://router.huggingface.co/v1"
    assert s.minimax_base_url == "https://api.minimax.io/v1"
    assert s.database_url == ""
    assert s.default_llm_tier == "worker"
    assert s.strategist_budget_daily_usd == 1.00
    assert s.enable_browser_use is False
    assert s.browser_backend == "browser_use"
    assert s.claude_cli_path == ""
    assert s.codex_cli_path == "codex"
    assert s.code_agent_backend == "codex_cli"
    assert s.code_agent_model == ""
    assert s.code_agent_reasoning_effort == "xhigh"
    assert s.code_agent_timeout_seconds == 7200.0
    assert s.self_coding_caps_per_hour == 0
    assert s.self_coding_caps_per_day == 0
    assert s.code_agent_max_concurrent == 1
    assert s.morning_session_model == "gpt-5.5"
    assert s.morning_session_reasoning_effort == "xhigh"
    assert s.worker_reasoning_effort == "medium"
    assert s.analyst_reasoning_effort == "high"
    assert s.strategist_reasoning_effort == "xhigh"
    assert s.chat_agentic_timeout_seconds == 300.0
    assert s.vscode_chat_enabled is True
    assert s.vscode_chat_host == "127.0.0.1"
    assert s.vscode_chat_port == 7331
    assert s.vscode_chat_token == ""
    assert s.telegram_mcp_enabled is False
    assert s.telegram_mcp_checkout_path == "~/.local/share/zhvusha/telegram-mcp"
    assert s.telegram_mcp_account_label == "personal"
    assert s.telegram_mcp_session_string_personal == ""
    assert s.telegram_mcp_session_name_personal == ""
    assert s.telegram_mcp_allowed_roots == "~/zhvusha-workspace/telegram-mcp"
    assert s.daivinchik_chat_id == "@leomatchbot"
    assert s.personal_telegram_inbound_enabled is False
    assert s.personal_telegram_inbound_store_path == (
        "~/zhvusha-workspace/telegram/inbound_events.jsonl"
    )
    assert s.personal_telegram_inbound_max_events_per_poll == 20
    assert s.personal_telegram_inbound_auto_reply_enabled is False
    assert s.personal_telegram_inbound_external_max_chars == 800
    assert s.personal_telegram_inbound_external_knowledge_categories == (
        "research,intel.channels,intel.youtube"
    )
    assert s.daemon_agent_runtime_enabled is False
    assert s.life_runtime_enabled is False
    assert s.life_runtime_state_path == "~/zhvusha-workspace/life_runtime"
    assert s.life_runtime_min_tick_interval_seconds == 300
    assert s.life_runtime_max_ticks_per_hour == 12
    assert s.voice_gateway_enabled is False
    assert s.voice_stt_provider == ""
    assert s.voice_tts_enabled is False
    assert s.desktop_control_enabled is False
    assert s.desktop_control_command_map_json == ""
    assert s.desktop_control_command_timeout_seconds == 10.0
    assert s.computer_use_enabled is False
    assert s.live_browser_control_enabled is False
    assert s.live_browser_backend == "chrome_devtools_mcp"
    assert s.live_browser_debug_url == "http://127.0.0.1:9222"
    assert s.live_browser_auto_launch is False
    assert s.live_browser_executable == "chromium"
    assert s.live_browser_user_data_dir == "~/zhvusha-workspace/live-chrome"
    assert s.live_browser_headless is False
    assert s.computer_use_irreversible_policy == "hard_stop"
    assert s.computer_use_shell_enabled is False
    assert s.agency_runtime_enabled is False
    assert s.agency_social_autonomy_enabled is False
    assert s.agency_permission_store_path == (
        "~/zhvusha-workspace/agency/permissions.jsonl"
    )
    assert s.agency_max_jobs_per_day == 12
    assert s.agency_max_social_messages_per_hour == 3
    assert s.autonomous_loop_budget_daily_usd == 1.0
    assert s.autonomous_loop_budget_weekly_usd == 5.0
    assert s.autonomous_self_coding_budget_daily_usd == 2.0
    assert s.autonomous_self_coding_budget_weekly_usd == 8.0
    assert s.autonomous_day_window_start_hour == 10
    assert s.autonomous_day_window_duration_hours == 12
    assert s.autonomous_loop_max_jobs_per_day == 12
    assert s.autonomous_loop_max_runtime_seconds_per_day == 43200
    assert s.autonomous_loop_max_retries_per_job == 2
    assert s.autonomous_loop_max_concurrent_jobs == 1
    assert s.autonomous_self_coding_max_jobs_per_day == 2
    assert s.autonomous_self_coding_max_runtime_seconds_per_day == 43200


def test_autonomous_budget_settings_overridable_from_env():
    env = {
        **REQUIRED_ENV,
        "AUTONOMOUS_LOOP_BUDGET_DAILY_USD": "0.25",
        "AUTONOMOUS_LOOP_BUDGET_WEEKLY_USD": "1.50",
        "AUTONOMOUS_SELF_CODING_BUDGET_DAILY_USD": "1.25",
        "AUTONOMOUS_SELF_CODING_BUDGET_WEEKLY_USD": "4.00",
        "AUTONOMOUS_DAY_WINDOW_START_HOUR": "9",
        "AUTONOMOUS_DAY_WINDOW_DURATION_HOURS": "12",
        "AUTONOMOUS_LOOP_MAX_JOBS_PER_DAY": "9",
        "AUTONOMOUS_LOOP_MAX_RUNTIME_SECONDS_PER_DAY": "36000",
        "AUTONOMOUS_LOOP_MAX_RETRIES_PER_JOB": "1",
        "AUTONOMOUS_LOOP_MAX_CONCURRENT_JOBS": "2",
        "AUTONOMOUS_SELF_CODING_MAX_JOBS_PER_DAY": "1",
        "AUTONOMOUS_SELF_CODING_MAX_RUNTIME_SECONDS_PER_DAY": "21600",
        "LIFE_RUNTIME_ENABLED": "true",
        "LIFE_RUNTIME_STATE_PATH": "/home/zhvusha/life-runtime",
        "LIFE_RUNTIME_MIN_TICK_INTERVAL_SECONDS": "60",
        "LIFE_RUNTIME_MAX_TICKS_PER_HOUR": "6",
        "PERSONAL_TELEGRAM_INBOUND_ENABLED": "true",
        "PERSONAL_TELEGRAM_INBOUND_STORE_PATH": "/home/zhvusha/inbound.jsonl",
        "PERSONAL_TELEGRAM_INBOUND_MAX_EVENTS_PER_POLL": "5",
        "PERSONAL_TELEGRAM_INBOUND_AUTO_REPLY_ENABLED": "true",
        "PERSONAL_TELEGRAM_INBOUND_EXTERNAL_MAX_CHARS": "240",
        "PERSONAL_TELEGRAM_INBOUND_EXTERNAL_KNOWLEDGE_CATEGORIES": "research,web",
    }
    with patch.dict(os.environ, env, clear=True):
        s = _settings_no_env()

    assert s.autonomous_loop_budget_daily_usd == 0.25
    assert s.autonomous_loop_budget_weekly_usd == 1.50
    assert s.autonomous_self_coding_budget_daily_usd == 1.25
    assert s.autonomous_self_coding_budget_weekly_usd == 4.00
    assert s.autonomous_day_window_start_hour == 9
    assert s.autonomous_day_window_duration_hours == 12
    assert s.autonomous_loop_max_jobs_per_day == 9
    assert s.autonomous_loop_max_runtime_seconds_per_day == 36000
    assert s.autonomous_loop_max_retries_per_job == 1
    assert s.autonomous_loop_max_concurrent_jobs == 2
    assert s.autonomous_self_coding_max_jobs_per_day == 1
    assert s.autonomous_self_coding_max_runtime_seconds_per_day == 21600
    assert s.life_runtime_enabled is True
    assert s.life_runtime_state_path == "/home/zhvusha/life-runtime"
    assert s.life_runtime_min_tick_interval_seconds == 60
    assert s.life_runtime_max_ticks_per_hour == 6
    assert s.personal_telegram_inbound_enabled is True
    assert s.personal_telegram_inbound_store_path == "/home/zhvusha/inbound.jsonl"
    assert s.personal_telegram_inbound_max_events_per_poll == 5
    assert s.personal_telegram_inbound_auto_reply_enabled is True
    assert s.personal_telegram_inbound_external_max_chars == 240
    assert s.personal_telegram_inbound_external_knowledge_categories == "research,web"


def test_telegram_mcp_settings_defaults_and_secret_masking():
    env = {
        **REQUIRED_ENV,
        "TELEGRAM_MCP_ENABLED": "true",
        "TELEGRAM_MCP_SESSION_STRING_PERSONAL": "personal_session_secret",
    }
    with patch.dict(os.environ, env, clear=True):
        s = _settings_no_env()

    rendered = repr(s)
    rich_repr_args = dict(s.__repr_args__())

    assert s.telegram_mcp_enabled is True
    assert s.telegram_mcp_account_label == "personal"
    assert "personal_session_secret" not in rendered
    assert rich_repr_args["telegram_mcp_session_string_personal"] == "pers***"


def test_chat_agentic_timeout_overridable_from_env():
    env = {**REQUIRED_ENV, "CHAT_AGENTIC_TIMEOUT_SECONDS": "240"}
    with patch.dict(os.environ, env, clear=True):
        s = _settings_no_env()
    assert s.chat_agentic_timeout_seconds == 240.0


def test_code_agent_timeout_overridable_from_env():
    env = {**REQUIRED_ENV, "CODE_AGENT_TIMEOUT_SECONDS": "7200"}
    with patch.dict(os.environ, env, clear=True):
        s = _settings_no_env()
    assert s.code_agent_timeout_seconds == 7200.0


def test_settings_repr_args_mask_secrets_for_tracebacks():
    env = {
        **REQUIRED_ENV,
        "GOOGLE_API_KEY": "google_secret_value",
        "ANTHROPIC_API_KEY": "anthropic_secret_value",
        "OPENROUTER_API_KEY": "openrouter_secret_value",
        "DATABASE_URL": "postgresql+asyncpg://user:password@localhost/db",
    }
    with patch.dict(os.environ, env, clear=True):
        s = _settings_no_env()

    rendered = repr(s)
    rich_repr_args = dict(s.__repr_args__())

    assert "google_secret_value" not in rendered
    assert "anthropic_secret_value" not in rendered
    assert "openrouter_secret_value" not in rendered
    assert "password@localhost" not in rendered
    assert rich_repr_args["google_api_key"] == "goog***"
    assert rich_repr_args["anthropic_api_key"] == "anth***"
    assert rich_repr_args["openrouter_api_key"] == "open***"
    assert rich_repr_args["database_url"] == "post***"


def test_kwork_settings_defaults():
    with patch.dict(os.environ, REQUIRED_ENV, clear=True):
        s = _settings_no_env()
    assert s.kwork_login == ""
    assert s.kwork_password == ""
    assert s.kwork_phone_last == ""
    assert s.kwork_poll_interval_seconds == 300
    assert s.kwork_min_budget == 3000
    assert s.kwork_max_offers == 15
    assert "python" in s.kwork_keywords
    assert "aiogram" in s.kwork_keywords


def test_settings_fails_without_bot_token():
    env = {k: v for k, v in REQUIRED_ENV.items() if k != "BOT_TOKEN"}
    with patch.dict(os.environ, env, clear=True), pytest.raises(ValidationError):
        _settings_no_env()


def test_settings_fails_without_admin_user_id():
    env = {k: v for k, v in REQUIRED_ENV.items() if k != "ADMIN_USER_ID"}
    with patch.dict(os.environ, env, clear=True), pytest.raises(ValidationError):
        _settings_no_env()


def test_settings_rejects_invalid_enrichment_tier():
    env = {**REQUIRED_ENV, "ENRICHMENT_TIER": "mega"}
    with patch.dict(os.environ, env, clear=True), pytest.raises(ValidationError):
        _settings_no_env()


def test_settings_rejects_invalid_dream_extraction_tier():
    env = {**REQUIRED_ENV, "DREAM_EXTRACTION_TIER": "ultra"}
    with patch.dict(os.environ, env, clear=True), pytest.raises(ValidationError):
        _settings_no_env()


def test_settings_rejects_invalid_reasoning_effort():
    env = {**REQUIRED_ENV, "STRATEGIST_REASONING_EFFORT": "maximum"}
    with patch.dict(os.environ, env, clear=True), pytest.raises(ValidationError):
        _settings_no_env()


def test_settings_provider_fields_default():
    """Provider fields default to Codex CLI subscription for text tiers."""
    with patch.dict(os.environ, REQUIRED_ENV, clear=True):
        s = _settings_no_env()
    assert s.worker_provider == "codex_cli"
    assert s.analyst_provider == "codex_cli"
    assert s.strategist_provider == "codex_cli"
    assert s.worker_model == "default"
    assert s.analyst_model == "gpt-5.5"
    assert s.strategist_model == "gpt-5.5"
    assert s.vision_provider == "gemini"


def test_settings_provider_fields_overridable_from_env():
    env = {
        **REQUIRED_ENV,
        "WORKER_PROVIDER": "claude_cli",
        "ANALYST_PROVIDER": "claude_cli",
        "STRATEGIST_PROVIDER": "anthropic_api",
        "VISION_PROVIDER": "gemini",
    }
    with patch.dict(os.environ, env, clear=True):
        s = _settings_no_env()
    assert s.worker_provider == "claude_cli"
    assert s.analyst_provider == "claude_cli"
    assert s.strategist_provider == "anthropic_api"

from functools import lru_cache
from typing import Any, ClassVar, Literal

from pydantic_settings import BaseSettings

Tier = Literal["worker", "analyst", "strategist"]
ReasoningEffort = Literal["low", "medium", "high", "xhigh"]


class Settings(BaseSettings):
    # Telegram
    bot_token: str
    telegram_bot_proxy: str = ""
    channel_id: str
    admin_user_id: int
    bot_restart_enabled: bool = False

    # LLM APIs (optional until Phase 2+)
    google_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""
    nvidia_nim_api_key: str = ""
    huggingface_api_key: str = ""
    minimax_api_key: str = ""
    nvidia_nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    huggingface_base_url: str = "https://router.huggingface.co/v1"
    minimax_base_url: str = "https://api.minimax.io/v1"

    # Database (optional until Phase 2+)
    database_url: str = ""
    redis_url: str = "redis://localhost:6379/0"

    # Kwork
    kwork_login: str = ""
    kwork_password: str = ""
    kwork_phone_last: str = ""
    kwork_poll_interval_seconds: int = 300
    kwork_min_budget: int = 3000
    kwork_max_offers: int = 15
    kwork_keywords: str = (
        "python,aiogram,telegram,bot,next.js,react,typescript,fastapi,ai"
    )

    # Legacy Claude CLI adapter path. Empty by default so CLI automation is
    # opt-in and cannot run accidentally.
    claude_cli_path: str = ""

    # Codex CLI for subscription-backed chat and delegated code-agent work.
    codex_cli_path: str = "codex"
    code_agent_backend: str = "codex_cli"
    code_agent_model: str = ""
    code_agent_reasoning_effort: ReasoningEffort = "xhigh"
    code_agent_timeout_seconds: float = 7200.0

    # Workspace
    workspace_path: str = "~/zhvusha-workspace"
    project_path: str = "~/Projects/ZHVUSHA"
    git_max_commits: int = 100
    morning_session_model: str = "gpt-5.5"
    morning_session_reasoning_effort: ReasoningEffort = "xhigh"
    morning_session_hour: int = 8
    morning_session_enabled: bool = False

    # /compare command — admin-only A/B testing of two LLMs in chat.
    # Empty values disable the command. Provider must exist in
    # src.llm.providers.PROVIDERS; model is an alias or full api_id.
    # compare_main_tier picks the local-side tier (worker/analyst/strategist)
    # — set to "analyst" to A/B Sonnet vs the shadow model.
    compare_main_tier: Tier = "worker"
    compare_provider: str = ""
    compare_model: str = ""

    # Per-user daily message cap for non-admin chats. Bot replies once with
    # a polite refusal and drops further messages from that user for the
    # remainder of the calendar day. Admin (admin_user_id) is never capped.
    # 0 disables the cap (legacy behaviour). Counter lives in Redis; if
    # Redis is unavailable, the cap is treated as disabled (fail-open).
    assistant_daily_message_limit: int = 30

    # Chat
    chat_assistant_tier: Tier = "analyst"
    chat_agentic_timeout_seconds: float = 300.0
    chat_decision_context_timeout_seconds: float = 8.0
    vscode_chat_enabled: bool = True
    vscode_chat_host: str = "127.0.0.1"
    vscode_chat_port: int = 7331
    vscode_chat_token: str = ""
    public_info_about_nikita: str = (
        "Никита — разработчик: Telegram-боты (Python/aiogram 3), "
        "сайты (Next.js/TypeScript), AI-интеграции. "
        "Боты от 5000 ₽, сайты от 7000 ₽, AI от 7000 ₽."
    )
    # Public Telegram handle (starts with @) shown to non-admin on request
    # or at the end of a development-related conversation. Empty = hidden.
    public_contact_nikita: str = ""

    # LLM tier providers — pair with *_model below.
    # Allowed values: "codex_cli" | "anthropic_api" | "gemini" | "openrouter"
    # | "nvidia_nim" | "huggingface" | "minimax" | "claude_cli" (legacy
    # explicit LLM fallback only; never a self-coding backend).
    # Architectural principle №1 (AGENTS.md/CLAUDE.md): swap provider for any tier
    # by changing one line in .env. The mapping (provider, model) → adapter
    # lives in src/llm/providers.py.
    worker_provider: str = "codex_cli"
    analyst_provider: str = "codex_cli"
    strategist_provider: str = "codex_cli"
    vision_provider: str = "gemini"

    # LLM tier models. Either a short alias ("haiku") declared in the
    # provider registry, or the provider's full api_id.
    worker_model: str = "default"
    analyst_model: str = "gpt-5.5"
    strategist_model: str = "gpt-5.5"
    vision_model: str = "gemini-2.5-flash-lite"

    # Reasoning effort is honored by providers that support it (Codex CLI).
    # Other providers ignore the field while preserving the tier contract.
    worker_reasoning_effort: ReasoningEffort = "medium"
    analyst_reasoning_effort: ReasoningEffort = "high"
    strategist_reasoning_effort: ReasoningEffort = "xhigh"

    # Agent
    default_llm_tier: str = "worker"
    strategist_budget_daily_usd: float = 1.00
    enable_browser_use: bool = False
    browser_backend: str = "browser_use"
    browser_http_proxy: str = ""
    image_generation_enabled: bool = False
    image_generation_provider: str = "cli"
    image_generation_model: str = ""
    image_generation_size: str = "1024x1024"
    image_generation_cli_command: str = ""
    image_generation_cli_timeout_seconds: float = 300.0

    # Memory (Phase 2)
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    recency_half_life_personal: int = 48
    recency_half_life_assistant: int = 24
    recency_half_life_social: int = 12
    reconsolidation_window_hours: int = 6

    # Enrichment (Phase 2)
    enrichment_tier: Tier = "worker"
    enrichment_min_length: int = 15
    enrichment_max_concurrent: int = 3
    dream_extraction_tier: Tier = "worker"

    # Decision engine (Phase 2)
    system1_default_threshold: float = 0.7

    # Consolidation (Phase 2)
    consolidation_top_n: int = 300

    # Personality (Phase 2)
    core_md_max_lines: int = 100
    memory_index_max_lines: int = 200
    memory_index_max_kb: int = 25

    # Dashboard
    dashboard_update_interval_seconds: int = 30

    # Browser history (Phase 3)
    firefox_profile_path: str = ""
    chrome_history_path: str = ""

    # YouTube (Phase 3)
    youtube_takeout_path: str = ""
    youtube_api_key: str = ""
    youtube_scan_enabled: bool = False
    youtube_transcribe_top_n: int = 3

    # Daemon (Phase 4)
    daemon_enabled: bool = False
    daemon_agent_runtime_enabled: bool = False
    daemon_max_llm_cost_per_day_usd: float = 5.0
    daemon_max_llm_calls_per_hour: int = 60
    daemon_decision_tier: str = "analyst"
    daemon_sleep_agent_tier: str = "worker"
    life_runtime_enabled: bool = False
    life_runtime_state_path: str = "~/zhvusha-workspace/life_runtime"
    life_runtime_min_tick_interval_seconds: int = 300
    life_runtime_max_ticks_per_hour: int = 12
    voice_gateway_enabled: bool = False
    voice_stt_provider: str = ""
    voice_tts_enabled: bool = False
    desktop_control_enabled: bool = False
    desktop_control_command_map_json: str = ""
    desktop_control_command_timeout_seconds: float = 10.0
    computer_use_enabled: bool = False
    live_browser_control_enabled: bool = False
    live_browser_backend: str = "chrome_devtools_mcp"
    live_browser_debug_url: str = "http://127.0.0.1:9222"
    live_browser_auto_launch: bool = False
    live_browser_executable: str = "chromium"
    live_browser_user_data_dir: str = "~/zhvusha-workspace/live-chrome"
    live_browser_headless: bool = False
    computer_use_irreversible_policy: Literal["hard_stop"] = "hard_stop"
    computer_use_shell_enabled: bool = False
    computer_use_shell_allowed_executables: str = ""
    computer_use_shell_timeout_seconds: float = 10.0

    # Agent SDK delegation
    delegate_enabled: bool = False
    delegate_cwd: str = "~/Projects/ZHVUSHA"
    delegate_timeout_seconds: int = 300
    delegate_max_concurrent: int = 1
    delegate_model: str = ""

    # MCP Server
    mcp_server_port: int = 8765
    telegram_mcp_enabled: bool = False
    telegram_mcp_checkout_path: str = "~/.local/share/zhvusha/telegram-mcp"
    telegram_mcp_uv_path: str = "uv"
    telegram_mcp_account_label: str = "personal"
    telegram_mcp_session_string_personal: str = ""
    telegram_mcp_session_name_personal: str = ""
    telegram_mcp_allowed_roots: str = "~/zhvusha-workspace/telegram-mcp"
    daivinchik_chat_id: str = "@leomatchbot"
    daivinchik_reference_sheets_enabled: bool = True
    daivinchik_liked_face_reference_dir: str = "~/Documents/нравится_лицо"
    daivinchik_disliked_face_reference_dir: str = "~/Documents/ненравятся_лицо"
    daivinchik_liked_body_reference_dir: str = "~/Documents/нравится_тело"
    daivinchik_disliked_body_reference_dir: str = "~/Documents/ненравится_тело"
    personal_telegram_inbound_enabled: bool = False
    personal_telegram_inbound_store_path: str = (
        "~/zhvusha-workspace/telegram/inbound_events.jsonl"
    )
    personal_telegram_inbound_max_events_per_poll: int = 20
    personal_telegram_inbound_auto_reply_enabled: bool = False
    personal_telegram_inbound_external_max_chars: int = 800
    personal_telegram_inbound_external_knowledge_categories: str = (
        "research,intel.channels,intel.youtube"
    )

    # Agency / self-complexification runtime. Disabled by default; this layer
    # can plan, draft and ask for scoped permissions, but does not grant live
    # external side effects by itself.
    agency_runtime_enabled: bool = False
    agency_social_autonomy_enabled: bool = False
    agency_permission_store_path: str = "~/zhvusha-workspace/agency/permissions.jsonl"
    agency_max_jobs_per_day: int = 12
    agency_max_social_messages_per_hour: int = 3

    # Telegram channels (Phase 3)
    telegram_api_id: int = 0
    telegram_api_hash: str = ""
    telethon_session_path: str = "~/.zhvusha_telethon.session"
    monitored_channel_ids: str = ""
    channel_read_delay_seconds: float = 1.5

    # News/topic pipeline (Phase 16-21)
    news_sources_enabled: bool = False
    news_poll_interval_seconds: int = 3600
    news_raw_stream: str = "news:raw"
    news_arxiv_rss_url: str = "https://rss.arxiv.org/rss/cs.AI+cs.CL+cs.SE"
    # Comma-separated RSS/Atom URLs for Tier B/C sources. Source metadata is
    # supplied by the collector wrapper; empty keeps the pipeline disabled.
    news_rss_urls: str = ""
    pillars_path: str = "~/zhvusha-workspace/personality/pillars.md"

    # === v4 self-coding controls (phase 1) ===
    # Off by default on phase 1 — Жвуша не должна пытаться кодить сама,
    # пока вся v4 инфраструктура не готова (KB #69, #82).
    self_coding_enabled: bool = False
    self_coding_max_tier: int = 1
    self_coding_require_approval: bool = True
    self_coding_audit_enabled: bool = True
    # Optional anti-loop caps for ImplementSpecSkill. ``0`` disables the
    # corresponding cap; safety still comes from approval, tier, whitelist,
    # env guard, tests and commit gates.
    self_coding_caps_per_hour: int = 0
    self_coding_caps_per_day: int = 0
    code_agent_max_concurrent: int = 1

    # Autonomous self-work loop. Disabled by default; when enabled, Жвуша
    # periodically creates an Agent Runtime self-improvement job that can
    # generate, self-approve and run bounded specs through the existing gates.
    autonomous_self_coding_enabled: bool = False
    autonomous_self_coding_interval_seconds: int = 21600
    autonomous_self_coding_initial_delay_seconds: int = 300
    autonomous_self_coding_state_path: str = ""
    autonomous_self_coding_restart_throttle_seconds: int = 21600
    autonomous_self_coding_morning_guard_enabled: bool = True
    autonomous_self_coding_user_idle_seconds: int = 7200
    autonomous_self_coding_user_activity_path: str = ""
    autonomous_self_coding_max_tier: int = 3
    autonomous_loop_budget_daily_usd: float = 1.0
    autonomous_loop_budget_weekly_usd: float = 5.0
    autonomous_self_coding_budget_daily_usd: float = 2.0
    autonomous_self_coding_budget_weekly_usd: float = 8.0
    autonomous_day_window_start_hour: int = 10
    autonomous_day_window_duration_hours: int = 12
    autonomous_loop_max_jobs_per_day: int = 12
    autonomous_loop_max_runtime_seconds_per_day: int = 43200
    autonomous_loop_max_retries_per_job: int = 2
    autonomous_loop_max_concurrent_jobs: int = 1
    autonomous_self_coding_max_jobs_per_day: int = 2
    autonomous_self_coding_max_runtime_seconds_per_day: int = 43200

    # extra='ignore' lets us retire deprecated settings without breaking
    # existing .env files (Pydantic v2 defaults to strict 'forbid').
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    _SECRET_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "bot_token",
            "google_api_key",
            "openai_api_key",
            "anthropic_api_key",
            "openrouter_api_key",
            "nvidia_nim_api_key",
            "huggingface_api_key",
            "minimax_api_key",
            "database_url",
            "redis_url",
            "kwork_login",
            "kwork_password",
            "kwork_phone_last",
            "youtube_api_key",
            "telegram_api_id",
            "telegram_api_hash",
            "telegram_mcp_session_string_personal",
        }
    )

    @classmethod
    def _repr_value(cls, name: str, value: Any) -> Any:
        if name in cls._SECRET_FIELDS and value:
            return f"{str(value)[:4]}***"
        return value

    def __repr_args__(self) -> list[tuple[str | None, Any]]:
        return [
            (name, self._repr_value(name, getattr(self, name)))
            for name in Settings.model_fields
        ]

    def __repr__(self) -> str:
        fields = []
        for name in Settings.model_fields:
            value = getattr(self, name)
            masked = self._repr_value(name, value)
            if name not in self._SECRET_FIELDS or not value:
                masked = repr(masked)
            fields.append(f"{name}={masked}")
        return f"Settings({', '.join(fields)})"

    def __str__(self) -> str:
        return self.__repr__()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

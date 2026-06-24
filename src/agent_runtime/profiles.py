"""Built-in Agent Runtime profiles."""

from __future__ import annotations

from src.agent_runtime.models import (
    AgentDefinition,
    CapabilityDefinition,
    InvocationProfile,
)
from src.agent_runtime.registry import AgentRegistry, CapabilityRegistry

BUILTIN_CAPABILITIES: tuple[CapabilityDefinition, ...] = (
    CapabilityDefinition(id="read_code", risk="low"),
    CapabilityDefinition(id="read_workspace", risk="low"),
    CapabilityDefinition(id="read_attachments", risk="low"),
    CapabilityDefinition(id="run_readonly_commands", risk="medium"),
    CapabilityDefinition(id="web_search_sources", risk="medium"),
    CapabilityDefinition(id="browser_read", risk="medium"),
    CapabilityDefinition(id="browser_screenshot", risk="medium"),
    CapabilityDefinition(id="browser_download", risk="medium"),
    CapabilityDefinition(id="channel_visual_image_generation", risk="medium"),
    CapabilityDefinition(id="browser_draft_form", risk="high"),
    CapabilityDefinition(id="browser_submit", risk="high", requires_approval=True),
    CapabilityDefinition(
        id="browser_live_control",
        description="Inspect and control an explicitly attachable live Chrome session.",
        risk="high",
    ),
    CapabilityDefinition(
        id="browser_navigate",
        description="Navigate an attachable live browser tab under an active computer-use grant.",
        risk="medium",
    ),
    CapabilityDefinition(
        id="browser_click",
        description="Click reversible UI targets in an attachable live browser session.",
        risk="high",
    ),
    CapabilityDefinition(
        id="browser_type",
        description="Type text into an attachable live browser session without final submit.",
        risk="high",
    ),
    CapabilityDefinition(
        id="browser_scroll",
        description="Scroll an attachable live browser session.",
        risk="medium",
    ),
    CapabilityDefinition(
        id="browser_tab_control",
        description="Switch or open tabs in an attachable live browser session.",
        risk="medium",
    ),
    CapabilityDefinition(
        id="browser_form_draft",
        description="Draft browser form values without final submit.",
        risk="high",
    ),
    CapabilityDefinition(
        id="browser_interactive_task",
        description=(
            "Complete bounded interactive web tasks in an isolated live browser, "
            "using task metadata and personality references without copying "
            "personality content into action payloads."
        ),
        risk="high",
    ),
    CapabilityDefinition(
        id="desktop_input",
        description="Send bounded GUI input without shell or terminal execution.",
        risk="high",
    ),
    CapabilityDefinition(
        id="desktop_window_control",
        description="Focus, move or close windows through GUI adapters.",
        risk="high",
    ),
    CapabilityDefinition(
        id="desktop_screenshot",
        description="Capture a local desktop screenshot for the active computer-use run.",
        risk="medium",
    ),
    CapabilityDefinition(
        id="desktop_app_launcher",
        description="Launch or focus a desktop application without shell execution.",
        risk="medium",
    ),
    CapabilityDefinition(
        id="desktop_hotkeys",
        description="Send bounded GUI hotkeys under an active computer-use grant.",
        risk="high",
    ),
    CapabilityDefinition(
        id="desktop_media_control",
        description="Control media playback through bounded GUI/media adapters.",
        risk="low",
        requires_approval=True,
    ),
    CapabilityDefinition(
        id="login",
        description="Authenticate in a browser or remote service after explicit approval.",
        risk="high",
        requires_approval=True,
    ),
    CapabilityDefinition(
        id="purchase",
        description="Perform a purchase or checkout action after explicit approval.",
        risk="high",
        requires_approval=True,
    ),
    CapabilityDefinition(
        id="delete",
        description="Delete remote or local user-visible data after explicit approval.",
        risk="high",
        requires_approval=True,
    ),
    CapabilityDefinition(id="write_files", risk="high", requires_approval=True),
    CapabilityDefinition(id="write_whitelisted_files_after_approval", risk="high"),
    CapabilityDefinition(id="run_tests", risk="medium"),
    CapabilityDefinition(id="commit", risk="high", requires_approval=True),
    CapabilityDefinition(id="commit_after_gate", risk="high"),
    CapabilityDefinition(id="self_approve_low_risk_specs", risk="high"),
    CapabilityDefinition(id="request_tier3_specs_for_nikita_approval", risk="high"),
    CapabilityDefinition(id="edit_env", risk="high", requires_approval=True),
    CapabilityDefinition(id="restart", risk="high", requires_approval=True),
    CapabilityDefinition(id="publish", risk="high", requires_approval=True),
    CapabilityDefinition(id="send_message", risk="high", requires_approval=True),
    CapabilityDefinition(id="telegram_mcp_read", risk="high"),
    CapabilityDefinition(
        id="telegram_mcp_send",
        risk="high",
        requires_approval=True,
    ),
    CapabilityDefinition(
        id="telegram_mcp_modify",
        risk="high",
        requires_approval=True,
    ),
    CapabilityDefinition(
        id="telegram_mcp_admin",
        risk="high",
        requires_approval=True,
    ),
    CapabilityDefinition(
        id="telegram_mcp_media_files",
        risk="high",
        requires_approval=True,
    ),
    CapabilityDefinition(
        id="telegram_mcp_media_read",
        description="Read Telegram media metadata and download media to bounded temp artifacts.",
        risk="medium",
    ),
    CapabilityDefinition(
        id="telegram_mcp_daivinchik_button",
        description="Press only the selected Daivinchik like/skip inline button.",
        risk="high",
        requires_approval=True,
    ),
    CapabilityDefinition(
        id="telegram_mcp_daivinchik_reply_button",
        description="Send only whitelisted Daivinchik reply-keyboard button text.",
        risk="high",
        requires_approval=True,
    ),
    CapabilityDefinition(
        id="telegram_mcp_daivinchik_notify",
        description="Send only bounded Daivinchik stop/attention notifications to Никита.",
        risk="high",
        requires_approval=True,
    ),
    CapabilityDefinition(
        id="telegram_mcp_daivinchik_forward_liked_profile",
        description="Forward only messages from a liked Daivinchik profile to Никита.",
        risk="high",
        requires_approval=True,
    ),
    CapabilityDefinition(
        id="agency_intent_plan",
        description="Plan self-complexification work without external side effects.",
        risk="low",
    ),
    CapabilityDefinition(
        id="agency_social_permission_request",
        description="Draft a scoped social permission request for Никита.",
        risk="medium",
    ),
    CapabilityDefinition(
        id="agency_social_judgement",
        description="Evaluate whether to speak, wait, draft or stay silent.",
        risk="medium",
    ),
    CapabilityDefinition(
        id="agency_stage_memory",
        description="Return memory candidates through Agent Runtime staging.",
        risk="medium",
    ),
    CapabilityDefinition(
        id="life_reflection",
        description="Produce a read-only LifeRuntime reflection capsule.",
        risk="low",
    ),
    CapabilityDefinition(
        id="life_stage_memory_candidate",
        description="Return LifeRuntime memory candidates for staging only.",
        risk="medium",
    ),
    CapabilityDefinition(
        id="external_skill_readonly",
        description=(
            "Use an approved external skill only as untrusted read-only procedural "
            "context through Agent Runtime."
        ),
        risk="medium",
    ),
    CapabilityDefinition(
        id="external_skill_execute",
        description=(
            "Execute one approved external-skill-derived tool call through "
            "ToolGateway after scoped execution approval."
        ),
        risk="high",
        requires_approval=True,
    ),
    CapabilityDefinition(
        id="desktop.app_launcher",
        description="Launch or focus a desktop application through the narrow desktop pack.",
        risk="medium",
        requires_approval=True,
    ),
    CapabilityDefinition(
        id="desktop.media_control",
        description="Control media playback, volume or player state.",
        risk="low",
        requires_approval=True,
    ),
    CapabilityDefinition(
        id="desktop.window_control",
        description="Move, focus or close a desktop window.",
        risk="medium",
        requires_approval=True,
    ),
    CapabilityDefinition(
        id="desktop.browser_open",
        description="Open a URL or browser tab without submitting forms.",
        risk="medium",
        requires_approval=True,
    ),
    CapabilityDefinition(
        id="desktop.screenshot",
        description="Capture a local screenshot for bounded inspection.",
        risk="medium",
        requires_approval=True,
    ),
    CapabilityDefinition(
        id="desktop.hotkeys",
        description="Send a bounded desktop hotkey sequence.",
        risk="high",
        requires_approval=True,
    ),
    CapabilityDefinition(
        id="desktop.system_power",
        description="Sleep, lock, restart or power-control the local desktop.",
        risk="high",
        requires_approval=True,
    ),
    CapabilityDefinition(
        id="desktop.shell",
        description=(
            "Approved structured argv command execution outside the Desktop "
            "Control convenience pack."
        ),
        risk="high",
        requires_approval=True,
    ),
    CapabilityDefinition(
        id="desktop.powershell",
        description="Explicitly outside the Desktop Control convenience pack.",
        risk="high",
        requires_approval=True,
    ),
)

SOURCE_COMPARE_AGENT = AgentDefinition(
    id="source_compare",
    purpose="Сравнить пост, ссылку, фото или вложение с проектным контекстом.",
    default_worker="source_compare",
    allowed_capabilities=(
        "read_code",
        "read_workspace",
        "read_attachments",
        "run_readonly_commands",
        "web_search_sources",
        "browser_read",
        "browser_screenshot",
        "browser_download",
    ),
    safety_policy="readonly_research.v1",
)

SOURCE_COMPARE_READONLY = InvocationProfile(
    id="source_compare.readonly",
    worker="source_compare",
    allowed_capabilities=SOURCE_COMPARE_AGENT.allowed_capabilities,
    denied_capabilities=(
        "write_files",
        "edit_env",
        "commit",
        "restart",
        "publish",
        "browser_submit",
    ),
)

SELF_CODING_AGENT = AgentDefinition(
    id="self_coding",
    purpose="Spec-first self-coding flow for /код and /code.",
    default_worker="codex_cli",
    allowed_capabilities=(
        "read_code",
        "read_workspace",
        "read_attachments",
        "run_readonly_commands",
        "write_whitelisted_files_after_approval",
        "run_tests",
        "commit_after_gate",
    ),
    safety_policy="self_coding_spec_first.v1",
)

SELF_CODING_READONLY = InvocationProfile(
    id="self_coding.readonly_discussion",
    worker="codex_cli",
    allowed_capabilities=(
        "read_code",
        "read_workspace",
        "read_attachments",
        "run_readonly_commands",
    ),
    denied_capabilities=(
        "write_files",
        "edit_env",
        "commit",
        "restart",
        "publish",
        "browser_submit",
    ),
    metadata={"phase": "discussion"},
)

SELF_CODING_IMPLEMENTATION = InvocationProfile(
    id="self_coding.implementation",
    worker="self_coding_native",
    allowed_capabilities=(
        "read_code",
        "read_workspace",
        "read_attachments",
        "run_readonly_commands",
        "write_whitelisted_files_after_approval",
        "run_tests",
        "commit_after_gate",
        "edit_env",
    ),
    denied_capabilities=(
        "restart",
        "publish",
        "browser_submit",
        "send_message",
    ),
    metadata={"phase": "implementation", "backend": "codex_cli"},
)

SELF_IMPROVEMENT_AGENT = AgentDefinition(
    id="self_improvement",
    purpose=(
        "Autonomous self-work loop: discover one bounded improvement, create a "
        "spec, self-approve only non-Tier-3 specs within the configured mandate, "
        "and leave Tier 3 specs for Никита's chat approval."
    ),
    default_worker="self_improvement",
    allowed_capabilities=(
        "read_code",
        "read_workspace",
        "read_attachments",
        "run_readonly_commands",
        "web_search_sources",
        "browser_read",
        "self_approve_low_risk_specs",
        "request_tier3_specs_for_nikita_approval",
        "write_whitelisted_files_after_approval",
        "run_tests",
        "commit_after_gate",
    ),
    safety_policy="self_improvement_autonomous_bounded.v1",
)

SELF_IMPROVEMENT_AUTONOMOUS = InvocationProfile(
    id="self_improvement.autonomous",
    worker="self_improvement",
    allowed_capabilities=SELF_IMPROVEMENT_AGENT.allowed_capabilities,
    denied_capabilities=(
        "write_files",
        "edit_env",
        "restart",
        "publish",
        "browser_submit",
        "send_message",
        "commit",
    ),
    metadata={"phase": "autonomous_self_work", "backend": "codex_cli"},
)

AGENCY_AGENT = AgentDefinition(
    id="agency",
    purpose=(
        "Self-complexification planner with scoped permission grants and social "
        "judgement. It may prepare Context Capsules, drafts and permission asks; "
        "live external actions stay behind explicit capabilities and policy."
    ),
    default_worker="agency",
    allowed_capabilities=(
        "read_workspace",
        "browser_read",
        "agency_intent_plan",
        "agency_social_permission_request",
        "agency_social_judgement",
        "agency_stage_memory",
    ),
    safety_policy="agency_permissioned_social_judgement.v1",
)

AGENCY_READONLY_DRAFT = InvocationProfile(
    id="agency.readonly_draft",
    worker="agency",
    allowed_capabilities=AGENCY_AGENT.allowed_capabilities,
    denied_capabilities=(
        "write_files",
        "write_whitelisted_files_after_approval",
        "commit",
        "commit_after_gate",
        "edit_env",
        "restart",
        "publish",
        "browser_submit",
        "send_message",
        "telegram_mcp_send",
        "telegram_mcp_modify",
        "telegram_mcp_admin",
        "telegram_mcp_media_files",
    ),
    metadata={"phase": "agency_planning", "side_effects": "disabled"},
)

LIFE_RUNTIME_AGENT = AgentDefinition(
    id="life_runtime",
    purpose=(
        "Durable read-only inner-life runtime: state, attention, drives, "
        "reflection proposals and memory candidates without external side effects."
    ),
    default_worker="life_runtime",
    allowed_capabilities=(
        "read_workspace",
        "life_reflection",
        "life_stage_memory_candidate",
    ),
    safety_policy="life_runtime_readonly.v1",
)

LIFE_REFLECTION_READONLY = InvocationProfile(
    id="life_reflection.readonly",
    worker="life_runtime",
    allowed_capabilities=(
        "read_workspace",
        "life_reflection",
    ),
    denied_capabilities=(
        "write_files",
        "write_whitelisted_files_after_approval",
        "commit",
        "commit_after_gate",
        "edit_env",
        "restart",
        "publish",
        "browser_submit",
        "send_message",
        "telegram_mcp_send",
        "telegram_mcp_modify",
        "telegram_mcp_admin",
        "telegram_mcp_media_files",
    ),
    metadata={"phase": "life_runtime_reflection", "side_effects": "disabled"},
)

LIFE_PROPOSAL_READONLY = InvocationProfile(
    id="life_proposal.readonly",
    worker="life_runtime",
    allowed_capabilities=LIFE_RUNTIME_AGENT.allowed_capabilities,
    denied_capabilities=LIFE_REFLECTION_READONLY.denied_capabilities,
    metadata={"phase": "life_runtime_proposal", "side_effects": "disabled"},
)

WEB_RESEARCH_AGENT = AgentDefinition(
    id="web_research",
    purpose="Read-only web/browser research for future source-backed answers.",
    default_worker="web_research",
    allowed_capabilities=(
        "web_search_sources",
        "browser_read",
        "browser_screenshot",
        "browser_download",
    ),
    safety_policy="web_readonly.v1",
)

WEB_RESEARCH_READONLY = InvocationProfile(
    id="web_research.readonly",
    worker="web_research",
    allowed_capabilities=WEB_RESEARCH_AGENT.allowed_capabilities,
    denied_capabilities=(
        "browser_draft_form",
        "browser_submit",
        "write_whitelisted_files_after_approval",
        "edit_env",
        "restart",
        "publish",
    ),
)

BROWSER_WORKFLOW_AGENT = AgentDefinition(
    id="browser_workflow",
    purpose=(
        "Read a browser form page and prepare a local draft artifact without "
        "submitting, logging in, publishing, sending or purchasing."
    ),
    default_worker="browser_workflow",
    allowed_capabilities=(
        "browser_read",
        "browser_draft_form",
    ),
    safety_policy="browser_workflow_draft_only.v1",
)

BROWSER_WORKFLOW_DRAFT = InvocationProfile(
    id="browser_workflow.draft",
    worker="browser_workflow",
    allowed_capabilities=BROWSER_WORKFLOW_AGENT.allowed_capabilities,
    denied_capabilities=(
        "browser_submit",
        "login",
        "purchase",
        "publish",
        "delete",
        "send_message",
        "write_files",
        "write_whitelisted_files_after_approval",
        "edit_env",
        "restart",
        "commit",
        "commit_after_gate",
        "telegram_mcp_send",
        "telegram_mcp_modify",
        "telegram_mcp_admin",
        "telegram_mcp_media_files",
    ),
    metadata={"mode": "draft_only", "side_effects": "local_draft_artifact"},
)

CHANNEL_VISUAL_AGENT = AgentDefinition(
    id="channel_visual",
    purpose="Prepare optional visual artifacts for approved channel post drafts.",
    default_worker="channel_visual",
    allowed_capabilities=(
        "read_workspace",
        "browser_screenshot",
        "browser_download",
        "channel_visual_image_generation",
    ),
    safety_policy="channel_visual_readonly_artifacts.v1",
)

CHANNEL_VISUAL_READONLY = InvocationProfile(
    id="channel_visual.readonly_artifacts",
    worker="channel_visual",
    allowed_capabilities=CHANNEL_VISUAL_AGENT.allowed_capabilities,
    denied_capabilities=(
        "write_files",
        "edit_env",
        "commit",
        "restart",
        "publish",
        "browser_submit",
        "send_message",
    ),
)

EXTERNAL_SKILL_AGENT = AgentDefinition(
    id="external_skill",
    purpose=(
        "Use approved Hermes/agentskills.io-compatible external skills as "
        "untrusted procedural context. Read-only use exposes only procedure; "
        "execution is a separate scoped ToolGateway call after approval. No "
        "scripts, tools, shell, browser submit, Telegram, file writes or memory "
        "writes are owned by the external skill."
    ),
    default_worker="external_skill",
    allowed_capabilities=(
        "external_skill_readonly",
        "external_skill_execute",
        "read_workspace",
        "run_readonly_commands",
        "web_search_sources",
        "browser_read",
        "browser_screenshot",
        "browser_download",
        "browser_draft_form",
        "browser_submit",
        "login",
        "purchase",
        "publish",
        "delete",
        "send_message",
        "telegram_mcp_read",
        "telegram_mcp_send",
        "write_whitelisted_files_after_approval",
    ),
    safety_policy="external_skill_untrusted_procedural_input.v1",
)

EXTERNAL_SKILL_READONLY = InvocationProfile(
    id="external_skill.readonly",
    worker="external_skill",
    allowed_capabilities=("external_skill_readonly",),
    denied_capabilities=(
        "write_files",
        "write_whitelisted_files_after_approval",
        "commit",
        "commit_after_gate",
        "edit_env",
        "restart",
        "publish",
        "browser_submit",
        "login",
        "purchase",
        "delete",
        "send_message",
        "telegram_mcp_send",
        "telegram_mcp_modify",
        "telegram_mcp_admin",
        "telegram_mcp_media_files",
    ),
    metadata={"mode": "read_only_procedural_input"},
)

EXTERNAL_SKILL_EXECUTION_BASE = InvocationProfile(
    id="external_skill.execution_base",
    worker="external_skill",
    allowed_capabilities=(
        "external_skill_readonly",
        "external_skill_execute",
    ),
    denied_capabilities=(
        "write_files",
        "write_whitelisted_files_after_approval",
        "commit",
        "commit_after_gate",
        "edit_env",
        "restart",
        "publish",
        "browser_submit",
        "login",
        "purchase",
        "delete",
        "send_message",
        "telegram_mcp_send",
        "telegram_mcp_modify",
        "telegram_mcp_admin",
        "telegram_mcp_media_files",
    ),
    metadata={"mode": "execution_requires_dynamic_scoped_tool_capabilities"},
)

DESKTOP_CONTROL_AGENT = AgentDefinition(
    id="desktop_control",
    purpose=(
        "Narrow desktop convenience actions selected by Жвуша and executed only "
        "through ToolGateway with scoped approval. Shells and broad computer "
        "ownership are outside this pack."
    ),
    default_worker="desktop_control",
    allowed_capabilities=(
        "desktop_app_launcher",
        "desktop_media_control",
        "desktop_window_control",
        "desktop_screenshot",
        "desktop_hotkeys",
    ),
    safety_policy="desktop_control_narrow_approval_gated.v1",
)

DESKTOP_CONTROL_CONVENIENCE = InvocationProfile(
    id="desktop_control.convenience",
    worker="desktop_control",
    allowed_capabilities=DESKTOP_CONTROL_AGENT.allowed_capabilities,
    denied_capabilities=(
        "desktop.shell",
        "desktop.powershell",
        "write_files",
        "write_whitelisted_files_after_approval",
        "commit",
        "commit_after_gate",
        "edit_env",
        "restart",
        "publish",
        "browser_submit",
        "send_message",
        "telegram_mcp_send",
        "telegram_mcp_modify",
        "telegram_mcp_admin",
        "telegram_mcp_media_files",
    ),
    metadata={
        "mode": "desktop_convenience",
        "dialogue_owner": "zhvusha",
        "side_effects": "tool_gateway_approval_required",
    },
)

COMPUTER_USE_AGENT = AgentDefinition(
    id="computer_use",
    purpose=(
        "Live browser and local GUI computer-use worker. Жвуша remains the "
        "orchestrator; this worker returns structured observations, artifacts "
        "and hard-stop conditions instead of final chat text."
    ),
    default_worker="computer_use",
    allowed_capabilities=(
        "browser_live_control",
        "browser_navigate",
        "browser_click",
        "browser_type",
        "browser_scroll",
        "browser_tab_control",
        "browser_form_draft",
        "browser_interactive_task",
        "browser_submit",
        "desktop_input",
        "desktop_window_control",
        "desktop_screenshot",
        "desktop_app_launcher",
        "desktop_hotkeys",
        "desktop_media_control",
        "login",
        "purchase",
        "publish",
        "delete",
        "send_message",
        "desktop.shell",
    ),
    safety_policy="computer_use_active_gui_scoped_approval.v1",
)

COMPUTER_USE_ACTIVE_GUI = InvocationProfile(
    id="computer_use.active_gui",
    worker="computer_use",
    allowed_capabilities=COMPUTER_USE_AGENT.allowed_capabilities,
    denied_capabilities=(
        "desktop.shell",
        "desktop.powershell",
        "write_files",
        "write_whitelisted_files_after_approval",
        "commit",
        "commit_after_gate",
        "edit_env",
        "restart",
        "telegram_mcp_send",
        "telegram_mcp_modify",
        "telegram_mcp_admin",
        "telegram_mcp_media_files",
    ),
    metadata={
        "mode": "active_gui",
        "backend": "chrome_devtools_mcp",
        "irreversible_policy": "scoped_approval",
        "shell": "disabled",
        "dialogue_owner": "zhvusha",
    },
)

COMPUTER_USE_APPROVED_SHELL = InvocationProfile(
    id="computer_use.approved_shell",
    worker="computer_use",
    allowed_capabilities=("desktop.shell",),
    denied_capabilities=(
        "browser_live_control",
        "browser_navigate",
        "browser_click",
        "browser_type",
        "browser_scroll",
        "browser_tab_control",
        "browser_form_draft",
        "browser_interactive_task",
        "browser_submit",
        "login",
        "purchase",
        "publish",
        "delete",
        "send_message",
        "desktop_input",
        "desktop_window_control",
        "desktop_screenshot",
        "desktop_app_launcher",
        "desktop_hotkeys",
        "desktop_media_control",
        "desktop.powershell",
        "write_files",
        "write_whitelisted_files_after_approval",
        "commit",
        "commit_after_gate",
        "edit_env",
        "restart",
        "telegram_mcp_send",
        "telegram_mcp_modify",
        "telegram_mcp_admin",
        "telegram_mcp_media_files",
    ),
    metadata={
        "mode": "approved_shell",
        "shell": "structured_argv_only",
        "dialogue_owner": "zhvusha",
    },
)

TELEGRAM_MCP_PERSONAL_AGENT = AgentDefinition(
    id="telegram_mcp_personal",
    purpose=(
        "Personal Telegram account access through chigwell/telegram-mcp "
        "and Telethon, separate from the Bot API path."
    ),
    default_worker="telegram_mcp",
    allowed_capabilities=(
        "telegram_mcp_read",
        "telegram_mcp_send",
        "telegram_mcp_modify",
        "telegram_mcp_admin",
        "telegram_mcp_media_files",
    ),
    safety_policy="telegram_mcp_personal_account.v1",
)

TELEGRAM_MCP_PERSONAL_READONLY = InvocationProfile(
    id="telegram_mcp.personal_readonly",
    worker="telegram_mcp",
    allowed_capabilities=("telegram_mcp_read",),
    denied_capabilities=(
        "telegram_mcp_send",
        "telegram_mcp_modify",
        "telegram_mcp_admin",
        "telegram_mcp_media_files",
        "send_message",
        "publish",
        "browser_submit",
        "edit_env",
        "restart",
    ),
    metadata={
        "account": "personal",
        "server": "chigwell/telegram-mcp",
        "transport": "stdio",
    },
)

TELEGRAM_MCP_PERSONAL_ACTIONS = InvocationProfile(
    id="telegram_mcp.personal_actions",
    worker="telegram_mcp",
    allowed_capabilities=TELEGRAM_MCP_PERSONAL_AGENT.allowed_capabilities,
    denied_capabilities=(
        "publish",
        "browser_submit",
        "edit_env",
        "restart",
    ),
    metadata={
        "account": "personal",
        "server": "chigwell/telegram-mcp",
        "transport": "stdio",
    },
)

DAIVINCHIK_TASTE_PROFILE_AGENT = AgentDefinition(
    id="daivinchik_taste_profile",
    purpose=(
        "Read/media-only profiling pass over Daivinchik Telegram history. "
        "Builds a private aggregate taste profile without pressing buttons, "
        "sending messages, modifying chats or persisting raw media."
    ),
    default_worker="daivinchik_taste_profile",
    allowed_capabilities=(
        "telegram_mcp_read",
        "telegram_mcp_media_read",
        "run_readonly_commands",
    ),
    safety_policy="telegram_mcp_daivinchik_profile_readonly.v1",
)

DAIVINCHIK_TASTE_PROFILE_READONLY = InvocationProfile(
    id="telegram_mcp.daivinchik_taste_profile",
    worker="daivinchik_taste_profile",
    allowed_capabilities=DAIVINCHIK_TASTE_PROFILE_AGENT.allowed_capabilities,
    denied_capabilities=(
        "telegram_mcp_send",
        "telegram_mcp_modify",
        "telegram_mcp_admin",
        "telegram_mcp_media_files",
        "send_message",
        "publish",
        "browser_submit",
        "edit_env",
        "restart",
        "write_files",
        "commit",
    ),
    metadata={
        "account": "personal",
        "server": "chigwell/telegram-mcp",
        "transport": "stdio",
        "vision": "terminal_codex_exec_image",
        "autolike_decision": "current_card_like_skip_manual_no_button_press",
        "media_retention": "temp_deleted_after_report",
        "attention_policy": "collect_or_stop_non_profile_messages",
        "output": "~/zhvusha-workspace/social/daivinchik/taste_profile.md",
    },
)

DAIVINCHIK_AUTOLIKE_AGENT = AgentDefinition(
    id="daivinchik_autolike",
    purpose=(
        "Bounded Daivinchik live MVP: read the current card, score it against "
        "the learned taste profile, press a like/skip inline button, and stop "
        "with a notification on non-profile/verification/service messages."
    ),
    default_worker="daivinchik_taste_profile",
    allowed_capabilities=(
        "telegram_mcp_read",
        "telegram_mcp_media_read",
        "run_readonly_commands",
        "telegram_mcp_daivinchik_button",
        "telegram_mcp_daivinchik_reply_button",
        "telegram_mcp_daivinchik_notify",
        "telegram_mcp_daivinchik_forward_liked_profile",
    ),
    safety_policy="telegram_mcp_daivinchik_autolike_mvp.v1",
)

DAIVINCHIK_AUTOLIKE_MVP = InvocationProfile(
    id="telegram_mcp.daivinchik_autolike_mvp",
    worker="daivinchik_taste_profile",
    allowed_capabilities=DAIVINCHIK_AUTOLIKE_AGENT.allowed_capabilities,
    denied_capabilities=(
        "telegram_mcp_send",
        "telegram_mcp_modify",
        "telegram_mcp_admin",
        "telegram_mcp_media_files",
        "send_message",
        "publish",
        "browser_submit",
        "edit_env",
        "restart",
        "write_files",
        "commit",
    ),
    metadata={
        "account": "personal",
        "server": "chigwell/telegram-mcp",
        "transport": "stdio",
        "vision": "terminal_codex_exec_image",
        "classifier": "terminal_codex_exec_text_low_effort",
        "scope": "bounded_current_card_loop",
        "button_policy": "only_like_skip_inline_button",
        "attention_policy": "stop_and_notify_non_profile_messages",
        "audit": "~/zhvusha-workspace/social/daivinchik/autolike_live.jsonl",
    },
)

DAIVINCHIK_AUTOLIKE_BOT_COMMAND = InvocationProfile(
    id="telegram_mcp.daivinchik_autolike_bot_command",
    worker="daivinchik_taste_profile",
    allowed_capabilities=(
        "telegram_mcp_read",
        "telegram_mcp_media_read",
        "run_readonly_commands",
        "telegram_mcp_daivinchik_button",
        "telegram_mcp_daivinchik_reply_button",
        "telegram_mcp_daivinchik_notify",
        "telegram_mcp_daivinchik_forward_liked_profile",
    ),
    denied_capabilities=(
        "telegram_mcp_send",
        "telegram_mcp_modify",
        "telegram_mcp_admin",
        "telegram_mcp_media_files",
        "send_message",
        "publish",
        "browser_submit",
        "edit_env",
        "restart",
        "write_files",
        "commit",
    ),
    metadata={
        "account": "personal",
        "server": "chigwell/telegram-mcp",
        "transport": "stdio",
        "vision": "terminal_codex_exec_image",
        "classifier": "terminal_codex_exec_text_low_effort",
        "scope": "bounded_current_card_loop_from_zhvusha_command",
        "button_policy": "only_like_skip_inline_button",
        "attention_policy": "stop_without_personal_telegram_send",
        "notify_policy": "bot_chat_only",
        "liked_forward_policy": "forward_liked_daivinchik_profile_to_nikita",
        "audit": "~/zhvusha-workspace/social/daivinchik/autolike_live.jsonl",
    },
)

BUILTIN_INVOCATION_PROFILES: tuple[InvocationProfile, ...] = (
    SOURCE_COMPARE_READONLY,
    SELF_CODING_READONLY,
    SELF_CODING_IMPLEMENTATION,
    SELF_IMPROVEMENT_AUTONOMOUS,
    AGENCY_READONLY_DRAFT,
    LIFE_REFLECTION_READONLY,
    LIFE_PROPOSAL_READONLY,
    WEB_RESEARCH_READONLY,
    BROWSER_WORKFLOW_DRAFT,
    CHANNEL_VISUAL_READONLY,
    EXTERNAL_SKILL_READONLY,
    EXTERNAL_SKILL_EXECUTION_BASE,
    DESKTOP_CONTROL_CONVENIENCE,
    COMPUTER_USE_ACTIVE_GUI,
    COMPUTER_USE_APPROVED_SHELL,
    TELEGRAM_MCP_PERSONAL_READONLY,
    TELEGRAM_MCP_PERSONAL_ACTIONS,
    DAIVINCHIK_TASTE_PROFILE_READONLY,
    DAIVINCHIK_AUTOLIKE_MVP,
    DAIVINCHIK_AUTOLIKE_BOT_COMMAND,
)

BUILTIN_AGENTS: tuple[AgentDefinition, ...] = (
    SOURCE_COMPARE_AGENT,
    SELF_CODING_AGENT,
    SELF_IMPROVEMENT_AGENT,
    AGENCY_AGENT,
    LIFE_RUNTIME_AGENT,
    WEB_RESEARCH_AGENT,
    BROWSER_WORKFLOW_AGENT,
    CHANNEL_VISUAL_AGENT,
    EXTERNAL_SKILL_AGENT,
    DESKTOP_CONTROL_AGENT,
    COMPUTER_USE_AGENT,
    TELEGRAM_MCP_PERSONAL_AGENT,
    DAIVINCHIK_TASTE_PROFILE_AGENT,
    DAIVINCHIK_AUTOLIKE_AGENT,
)


def build_builtin_agent_registry() -> AgentRegistry:
    """Build a registry with built-in agent definitions."""
    return AgentRegistry(agents=BUILTIN_AGENTS)


def build_builtin_capability_registry() -> CapabilityRegistry:
    """Build and validate the built-in runtime capability registry."""
    registry = CapabilityRegistry(capabilities=BUILTIN_CAPABILITIES)
    for profile in BUILTIN_INVOCATION_PROFILES:
        registry.validate_profile(profile)
    return registry

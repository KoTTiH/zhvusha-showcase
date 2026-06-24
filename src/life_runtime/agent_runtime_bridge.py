"""Read-only bridge objects from LifeRuntime to Agent Runtime."""

from __future__ import annotations

from src.life_runtime.models import LifeActionRequest, LifeActionRequestKind

LIFE_RUNTIME_READONLY_DENIED_CAPABILITIES: tuple[str, ...] = (
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
)


def build_life_reflection_action_request(
    *,
    tick_id: str,
    reason: str,
) -> LifeActionRequest:
    """Build a read-only Agent Runtime reflection request."""

    return LifeActionRequest(
        requested_by_tick_id=tick_id,
        kind=LifeActionRequestKind.AGENT_RUNTIME_JOB,
        profile_id="life_reflection.readonly",
        capabilities_requested=("read_workspace", "life_reflection"),
        denied_capabilities=LIFE_RUNTIME_READONLY_DENIED_CAPABILITIES,
        reason=reason,
        requires_approval=False,
    )

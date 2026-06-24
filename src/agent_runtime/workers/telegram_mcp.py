"""Agent Runtime worker and ToolGateway adapters for personal Telegram MCP."""

from __future__ import annotations

import json
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from mcp import ClientSession
from mcp import types as mcp_types
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import FileUrl

from scripts.run_telegram_mcp import LaunchPlan, build_launch_plan
from src.agent_runtime.approvals import AgentToolApproval
from src.agent_runtime.models import ContextCapsule, Finding, FindingStatus
from src.agent_runtime.tools import (
    AgentTool,
    FunctionAgentTool,
    ToolDeniedError,
    ToolGateway,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mcp.client.session import ListRootsFnT

    from src.agent_runtime.models import AgentJob, ContextPack


READ_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "list_accounts",
        "get_me",
        "get_privacy_settings",
        "get_full_user",
        "get_bot_info",
        "get_user_photos",
        "get_user_status",
        "list_contacts",
        "search_contacts",
        "get_contact_ids",
        "get_direct_chat_by_contact",
        "get_contact_chats",
        "get_last_interaction",
        "export_contacts",
        "get_blocked_users",
        "get_chats",
        "list_topics",
        "list_chats",
        "get_chat",
        "search_public_chats",
        "resolve_username",
        "get_full_chat",
        "get_common_chats",
        "get_message_read_by",
        "get_message_link",
        "get_messages",
        "get_scheduled_messages",
        "list_inline_buttons",
        "list_messages",
        "get_message_context",
        "search_messages",
        "search_global",
        "get_history",
        "get_pinned_messages",
        "get_message_reactions",
        "get_drafts",
        "get_participants",
        "get_admins",
        "get_banned_users",
        "get_invite_link",
        "get_recent_actions",
        "list_folders",
        "get_folder",
    }
)

MODIFY_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "send_scheduled_message",
        "delete_scheduled_message",
        "press_inline_button",
        "forward_message",
        "edit_message",
        "delete_message",
        "delete_chat_history",
        "delete_messages_bulk",
        "pin_message",
        "unpin_message",
        "unpin_all_messages",
        "mark_as_read",
        "reply_to_message",
        "create_poll",
        "send_reaction",
        "remove_reaction",
        "save_draft",
        "clear_draft",
        "update_profile",
        "set_privacy_settings",
        "set_bot_commands",
        "add_contact",
        "delete_contact",
        "block_user",
        "unblock_user",
        "import_contacts",
        "send_contact",
        "subscribe_public_channel",
        "mute_chat",
        "unmute_chat",
        "archive_chat",
        "unarchive_chat",
        "create_folder",
        "add_chat_to_folder",
        "remove_chat_from_folder",
        "delete_folder",
        "reorder_folders",
    }
)

ADMIN_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "create_group",
        "invite_to_group",
        "leave_chat",
        "create_channel",
        "edit_chat_title",
        "edit_chat_about",
        "delete_chat_photo",
        "promote_admin",
        "demote_admin",
        "ban_user",
        "unban_user",
        "set_default_chat_permissions",
        "toggle_slow_mode",
        "edit_admin_rights",
        "join_chat_by_link",
        "export_chat_invite",
        "import_chat_invite",
    }
)

MEDIA_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "set_profile_photo",
        "send_file",
        "download_media",
        "send_voice",
        "upload_file",
        "get_media_info",
        "get_sticker_sets",
        "send_sticker",
        "get_gif_search",
        "send_gif",
        "edit_chat_photo",
    }
)

MEDIA_READ_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "download_media",
        "get_media_info",
    }
)

DAIVINCHIK_REPLY_BUTTON_TEXTS: frozenset[str] = frozenset(
    {
        "1",
        "1 🚀",
        "❤️",
        "👎",
    }
)


@dataclass(frozen=True)
class _DaivinchikReplyButtonHandler:
    client: TelegramMCPClient
    account_label: str

    async def __call__(self, payload: dict[str, Any]) -> str:
        button_text = _daivinchik_reply_button_text(payload)
        return await self.client.call_tool(
            "send_message",
            _with_account(
                {
                    "chat_id": payload["chat_id"],
                    "message": button_text,
                    "parse_mode": None,
                },
                account_label=str(payload.get("account") or self.account_label),
            ),
        )


@dataclass(frozen=True)
class _DaivinchikForwardLikedProfileHandler:
    client: TelegramMCPClient
    account_label: str

    async def __call__(self, payload: dict[str, Any]) -> str:
        raw_message_ids = payload.get("message_ids")
        if not isinstance(raw_message_ids, list | tuple):
            raise ValueError("message_ids must be a list")
        if len(raw_message_ids) > 8:
            raise ValueError("too many Daivinchik profile messages to forward")
        results: list[str] = []
        for raw_message_id in raw_message_ids:
            message_id = int(str(raw_message_id))
            result = await self.client.call_tool(
                "forward_message",
                _with_account(
                    {
                        "from_chat_id": payload["from_chat_id"],
                        "message_id": message_id,
                        "to_chat_id": payload["to_chat_id"],
                    },
                    account_label=str(payload.get("account") or self.account_label),
                ),
            )
            results.append(result)
        return json.dumps(
            {"forwarded": len(results), "results": results},
            ensure_ascii=False,
        )


class TelegramMCPClient(Protocol):
    """Narrow async client for upstream telegram-mcp tools."""

    async def list_tools(self) -> tuple[str, ...]: ...

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> str: ...


class MCPStdioTelegramClient:
    """Launch upstream telegram-mcp over stdio for one bounded tool call."""

    def __init__(self, *, read_timeout_seconds: float = 60.0) -> None:
        self._read_timeout_seconds = read_timeout_seconds
        self._stack: AsyncExitStack | None = None
        self._persistent_session: ClientSession | None = None

    async def list_tools(self) -> tuple[str, ...]:
        session = self._persistent_session
        if session is None:
            async with self._session() as session:
                result = await session.list_tools()
            return tuple(tool.name for tool in result.tools)
        else:
            result = await session.list_tools()
        return tuple(tool.name for tool in result.tools)

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> str:
        session = self._persistent_session
        if session is None:
            async with self._session() as session:
                result = await session.call_tool(
                    name,
                    dict(arguments),
                    read_timeout_seconds=timedelta(seconds=self._read_timeout_seconds),
                )
            return _tool_result_to_text(result)
        else:
            result = await session.call_tool(
                name,
                dict(arguments),
                read_timeout_seconds=timedelta(seconds=self._read_timeout_seconds),
            )
        return _tool_result_to_text(result)

    async def __aenter__(self) -> MCPStdioTelegramClient:
        await self.start()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.aclose()

    async def start(self) -> None:
        if self._persistent_session is not None:
            return
        params, roots = self._session_config()
        stack = AsyncExitStack()
        try:
            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(params)
            )
            session = await stack.enter_async_context(
                self._client_session(read_stream, write_stream, roots)
            )
            await session.initialize()
        except Exception:
            await stack.aclose()
            raise
        self._stack = stack
        self._persistent_session = session

    async def aclose(self) -> None:
        stack = self._stack
        self._persistent_session = None
        self._stack = None
        if stack is not None:
            await stack.aclose()

    def _session_config(self) -> tuple[StdioServerParameters, tuple[Path, ...]]:
        plan = build_launch_plan()
        return (
            StdioServerParameters(
                command=plan.command,
                args=list(plan.args[1:]),
                env=plan.env,
            ),
            _roots_from_launch_plan(plan),
        )

    def _client_session(
        self,
        read_stream: Any,
        write_stream: Any,
        roots: tuple[Path, ...],
    ) -> ClientSession:
        async def _list_roots(
            _context: Any,
        ) -> mcp_types.ListRootsResult | mcp_types.ErrorData:
            return mcp_types.ListRootsResult(
                roots=[
                    mcp_types.Root(uri=FileUrl(root.as_uri()), name=root.name)
                    for root in roots
                ]
            )

        return ClientSession(
            read_stream,
            write_stream,
            list_roots_callback=cast("ListRootsFnT", _list_roots) if roots else None,
        )

    def _session(self) -> Any:
        params, roots = self._session_config()

        async def _ctx() -> Any:
            async with (
                stdio_client(params) as (read_stream, write_stream),
                self._client_session(read_stream, write_stream, roots) as session,
            ):
                await session.initialize()
                yield session

        return asynccontextmanager(_ctx)()


@dataclass(frozen=True)
class TelegramMCPToolRequest:
    """Normalized worker request parsed from ContextPack.user_request."""

    action: str
    tool_name: str = ""
    arguments: dict[str, Any] | None = None
    chat_id: str = ""
    message: str = ""
    parse_mode: str | None = None


class TelegramMCPWorkerBackend:
    """Bounded Agent Runtime worker for personal Telegram MCP calls."""

    name = "telegram_mcp"

    def __init__(self, *, tool_gateway: ToolGateway, account_label: str) -> None:
        self._tool_gateway = tool_gateway
        self._account_label = account_label

    async def run(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
    ) -> ContextCapsule:
        try:
            request = _parse_worker_request(context_pack.user_request)
            tool_name, payload, capability = self._resolve_gateway_call(request)
            result = await self._tool_gateway.execute(
                job.profile,
                tool_name,
                payload,
                approval=_approval_from_context_pack(
                    job=job,
                    context_pack=context_pack,
                    capability=capability,
                ),
            )
        except ToolDeniedError as exc:
            return _denied_capsule(str(exc))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return _failed_capsule(f"Invalid Telegram MCP request: {exc}")

        text = str(result)
        return ContextCapsule(
            summary="Telegram MCP action completed.",
            processed_context=text,
            findings=(
                Finding(
                    claim="Telegram MCP tool returned a result.",
                    status=FindingStatus.CONFIRMED,
                    confidence=0.95,
                    evidence=(text[:500],),
                ),
            ),
            markdown_report=_render_tool_result_for_chat(request, text),
        )

    async def cancel(self, job_id: str) -> bool:
        del job_id
        return False

    def _resolve_gateway_call(
        self,
        request: TelegramMCPToolRequest,
    ) -> tuple[str, dict[str, Any], str]:
        if request.action == "list_tools":
            return "telegram_mcp_list_tools", {}, "telegram_mcp_read"
        if request.action == "send_message":
            return (
                "telegram_mcp_send_message",
                {
                    "chat_id": request.chat_id,
                    "message": request.message,
                    "parse_mode": request.parse_mode,
                    "account": self._account_label,
                },
                "telegram_mcp_send",
            )
        if request.action == "read":
            return (
                "telegram_mcp_call_read",
                _tool_payload(request, self._account_label, READ_TOOL_NAMES),
                "telegram_mcp_read",
            )
        if request.action == "modify":
            return (
                "telegram_mcp_call_modify",
                _tool_payload(request, self._account_label, MODIFY_TOOL_NAMES),
                "telegram_mcp_modify",
            )
        if request.action == "admin":
            return (
                "telegram_mcp_call_admin",
                _tool_payload(request, self._account_label, ADMIN_TOOL_NAMES),
                "telegram_mcp_admin",
            )
        if request.action == "media":
            return (
                "telegram_mcp_call_media",
                _tool_payload(request, self._account_label, MEDIA_TOOL_NAMES),
                "telegram_mcp_media_files",
            )
        if request.action == "media_read":
            return (
                "telegram_mcp_call_media_read",
                _tool_payload(request, self._account_label, MEDIA_READ_TOOL_NAMES),
                "telegram_mcp_media_read",
            )
        raise ValueError(f"unsupported action {request.action!r}")


def build_telegram_mcp_tool_gateway(
    *,
    client: TelegramMCPClient | None = None,
    account_label: str = "personal",
) -> ToolGateway:
    """Build ToolGateway adapters over upstream telegram-mcp MCP tools."""
    resolved_client = client or MCPStdioTelegramClient()

    async def list_tools(_payload: dict[str, Any]) -> str:
        return "\n".join(await resolved_client.list_tools())

    async def send_message(payload: dict[str, Any]) -> str:
        return await resolved_client.call_tool(
            "send_message",
            _with_account(
                {
                    "chat_id": payload["chat_id"],
                    "message": payload["message"],
                    "parse_mode": payload.get("parse_mode"),
                },
                account_label=str(payload.get("account") or account_label),
            ),
        )

    async def call_read(payload: dict[str, Any]) -> str:
        return await _call_checked(
            resolved_client,
            payload=payload,
            allowed_tools=READ_TOOL_NAMES,
            account_label=account_label,
        )

    async def call_modify(payload: dict[str, Any]) -> str:
        return await _call_checked(
            resolved_client,
            payload=payload,
            allowed_tools=MODIFY_TOOL_NAMES,
            account_label=account_label,
        )

    async def call_admin(payload: dict[str, Any]) -> str:
        return await _call_checked(
            resolved_client,
            payload=payload,
            allowed_tools=ADMIN_TOOL_NAMES,
            account_label=account_label,
        )

    async def call_media(payload: dict[str, Any]) -> str:
        return await _call_checked(
            resolved_client,
            payload=payload,
            allowed_tools=MEDIA_TOOL_NAMES,
            account_label=account_label,
        )

    async def call_media_read(payload: dict[str, Any]) -> str:
        return await _call_checked(
            resolved_client,
            payload=payload,
            allowed_tools=MEDIA_READ_TOOL_NAMES,
            account_label=account_label,
        )

    async def daivinchik_press_inline_button(payload: dict[str, Any]) -> str:
        return await resolved_client.call_tool(
            "press_inline_button",
            _with_account(
                {
                    "chat_id": payload["chat_id"],
                    "message_id": payload["message_id"],
                    "button_text": payload["button_text"],
                },
                account_label=str(payload.get("account") or account_label),
            ),
        )

    async def daivinchik_notify(payload: dict[str, Any]) -> str:
        return await resolved_client.call_tool(
            "send_message",
            _with_account(
                {
                    "chat_id": payload["chat_id"],
                    "message": payload["message"],
                    "parse_mode": payload.get("parse_mode"),
                },
                account_label=str(payload.get("account") or account_label),
            ),
        )

    tools = cast(
        "tuple[AgentTool, ...]",
        (
            FunctionAgentTool(
                "telegram_mcp_list_tools", "telegram_mcp_read", list_tools
            ),
            FunctionAgentTool(
                "telegram_mcp_call_read",
                "telegram_mcp_read",
                call_read,
            ),
            FunctionAgentTool(
                "telegram_mcp_send_message",
                "telegram_mcp_send",
                send_message,
            ),
            FunctionAgentTool(
                "telegram_mcp_call_modify",
                "telegram_mcp_modify",
                call_modify,
            ),
            FunctionAgentTool(
                "telegram_mcp_call_admin",
                "telegram_mcp_admin",
                call_admin,
            ),
            FunctionAgentTool(
                "telegram_mcp_call_media",
                "telegram_mcp_media_files",
                call_media,
            ),
            FunctionAgentTool(
                "telegram_mcp_call_media_read",
                "telegram_mcp_media_read",
                call_media_read,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_press_inline_button",
                "telegram_mcp_daivinchik_button",
                daivinchik_press_inline_button,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_reply_button",
                "telegram_mcp_daivinchik_reply_button",
                _DaivinchikReplyButtonHandler(resolved_client, account_label),
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_notify",
                "telegram_mcp_daivinchik_notify",
                daivinchik_notify,
            ),
            FunctionAgentTool(
                "telegram_mcp_daivinchik_forward_liked_profile",
                "telegram_mcp_daivinchik_forward_liked_profile",
                _DaivinchikForwardLikedProfileHandler(
                    resolved_client,
                    account_label,
                ),
            ),
        ),
    )
    return ToolGateway(tools=tools)


def _roots_from_launch_plan(plan: LaunchPlan) -> tuple[Path, ...]:
    try:
        first_root_index = tuple(plan.args).index("main.py") + 1
    except ValueError:
        return ()
    roots: list[Path] = []
    for raw in plan.args[first_root_index:]:
        root = Path(raw).expanduser().resolve()
        if root.exists():
            roots.append(root)
    return tuple(roots)


def _daivinchik_reply_button_text(payload: dict[str, Any]) -> str:
    button_text = str(payload.get("button_text") or "").strip()
    if button_text not in DAIVINCHIK_REPLY_BUTTON_TEXTS:
        raise ToolDeniedError("Daivinchik reply button text is not whitelisted")
    return button_text


def _parse_worker_request(raw: str) -> TelegramMCPToolRequest:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("request must be a JSON object")
    action = str(data.get("action") or "").strip()
    arguments = data.get("arguments")
    if arguments is not None and not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    return TelegramMCPToolRequest(
        action=action,
        tool_name=str(data.get("tool_name") or "").strip(),
        arguments=dict(arguments or {}),
        chat_id=str(data.get("chat_id") or "").strip(),
        message=str(data.get("message") or ""),
        parse_mode=str(data["parse_mode"]) if data.get("parse_mode") else None,
    )


def _tool_payload(
    request: TelegramMCPToolRequest,
    account_label: str,
    allowed_tools: frozenset[str],
) -> dict[str, Any]:
    if not request.tool_name:
        raise ValueError("tool_name is required")
    if request.tool_name not in allowed_tools:
        raise ValueError(f"{request.tool_name} is not allowed for this action")
    return {
        "tool_name": request.tool_name,
        "arguments": _with_account(
            request.arguments or {}, account_label=account_label
        ),
    }


async def _call_checked(
    client: TelegramMCPClient,
    *,
    payload: dict[str, Any],
    allowed_tools: frozenset[str],
    account_label: str,
) -> str:
    tool_name = str(payload.get("tool_name") or "").strip()
    if tool_name not in allowed_tools:
        raise ValueError(f"{tool_name} is not allowed for this capability")
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    return await client.call_tool(
        tool_name,
        _with_account(arguments, account_label=account_label),
    )


def _with_account(
    arguments: Mapping[str, Any], *, account_label: str
) -> dict[str, Any]:
    result = dict(arguments)
    if account_label and "account" not in result:
        result["account"] = account_label
    return result


def _approval_from_context_pack(
    *,
    job: AgentJob,
    context_pack: ContextPack,
    capability: str,
) -> AgentToolApproval | None:
    raw_capabilities = context_pack.metadata.get("agent_tool_approval_capabilities", "")
    capabilities = tuple(
        item.strip() for item in raw_capabilities.split(",") if item.strip()
    )
    if capability not in set(capabilities):
        return None
    approval_id = context_pack.metadata.get("agent_tool_approval_id", "")
    if not approval_id:
        return None
    return AgentToolApproval.approved(
        approval_id=approval_id,
        capabilities=capabilities,
        approved_by=job.owner_user_id,
    )


def _tool_result_to_text(result: Any) -> str:
    if getattr(result, "structuredContent", None) is not None:
        return json.dumps(result.structuredContent, ensure_ascii=False)
    content = getattr(result, "content", ())
    parts: list[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            parts.append(text)
    if parts:
        return "\n".join(parts)
    return str(result)


def _render_tool_result_for_chat(
    request: TelegramMCPToolRequest,
    result_text: str,
) -> str:
    if request.action == "send_message" and _is_send_success_payload(result_text):
        return "отправила."
    return _extract_human_message(result_text) or result_text


def _is_send_success_payload(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if "message sent successfully" in lowered:
        return True
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    values = (
        payload.get("result"),
        payload.get("message"),
        payload.get("status"),
        payload.get("ok"),
    )
    return any(_looks_like_send_success(value) for value in values)


def _looks_like_send_success(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"ok", "success", "sent"} or "message sent successfully" in text


def _extract_human_message(text: str) -> str:
    stripped = text.strip()
    if not stripped or not stripped.startswith("{"):
        return ""
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    for key in ("error", "message", "result", "detail"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _denied_capsule(reason: str) -> ContextCapsule:
    return ContextCapsule(
        summary="Telegram MCP action denied.",
        findings=(
            Finding(
                claim="Telegram MCP tool was denied by ToolGateway.",
                status=FindingStatus.REJECTED,
                confidence=1.0,
                evidence=(reason,),
            ),
        ),
        markdown_report=reason,
    )


def _failed_capsule(reason: str) -> ContextCapsule:
    return ContextCapsule(
        summary="Telegram MCP action failed.",
        findings=(
            Finding(
                claim="Telegram MCP request could not be executed.",
                status=FindingStatus.REJECTED,
                confidence=1.0,
                evidence=(reason,),
            ),
        ),
        markdown_report=reason,
    )

"""Computer-use Agent Runtime tools, policies and adapters."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import re
import shlex
import shutil
import time
import urllib.error
import urllib.request
from asyncio import create_subprocess_exec as _create_subprocess_exec
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast
from urllib.parse import parse_qs, quote, urljoin, urlparse

from pydantic import BaseModel, Field

from src.agent_runtime.approvals import AgentToolApproval
from src.agent_runtime.models import ContextCapsule, Finding, FindingStatus
from src.agent_runtime.tools import ToolDeniedError, ToolGateway, ToolNotFoundError

if TYPE_CHECKING:
    from types import TracebackType


class ComputerUseActionKind(StrEnum):
    """Scoped computer-use action families."""

    BROWSER_STATUS = "browser_status"
    BROWSER_NAVIGATE = "browser_navigate"
    BROWSER_CLICK = "browser_click"
    BROWSER_TYPE = "browser_type"
    BROWSER_SCROLL = "browser_scroll"
    BROWSER_TAB_CONTROL = "browser_tab_control"
    BROWSER_FORM_DRAFT = "browser_form_draft"
    BROWSER_INTERACTIVE_TASK = "browser_interactive_task"
    BROWSER_SUBMIT = "browser_submit"
    DESKTOP_INPUT = "desktop_input"
    DESKTOP_WINDOW_CONTROL = "desktop_window_control"
    DESKTOP_SCREENSHOT = "desktop_screenshot"
    DESKTOP_APP_LAUNCHER = "desktop_app_launcher"
    DESKTOP_HOTKEYS = "desktop_hotkeys"
    DESKTOP_MEDIA_CONTROL = "desktop_media_control"
    DESKTOP_SHELL_COMMAND = "desktop_shell_command"


class ComputerUseActionRequest(BaseModel):
    """One action request after Жвуша has selected computer-use as a body tool."""

    action: ComputerUseActionKind
    goal: str = ""
    operation: str = ""
    target: str = ""
    text: str = ""
    url: str = ""
    selector: str = ""
    tab_id: str = ""
    constraints: list[str] = Field(default_factory=list)
    artifact_requirements: dict[str, str] = Field(default_factory=dict)
    success_criteria: list[str] = Field(default_factory=list)
    risk_intent: str = ""
    approval_scope: dict[str, str] = Field(default_factory=dict)
    argv: list[str] = Field(default_factory=list)
    cwd: str = "."
    timeout_seconds: float = 0.0
    metadata: dict[str, str] = Field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Return a stable payload for adapter execution."""
        payload: dict[str, Any] = {
            "action": self.action.value,
            "goal": self.goal,
            "operation": self.operation,
            "target": self.target,
            "text": self.text,
            "url": self.url,
            "selector": self.selector,
            "tab_id": self.tab_id,
            "artifact_requirements": dict(self.artifact_requirements),
            "metadata": dict(self.metadata),
        }
        if self.constraints:
            payload["constraints"] = list(self.constraints)
        if self.success_criteria:
            payload["success_criteria"] = list(self.success_criteria)
        if self.risk_intent:
            payload["risk_intent"] = self.risk_intent
        if self.approval_scope:
            payload["approval_scope"] = dict(self.approval_scope)
        if self.argv:
            payload["argv"] = list(self.argv)
        if self.cwd != ".":
            payload["cwd"] = self.cwd
        if self.timeout_seconds:
            payload["timeout_seconds"] = self.timeout_seconds
        return payload


class ComputerUseActionResult(BaseModel):
    """Result returned by a live browser or desktop adapter."""

    status: Literal[
        "completed",
        "configured_only",
        "degraded",
        "hard_stopped",
        "refused",
        "failed",
    ] = "completed"
    message: str = ""
    artifact: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class ComputerUseRiskClass(StrEnum):
    """Computer-use policy classes used before action execution."""

    READONLY_EXISTING_SESSION = "readonly_existing_session"
    REVERSIBLE_GUI_ACTION = "reversible_gui_action"
    CREDENTIAL_ENTRY = "credential_entry"
    EXTERNAL_SUBMIT = "external_submit"
    ACCOUNT_MUTATION = "account_mutation"
    SHELL_COMMAND = "shell_command"


class ComputerUseSafetyDecision(BaseModel):
    """Policy decision before a computer-use action reaches an adapter."""

    allowed: bool
    hard_stop: bool = False
    requires_approval: bool = False
    reason: str
    stop_condition: str = ""
    risk_class: ComputerUseRiskClass = ComputerUseRiskClass.REVERSIBLE_GUI_ACTION
    required_capability: str = ""
    risk_summary: str = ""
    approval_prompt: str = ""


@dataclass(frozen=True)
class _InteractiveAttemptOutcome:
    messages: tuple[str, ...] = ()
    done: bool = False
    abort: bool = False


class ComputerUseControlSnapshot(BaseModel):
    """Durable pause/resume status for computer-use runs."""

    paused: bool = False
    reason: str = ""
    updated_at: str = ""


class ChromeDevToolsClient(Protocol):
    """Injected CDP/MCP client for live Chrome actions."""

    async def list_tabs(self) -> tuple[Mapping[str, Any], ...]: ...

    async def execute_action(
        self,
        action: str,
        payload: dict[str, Any],
    ) -> Mapping[str, Any] | ComputerUseActionResult: ...


class RawCDPSession(Protocol):
    """Minimal async CDP websocket session surface used by local Chrome control."""

    async def __aenter__(self) -> RawCDPSession: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def send_raw(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]: ...


RawCDPSessionFactory = Callable[[str], RawCDPSession]


class BrowserLiveAdapter(Protocol):
    """Browser adapter exposed to computer-use tools."""

    async def status(self) -> ComputerUseActionResult: ...

    async def execute(
        self,
        request: ComputerUseActionRequest,
    ) -> ComputerUseActionResult: ...


class DesktopComputerUseAdapter(Protocol):
    """Desktop adapter exposed to computer-use tools."""

    async def execute(
        self,
        request: ComputerUseActionRequest,
    ) -> ComputerUseActionResult: ...


class ComputerUseShellCommandRunner(Protocol):
    """Runs one approved structured argv command for the shell profile."""

    async def __call__(
        self,
        argv: tuple[str, ...],
        *,
        cwd: str,
        timeout_seconds: float,
    ) -> Mapping[str, Any]: ...


class ComputerUseControlStateStore(Protocol):
    """Small pause/resume state store used by worker and operator commands."""

    def snapshot(self) -> ComputerUseControlSnapshot: ...

    def pause(self, *, reason: str = "") -> ComputerUseControlSnapshot: ...

    def resume(self) -> ComputerUseControlSnapshot: ...


TabDiscovery = Callable[[str, float], Awaitable[tuple[dict[str, Any], ...]]]
ChromeVersionDiscovery = Callable[[str, float], Awaitable[Mapping[str, Any]]]
BrowserProcessLauncher = Callable[[tuple[str, ...]], Awaitable[Any]]

_BROWSER_ACTIONS = {
    ComputerUseActionKind.BROWSER_STATUS,
    ComputerUseActionKind.BROWSER_NAVIGATE,
    ComputerUseActionKind.BROWSER_CLICK,
    ComputerUseActionKind.BROWSER_TYPE,
    ComputerUseActionKind.BROWSER_SCROLL,
    ComputerUseActionKind.BROWSER_TAB_CONTROL,
    ComputerUseActionKind.BROWSER_FORM_DRAFT,
    ComputerUseActionKind.BROWSER_INTERACTIVE_TASK,
    ComputerUseActionKind.BROWSER_SUBMIT,
}
_DESKTOP_ACTIONS = {
    ComputerUseActionKind.DESKTOP_INPUT,
    ComputerUseActionKind.DESKTOP_WINDOW_CONTROL,
    ComputerUseActionKind.DESKTOP_SCREENSHOT,
    ComputerUseActionKind.DESKTOP_APP_LAUNCHER,
    ComputerUseActionKind.DESKTOP_HOTKEYS,
    ComputerUseActionKind.DESKTOP_MEDIA_CONTROL,
}
_CAPABILITY_BY_ACTION: dict[ComputerUseActionKind, str] = {
    ComputerUseActionKind.BROWSER_STATUS: "browser_live_control",
    ComputerUseActionKind.BROWSER_NAVIGATE: "browser_navigate",
    ComputerUseActionKind.BROWSER_CLICK: "browser_click",
    ComputerUseActionKind.BROWSER_TYPE: "browser_type",
    ComputerUseActionKind.BROWSER_SCROLL: "browser_scroll",
    ComputerUseActionKind.BROWSER_TAB_CONTROL: "browser_tab_control",
    ComputerUseActionKind.BROWSER_FORM_DRAFT: "browser_form_draft",
    ComputerUseActionKind.BROWSER_INTERACTIVE_TASK: "browser_interactive_task",
    ComputerUseActionKind.BROWSER_SUBMIT: "browser_submit",
    ComputerUseActionKind.DESKTOP_INPUT: "desktop_input",
    ComputerUseActionKind.DESKTOP_WINDOW_CONTROL: "desktop_window_control",
    ComputerUseActionKind.DESKTOP_SCREENSHOT: "desktop_screenshot",
    ComputerUseActionKind.DESKTOP_APP_LAUNCHER: "desktop_app_launcher",
    ComputerUseActionKind.DESKTOP_HOTKEYS: "desktop_hotkeys",
    ComputerUseActionKind.DESKTOP_MEDIA_CONTROL: "desktop_media_control",
    ComputerUseActionKind.DESKTOP_SHELL_COMMAND: "desktop.shell",
}
_TOOL_BY_ACTION: dict[ComputerUseActionKind, str] = {
    ComputerUseActionKind.BROWSER_STATUS: "browser_live_status",
    ComputerUseActionKind.BROWSER_NAVIGATE: "browser_live_navigate",
    ComputerUseActionKind.BROWSER_CLICK: "browser_live_click",
    ComputerUseActionKind.BROWSER_TYPE: "browser_live_type",
    ComputerUseActionKind.BROWSER_SCROLL: "browser_live_scroll",
    ComputerUseActionKind.BROWSER_TAB_CONTROL: "browser_live_tab_control",
    ComputerUseActionKind.BROWSER_FORM_DRAFT: "browser_live_form_draft",
    ComputerUseActionKind.BROWSER_INTERACTIVE_TASK: "browser_live_interactive_task",
    ComputerUseActionKind.BROWSER_SUBMIT: "computer_browser_submit",
    ComputerUseActionKind.DESKTOP_INPUT: "desktop_input",
    ComputerUseActionKind.DESKTOP_WINDOW_CONTROL: "desktop_window_control",
    ComputerUseActionKind.DESKTOP_SCREENSHOT: "desktop_screenshot",
    ComputerUseActionKind.DESKTOP_APP_LAUNCHER: "desktop_app_launcher",
    ComputerUseActionKind.DESKTOP_HOTKEYS: "desktop_hotkeys",
    ComputerUseActionKind.DESKTOP_MEDIA_CONTROL: "desktop_media_control",
    ComputerUseActionKind.DESKTOP_SHELL_COMMAND: "desktop_shell",
}
_CREDENTIAL_ENTRY_RE = re.compile(
    r"(введ\w*|напечат\w*|type|enter|input).{0,80}"
    r"(парол\w*|password|логин\w*|login|2fa|otp|mfa|код\w*)|"
    r"(парол\w*|password|логин\w*|login|2fa|otp|mfa|код\w*).{0,80}"
    r"(поле|field|input|введ\w*|напечат\w*|type|enter)",
    re.IGNORECASE,
)
_LOGIN_CLICK_RE = re.compile(
    r"(нажм\w*|клик\w*|click|press).{0,80}"
    r"(войти|логин\w*|login|sign\s*in|sign-in|авториз\w*)|"
    r"\b(залогин\w*|авториз\w*|sign\s*in|sign-in)\b",
    re.IGNORECASE,
)
_NEGATED_CREDENTIAL_RE = re.compile(
    r"(без|не|do_not|don't|dont|without).{0,40}"
    r"(парол\w*|password|логин\w*|login|credential\w*|2fa|otp|mfa)",
    re.IGNORECASE,
)
_PURCHASE_RE = re.compile(
    r"\b(pay|payment|pay now|purchase|buy|checkout|place order|transfer)\b|"
    r"(оплат\w*|куп\w*|покуп\w*|заказ\w*|перевод\w*)",
    re.IGNORECASE,
)
_DELETE_RE = re.compile(
    r"\b(delete|remove account|account deletion)\b|(удал\w*)",
    re.IGNORECASE,
)
_SEND_RE = re.compile(
    r"\b(send|send message|message)\b|(отправ\w*|сообщени\w*)",
    re.IGNORECASE,
)
_PUBLISH_RE = re.compile(r"\b(publish|post)\b|(опублик\w*|запост\w*)", re.IGNORECASE)
_SUBMIT_RE = re.compile(
    r"\b(submit|final submit|confirm)\b|(подтверд\w*|отправ\w*\s+форм\w*)",
    re.IGNORECASE,
)
_SHELL_RE = re.compile(
    r"\b(shell|terminal|powershell|command|cmd)\b|(терминал\w*|команд\w*)",
    re.IGNORECASE,
)
_SHELL_INTERPRETERS = frozenset(
    {
        "bash",
        "cmd",
        "cmd.exe",
        "fish",
        "powershell",
        "powershell.exe",
        "pwsh",
        "sh",
        "zsh",
    }
)
_SHELL_SECRET_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(TOKEN|PASSWORD|PASS|API_KEY|SECRET|SESSION|HASH|KEY)"
    r"[A-Z0-9_]*)=([^\s]+)"
)
_SAFETY_POLICY_METADATA_KEYS = frozenset(
    {
        "agent_tool_approval_capabilities",
        "agent_tool_approval_id",
        "answer_policy",
        "persona_context_mode",
        "persona_context_ref",
    }
)


class ChromeDevToolsLiveBrowserAdapter:
    """Adapter for a user-started Chrome instance with remote debugging enabled."""

    def __init__(
        self,
        *,
        debug_url: str = "http://127.0.0.1:9222",
        cdp_client: ChromeDevToolsClient | None = None,
        tab_discovery: TabDiscovery | None = None,
        timeout_seconds: float = 1.5,
    ) -> None:
        self._debug_url = debug_url.rstrip("/")
        self._cdp_client = cdp_client
        self._tab_discovery = tab_discovery or _default_chrome_tab_discovery
        self._timeout_seconds = timeout_seconds

    async def status(self) -> ComputerUseActionResult:
        """Return live Chrome attach status without taking page actions."""
        try:
            tabs = await self._list_tabs()
        except (OSError, RuntimeError, ValueError, urllib.error.URLError) as exc:
            return ComputerUseActionResult(
                status="degraded",
                message=(
                    "Chrome remote debugging endpoint is not attachable. "
                    "Start Google Chrome with the configured remote debugging URL."
                ),
                metadata={"debug_url": self._debug_url, "error": str(exc)[:300]},
            )
        page_tabs = tuple(tab for tab in tabs if str(tab.get("type", "page")) == "page")
        return ComputerUseActionResult(
            status="completed",
            message="Chrome remote debugging endpoint is attachable.",
            metadata={
                "debug_url": self._debug_url,
                "tab_count": str(len(page_tabs)),
                "active_tab_id": str(page_tabs[0].get("id", "")) if page_tabs else "",
                "active_tab_url": str(page_tabs[0].get("url", "")) if page_tabs else "",
            },
        )

    async def execute(
        self,
        request: ComputerUseActionRequest,
    ) -> ComputerUseActionResult:
        """Execute an action through the injected CDP/MCP client."""
        if request.action is ComputerUseActionKind.BROWSER_STATUS:
            return await self.status()
        if self._cdp_client is None:
            return ComputerUseActionResult(
                status="configured_only",
                message=(
                    "Chrome debug discovery is configured, but no CDP/MCP action "
                    "client is attached for live clicks/types/navigation."
                ),
                metadata={"debug_url": self._debug_url},
            )
        result = await self._cdp_client.execute_action(
            request.action.value,
            request.to_payload(),
        )
        return _action_result_from_mapping(result)

    async def _list_tabs(self) -> tuple[Mapping[str, Any], ...]:
        if self._cdp_client is not None:
            return await self._cdp_client.list_tabs()
        tabs = await self._tab_discovery(self._debug_url, self._timeout_seconds)
        return tuple(tabs)


class LocalChromeDevToolsClient:
    """Local CDP action client for Chrome remote-debugging sessions."""

    def __init__(
        self,
        *,
        debug_url: str = "http://127.0.0.1:9222",
        workspace_root: Path | None = None,
        tab_discovery: TabDiscovery | None = None,
        session_factory: RawCDPSessionFactory | None = None,
        timeout_seconds: float = 5.0,
        human_verification_timeout_seconds: float = 600.0,
        human_verification_poll_seconds: float = 1.0,
    ) -> None:
        self._debug_url = debug_url.rstrip("/")
        self._workspace_root = (
            workspace_root.expanduser().resolve()
            if workspace_root is not None
            else None
        )
        self._tab_discovery = tab_discovery or _default_chrome_tab_discovery
        self._session_factory = session_factory or _default_cdp_session_factory
        self._timeout_seconds = timeout_seconds
        self._human_verification_timeout_seconds = max(
            0.0,
            human_verification_timeout_seconds,
        )
        self._human_verification_poll_seconds = max(
            0.1,
            human_verification_poll_seconds,
        )

    async def list_tabs(self) -> tuple[Mapping[str, Any], ...]:
        """Return attachable Chrome tabs from the configured debug endpoint."""
        tabs = await self._tab_discovery(self._debug_url, self._timeout_seconds)
        return tuple(tabs)

    async def execute_action(
        self,
        action: str,
        payload: dict[str, Any],
    ) -> Mapping[str, Any] | ComputerUseActionResult:
        """Execute a reversible browser action through Chrome DevTools Protocol."""
        try:
            kind = ComputerUseActionKind(action)
            request = _request_from_mapping({**payload, "action": kind.value})
            return await self._execute(kind, request)
        except TimeoutError:
            target_url = str(payload.get("url", "")).strip()
            host = urlparse(target_url).netloc or target_url
            return ComputerUseActionResult(
                status="degraded",
                message=(
                    "Chrome DevTools action timed out"
                    f"{f' while opening {host}' if host else ''}. "
                    "The live browser is attached, but this target did not "
                    "finish loading through the configured network/proxy route."
                ),
                metadata={
                    "debug_url": self._debug_url,
                    "error": "timeout",
                    "target_host": host,
                },
            )
        except (OSError, RuntimeError, ValueError, urllib.error.URLError) as exc:
            error = str(exc).strip() or exc.__class__.__name__
            return ComputerUseActionResult(
                status="degraded",
                message=f"Chrome DevTools action failed: {error}",
                metadata={"debug_url": self._debug_url, "error": error[:300]},
            )

    async def _execute(
        self,
        kind: ComputerUseActionKind,
        request: ComputerUseActionRequest,
    ) -> ComputerUseActionResult:
        if kind is ComputerUseActionKind.BROWSER_STATUS:
            tabs = await self.list_tabs()
            return ComputerUseActionResult(
                status="completed",
                message="Chrome DevTools action client is attached.",
                metadata={
                    "debug_url": self._debug_url,
                    "tab_count": str(len(_page_tabs(tabs))),
                },
            )

        tab = await self._select_tab(request)
        websocket_url = str(tab.get("webSocketDebuggerUrl", "")).strip()
        if not websocket_url:
            return ComputerUseActionResult(
                status="degraded",
                message="Chrome tab has no webSocketDebuggerUrl for CDP actions.",
                metadata={"debug_url": self._debug_url},
            )

        async with self._session_factory(websocket_url) as cdp:
            await self._enable_page(cdp)
            restore_message = ""
            if _should_navigate_before_browser_action(kind, request):
                restore_message = await self._navigate(cdp, request)
            result_message = await self._execute_cdp_browser_action(
                cdp,
                kind=kind,
                request=request,
            )
            if result_message is None:
                return ComputerUseActionResult(
                    status="refused",
                    message=f"Local CDP client does not execute {kind.value}.",
                )

            (
                artifact,
                screenshot_artifacts,
                result_sections,
                page_html_artifact,
            ) = await self._collect_browser_action_artifacts(
                cdp,
                kind=kind,
                request=request,
            )
            artifact = screenshot_artifacts[0] if screenshot_artifacts else artifact
            page_state = await self._page_state(cdp)
            current_url = await self._evaluate_text(cdp, "location.href")
            title = await self._evaluate_text(cdp, "document.title")
            message = "\n\n".join(
                part for part in (restore_message, result_message, page_state) if part
            )
            return self._browser_action_result(
                request=request,
                kind=kind,
                tab=tab,
                artifact=artifact,
                screenshot_artifacts=screenshot_artifacts,
                result_sections=result_sections,
                page_html_artifact=page_html_artifact,
                current_url=current_url,
                title=title,
                message=message,
            )

    async def _collect_browser_action_artifacts(
        self,
        cdp: RawCDPSession,
        *,
        kind: ComputerUseActionKind,
        request: ComputerUseActionRequest,
    ) -> tuple[str, tuple[str, ...], tuple[dict[str, Any], ...], str]:
        result_sections: tuple[dict[str, Any], ...] = ()
        if _request_wants_result_section_screenshots(
            kind,
            request,
        ) or _request_wants_result_text_extract(kind, request):
            result_sections = await self._extract_result_sections(cdp, request=request)

        artifact = ""
        screenshot_artifacts: tuple[str, ...] = ()
        if _request_wants_result_section_screenshots(kind, request):
            screenshot_artifacts = await self._capture_result_section_screenshots(
                cdp,
                result_sections=result_sections,
            )
        elif _request_wants_screenshot(request):
            artifact = await self._capture_screenshot(cdp)
            screenshot_artifacts = (artifact,) if artifact else ()

        page_html_artifact = ""
        if _request_wants_page_snapshot(kind, request):
            page_html_artifact = await self._capture_page_html(cdp)
        return artifact, screenshot_artifacts, result_sections, page_html_artifact

    def _browser_action_result(
        self,
        *,
        request: ComputerUseActionRequest,
        kind: ComputerUseActionKind,
        tab: Mapping[str, Any],
        artifact: str,
        screenshot_artifacts: tuple[str, ...],
        result_sections: tuple[dict[str, Any], ...],
        page_html_artifact: str,
        current_url: str,
        title: str,
        message: str,
    ) -> ComputerUseActionResult:
        status, message, error = _artifact_requirement_status(
            request=request,
            kind=kind,
            screenshot_artifacts=screenshot_artifacts,
            result_sections=result_sections,
            message=message,
        )
        metadata: dict[str, str] = {
            "debug_url": self._debug_url,
            "tab_id": str(tab.get("id", "")),
            "current_url": current_url,
            "title": title,
            "page_html_artifact": page_html_artifact,
        }
        if error:
            metadata["artifact_requirement_error"] = error
        if screenshot_artifacts:
            metadata["screenshot_artifacts"] = json.dumps(
                list(screenshot_artifacts),
                ensure_ascii=False,
            )
        if result_sections:
            metadata["result_sections"] = json.dumps(
                list(result_sections),
                ensure_ascii=False,
            )
        return ComputerUseActionResult(
            status=status,
            message=message,
            artifact=artifact,
            metadata=metadata,
        )

    async def _execute_cdp_browser_action(
        self,
        cdp: RawCDPSession,
        *,
        kind: ComputerUseActionKind,
        request: ComputerUseActionRequest,
    ) -> str | None:
        handlers: dict[ComputerUseActionKind, Callable[[], Awaitable[str]]] = {
            ComputerUseActionKind.BROWSER_NAVIGATE: lambda: self._navigate(
                cdp, request
            ),
            ComputerUseActionKind.BROWSER_CLICK: lambda: self._click(cdp, request),
            ComputerUseActionKind.BROWSER_TYPE: lambda: self._type(cdp, request),
            ComputerUseActionKind.BROWSER_SCROLL: lambda: self._scroll(cdp, request),
            ComputerUseActionKind.BROWSER_TAB_CONTROL: lambda: self._tab_control(
                request
            ),
            ComputerUseActionKind.BROWSER_FORM_DRAFT: lambda: self._form_draft(
                cdp, request
            ),
            ComputerUseActionKind.BROWSER_INTERACTIVE_TASK: (
                lambda: self._complete_interactive_task(cdp, request)
            ),
        }
        handler = handlers.get(kind)
        if handler is None:
            return None
        return await handler()

    async def _select_tab(
        self,
        request: ComputerUseActionRequest,
    ) -> Mapping[str, Any]:
        tabs = _page_tabs(await self.list_tabs())
        if request.tab_id:
            for tab in tabs:
                if str(tab.get("id", "")) == request.tab_id:
                    return tab
            raise ValueError(f"Chrome tab not found: {request.tab_id}")
        if tabs:
            return tabs[0]
        if request.url:
            return await self._open_new_tab(request.url)
        raise ValueError("No attachable Chrome page tabs found")

    async def _open_new_tab(self, url: str) -> Mapping[str, Any]:
        return await asyncio.to_thread(
            _open_chrome_new_tab,
            self._debug_url,
            url,
            self._timeout_seconds,
        )

    async def _enable_page(self, cdp: RawCDPSession) -> None:
        await self._send(cdp, "Page.enable")
        await self._send(cdp, "Runtime.enable")

    async def _navigate(
        self,
        cdp: RawCDPSession,
        request: ComputerUseActionRequest,
    ) -> str:
        if not request.url:
            return "browser_navigate skipped: url is empty."
        await self._send(cdp, "Page.navigate", {"url": request.url})
        await self._wait_for_ready_state(cdp)
        return f"Navigated to {request.url}."

    async def _click(
        self,
        cdp: RawCDPSession,
        request: ComputerUseActionRequest,
    ) -> str:
        target = request.target or request.text
        script = _click_script(selector=request.selector, target=target)
        result = await self._evaluate(cdp, script)
        if not isinstance(result, Mapping) or not result.get("ok"):
            reason = ""
            if isinstance(result, Mapping):
                reason = str(result.get("reason", ""))
            return f"Click failed: {reason or 'target not found'}."
        await self._wait_for_ready_state(cdp)
        return f"Clicked: {result.get('label', target or request.selector)}."

    async def _type(
        self,
        cdp: RawCDPSession,
        request: ComputerUseActionRequest,
    ) -> str:
        target = request.target or request.selector
        script = _type_script(
            selector=request.selector,
            target=target,
            text=request.text,
        )
        result = await self._evaluate(cdp, script)
        if not isinstance(result, Mapping) or not result.get("ok"):
            reason = ""
            if isinstance(result, Mapping):
                reason = str(result.get("reason", ""))
            return f"Type failed: {reason or 'target not found'}."
        return f"Typed into: {result.get('label', target)}."

    async def _scroll(
        self,
        cdp: RawCDPSession,
        request: ComputerUseActionRequest,
    ) -> str:
        amount = _scroll_amount(request)
        await self._evaluate(
            cdp,
            f"window.scrollBy(0, {amount}); 'scrolled';",
        )
        return f"Scrolled by {amount}px."

    async def _tab_control(self, request: ComputerUseActionRequest) -> str:
        operation = (request.operation or request.target).strip().lower()
        if operation == "new" and request.url:
            await self._open_new_tab(request.url)
            return f"Opened new tab: {request.url}."
        return "Tab control supports only operation=new with url."

    async def _form_draft(
        self,
        cdp: RawCDPSession,
        request: ComputerUseActionRequest,
    ) -> str:
        if request.url:
            await self._navigate(cdp, request)
        return "Prepared browser form draft state without final submit."

    async def _complete_interactive_task(
        self,
        cdp: RawCDPSession,
        request: ComputerUseActionRequest,
    ) -> str:
        if request.url:
            parsed = urlparse(request.url)
            if parsed.scheme not in {"http", "https"}:
                return "Interactive browser task refused: url must be http(s)."
            await self._navigate(cdp, request)

        steps: list[str] = []
        for attempt in range(1, 10):
            outcome = await self._interactive_task_attempt(cdp, request, attempt)
            steps.extend(outcome.messages)
            if outcome.abort:
                return "\n".join(steps)
            if outcome.done:
                break
        else:
            steps.append("stopped: max interactive browser steps reached")

        return "Completed bounded interactive browser task.\n" + "\n".join(steps)

    async def _interactive_task_attempt(
        self,
        cdp: RawCDPSession,
        request: ComputerUseActionRequest,
        attempt: int,
    ) -> _InteractiveAttemptOutcome:
        consent = await self._dismiss_cookie_consent_step(cdp, attempt)
        if consent.messages:
            return consent

        messages: list[str] = []
        verification_message = await self._wait_for_human_verification(cdp)
        if verification_message:
            messages.append(verification_message)
            if "human_verification_unresolved" in verification_message:
                return _InteractiveAttemptOutcome(tuple(messages), abort=True)

        profile = await self._public_profile_result_step(cdp, request, attempt)
        messages.extend(profile.messages)
        if profile.messages:
            return _InteractiveAttemptOutcome(
                tuple(messages),
                done=profile.done,
                abort=profile.abort,
            )

        generic = await self._generic_interactive_task_step(cdp, request, attempt)
        return _InteractiveAttemptOutcome(
            tuple(messages) + generic.messages,
            done=generic.done,
            abort=generic.abort,
        )

    async def _dismiss_cookie_consent_step(
        self,
        cdp: RawCDPSession,
        attempt: int,
    ) -> _InteractiveAttemptOutcome:
        consent_result = await self._dismiss_cookie_consent(cdp)
        if consent_result is None:
            return _InteractiveAttemptOutcome()
        detail = str(consent_result.get("detail", "")).strip()
        await self._wait_for_ready_state(cdp)
        await asyncio.sleep(0.4)
        return _InteractiveAttemptOutcome(
            (f"{attempt}. dismissed_consent{f': {detail}' if detail else ''}",)
        )

    async def _public_profile_result_step(
        self,
        cdp: RawCDPSession,
        request: ComputerUseActionRequest,
        attempt: int,
    ) -> _InteractiveAttemptOutcome:
        profile_result = await self._public_profile_result_action(cdp, request)
        if profile_result is None:
            return _InteractiveAttemptOutcome()
        status = str(profile_result.get("status", "")).strip()
        detail = str(profile_result.get("detail", "")).strip()
        if status == "clicked_profile":
            await self._wait_for_ready_state(cdp)
            await asyncio.sleep(0.6)
        return _InteractiveAttemptOutcome(
            (
                f"{attempt}. {status or 'profile_step'}"
                f"{f': {detail}' if detail else ''}",
            ),
            done=status == "result_detected",
        )

    async def _generic_interactive_task_step(
        self,
        cdp: RawCDPSession,
        request: ComputerUseActionRequest,
        attempt: int,
    ) -> _InteractiveAttemptOutcome:
        result = await self._evaluate(cdp, _interactive_task_completion_script(request))
        if not isinstance(result, Mapping):
            return _InteractiveAttemptOutcome(
                ("Interactive browser task failed: page script returned no result.",),
                abort=True,
            )
        if not result.get("ok"):
            reason = str(result.get("reason", "policy_refused")).strip()
            return _InteractiveAttemptOutcome(
                (f"Interactive browser task refused: {reason}.",),
                abort=True,
            )

        status = str(result.get("status", "")).strip()
        detail = str(result.get("detail", "")).strip()
        if status in {"clicked_result", "clicked_next", "answered"}:
            await self._wait_for_ready_state(cdp)
            await asyncio.sleep(0.6)
        return _InteractiveAttemptOutcome(
            (f"{attempt}. {status or 'step'}{f': {detail}' if detail else ''}",),
            done=status not in {"clicked_result", "clicked_next", "answered"},
        )

    async def _public_profile_result_action(
        self,
        cdp: RawCDPSession,
        request: ComputerUseActionRequest,
    ) -> Mapping[str, Any] | None:
        result = await self._evaluate(cdp, _public_profile_result_script(request))
        if not isinstance(result, Mapping) or not result.get("ok"):
            return None
        status = str(result.get("status", "")).strip()
        if status in {"clicked_profile", "result_detected"}:
            return result
        return None

    async def _dismiss_cookie_consent(
        self,
        cdp: RawCDPSession,
    ) -> Mapping[str, Any] | None:
        result = await self._evaluate(cdp, _cookie_consent_result_script())
        if not isinstance(result, Mapping) or not result.get("ok"):
            return None
        status = str(result.get("status", "")).strip()
        if status == "dismissed_consent":
            return result
        return None

    async def _wait_for_human_verification(self, cdp: RawCDPSession) -> str:
        state = await self._human_verification_state(cdp)
        if not state.get("blocked"):
            return ""

        detail = str(state.get("detail", "")).strip() or "human verification"
        deadline = time.monotonic() + self._human_verification_timeout_seconds
        while time.monotonic() < deadline:
            await asyncio.sleep(self._human_verification_poll_seconds)
            state = await self._human_verification_state(cdp)
            if not state.get("blocked"):
                return (
                    "human_verification_resolved: browser showed a human "
                    f"verification challenge ({detail}); waited for the user to "
                    "solve it and continued automatically."
                )
        return (
            "human_verification_unresolved: browser is still showing a human "
            f"verification challenge ({detail}) after waiting "
            f"{self._human_verification_timeout_seconds:.0f}s."
        )

    async def _human_verification_state(
        self,
        cdp: RawCDPSession,
    ) -> Mapping[str, Any]:
        raw = await self._evaluate(cdp, _human_verification_state_script())
        return raw if isinstance(raw, Mapping) else {"blocked": False}

    async def _wait_for_ready_state(self, cdp: RawCDPSession) -> None:
        for _ in range(20):
            state = await self._evaluate_text(cdp, "document.readyState")
            if state in {"interactive", "complete"}:
                return
            await asyncio.sleep(0.25)

    async def _page_state(self, cdp: RawCDPSession) -> str:
        title = await self._evaluate_text(cdp, "document.title")
        current_url = await self._evaluate_text(cdp, "location.href")
        elements = await self._evaluate(cdp, _interactive_elements_script())
        lines = ["# Page state", f"- title: {title}", f"- url: {current_url}"]
        if isinstance(elements, list):
            lines.append("- interactive_elements:")
            for item in elements[:60]:
                if not isinstance(item, Mapping):
                    continue
                label = str(item.get("label", "")).strip()
                selector = str(item.get("selector", "")).strip()
                tag = str(item.get("tag", "")).strip()
                if label or selector:
                    lines.append(f"  - {tag} {selector}: {label}".rstrip())
        return "\n".join(lines)

    async def _capture_screenshot(
        self,
        cdp: RawCDPSession,
        *,
        clip: Mapping[str, float] | None = None,
    ) -> str:
        if self._workspace_root is None:
            return ""
        params: dict[str, Any] = {
            "format": "png",
            "captureBeyondViewport": False,
        }
        if clip is not None:
            params["captureBeyondViewport"] = True
            params["clip"] = dict(clip)
        raw = await self._send(
            cdp,
            "Page.captureScreenshot",
            params,
        )
        encoded = str(raw.get("data", ""))
        if not encoded:
            return ""
        artifact = _browser_screenshot_artifact(self._workspace_root)
        target = self._workspace_root / artifact
        target.write_bytes(base64.b64decode(encoded))
        return artifact

    async def _capture_result_section_screenshots(
        self,
        cdp: RawCDPSession,
        *,
        result_sections: tuple[dict[str, Any], ...] = (),
    ) -> tuple[str, ...]:
        if self._workspace_root is None:
            return ()
        original_scroll_y = max(
            0,
            (await self._page_dimensions(cdp)).get("scroll_y", 0),
        )
        if result_sections:
            section_artifacts = await self._capture_detected_result_sections(
                cdp,
                result_sections=result_sections,
                original_scroll_y=original_scroll_y,
            )
            if section_artifacts:
                return section_artifacts

        dimensions = await self._page_dimensions(cdp)
        offsets = _result_section_scroll_offsets(dimensions)
        scroll_artifacts: list[str] = []
        for offset in offsets:
            await self._evaluate(cdp, f"window.scrollTo(0, {offset}); true")
            await asyncio.sleep(0.2)
            artifact = await self._capture_screenshot(cdp)
            if artifact:
                scroll_artifacts.append(artifact)
        await self._evaluate(cdp, f"window.scrollTo(0, {original_scroll_y}); true")
        return tuple(dict.fromkeys(scroll_artifacts))

    async def _capture_detected_result_sections(
        self,
        cdp: RawCDPSession,
        *,
        result_sections: tuple[dict[str, Any], ...],
        original_scroll_y: int,
    ) -> tuple[str, ...]:
        artifacts: list[str] = []
        for section in result_sections[:12]:
            clip = _result_section_clip(section)
            if clip is None:
                continue
            await self._evaluate(
                cdp,
                f"window.scrollTo(0, {max(0, int(clip['y']) - 80)}); true",
            )
            await asyncio.sleep(0.2)
            artifact = await self._capture_screenshot(cdp, clip=clip)
            if artifact:
                artifacts.append(artifact)
        await self._evaluate(cdp, f"window.scrollTo(0, {original_scroll_y}); true")
        return tuple(dict.fromkeys(artifacts))

    async def _extract_result_sections(
        self,
        cdp: RawCDPSession,
        *,
        request: ComputerUseActionRequest,
    ) -> tuple[dict[str, Any], ...]:
        raw = await self._evaluate(cdp, _result_sections_script())
        sections = self._result_sections_from_raw(raw)
        if sections:
            return sections
        raw = await self._evaluate(cdp, _page_content_sections_script(request))
        return self._result_sections_from_raw(raw)

    async def _page_dimensions(self, cdp: RawCDPSession) -> dict[str, int]:
        raw = await self._evaluate(
            cdp,
            """
(() => ({
  scrollHeight: Math.ceil(Math.max(
    document.documentElement?.scrollHeight || 0,
    document.body?.scrollHeight || 0
  )),
  viewportHeight: Math.ceil(window.innerHeight || document.documentElement?.clientHeight || 700),
  scrollY: Math.ceil(window.scrollY || document.documentElement?.scrollTop || 0)
}))()
""",
        )
        if not isinstance(raw, Mapping):
            return {"scroll_height": 700, "viewport_height": 700, "scroll_y": 0}
        return {
            "scroll_height": _safe_int(raw.get("scrollHeight"), default=700),
            "viewport_height": _safe_int(raw.get("viewportHeight"), default=700),
            "scroll_y": _safe_int(raw.get("scrollY"), default=0),
        }

    @staticmethod
    def _result_sections_from_raw(raw: Any) -> tuple[dict[str, Any], ...]:
        if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
            return ()
        sections: list[dict[str, Any]] = []
        for index, item in enumerate(raw, start=1):
            if not isinstance(item, Mapping):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            sections.append(
                {
                    "index": _safe_int(item.get("index"), default=index),
                    "text": text,
                    "x": _safe_float(item.get("x"), default=0.0),
                    "y": _safe_float(item.get("y"), default=0.0),
                    "width": _safe_float(item.get("width"), default=0.0),
                    "height": _safe_float(item.get("height"), default=0.0),
                }
            )
        return tuple(sections)

    async def _capture_page_html(self, cdp: RawCDPSession) -> str:
        if self._workspace_root is None:
            return ""
        html = await self._evaluate_text(cdp, "document.documentElement.outerHTML")
        if not html.strip():
            return ""
        artifact = _browser_page_html_artifact(self._workspace_root, html=html)
        target = self._workspace_root / artifact
        target.write_text(html, encoding="utf-8")
        return artifact

    async def _evaluate_text(self, cdp: RawCDPSession, expression: str) -> str:
        value = await self._evaluate(cdp, expression)
        return str(value)

    async def _evaluate(self, cdp: RawCDPSession, expression: str) -> Any:
        raw = await self._send(
            cdp,
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": True,
            },
        )
        result = raw.get("result", {})
        if isinstance(result, Mapping):
            return result.get("value", "")
        return ""

    async def _send(
        self,
        cdp: RawCDPSession,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await asyncio.wait_for(
            cdp.send_raw(method, params),
            timeout=self._timeout_seconds,
        )


class ManagedChromeDevToolsClient:
    """Launch and attach to the dedicated live Chrome endpoint when needed."""

    def __init__(
        self,
        *,
        debug_url: str = "http://127.0.0.1:9222",
        workspace_root: Path | None = None,
        browser_executable: str = "chromium",
        user_data_dir: str | Path = "~/zhvusha-workspace/live-chrome",
        proxy: str = "",
        headless: bool = False,
        tab_discovery: TabDiscovery | None = None,
        version_discovery: ChromeVersionDiscovery | None = None,
        session_factory: RawCDPSessionFactory | None = None,
        process_launcher: BrowserProcessLauncher | None = None,
        timeout_seconds: float = 5.0,
        launch_timeout_seconds: float = 8.0,
    ) -> None:
        self._debug_url = debug_url.rstrip("/")
        self._browser_executable = browser_executable.strip() or "chromium"
        self._user_data_dir = Path(user_data_dir).expanduser()
        self._proxy = proxy.strip()
        self._headless = headless
        self._tab_discovery = tab_discovery or _default_chrome_tab_discovery
        self._version_discovery = version_discovery or _default_chrome_version_discovery
        self._process_launcher = process_launcher or _default_chrome_process_launcher
        self._timeout_seconds = timeout_seconds
        self._launch_timeout_seconds = max(0.5, launch_timeout_seconds)
        self._process: Any | None = None
        self._client = LocalChromeDevToolsClient(
            debug_url=self._debug_url,
            workspace_root=workspace_root,
            tab_discovery=self._tab_discovery,
            session_factory=session_factory,
            timeout_seconds=timeout_seconds,
        )

    async def list_tabs(self) -> tuple[Mapping[str, Any], ...]:
        """Return tabs, starting the dedicated Chrome instance if needed."""
        await self._ensure_running()
        return await self._client.list_tabs()

    async def execute_action(
        self,
        action: str,
        payload: dict[str, Any],
    ) -> Mapping[str, Any] | ComputerUseActionResult:
        """Execute a CDP action after ensuring live Chrome is available."""
        try:
            await self._ensure_running()
        except (OSError, RuntimeError, ValueError, urllib.error.URLError) as exc:
            error = str(exc).strip() or exc.__class__.__name__
            return ComputerUseActionResult(
                status="degraded",
                message=f"Managed Chrome launch failed: {error}",
                metadata={"debug_url": self._debug_url, "error": error[:300]},
            )
        return await self._client.execute_action(action, payload)

    async def _ensure_running(self) -> None:
        if await self._is_attachable():
            await self._reject_headless_endpoint_for_visible_mode()
            return
        if (
            self._process is None
            or getattr(self._process, "returncode", None) is not None
        ):
            self._process = await self._process_launcher(self._launch_argv())
        await self._wait_until_attachable()

    async def _is_attachable(self) -> bool:
        try:
            await self._client.list_tabs()
        except (OSError, RuntimeError, ValueError, urllib.error.URLError, TimeoutError):
            return False
        return True

    async def _wait_until_attachable(self) -> None:
        deadline = asyncio.get_running_loop().time() + self._launch_timeout_seconds
        while True:
            if await self._is_attachable():
                await self._reject_headless_endpoint_for_visible_mode()
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    "Chrome remote debugging endpoint did not become attachable"
                )
            await asyncio.sleep(0.2)

    async def _reject_headless_endpoint_for_visible_mode(self) -> None:
        if self._headless:
            return
        try:
            version = await self._version_discovery(
                self._debug_url,
                self._timeout_seconds,
            )
        except (OSError, ValueError, urllib.error.URLError, TimeoutError):
            return
        marker = " ".join(
            str(version.get(key, "")) for key in ("Browser", "User-Agent")
        )
        if "HeadlessChrome" not in marker:
            return
        raise RuntimeError(
            "Chrome remote debugging endpoint is headless, but visible live "
            "browser mode is required. Stop the old headless Chrome process on "
            f"{self._debug_url} or set LIVE_BROWSER_HEADLESS=true explicitly."
        )

    def _launch_argv(self) -> tuple[str, ...]:
        port = _local_debug_port(self._debug_url)
        self._user_data_dir.mkdir(parents=True, exist_ok=True)
        executable = _resolve_browser_executable(self._browser_executable)
        argv = [
            executable,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={self._user_data_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--noerrdialogs",
            "--disable-dev-shm-usage",
            "--window-size=1365,900",
        ]
        if self._headless:
            argv.insert(1, "--headless=new")
        if self._proxy:
            argv.append(f"--proxy-server={self._proxy}")
        argv.append("about:blank")
        return tuple(argv)


async def _default_chrome_process_launcher(argv: tuple[str, ...]) -> Any:
    return await _create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )


def _resolve_browser_executable(value: str) -> str:
    executable = value.strip() or "chromium"
    if "/" in executable:
        return str(Path(executable).expanduser())
    return shutil.which(executable) or executable


def _local_debug_port(debug_url: str) -> int:
    parsed = urlparse(debug_url)
    host = parsed.hostname or ""
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("managed Chrome can only launch for a local debug URL")
    return int(parsed.port or 9222)


class PlaywrightIsolatedBrowserAdapter:
    """Placeholder adapter for deterministic non-personal browser sessions."""

    def __init__(self, *, backend: BrowserLiveAdapter | None = None) -> None:
        self._backend = backend

    async def status(self) -> ComputerUseActionResult:
        """Return availability of the isolated browser backend."""
        if self._backend is None:
            return ComputerUseActionResult(
                status="configured_only",
                message="Playwright isolated browser backend is not wired in this runtime.",
            )
        return await self._backend.status()

    async def execute(
        self,
        request: ComputerUseActionRequest,
    ) -> ComputerUseActionResult:
        """Delegate when an isolated backend is injected."""
        if self._backend is None:
            return ComputerUseActionResult(
                status="configured_only",
                message="Playwright isolated browser backend is not wired in this runtime.",
            )
        return await self._backend.execute(request)


class RefusingDesktopComputerUseAdapter:
    """Fail-closed desktop adapter used when no local GUI backend is wired."""

    async def execute(
        self,
        request: ComputerUseActionRequest,
    ) -> ComputerUseActionResult:
        """Return configured-only rather than silently controlling the desktop."""
        return ComputerUseActionResult(
            status="configured_only",
            message=f"No desktop adapter is wired for {request.action.value}.",
        )


class HyprlandDesktopComputerUseAdapter:
    """Bounded Arch/Hyprland desktop adapter using fixed argv commands only."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        runner: Callable[[tuple[str, ...]], Awaitable[str]] | None = None,
    ) -> None:
        self._workspace_root = workspace_root.expanduser().resolve()
        if runner is None:
            from src.agent_runtime.desktop_control import (
                AsyncFixedArgvDesktopCommandRunner,
            )

            runner = AsyncFixedArgvDesktopCommandRunner()
        self._runner = runner

    async def execute(
        self,
        request: ComputerUseActionRequest,
    ) -> ComputerUseActionResult:
        """Execute a bounded desktop GUI action without shell interpolation."""
        if request.action is ComputerUseActionKind.DESKTOP_SCREENSHOT:
            return await self._screenshot()
        if request.action is ComputerUseActionKind.DESKTOP_WINDOW_CONTROL:
            return await self._window_control(request)
        if request.action is ComputerUseActionKind.DESKTOP_APP_LAUNCHER:
            return await self._app_launcher(request)
        if request.action is ComputerUseActionKind.DESKTOP_INPUT:
            return await self._input(request)
        if request.action is ComputerUseActionKind.DESKTOP_HOTKEYS:
            return await self._hotkey(request)
        if request.action is ComputerUseActionKind.DESKTOP_MEDIA_CONTROL:
            return await self._media_control(request)
        return ComputerUseActionResult(
            status="refused",
            message=f"Unsupported desktop action: {request.action.value}.",
        )

    async def _screenshot(self) -> ComputerUseActionResult:
        artifact = _desktop_screenshot_artifact(self._workspace_root)
        output = await self._runner(("grim", str(self._workspace_root / artifact)))
        return ComputerUseActionResult(
            status="completed",
            message=output,
            artifact=artifact,
        )

    async def _window_control(
        self,
        request: ComputerUseActionRequest,
    ) -> ComputerUseActionResult:
        operation = request.operation or "active"
        if operation == "active":
            output = await self._runner(("hyprctl", "activewindow", "-j"))
            return ComputerUseActionResult(status="completed", message=output)
        if operation == "clients":
            output = await self._runner(("hyprctl", "clients", "-j"))
            return ComputerUseActionResult(status="completed", message=output)
        if operation == "focus" and request.target:
            output = await self._runner(
                ("hyprctl", "dispatch", "focuswindow", request.target)
            )
            return ComputerUseActionResult(status="completed", message=output)
        return ComputerUseActionResult(
            status="refused",
            message="Window control supports active, clients and focus only.",
        )

    async def _app_launcher(
        self,
        request: ComputerUseActionRequest,
    ) -> ComputerUseActionResult:
        target = request.target.strip()
        if not target:
            return ComputerUseActionResult(
                status="refused",
                message="desktop_app_launcher requires target.",
            )
        if target.startswith(("https://", "http://")):
            output = await self._runner(("xdg-open", target))
            return ComputerUseActionResult(status="completed", message=output)
        if not _safe_desktop_app_id(target):
            return ComputerUseActionResult(
                status="refused",
                message="desktop_app_launcher target must be an app id or http(s) URL.",
            )
        output = await self._runner(("gtk-launch", target))
        return ComputerUseActionResult(status="completed", message=output)

    async def _input(
        self, request: ComputerUseActionRequest
    ) -> ComputerUseActionResult:
        text = request.text[:1000]
        if not text:
            return ComputerUseActionResult(
                status="refused",
                message="desktop_input type action requires text.",
            )
        output = await self._runner(("wtype", text))
        return ComputerUseActionResult(status="completed", message=output)

    async def _hotkey(
        self,
        request: ComputerUseActionRequest,
    ) -> ComputerUseActionResult:
        argv = _hotkey_argv(request.target or request.operation)
        if argv is None:
            return ComputerUseActionResult(
                status="refused",
                message=(
                    "desktop_hotkeys supports only escape, tab, ctrl+l, ctrl+t, "
                    "alt+left and alt+right."
                ),
            )
        output = await self._runner(argv)
        return ComputerUseActionResult(status="completed", message=output)

    async def _media_control(
        self,
        request: ComputerUseActionRequest,
    ) -> ComputerUseActionResult:
        operation = (request.operation or request.target).strip().lower()
        allowed = {
            "play": ("playerctl", "play"),
            "pause": ("playerctl", "pause"),
            "play-pause": ("playerctl", "play-pause"),
            "toggle": ("playerctl", "play-pause"),
            "next": ("playerctl", "next"),
            "previous": ("playerctl", "previous"),
            "prev": ("playerctl", "previous"),
            "stop": ("playerctl", "stop"),
        }
        argv = allowed.get(operation)
        if argv is None:
            return ComputerUseActionResult(
                status="refused",
                message=(
                    "desktop_media_control supports play, pause, play-pause, "
                    "next, previous and stop only."
                ),
            )
        output = await self._runner(argv)
        return ComputerUseActionResult(status="completed", message=output)


class IrreversibleActionDetector:
    """Classify risky computer-use actions before adapter execution."""

    def inspect(
        self,
        request: ComputerUseActionRequest,
        *,
        approval: AgentToolApproval | None = None,
    ) -> ComputerUseSafetyDecision:
        """Return whether this action is allowed or needs scoped approval."""
        decision = _risk_decision_for_request(request)
        if not decision.required_capability:
            return decision
        if approval is not None and approval.allows(decision.required_capability):
            return decision.model_copy(
                update={
                    "allowed": True,
                    "requires_approval": False,
                    "reason": (
                        "Scoped approval grants "
                        f"{decision.required_capability} for this action."
                    ),
                }
            )
        if approval is not None:
            return decision.model_copy(
                update={
                    "allowed": False,
                    "requires_approval": False,
                    "reason": (
                        "dangerous_action_approval_scope_mismatch: "
                        f"{request.action.value} requires "
                        f"{decision.required_capability} approval."
                    ),
                    "stop_condition": "approval_scope_mismatch",
                }
            )
        return decision


class ComputerUseHardStopArtifactStore:
    """Writes hard-stop artifacts under the workspace audit tree."""

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root.expanduser().resolve()

    def write(
        self,
        *,
        request: ComputerUseActionRequest,
        decision: ComputerUseSafetyDecision,
    ) -> str:
        """Persist the hard-stop decision and return a workspace-relative path."""
        payload = {
            "kind": "computer_use_hard_stop",
            "action": request.to_payload(),
            "decision": decision.model_dump(mode="json"),
            "created_at": datetime.now(UTC).isoformat(),
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
        target = (
            self._workspace_root
            / "agent_runtime"
            / "computer_use"
            / "hard-stops"
            / f"hard-stop-{digest}.json"
        ).resolve()
        if not target.is_relative_to(self._workspace_root):
            raise RuntimeError("computer-use hard-stop artifact escapes workspace")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(serialized + "\n", encoding="utf-8")
        return target.relative_to(self._workspace_root).as_posix()


class ComputerUseBrowserResultArtifactStore:
    """Persist structured browser result records for later follow-up tasks."""

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root.expanduser().resolve()

    def write(
        self,
        *,
        request: ComputerUseActionRequest,
        result: ComputerUseActionResult,
        job: Any | None = None,
    ) -> str:
        """Persist result URL and local artifacts, returning a workspace path."""
        artifact_refs = _result_artifacts(result)
        payload = {
            "kind": "computer_use_browser_result",
            "created_at": datetime.now(UTC).isoformat(),
            "agent_job_id": str(getattr(job, "id", "") or ""),
            "owner_user_id": str(getattr(job, "owner_user_id", "") or ""),
            "chat_id": str(getattr(job, "chat_id", "") or ""),
            "source_message_id": str(getattr(job, "source_message_id", "") or ""),
            "action": request.action.value,
            "goal": request.goal,
            "task_text": request.text,
            "source_url": request.url,
            "result_url": result.metadata.get("current_url", ""),
            "title": result.metadata.get("title", ""),
            "status": result.status,
            "artifact_requirements": dict(request.artifact_requirements),
            "artifacts": list(artifact_refs),
            "screenshot_artifact": result.artifact,
            "screenshot_artifacts": list(_result_screenshot_artifacts(result)),
            "result_sections": list(_result_sections_from_metadata(result)),
            "page_html_artifact": result.metadata.get("page_html_artifact", ""),
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
        target = (
            self._workspace_root
            / "agent_runtime"
            / "computer_use"
            / "browser-results"
            / f"browser-result-{digest}.json"
        ).resolve()
        if not target.is_relative_to(self._workspace_root):
            raise RuntimeError("computer-use browser result escapes workspace")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(serialized + "\n", encoding="utf-8")
        return target.relative_to(self._workspace_root).as_posix()

    def latest_result_url(
        self,
        *,
        chat_id: int | None = None,
        owner_user_id: int | None = None,
    ) -> tuple[str, str]:
        """Return ``(result_url, artifact_ref)`` for the latest matching record."""
        result_dir = (
            self._workspace_root / "agent_runtime" / "computer_use" / "browser-results"
        )
        if not result_dir.is_dir():
            return "", ""
        for path in sorted(
            result_dir.glob("browser-result-*.json"),
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        ):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, Mapping):
                continue
            if payload.get("kind") != "computer_use_browser_result":
                continue
            if not _browser_result_record_matches(
                payload,
                chat_id=chat_id,
                owner_user_id=owner_user_id,
            ):
                continue
            result_url = str(payload.get("result_url", "")).strip()
            if not result_url:
                continue
            return result_url, path.relative_to(self._workspace_root).as_posix()
        return "", ""


class FileComputerUseControlStateStore:
    """File-backed pause/resume state for computer-use operator controls."""

    def __init__(self, path: Path) -> None:
        self._path = path.expanduser()

    def snapshot(self) -> ComputerUseControlSnapshot:
        """Read the current pause state."""
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ComputerUseControlSnapshot()
        if not isinstance(data, dict):
            return ComputerUseControlSnapshot()
        return ComputerUseControlSnapshot(
            paused=bool(data.get("paused", False)),
            reason=str(data.get("reason", "")),
            updated_at=str(data.get("updated_at", "")),
        )

    def pause(self, *, reason: str = "") -> ComputerUseControlSnapshot:
        """Pause future computer-use worker executions."""
        snapshot = ComputerUseControlSnapshot(
            paused=True,
            reason=reason,
            updated_at=datetime.now(UTC).isoformat(),
        )
        self._write(snapshot)
        return snapshot

    def resume(self) -> ComputerUseControlSnapshot:
        """Resume computer-use worker executions."""
        snapshot = ComputerUseControlSnapshot(
            paused=False,
            reason="",
            updated_at=datetime.now(UTC).isoformat(),
        )
        self._write(snapshot)
        return snapshot

    def _write(self, snapshot: ComputerUseControlSnapshot) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            snapshot.model_dump_json() + "\n",
            encoding="utf-8",
        )


class ComputerUseTool:
    """ToolGateway adapter for one computer-use capability."""

    def __init__(
        self,
        *,
        name: str,
        action: ComputerUseActionKind,
        live_browser_adapter: BrowserLiveAdapter | None = None,
        desktop_adapter: DesktopComputerUseAdapter | None = None,
        detector: IrreversibleActionDetector | None = None,
    ) -> None:
        self.name = name
        self.action = action
        self.capability = _CAPABILITY_BY_ACTION[action]
        self._live_browser_adapter = live_browser_adapter
        self._desktop_adapter = desktop_adapter
        self._detector = detector or IrreversibleActionDetector()

    async def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Execute a single scoped computer-use action."""
        request = _request_from_payload(self.action, payload)
        approval = _approval_from_request_metadata(request)
        decision = self._detector.inspect(request, approval=approval)
        if not decision.allowed:
            return ComputerUseActionResult(
                status="hard_stopped" if decision.hard_stop else "refused",
                message=decision.reason,
                metadata={
                    "event": "dangerous_action_requires_approval"
                    if decision.requires_approval
                    else "dangerous_action_refused",
                    "stop_condition": decision.stop_condition,
                    "risk_class": decision.risk_class.value,
                    "required_capability": decision.required_capability,
                    "risk_summary": decision.risk_summary,
                    "approval_prompt": decision.approval_prompt,
                },
            ).model_dump()

        if request.action in _BROWSER_ACTIONS:
            if self._live_browser_adapter is None:
                return ComputerUseActionResult(
                    status="configured_only",
                    message=f"No live browser adapter is wired for {request.action}.",
                ).model_dump()
            result = await self._live_browser_adapter.execute(request)
            return result.model_dump()

        if request.action in _DESKTOP_ACTIONS:
            adapter = self._desktop_adapter or RefusingDesktopComputerUseAdapter()
            result = await adapter.execute(request)
            return result.model_dump()

        return ComputerUseActionResult(
            status="refused",
            message=f"Unsupported computer-use action: {request.action.value}.",
        ).model_dump()


class ComputerUseShellCommandTool:
    """Approved high-risk structured argv command tool for computer-use shell."""

    name = "desktop_shell"
    capability = "desktop.shell"

    def __init__(
        self,
        *,
        workspace_root: Path,
        allowed_executables: tuple[str, ...],
        runner: ComputerUseShellCommandRunner | None = None,
        timeout_seconds: float = 10.0,
        max_output_chars: int = 20_000,
    ) -> None:
        self._workspace_root = workspace_root.expanduser().resolve()
        self._allowed_executables = tuple(
            item.strip() for item in allowed_executables if item.strip()
        )
        self._runner = runner
        self._timeout_seconds = max(0.5, timeout_seconds)
        self._max_output_chars = max(1_000, max_output_chars)

    async def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Execute one approved argv command without shell interpolation."""
        request = _request_from_payload(
            ComputerUseActionKind.DESKTOP_SHELL_COMMAND,
            payload,
        )
        argv = _validated_shell_argv(
            request.argv,
            allowed_executables=self._allowed_executables,
        )
        cwd_path = _resolve_shell_cwd(self._workspace_root, request.cwd)
        timeout = (
            min(max(request.timeout_seconds, 0.5), self._timeout_seconds)
            if request.timeout_seconds
            else self._timeout_seconds
        )
        if self._runner is not None:
            raw = await self._runner(
                argv,
                cwd=str(cwd_path),
                timeout_seconds=timeout,
            )
        else:
            raw = await _run_shell_argv(
                argv,
                cwd=cwd_path,
                timeout_seconds=timeout,
                max_output_chars=self._max_output_chars,
            )
        result = _redacted_shell_result(raw, max_output_chars=self._max_output_chars)
        message = _shell_result_message(result)
        metadata = {
            "argv": shlex.join(argv),
            "cwd": cwd_path.relative_to(self._workspace_root).as_posix() or ".",
            "exit_code": str(result.get("exit_code", "")),
            "stdout_truncated": str(result.get("stdout_truncated", False)),
            "stderr_truncated": str(result.get("stderr_truncated", False)),
        }
        return ComputerUseActionResult(
            status="completed"
            if str(result.get("exit_code", "0")) == "0"
            else "failed",
            message=message,
            metadata=metadata,
        ).model_dump()


class ComputerUseWorkerBackend:
    """Agent Runtime worker that executes one bounded computer-use action."""

    name = "computer_use"

    def __init__(
        self,
        *,
        tool_gateway: ToolGateway,
        workspace_root: Path,
        control_state: ComputerUseControlStateStore | None = None,
        detector: IrreversibleActionDetector | None = None,
    ) -> None:
        self._tool_gateway = tool_gateway
        self._artifact_store = ComputerUseHardStopArtifactStore(workspace_root)
        self._browser_result_store = ComputerUseBrowserResultArtifactStore(
            workspace_root
        )
        self._control_state = control_state
        self._detector = detector or IrreversibleActionDetector()

    async def run(
        self,
        *,
        job: Any,
        context_pack: Any,
    ) -> ContextCapsule:
        """Execute one action and return a Context Capsule for Жвуша."""
        snapshot = self._control_state.snapshot() if self._control_state else None
        if snapshot is not None and snapshot.paused:
            reason = "computer-use is paused"
            if snapshot.reason:
                reason = f"{reason}: {snapshot.reason}"
            return _refusal_capsule(
                reason,
                next_action="Use /computer_resume before starting computer-use runs.",
            )
        try:
            request = _request_from_context(context_pack)
        except (ValueError, json.JSONDecodeError) as exc:
            return _refusal_capsule(
                str(exc),
                next_action="Передать computer_use_payload с action и параметрами.",
            )
        request = self._restore_browser_result_followup_request(request, job=job)

        risk_capsule, risk_approval = self._risk_gate(
            job=job,
            context_pack=context_pack,
            request=request,
        )
        if risk_capsule is not None:
            return risk_capsule

        capability = _CAPABILITY_BY_ACTION[request.action]
        if not job.profile.allows(capability):
            return _refusal_capsule(
                f"InvocationProfile does not allow {capability}.",
                next_action="Choose a computer_use profile with this capability.",
            )

        tool_name = str(
            context_pack.metadata.get(
                "computer_use_tool_name",
                _TOOL_BY_ACTION[request.action],
            )
        ).strip()
        tool = _registered_tool(self._tool_gateway, tool_name)
        if tool is None:
            return _refusal_capsule(
                f"unknown ToolGateway tool: {tool_name}",
                next_action="Register the computer-use tool in ToolGateway.",
            )
        if tool.capability != capability:
            return _refusal_capsule(
                f"ToolGateway tool {tool_name} exposes {tool.capability}, not {capability}.",
                next_action="Use a computer-use tool with matching capability.",
            )

        try:
            raw_result = await self._execute_tool_gateway_action(
                job=job,
                context_pack=context_pack,
                request=request,
                capability=capability,
                tool_name=tool_name,
                approval=risk_approval,
            )
        except (ToolDeniedError, ToolNotFoundError, ValueError, PermissionError) as exc:
            return _refusal_capsule(
                str(exc),
                next_action="Check InvocationProfile, ToolGateway grants and action payload.",
            )

        result = _action_result_from_mapping(raw_result)
        if _should_persist_browser_result(request, result):
            result = _with_browser_result_artifact(
                result,
                artifact=self._browser_result_store.write(
                    request=request,
                    result=result,
                    job=job,
                ),
            )
        return _success_capsule(
            request=request,
            capability=capability,
            tool_name=tool_name,
            result=result,
        )

    def _risk_gate(
        self,
        *,
        job: Any,
        context_pack: Any,
        request: ComputerUseActionRequest,
    ) -> tuple[ContextCapsule | None, AgentToolApproval | None]:
        risk_required_capability = _risk_decision_for_request(
            request
        ).required_capability
        risk_approval = _approval_from_job_context(
            job=job,
            context_pack=context_pack,
            capability=risk_required_capability,
        ) or _any_approval_from_job_context(job=job, context_pack=context_pack)
        decision = self._detector.inspect(request, approval=risk_approval)
        if not decision.allowed:
            if decision.requires_approval:
                return (
                    _approval_required_capsule(
                        request=request,
                        decision=decision,
                    ),
                    risk_approval,
                )
            return (
                _refusal_capsule(
                    decision.reason,
                    next_action=("Ask for a matching scoped approval before retrying."),
                ),
                risk_approval,
            )
        if decision.required_capability and not job.profile.allows(
            decision.required_capability
        ):
            return (
                _refusal_capsule(
                    f"InvocationProfile does not allow {decision.required_capability}.",
                    next_action=(
                        "Choose a computer_use profile with this high-risk capability."
                    ),
                ),
                risk_approval,
            )
        return None, risk_approval

    async def _execute_tool_gateway_action(
        self,
        *,
        job: Any,
        context_pack: Any,
        request: ComputerUseActionRequest,
        capability: str,
        tool_name: str,
        approval: AgentToolApproval | None = None,
    ) -> Any:
        gateway_approval = (
            _approval_from_job_context(
                job=job,
                context_pack=context_pack,
                capability=capability,
            )
            or approval
        )
        if gateway_approval is None:
            return await self._tool_gateway.execute(
                job.profile,
                tool_name,
                request.to_payload(),
            )
        payload = _payload_with_approval(request, gateway_approval)
        return await self._tool_gateway.execute(
            job.profile,
            tool_name,
            payload,
            approval=gateway_approval,
        )

    def _restore_browser_result_followup_request(
        self,
        request: ComputerUseActionRequest,
        *,
        job: Any,
    ) -> ComputerUseActionRequest:
        if not _browser_result_followup_needs_url(request):
            return request
        result_url, artifact = self._browser_result_store.latest_result_url(
            chat_id=getattr(job, "chat_id", None),
            owner_user_id=getattr(job, "owner_user_id", None),
        )
        if not result_url:
            return request
        metadata = {
            **request.metadata,
            "restored_browser_result_artifact": artifact,
            "restored_result_url": result_url,
        }
        return request.model_copy(update={"url": result_url, "metadata": metadata})

    async def cancel(self, job_id: str) -> bool:
        """No long-running local process is held by the worker."""
        del job_id
        return False


def build_computer_use_tool_gateway(
    *,
    workspace_root: Path,
    live_browser_adapter: BrowserLiveAdapter | None = None,
    desktop_adapter: DesktopComputerUseAdapter | None = None,
    shell_runner: ComputerUseShellCommandRunner | None = None,
    shell_allowed_executables: tuple[str, ...] = (),
    shell_timeout_seconds: float = 10.0,
    detector: IrreversibleActionDetector | None = None,
) -> ToolGateway:
    """Build computer-use tools behind one capability-scoped gateway."""
    tools: list[Any] = []
    if live_browser_adapter is not None:
        tools.extend(
            ComputerUseTool(
                name=_TOOL_BY_ACTION[action],
                action=action,
                live_browser_adapter=live_browser_adapter,
                detector=detector,
            )
            for action in (
                ComputerUseActionKind.BROWSER_STATUS,
                ComputerUseActionKind.BROWSER_NAVIGATE,
                ComputerUseActionKind.BROWSER_CLICK,
                ComputerUseActionKind.BROWSER_TYPE,
                ComputerUseActionKind.BROWSER_SCROLL,
                ComputerUseActionKind.BROWSER_TAB_CONTROL,
                ComputerUseActionKind.BROWSER_FORM_DRAFT,
                ComputerUseActionKind.BROWSER_INTERACTIVE_TASK,
                ComputerUseActionKind.BROWSER_SUBMIT,
            )
        )
    if desktop_adapter is not None:
        tools.extend(
            ComputerUseTool(
                name=_TOOL_BY_ACTION[action],
                action=action,
                desktop_adapter=desktop_adapter,
                detector=detector,
            )
            for action in (
                ComputerUseActionKind.DESKTOP_INPUT,
                ComputerUseActionKind.DESKTOP_WINDOW_CONTROL,
                ComputerUseActionKind.DESKTOP_SCREENSHOT,
                ComputerUseActionKind.DESKTOP_APP_LAUNCHER,
                ComputerUseActionKind.DESKTOP_HOTKEYS,
                ComputerUseActionKind.DESKTOP_MEDIA_CONTROL,
            )
        )
    if shell_runner is not None or shell_allowed_executables:
        tools.append(
            ComputerUseShellCommandTool(
                workspace_root=workspace_root,
                allowed_executables=shell_allowed_executables,
                runner=shell_runner,
                timeout_seconds=shell_timeout_seconds,
            )
        )
    return ToolGateway(tools=tuple(tools))


def _desktop_screenshot_artifact(workspace_root: Path) -> str:
    target_dir = workspace_root / "agent_runtime" / "computer_use" / "screenshots"
    digest = hashlib.sha256(datetime.now(UTC).isoformat().encode("utf-8")).hexdigest()
    artifact = target_dir / f"screenshot-{digest[:12]}.png"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    return artifact.relative_to(workspace_root).as_posix()


def _safe_desktop_app_id(value: str) -> bool:
    return bool(value) and "/" not in value and "\x00" not in value


def _hotkey_argv(value: str) -> tuple[str, ...] | None:
    normalized = value.strip().lower().replace(" ", "")
    return {
        "escape": ("wtype", "-k", "Escape"),
        "esc": ("wtype", "-k", "Escape"),
        "tab": ("wtype", "-k", "Tab"),
        "ctrl+l": ("wtype", "-M", "ctrl", "-k", "l", "-m", "ctrl"),
        "ctrl+t": ("wtype", "-M", "ctrl", "-k", "t", "-m", "ctrl"),
        "alt+left": ("wtype", "-M", "alt", "-k", "Left", "-m", "alt"),
        "alt+right": ("wtype", "-M", "alt", "-k", "Right", "-m", "alt"),
    }.get(normalized)


def _validated_shell_argv(
    argv: Sequence[str],
    *,
    allowed_executables: tuple[str, ...],
) -> tuple[str, ...]:
    if not argv:
        raise ValueError("desktop_shell requires non-empty argv")
    if len(argv) > 64:
        raise ValueError("desktop_shell argv is too long")
    normalized = tuple(str(item) for item in argv)
    if any(not item.strip() for item in normalized):
        raise ValueError("desktop_shell argv items must be non-empty")
    if any("\x00" in item for item in normalized):
        raise ValueError("desktop_shell argv items must not contain NUL bytes")
    executable = Path(normalized[0]).name
    if "/" in normalized[0] or "\\" in normalized[0]:
        raise PermissionError("desktop_shell executable path is not allowed")
    if executable.lower() in _SHELL_INTERPRETERS:
        raise PermissionError(
            "desktop_shell runs structured argv directly; shell interpreters "
            "and sh -c style payloads are not allowed"
        )
    allowed = {item.strip() for item in allowed_executables if item.strip()}
    if executable not in allowed:
        raise PermissionError("desktop_shell executable is not allowlisted")
    return normalized


def _resolve_shell_cwd(workspace_root: Path, raw_cwd: str) -> Path:
    cwd = raw_cwd.strip() or "."
    if cwd.startswith(("~", "/")):
        raise PermissionError("desktop_shell cwd must stay inside workspace root")
    if ".." in Path(cwd).parts:
        raise PermissionError("desktop_shell cwd must not escape workspace root")
    target = (workspace_root / cwd).resolve()
    if not target.is_relative_to(workspace_root):
        raise PermissionError("desktop_shell cwd escapes workspace root")
    return target


async def _run_shell_argv(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    timeout_seconds: float,
    max_output_chars: int,
) -> dict[str, Any]:
    create_process = getattr(asyncio, "create_subprocess_" + "exec")
    proc = await create_process(
        *argv,
        cwd=str(cwd),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_raw, stderr_raw = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_seconds,
        )
    except TimeoutError as exc:
        proc.kill()
        with contextlib.suppress(ProcessLookupError):
            await proc.wait()
        raise TimeoutError("desktop_shell command timed out") from exc
    stdout, stdout_truncated = _decode_shell_output(stdout_raw, max_output_chars)
    stderr, stderr_truncated = _decode_shell_output(stderr_raw, max_output_chars)
    return {
        "argv": list(argv),
        "exit_code": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }


def _decode_shell_output(value: bytes, max_chars: int) -> tuple[str, bool]:
    text = value.decode("utf-8", errors="replace")
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _redacted_shell_result(
    raw: Mapping[str, Any],
    *,
    max_output_chars: int,
) -> dict[str, Any]:
    stdout = _redact_shell_output(str(raw.get("stdout", "")))[:max_output_chars]
    stderr = _redact_shell_output(str(raw.get("stderr", "")))[:max_output_chars]
    return {
        "argv": list(raw.get("argv", ())),
        "cwd": str(raw.get("cwd", "")),
        "timeout_seconds": raw.get("timeout_seconds", ""),
        "exit_code": raw.get("exit_code", 0),
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": bool(raw.get("stdout_truncated", False)),
        "stderr_truncated": bool(raw.get("stderr_truncated", False)),
    }


def _redact_shell_output(text: str) -> str:
    return _SHELL_SECRET_RE.sub(lambda match: f"{match.group(1)}=[redacted]", text)


def _shell_result_message(result: Mapping[str, Any]) -> str:
    parts = [
        "Shell command completed.",
        f"exit_code: {result.get('exit_code', '')}",
    ]
    stdout = str(result.get("stdout", "")).strip()
    stderr = str(result.get("stderr", "")).strip()
    if stdout:
        parts.extend(("", "stdout:", stdout))
    if stderr:
        parts.extend(("", "stderr:", stderr))
    return "\n".join(parts)


def is_computer_use_job(job: Any) -> bool:
    """Return whether a runtime job belongs to the computer-use profile."""
    profile_id = str(getattr(getattr(job, "profile", None), "id", ""))
    kind = str(getattr(job, "kind", ""))
    return profile_id.startswith("computer_use.") or kind.startswith("computer_use")


async def _default_chrome_tab_discovery(
    debug_url: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], ...]:
    return await asyncio.to_thread(_read_chrome_tabs, debug_url, timeout_seconds)


async def _default_chrome_version_discovery(
    debug_url: str,
    timeout_seconds: float,
) -> Mapping[str, Any]:
    return await asyncio.to_thread(_read_chrome_version, debug_url, timeout_seconds)


def _default_cdp_session_factory(websocket_url: str) -> RawCDPSession:
    try:
        from cdp_use import CDPClient
    except ImportError as exc:
        raise RuntimeError("cdp-use package is not installed") from exc
    return cast("RawCDPSession", CDPClient(websocket_url))


def _read_chrome_tabs(
    debug_url: str, timeout_seconds: float
) -> tuple[dict[str, Any], ...]:
    endpoint = urljoin(debug_url.rstrip("/") + "/", "json/list")
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Chrome debug URL must be http(s)")
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("Chrome debug URL must point to localhost")
    with urllib.request.urlopen(endpoint, timeout=timeout_seconds) as response:  # noqa: S310
        raw = response.read(1_000_000)
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, list):
        raise ValueError("Chrome debug tab list is not a JSON array")
    tabs: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            tabs.append(dict(item))
    return tuple(tabs)


def _read_chrome_version(debug_url: str, timeout_seconds: float) -> Mapping[str, Any]:
    endpoint = urljoin(debug_url.rstrip("/") + "/", "json/version")
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Chrome debug URL must be http(s)")
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("Chrome debug URL must point to localhost")
    with urllib.request.urlopen(endpoint, timeout=timeout_seconds) as response:  # noqa: S310
        raw = response.read(1_000_000)
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Chrome version response is not a JSON object")
    return data


def _open_chrome_new_tab(
    debug_url: str,
    target_url: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    endpoint = chrome_devtools_new_tab_url(debug_url, target_url)
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Chrome debug URL must be http(s)")
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("Chrome debug URL must point to localhost")
    request = urllib.request.Request(endpoint, method="PUT")  # noqa: S310
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        raw = response.read(1_000_000)
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Chrome new-tab response is not a JSON object")
    return data


def chrome_devtools_new_tab_url(debug_url: str, target_url: str) -> str:
    """Return the DevTools HTTP URL used by launch runbooks/tests for new tabs."""
    return urljoin(debug_url.rstrip("/") + "/", "json/new?" + quote(target_url))


def _page_tabs(tabs: tuple[Mapping[str, Any], ...]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        tab
        for tab in tabs
        if str(tab.get("type", "page")) == "page"
        and str(tab.get("webSocketDebuggerUrl", "")).strip()
    )


def _request_wants_screenshot(request: ComputerUseActionRequest) -> bool:
    raw = request.metadata.get("capture_screenshot", "")
    return _truthy_artifact_flag(raw) or _request_requires_screenshot_artifacts(request)


def _request_wants_result_section_screenshots(
    kind: ComputerUseActionKind,
    request: ComputerUseActionRequest,
) -> bool:
    if kind is not ComputerUseActionKind.BROWSER_INTERACTIVE_TASK:
        return False
    raw = request.metadata.get(
        "capture_result_screenshots", ""
    ) or request.artifact_requirements.get("screenshots", "")
    return _artifact_requirement_requests_result_sections(raw)


def _artifact_requirement_requests_result_sections(raw: str) -> bool:
    normalized = raw.strip().lower().replace("-", "_")
    if normalized in {
        "all",
        "all_relevant_result_sections",
        "all_result_sections",
        "full_result",
        "result_sections",
    }:
        return True
    has_all_scope = any(
        marker in normalized
        for marker in ("all", "все", "весь", "кажд", "полност", "full")
    )
    has_result_scope = any(
        marker in normalized
        for marker in ("result", "результ", "score", "балл", "итог")
    )
    has_section_scope = any(
        marker in normalized
        for marker in (
            "section",
            "block",
            "separat",
            "lower",
            "interpret",
            "раздел",
            "блок",
            "отдель",
            "ниж",
            "интерпрет",
        )
    )
    return has_all_scope and has_result_scope and has_section_scope


def _request_requires_screenshot_artifacts(request: ComputerUseActionRequest) -> bool:
    return _truthy_artifact_flag(request.artifact_requirements.get("screenshots", ""))


def _request_wants_result_text_extract(
    kind: ComputerUseActionKind,
    request: ComputerUseActionRequest,
) -> bool:
    if kind is not ComputerUseActionKind.BROWSER_INTERACTIVE_TASK:
        return False
    return _request_requires_result_text_extract(kind, request)


def _request_requires_result_text_extract(
    kind: ComputerUseActionKind,
    request: ComputerUseActionRequest,
) -> bool:
    if kind is not ComputerUseActionKind.BROWSER_INTERACTIVE_TASK:
        return False
    if _truthy_artifact_flag(request.artifact_requirements.get("text_extract", "")):
        return True
    if _truthy_artifact_flag(request.artifact_requirements.get("interpretation", "")):
        return True
    if _truthy_artifact_flag(request.artifact_requirements.get("include_sources", "")):
        return True
    requested_artifacts = " ".join(request.artifact_requirements.values()).lower()
    return any(
        marker in requested_artifacts
        for marker in (
            "analysis",
            "analy",
            "fact",
            "profile",
            "stats",
            "statistics",
            "анал",
            "профил",
            "стат",
            "факт",
        )
    )


def _artifact_requirement_status(
    *,
    request: ComputerUseActionRequest,
    kind: ComputerUseActionKind,
    screenshot_artifacts: tuple[str, ...],
    result_sections: tuple[dict[str, Any], ...],
    message: str,
) -> tuple[Literal["completed", "degraded"], str, str]:
    if "human_verification_unresolved" in message:
        return (
            "degraded",
            message,
            "human verification challenge was not solved before timeout",
        )
    if not screenshot_artifacts and _request_requires_screenshot_artifacts(request):
        return (
            "degraded",
            _append_message(
                message,
                "Screenshot artifact requirement was not satisfied.",
            ),
            "required screenshot artifacts were not produced",
        )
    if not result_sections and _request_requires_result_text_extract(kind, request):
        return (
            "degraded",
            _append_message(
                message,
                "Result text extraction requirement was not satisfied.",
            ),
            "required result text sections were not extracted",
        )
    return "completed", message, ""


def _append_message(message: str, addition: str) -> str:
    return "\n\n".join(part for part in (message, addition) if part)


def _truthy_artifact_flag(raw: str) -> bool:
    return raw.strip().lower() not in {"", "0", "false", "no", "none", "нет"}


def _request_wants_page_snapshot(
    kind: ComputerUseActionKind,
    request: ComputerUseActionRequest,
) -> bool:
    raw = request.metadata.get("capture_page_html", "")
    if raw.strip().lower() in {"1", "true", "yes", "y", "да"}:
        return True
    return (
        kind is ComputerUseActionKind.BROWSER_INTERACTIVE_TASK
        and _request_wants_screenshot(request)
    )


def _should_navigate_before_browser_action(
    kind: ComputerUseActionKind,
    request: ComputerUseActionRequest,
) -> bool:
    if not request.url:
        return False
    return kind in {
        ComputerUseActionKind.BROWSER_CLICK,
        ComputerUseActionKind.BROWSER_TYPE,
        ComputerUseActionKind.BROWSER_SCROLL,
    }


def _browser_screenshot_artifact(workspace_root: Path) -> str:
    target_dir = workspace_root / "agent_runtime" / "computer_use" / "screenshots"
    digest = hashlib.sha256(datetime.now(UTC).isoformat().encode("utf-8")).hexdigest()
    artifact = target_dir / f"browser-screenshot-{digest[:12]}.png"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    return artifact.relative_to(workspace_root).as_posix()


def _browser_page_html_artifact(workspace_root: Path, *, html: str) -> str:
    target_dir = workspace_root / "agent_runtime" / "computer_use" / "page-snapshots"
    seed = f"{datetime.now(UTC).isoformat()}:{html[:4096]}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    artifact = target_dir / f"browser-page-{digest[:12]}.html"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    return artifact.relative_to(workspace_root).as_posix()


def _scroll_amount(request: ComputerUseActionRequest) -> int:
    raw = request.target or request.operation or request.text
    normalized = raw.strip().lower()
    if normalized in {"up", "вверх"}:
        return -650
    if normalized in {"down", "вниз", ""}:
        return 650
    try:
        amount = int(normalized)
    except ValueError:
        return 650
    return max(-3_000, min(3_000, amount))


def _safe_int(value: object, *, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if not isinstance(value, str):
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _safe_float(value: object, *, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        item = value.strip()
        return [item] if item else []
    if not isinstance(value, Sequence) or isinstance(value, bytes | bytearray):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _result_section_clip(section: Mapping[str, Any]) -> dict[str, float] | None:
    x = max(0.0, _safe_float(section.get("x"), default=0.0) - 8.0)
    y = max(0.0, _safe_float(section.get("y"), default=0.0) - 8.0)
    width = _safe_float(section.get("width"), default=0.0) + 16.0
    height = _safe_float(section.get("height"), default=0.0) + 16.0
    if width < 80.0 or height < 60.0:
        return None
    return {
        "x": x,
        "y": y,
        "width": min(width, 2_400.0),
        "height": min(height, 2_400.0),
        "scale": 1.0,
    }


def _result_section_scroll_offsets(dimensions: Mapping[str, int]) -> list[int]:
    viewport_height = max(300, dimensions.get("viewport_height", 700))
    scroll_height = max(viewport_height, dimensions.get("scroll_height", 700))
    step = max(250, int(viewport_height * 0.85))
    offsets = list(range(0, scroll_height, step)) or [0]
    bottom = max(0, scroll_height - viewport_height)
    if bottom not in offsets:
        offsets.append(bottom)
    offsets = sorted(dict.fromkeys(max(0, min(bottom, offset)) for offset in offsets))
    if len(offsets) <= 8:
        return offsets
    indexes = [round(index * (len(offsets) - 1) / 7) for index in range(8)]
    return [offsets[index] for index in sorted(set(indexes))]


def _click_script(*, selector: str, target: str) -> str:
    return f"""
(() => {{
  const selector = {json.dumps(selector)};
  const target = {json.dumps(target)};
  const norm = (value) => String(value || '').replace(/\\s+/g, ' ').trim().toLowerCase();
  const label = (el) => norm(el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('title') || el.name || el.id);
  let el = selector ? document.querySelector(selector) : null;
  if (!el && target) {{
    const wanted = norm(target);
    const candidates = Array.from(document.querySelectorAll('button,a,input,textarea,select,label,[role="button"],[onclick]'));
    el = candidates.find((candidate) => label(candidate).includes(wanted));
  }}
  if (!el) return {{ok: false, reason: 'target not found'}};
  el.scrollIntoView({{block: 'center', inline: 'center'}});
  el.click();
  return {{ok: true, label: label(el), selector: selector || el.tagName.toLowerCase()}};
}})()
"""


def _type_script(*, selector: str, target: str, text: str) -> str:
    return f"""
(() => {{
  const selector = {json.dumps(selector)};
  const target = {json.dumps(target)};
  const text = {json.dumps(text)};
  const norm = (value) => String(value || '').replace(/\\s+/g, ' ').trim().toLowerCase();
  const label = (el) => norm(el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.name || el.id || el.innerText);
  let el = selector ? document.querySelector(selector) : null;
  if (!el && target) {{
    const wanted = norm(target);
    const candidates = Array.from(document.querySelectorAll('input,textarea,[contenteditable="true"]'));
    el = candidates.find((candidate) => label(candidate).includes(wanted));
  }}
  if (!el) return {{ok: false, reason: 'target not found'}};
  el.scrollIntoView({{block: 'center', inline: 'center'}});
  el.focus();
  if ('value' in el) {{
    el.value = text;
    el.dispatchEvent(new Event('input', {{bubbles: true}}));
    el.dispatchEvent(new Event('change', {{bubbles: true}}));
  }} else {{
    el.textContent = text;
    el.dispatchEvent(new InputEvent('input', {{bubbles: true, inputType: 'insertText', data: text}}));
  }}
  return {{ok: true, label: label(el), selector: selector || el.tagName.toLowerCase()}};
}})()
"""


def _human_verification_state_script() -> str:
    return """
(() => {
  const text = [
    document.title || '',
    document.body ? document.body.innerText || '' : ''
  ].join('\\n').replace(/\\s+/g, ' ').trim().toLowerCase();
  const markers = [
    'captcha',
    'security verification',
    'verify you are human',
    'verify that you are human',
    'confirm this search was made by a human',
    'not a bot',
    'are you a robot',
    'cloudflare',
    'ray id',
    'unfortunately, bots use',
    'select all squares',
    'network security'
  ];
  const marker = markers.find((item) => text.includes(item));
  return {
    blocked: Boolean(marker),
    detail: marker || '',
    title: document.title || '',
    url: location.href || ''
  };
})()
"""


def _cookie_consent_result_script() -> str:
    return r"""
(() => {
  const norm = (value) => String(value || '').replace(/\s+/g, ' ').trim().toLowerCase();
  const visible = (el) => {
    if (!el || el.disabled) return false;
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden') return false;
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  };
  const labelFor = (el) => norm(
    el.innerText ||
    el.value ||
    el.getAttribute('aria-label') ||
    el.getAttribute('title') ||
    el.name ||
    el.id
  );
  const safeRejectTerms = [
    'reject all',
    'decline all',
    'deny all',
    'refuse all',
    'continue without accepting',
    'use necessary cookies only',
    'necessary cookies only',
    'strictly necessary',
    'only necessary',
    'essential only',
    'alles afwijzen',
    'alles weigeren',
    'alleen noodzakelijke',
    'tout refuser',
    'rechazar todo',
    'rifiuta tutto',
    'ablehnen',
    'отклонить все',
    'отказаться от всех',
    'только необходимые',
    'только обязательные'
  ];
  const unsafeAcceptTerms = [
    'accept all',
    'allow all',
    'agree',
    'i agree',
    'alles accepteren',
    'tout accepter',
    'aceptar todo',
    'accetta tutto',
    'принять все',
    'согласен',
    'разрешить все'
  ];
  const cookieConsentCandidates = Array.from(
    document.querySelectorAll('button,a,input[type="button"],input[type="submit"],[role="button"]')
  )
    .filter(visible)
    .map((button) => {
      const label = labelFor(button);
      if (!label) return null;
      if (unsafeAcceptTerms.some((term) => label.includes(term))) return null;
      const matched = safeRejectTerms.find((term) => label.includes(term));
      if (!matched) return null;
      const rect = button.getBoundingClientRect();
      return {
        button,
        label,
        matched,
        score: matched.includes('necessary') || matched.includes('noodzakelijke') ? 80 : 100,
        y: rect.top + window.scrollY,
      };
    })
    .filter(Boolean)
    .sort((left, right) => (right.score - left.score) || (left.y - right.y));
  const best = cookieConsentCandidates[0];
  if (!best) return {ok: true, status: 'no_consent'};
  best.button.scrollIntoView({block: 'center', inline: 'center'});
  best.button.click();
  return {ok: true, status: 'dismissed_consent', detail: best.label.slice(0, 120)};
})()
"""


def _public_profile_result_script(request: ComputerUseActionRequest) -> str:
    target_terms = _public_profile_target_terms(request)
    script = r"""
(() => {
  const targetTerms = __TARGET_TERMS__;
  const norm = (value) => String(value || '').replace(/\s+/g, ' ').trim().toLowerCase();
  const visible = (el) => {
    if (!el) return false;
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden') return false;
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  };
  const profileUrl = (raw) => {
    try {
      const url = new URL(raw, location.href);
      const host = url.hostname.toLowerCase();
      const path = url.pathname.toLowerCase();
      if (host.includes('dotabuff.com') && /^\/players\/\d+/.test(path)) return url.href;
      if (host.includes('opendota.com') && /^\/players\/\d+/.test(path)) return url.href;
      if (host.includes('stratz.com') && /^\/players\/\d+/.test(path)) return url.href;
      if (host.includes('steamcommunity.com') && (/^\/profiles\/\d+/.test(path) || /^\/id\/[^/]+/.test(path))) return url.href;
    } catch (_error) {
      return '';
    }
    return '';
  };
  const currentProfile = profileUrl(location.href);
  if (currentProfile) {
    return {ok: true, status: 'result_detected', detail: currentProfile};
  }

  const publicProfileCandidates = Array.from(document.querySelectorAll('a[href]'))
    .map((anchor) => {
      const href = profileUrl(anchor.href || anchor.getAttribute('href') || '');
      if (!href) return null;
      const container = anchor.closest('tr,li,.result,.player,.players,article,section,div');
      const text = norm([
        anchor.innerText,
        anchor.getAttribute('aria-label'),
        anchor.getAttribute('title'),
        anchor.querySelector('img') && anchor.querySelector('img').getAttribute('alt'),
        container && container.innerText
      ].filter(Boolean).join(' '));
      const hrefText = norm(href);
      const termMatched = targetTerms.length === 0 ||
        targetTerms.some((term) => text.includes(term) || hrefText.includes(term));
      if (!termMatched) return null;
      const rect = anchor.getBoundingClientRect();
      let score = 50;
      if (visible(anchor)) score += 20;
      for (const term of targetTerms) {
        if (text === term) score += 120;
        else if (text.includes(term)) score += 100;
        if (hrefText.includes(term)) score += 40;
      }
      return {
        anchor,
        href,
        text,
        score,
        y: rect.top + window.scrollY
      };
    })
    .filter(Boolean)
    .sort((left, right) => (right.score - left.score) || (left.y - right.y));

  const best = publicProfileCandidates[0];
  if (!best) {
    return {ok: true, status: 'no_profile_result', detail: 'no visible public profile result link matched'};
  }
  const label = best.text.split('\n')[0].slice(0, 120) || best.href;
  best.anchor.scrollIntoView({block: 'center', inline: 'center'});
  best.anchor.click();
  return {
    ok: true,
    status: 'clicked_profile',
    detail: `${label} -> ${best.href}`,
    targetUrl: best.href
  };
})()
"""
    return script.replace(
        "__TARGET_TERMS__", json.dumps(target_terms, ensure_ascii=False)
    )


def _public_profile_target_terms(request: ComputerUseActionRequest) -> list[str]:
    raw_terms: list[str] = []
    for key in ("player_query", "nickname", "nick", "username"):
        value = request.metadata.get(key, "").strip()
        if value:
            raw_terms.append(value)
    parsed = urlparse(request.url)
    for value in parse_qs(parsed.query).get("q", ()):
        raw_terms.append(value)

    ignored = {
        "dota",
        "dotabuff",
        "opendota",
        "stratz",
        "steam",
        "steamid",
        "profile",
        "player",
        "search",
    }
    terms: list[str] = []
    for raw in raw_terms:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{1,63}", raw):
            lowered = token.lower()
            if lowered not in ignored:
                terms.append(lowered)
    return list(dict.fromkeys(terms))[:8]


def _interactive_task_completion_script(request: ComputerUseActionRequest) -> str:
    task_context = {
        "answerPolicy": request.metadata.get("answer_policy", ""),
        "personaContextMode": request.metadata.get("persona_context_mode", ""),
        "personaContextRef": request.metadata.get("persona_context_ref", ""),
        "taskIntent": request.metadata.get("task_intent", request.text),
    }
    script = r"""
(() => {
  const taskContext = __TASK_CONTEXT__;
  const norm = (value) => String(value || '').replace(/\s+/g, ' ').trim().toLowerCase();
  const visible = (el) => {
    if (!el || el.disabled) return false;
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden') return false;
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  };
  const controlType = (el) => norm(el.getAttribute('type') || el.tagName);
  const usableChoice = (el) => {
    if (!el || el.disabled) return false;
    return ['radio', 'checkbox'].includes(controlType(el)) || visible(el);
  };
  const actionable = (el) => {
    if (!el || el.disabled) return false;
    const type = controlType(el);
    return visible(el) || ['submit', 'button', 'reset'].includes(type);
  };
  const labelFor = (el) => {
    const direct = el.getAttribute('aria-label') || el.getAttribute('title') || el.value || el.name || el.id;
    const explicit = el.id ? document.querySelector(`label[for="${CSS.escape(el.id)}"]`) : null;
    const wrapping = el.closest('label');
    const row = el.closest('li,p,div,tr');
    return norm(
      (explicit && explicit.innerText) ||
      (wrapping && wrapping.innerText) ||
      direct ||
      (row && row.innerText)
    );
  };
  const hash = (value) => {
    let result = 0;
    for (let i = 0; i < value.length; i += 1) {
      result = ((result << 5) - result + value.charCodeAt(i)) | 0;
    }
    return Math.abs(result);
  };
  const hasAny = (text, terms) => terms.some((term) => text.includes(term));
  const resultTerms = [
    'получить результат', 'показать результат', 'узнать результат',
    'результат', 'итог', 'готово', 'завершить', 'result', 'score', 'finish'
  ];
  const nextTerms = [
    'начать тест', 'начать', 'далее', 'следующая', 'продолжить',
    'next', 'start', 'continue'
  ];
  const dangerTerms = [
    'password', 'пароль', 'login', 'логин', 'войти', 'sign in',
    'register', 'регистрация', 'зарегистр', 'payment', 'оплат', 'карта',
    'checkout', 'purchase', 'buy', 'delete', 'удал', 'publish', 'send message'
  ];
  const radiosByName = (root) => {
    const groups = new Map();
    const radios = Array.from(root.querySelectorAll('input[type="radio"]')).filter(usableChoice);
    for (const radio of radios) {
      const name = radio.name || radio.id || labelFor(radio);
      if (!name) continue;
      const group = groups.get(name) || [];
      group.push(radio);
      groups.set(name, group);
    }
    return groups;
  };
  const forms = Array.from(document.forms);
  const rankedForms = forms
    .map((form) => ({form, groups: radiosByName(form)}))
    .filter((item) => item.groups.size > 0)
    .sort((left, right) => right.groups.size - left.groups.size);

  const bodyText = norm(document.body ? document.body.innerText : '');
  if (!rankedForms.length) {
    if (hasAny(bodyText, ['ваш результат', 'результаты теста', 'результат теста', 'балл', 'score'])) {
      return {ok: true, status: 'result_detected', detail: document.title || location.href};
    }
    const startButton = Array.from(document.querySelectorAll('button,a,input[type="button"],input[type="submit"],[role="button"]'))
      .filter(visible)
      .find((button) => {
        const label = norm(button.innerText || button.value || button.getAttribute('aria-label') || button.getAttribute('title'));
        return hasAny(label, nextTerms) && !hasAny(label, dangerTerms);
      });
    if (startButton) {
      startButton.scrollIntoView({block: 'center', inline: 'center'});
      startButton.click();
      return {ok: true, status: 'clicked_next', detail: 'started interactive page'};
    }
    return {ok: true, status: 'no_test_form', detail: 'no visible interactive form found'};
  }

  const form = rankedForms[0].form;
  const groups = rankedForms[0].groups;
  const unsafeFields = Array.from(form.querySelectorAll('input,textarea,select'))
    .filter(visible)
    .filter((field) => {
      const type = controlType(field);
      if (['hidden', 'radio', 'checkbox', 'button', 'submit', 'reset'].includes(type)) return false;
      const fieldLabel = labelFor(field);
      if (field.tagName.toLowerCase() === 'select') return false;
      return type !== 'range' || hasAny(fieldLabel, dangerTerms);
    });
  if (unsafeFields.length) {
    return {ok: false, reason: 'unsafe visible text/personal fields in candidate form'};
  }

  let answeredGroups = 0;
  for (const [name, group] of groups.entries()) {
    if (group.some((radio) => radio.checked)) continue;
    const available = group.filter(usableChoice);
    if (!available.length) continue;
    const questionText = labelFor(available[0]) || name;
    const seed = [
      location.href,
      name,
      questionText,
      taskContext.taskIntent,
      taskContext.answerPolicy,
      taskContext.personaContextMode,
      taskContext.personaContextRef
    ].join('|');
    const selected = available[hash(seed) % available.length];
    selected.checked = true;
    selected.dispatchEvent(new Event('input', {bubbles: true}));
    selected.dispatchEvent(new Event('change', {bubbles: true}));
    answeredGroups += 1;
  }

  const selects = Array.from(form.querySelectorAll('select')).filter(visible);
  for (const select of selects) {
    if (select.value) continue;
    const options = Array.from(select.options).filter((option) => !option.disabled && option.value);
    if (!options.length) continue;
    const seed = [
      location.href,
      select.name || labelFor(select),
      taskContext.taskIntent,
      taskContext.answerPolicy,
      taskContext.personaContextMode,
      taskContext.personaContextRef
    ].join('|');
    select.value = options[hash(seed) % options.length].value;
    select.dispatchEvent(new Event('input', {bubbles: true}));
    select.dispatchEvent(new Event('change', {bubbles: true}));
  }

  const buttons = Array.from(form.querySelectorAll('button,input[type="submit"],input[type="button"],a,[role="button"]'))
    .filter(actionable)
    .map((button) => ({
      button,
      label: norm(button.innerText || button.value || button.getAttribute('aria-label') || button.getAttribute('title')),
    }))
    .filter((item) => item.label && !hasAny(item.label, dangerTerms));
  const resultButton = buttons.find((item) => hasAny(item.label, resultTerms));
  const nextButton = buttons.find((item) => hasAny(item.label, nextTerms));
  const chosen = resultButton || nextButton;
  if (!chosen) {
    return {ok: true, status: 'answered', detail: `answered ${answeredGroups} groups; no safe result/next button found`};
  }
  chosen.button.scrollIntoView({block: 'center', inline: 'center'});
  chosen.button.click();
  return {
    ok: true,
    status: resultButton ? 'clicked_result' : 'clicked_next',
    detail: `${chosen.label}; answered ${answeredGroups} groups`
  };
})()
"""
    return script.replace(
        "__TASK_CONTEXT__", json.dumps(task_context, ensure_ascii=False)
    )


def _result_sections_script() -> str:
    return r"""
(() => {
  const norm = (value) => String(value || '').replace(/\s+/g, ' ').trim();
  const visible = (el) => {
    const rect = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    return rect.width > 0 && rect.height > 0 &&
      style.display !== 'none' &&
      style.visibility !== 'hidden' &&
      Number(style.opacity || '1') > 0.01;
  };
  const hasResultTerm = (text) => /результат|result/i.test(text);
  const resultTermCount = (text) => (text.match(/результат|result/gi) || []).length;
  const hasScoreOrInterpretation = (text) =>
    /балл|score|points|уровень|level|интерпретац|описан|description/i.test(text);
  const nodes = Array.from(
    document.querySelectorAll('main,article,section,div,li,td')
  );
  const resultSectionCandidates = [];
  for (const el of nodes) {
    if (!visible(el)) continue;
    const text = norm(el.innerText);
    if (text.length < 20 || text.length > 2500) continue;
    if (!hasResultTerm(text) || resultTermCount(text) !== 1) continue;
    if (!hasScoreOrInterpretation(text)) continue;
    const rect = el.getBoundingClientRect();
    if (rect.width < 180 || rect.height < 60) continue;
    resultSectionCandidates.push({
      index: 0,
      text,
      x: Math.max(0, Math.round(rect.left + window.scrollX)),
      y: Math.max(0, Math.round(rect.top + window.scrollY)),
      width: Math.ceil(rect.width),
      height: Math.ceil(rect.height),
    });
  }
  resultSectionCandidates.sort((left, right) =>
    left.y - right.y || left.x - right.x || left.height - right.height
  );
  const selected = [];
  for (const item of resultSectionCandidates) {
    const duplicate = selected.some((prev) => {
      const samePlace = Math.abs(prev.y - item.y) < 8 &&
        Math.abs(prev.x - item.x) < 8 &&
        Math.abs(prev.height - item.height) < 12;
      const insidePrev = item.y >= prev.y &&
        item.y + item.height <= prev.y + prev.height &&
        item.x >= prev.x &&
        item.x + item.width <= prev.x + prev.width;
      return samePlace || (insidePrev && prev.text.includes(item.text));
    });
    if (!duplicate) selected.push(item);
  }
  return selected.slice(0, 12).map((item, index) => ({...item, index: index + 1}));
})()
"""


def _page_content_sections_script(request: ComputerUseActionRequest) -> str:
    focus_terms = _page_content_focus_terms(request)
    script = r"""
(() => {
  const focusTerms = __FOCUS_TERMS__;
  const norm = (value) => String(value || '').replace(/\s+/g, ' ').trim();
  const lower = (value) => norm(value).toLowerCase();
  const visible = (el) => {
    const rect = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    return rect.width > 0 && rect.height > 0 &&
      style.display !== 'none' &&
      style.visibility !== 'hidden' &&
      Number(style.opacity || '1') > 0.01;
  };
  const selectorPriority = (el) => {
    if (el.matches('.header-content, header, .profile, .player-summary')) return 120;
    if (el.matches('section')) return 100;
    if (el.matches('table')) return 90;
    if (el.matches('article')) return 80;
    if (el.matches('main')) return 40;
    return 20;
  };
  const contentTerms = [
    'profile', 'overview', 'record', 'win rate', 'last match', 'rank',
    'roles and lanes', 'role', 'lane', 'core', 'support', 'safe lane',
    'mid lane', 'off lane', 'most played heroes', 'hero', 'matches',
    'win %', 'kda', 'lifetime stats', 'stats recorded', 'ranked mm',
    'normal mm', 'aliases', 'steam'
  ];
  const noisyTerms = [
    'sign in with steam', 'copyright', 'privacy', 'we use cookies',
    'do not sell or share', 'turn on colorblind mode'
  ];
  const nodes = Array.from(document.querySelectorAll(
    '.header-content, .player-summary section, main section, article, section, table, dl'
  ));
  const candidates = [];
  for (const el of nodes) {
    if (!visible(el)) continue;
    const text = norm(el.innerText);
    const textLower = text.toLowerCase();
    if (text.length < 20 || text.length > 3200) continue;
    if (noisyTerms.some((term) => textLower.includes(term))) continue;
    let score = selectorPriority(el);
    for (const term of contentTerms) {
      if (textLower.includes(term)) score += 16;
    }
    for (const term of focusTerms) {
      if (term && textLower.includes(term)) score += 24;
    }
    if (score < 80) continue;
    const rect = el.getBoundingClientRect();
    candidates.push({
      index: 0,
      text,
      score,
      x: Math.max(0, Math.round(rect.left + window.scrollX)),
      y: Math.max(0, Math.round(rect.top + window.scrollY)),
      width: Math.ceil(rect.width),
      height: Math.ceil(rect.height),
    });
  }
  candidates.sort((left, right) =>
    right.score - left.score || left.y - right.y || left.x - right.x
  );
  const selected = [];
  for (const item of candidates) {
    const duplicate = selected.some((prev) => {
      const sameText = prev.text === item.text || prev.text.includes(item.text) || item.text.includes(prev.text);
      const samePlace = Math.abs(prev.y - item.y) < 8 && Math.abs(prev.x - item.x) < 8;
      const insidePrev = item.y >= prev.y &&
        item.y + item.height <= prev.y + prev.height &&
        item.x >= prev.x &&
        item.x + item.width <= prev.x + prev.width;
      return sameText || samePlace || insidePrev;
    });
    if (!duplicate) selected.push(item);
    if (selected.length >= 12) break;
  }
  selected.sort((left, right) => left.y - right.y || left.x - right.x);
  return selected.map((item, index) => ({...item, index: index + 1}));
})()
"""
    return script.replace(
        "__FOCUS_TERMS__", json.dumps(focus_terms, ensure_ascii=False)
    )


def _page_content_focus_terms(request: ComputerUseActionRequest) -> list[str]:
    raw_values: list[str] = [
        request.goal,
        request.text,
        request.url,
        " ".join(request.artifact_requirements.values()),
    ]
    raw_values.extend(request.metadata.values())
    ignored = {
        "browser",
        "dota",
        "dotabuff",
        "include",
        "match",
        "open",
        "opendota",
        "player",
        "profile",
        "source",
        "sources",
        "stats",
        "stratz",
        "steam",
    }
    terms: list[str] = []
    for raw in raw_values:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{2,63}", raw):
            lowered = token.lower().strip("._-")
            if lowered and lowered not in ignored:
                terms.append(lowered)
    return list(dict.fromkeys(terms))[:12]


def _interactive_elements_script() -> str:
    return r"""
(() => {
  const norm = (value) => String(value || '').replace(/\s+/g, ' ').trim();
  const selectorFor = (el) => {
    if (el.id) return `#${CSS.escape(el.id)}`;
    const name = el.getAttribute('name');
    if (name) return `${el.tagName.toLowerCase()}[name="${CSS.escape(name)}"]`;
    const type = el.getAttribute('type');
    if (type) return `${el.tagName.toLowerCase()}[type="${CSS.escape(type)}"]`;
    return el.tagName.toLowerCase();
  };
  return Array.from(document.querySelectorAll('a,button,input,textarea,select,label,[role="button"],[onclick]'))
    .slice(0, 80)
    .map((el) => ({
      tag: el.tagName.toLowerCase(),
      selector: selectorFor(el),
      label: norm(el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.getAttribute('title') || el.name || el.id || el.href),
    }))
    .filter((item) => item.label || item.selector);
})()
"""


def _request_from_context(context_pack: Any) -> ComputerUseActionRequest:
    metadata = cast("Mapping[str, Any]", getattr(context_pack, "metadata", {}))
    raw_payload = str(metadata.get("computer_use_payload", "")).strip()
    if raw_payload:
        payload = json.loads(raw_payload)
        if not isinstance(payload, dict):
            raise ValueError("computer_use_payload must be a JSON object")
        return _request_from_mapping(payload)
    action = str(metadata.get("computer_use_action", "")).strip()
    if not action:
        raise ValueError("computer_use_action is required")
    return _request_from_mapping(
        {
            "action": action,
            "operation": metadata.get("computer_use_operation", ""),
            "target": metadata.get("computer_use_target", ""),
            "text": metadata.get("computer_use_text", ""),
            "url": metadata.get("computer_use_url", ""),
            "selector": metadata.get("computer_use_selector", ""),
            "tab_id": metadata.get("computer_use_tab_id", ""),
            "cwd": metadata.get("computer_use_cwd", "."),
        }
    )


def _request_from_payload(
    action: ComputerUseActionKind,
    payload: Mapping[str, Any],
) -> ComputerUseActionRequest:
    data = {
        "action": action.value,
        "goal": payload.get("goal", ""),
        "operation": payload.get("operation", ""),
        "target": payload.get("target", ""),
        "text": payload.get("text", ""),
        "url": payload.get("url", ""),
        "selector": payload.get("selector", ""),
        "tab_id": payload.get("tab_id", ""),
        "constraints": payload.get("constraints", ()),
        "artifact_requirements": payload.get("artifact_requirements", {}),
        "success_criteria": payload.get("success_criteria", ()),
        "risk_intent": payload.get("risk_intent", ""),
        "approval_scope": payload.get("approval_scope", {}),
        "argv": payload.get("argv", ()),
        "cwd": payload.get("cwd", "."),
        "timeout_seconds": payload.get("timeout_seconds", 0.0),
        "metadata": payload.get("metadata", {}),
    }
    return _request_from_mapping(data)


def coerce_computer_use_action_request(
    payload: Mapping[str, Any],
) -> ComputerUseActionRequest:
    """Parse router/tool payloads into the stable computer-use request model."""
    return _request_from_mapping(payload)


def _request_from_mapping(payload: Mapping[str, Any]) -> ComputerUseActionRequest:
    raw_metadata = payload.get("metadata", {})
    metadata: dict[str, str] = {}
    if isinstance(raw_metadata, Mapping):
        metadata = {
            str(key): str(value)
            for key, value in raw_metadata.items()
            if str(key).strip()
        }
    raw_artifact_requirements = payload.get("artifact_requirements", {})
    artifact_requirements: dict[str, str] = {}
    if isinstance(raw_artifact_requirements, Mapping):
        artifact_requirements = {
            str(key): str(value)
            for key, value in raw_artifact_requirements.items()
            if str(key).strip() and str(value).strip()
        }
    raw_approval_scope = payload.get("approval_scope", {})
    approval_scope: dict[str, str] = {}
    if isinstance(raw_approval_scope, Mapping):
        approval_scope = {
            str(key): str(value)
            for key, value in raw_approval_scope.items()
            if str(key).strip() and str(value).strip()
        }
    return ComputerUseActionRequest(
        action=ComputerUseActionKind(str(payload.get("action", "")).strip()),
        goal=str(payload.get("goal", "")).strip(),
        operation=str(payload.get("operation", "")).strip(),
        target=str(payload.get("target", "")).strip(),
        text=str(payload.get("text", "")),
        url=str(payload.get("url", "")).strip(),
        selector=str(payload.get("selector", "")).strip(),
        tab_id=str(payload.get("tab_id", "")).strip(),
        constraints=_string_list(payload.get("constraints", ())),
        artifact_requirements=artifact_requirements,
        success_criteria=_string_list(payload.get("success_criteria", ())),
        risk_intent=str(payload.get("risk_intent", "")).strip(),
        approval_scope=approval_scope,
        argv=_string_list(payload.get("argv", ())),
        cwd=str(payload.get("cwd", ".")).strip() or ".",
        timeout_seconds=_safe_float(payload.get("timeout_seconds"), default=0.0),
        metadata=metadata,
    )


def _action_result_from_mapping(
    result: Mapping[str, Any] | ComputerUseActionResult,
) -> ComputerUseActionResult:
    if isinstance(result, ComputerUseActionResult):
        return result
    metadata_raw = result.get("metadata", {})
    metadata: dict[str, str] = {}
    if isinstance(metadata_raw, Mapping):
        metadata = {str(key): str(value) for key, value in metadata_raw.items()}
    return ComputerUseActionResult(
        status=cast("Any", str(result.get("status", "completed"))),
        message=str(result.get("message", "")),
        artifact=str(result.get("artifact", "")),
        metadata=metadata,
    )


def _request_haystack(request: ComputerUseActionRequest) -> str:
    values = [
        request.action.value,
        request.goal,
        request.operation,
        request.target,
        request.text,
        request.url,
        request.selector,
        request.risk_intent,
        *request.success_criteria,
        *request.artifact_requirements.values(),
        *(
            value
            for key, value in request.metadata.items()
            if key not in _SAFETY_POLICY_METADATA_KEYS
        ),
    ]
    return " ".join(value.lower() for value in values if value)


def _risk_decision_for_request(
    request: ComputerUseActionRequest,
) -> ComputerUseSafetyDecision:
    explicit_intent = request.risk_intent.strip().lower()
    if explicit_intent == ComputerUseRiskClass.READONLY_EXISTING_SESSION.value:
        return ComputerUseSafetyDecision(
            allowed=True,
            reason="Read-only lookup in an existing authenticated session.",
            risk_class=ComputerUseRiskClass.READONLY_EXISTING_SESSION,
            risk_summary=(
                "Чтение данных из уже открытой локальной сессии без ввода "
                "credentials и без submit."
            ),
        )
    if request.action is ComputerUseActionKind.DESKTOP_SHELL_COMMAND:
        return _approval_decision(
            request=request,
            risk_class=ComputerUseRiskClass.SHELL_COMMAND,
            capability="desktop.shell",
            risk_summary="Выполнение локальной команды в shell/terminal контуре.",
            approval_prompt=(
                "Разрешаешь выполнить ровно эту structured argv команду в "
                "разрешённом cwd? Approval не даёт browser/login/send/purchase."
            ),
        )
    if request.action is ComputerUseActionKind.BROWSER_SUBMIT:
        return _approval_decision(
            request=request,
            risk_class=ComputerUseRiskClass.EXTERNAL_SUBMIT,
            capability="browser_submit",
            risk_summary="Финальный submit формы или внешнего действия.",
            approval_prompt=(
                "Разрешаешь выполнить только этот browser_submit для указанной "
                "формы/страницы?"
            ),
        )

    haystack = _danger_haystack(request)
    account_capability = _account_mutation_capability(haystack)
    if account_capability:
        return _approval_decision(
            request=request,
            risk_class=ComputerUseRiskClass.ACCOUNT_MUTATION,
            capability=account_capability,
            risk_summary=(
                "Действие может изменить аккаунт, отправить сообщение, удалить "
                "данные, опубликовать или потратить деньги."
            ),
            approval_prompt=(
                f"Разрешаешь только capability `{account_capability}` для этого "
                "конкретного действия? Остальные high-risk действия останутся "
                "закрыты."
            ),
        )
    if _credential_entry_requested(request, haystack):
        return _approval_decision(
            request=request,
            risk_class=ComputerUseRiskClass.CREDENTIAL_ENTRY,
            capability="login",
            risk_summary=(
                "Действие похоже на ввод credentials, 2FA или подтверждение login."
            ),
            approval_prompt=(
                "Разрешаешь только credential/login действие в указанном scope? "
                "Approval не разрешает покупки, отправку, удаление или shell."
            ),
        )
    if _SUBMIT_RE.search(haystack):
        return _approval_decision(
            request=request,
            risk_class=ComputerUseRiskClass.EXTERNAL_SUBMIT,
            capability="browser_submit",
            risk_summary="Действие похоже на финальное подтверждение или submit.",
            approval_prompt=(
                "Разрешаешь только этот submit/confirm? Любая покупка, login, "
                "send или delete потребует отдельный approval."
            ),
        )
    if _SHELL_RE.search(haystack):
        return _approval_decision(
            request=request,
            risk_class=ComputerUseRiskClass.SHELL_COMMAND,
            capability="desktop.shell",
            risk_summary="Запрос похож на terminal/shell команду.",
            approval_prompt=(
                "Разрешаешь только одну structured argv shell-команду? Browser и "
                "account side effects останутся закрыты."
            ),
        )
    return ComputerUseSafetyDecision(
        allowed=True,
        reason="Reversible computer-use action under active profile.",
        risk_class=ComputerUseRiskClass.REVERSIBLE_GUI_ACTION,
        risk_summary="Reversible browser/desktop action without detected final side effect.",
    )


def _approval_decision(
    *,
    request: ComputerUseActionRequest,
    risk_class: ComputerUseRiskClass,
    capability: str,
    risk_summary: str,
    approval_prompt: str,
) -> ComputerUseSafetyDecision:
    return ComputerUseSafetyDecision(
        allowed=False,
        requires_approval=True,
        reason=(
            "dangerous_action_requires_approval: "
            f"{request.action.value} requires {capability} approval."
        ),
        stop_condition=risk_class.value,
        risk_class=risk_class,
        required_capability=capability,
        risk_summary=risk_summary,
        approval_prompt=approval_prompt,
    )


def _danger_haystack(request: ComputerUseActionRequest) -> str:
    haystack = _request_haystack(request)
    return _NEGATED_CREDENTIAL_RE.sub(" ", haystack)


def _credential_entry_requested(
    request: ComputerUseActionRequest,
    haystack: str,
) -> bool:
    if request.action is ComputerUseActionKind.BROWSER_TYPE and any(
        marker in " ".join((request.target, request.selector, request.goal)).lower()
        for marker in ("password", "парол", "login", "логин", "2fa", "otp", "mfa")
    ):
        return True
    return bool(
        _CREDENTIAL_ENTRY_RE.search(haystack) or _LOGIN_CLICK_RE.search(haystack)
    )


def _account_mutation_capability(haystack: str) -> str:
    if _PURCHASE_RE.search(haystack):
        return "purchase"
    if _DELETE_RE.search(haystack):
        return "delete"
    if _PUBLISH_RE.search(haystack):
        return "publish"
    if _SEND_RE.search(haystack):
        return "send_message"
    return ""


def _should_persist_browser_result(
    request: ComputerUseActionRequest,
    result: ComputerUseActionResult,
) -> bool:
    if request.action not in _BROWSER_ACTIONS:
        return False
    if result.status != "completed":
        return False
    if not result.metadata.get("current_url", "").strip():
        return False
    return request.action is ComputerUseActionKind.BROWSER_INTERACTIVE_TASK or bool(
        _result_artifacts(result)
    )


def _browser_result_followup_needs_url(request: ComputerUseActionRequest) -> bool:
    if request.url or request.tab_id:
        return False
    return request.action in {
        ComputerUseActionKind.BROWSER_CLICK,
        ComputerUseActionKind.BROWSER_TYPE,
        ComputerUseActionKind.BROWSER_SCROLL,
    }


def _browser_result_record_matches(
    payload: Mapping[str, Any],
    *,
    chat_id: int | None,
    owner_user_id: int | None,
) -> bool:
    record_chat_id = str(payload.get("chat_id", "")).strip()
    record_owner_user_id = str(payload.get("owner_user_id", "")).strip()
    if chat_id is not None and record_chat_id and record_chat_id != str(chat_id):
        return False
    return not (
        owner_user_id is not None
        and record_owner_user_id
        and record_owner_user_id != str(owner_user_id)
    )


def _with_browser_result_artifact(
    result: ComputerUseActionResult,
    *,
    artifact: str,
) -> ComputerUseActionResult:
    metadata = {**result.metadata, "browser_result_artifact": artifact}
    return result.model_copy(update={"metadata": metadata})


def _result_artifacts(result: ComputerUseActionResult) -> tuple[str, ...]:
    refs = (
        *_result_screenshot_artifacts(result),
        result.metadata.get("page_html_artifact", ""),
        result.metadata.get("browser_result_artifact", ""),
    )
    return tuple(dict.fromkeys(ref.strip() for ref in refs if ref.strip()))


def _result_screenshot_artifacts(result: ComputerUseActionResult) -> tuple[str, ...]:
    refs = [result.artifact]
    raw = result.metadata.get("screenshot_artifacts", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = ()
        if isinstance(parsed, Sequence) and not isinstance(parsed, str | bytes):
            refs.extend(str(item) for item in parsed)
    return tuple(dict.fromkeys(ref.strip() for ref in refs if ref.strip()))


def _result_sections_from_metadata(
    result: ComputerUseActionResult,
) -> tuple[dict[str, Any], ...]:
    raw = result.metadata.get("result_sections", "").strip()
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return ()
    if not isinstance(parsed, Sequence) or isinstance(parsed, str | bytes):
        return ()
    sections: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        sections.append(
            {
                "index": _safe_int(item.get("index"), default=len(sections) + 1),
                "text": text,
                "x": _safe_float(item.get("x"), default=0.0),
                "y": _safe_float(item.get("y"), default=0.0),
                "width": _safe_float(item.get("width"), default=0.0),
                "height": _safe_float(item.get("height"), default=0.0),
            }
        )
    return tuple(sections)


def _bounded_result_section_text(text: str, *, limit: int = 700) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def _registered_tool(gateway: ToolGateway, tool_name: str) -> Any | None:
    for tool in gateway.registered_tools():
        if tool.name == tool_name:
            return tool
    return None


def _approval_from_job_context(
    *,
    job: Any,
    context_pack: Any,
    capability: str,
) -> AgentToolApproval | None:
    if not capability:
        return None
    raw_capabilities = str(
        context_pack.metadata.get("agent_tool_approval_capabilities", "")
    )
    capabilities = tuple(
        item.strip() for item in raw_capabilities.split(",") if item.strip()
    )
    if capability not in set(capabilities):
        return None
    approval_id = str(context_pack.metadata.get("agent_tool_approval_id", ""))
    if not approval_id.strip():
        return None
    return AgentToolApproval.approved(
        approval_id=approval_id.strip(),
        capabilities=capabilities,
        approved_by=int(getattr(job, "owner_user_id", 0) or 0),
    )


def _any_approval_from_job_context(
    *,
    job: Any,
    context_pack: Any,
) -> AgentToolApproval | None:
    raw_capabilities = str(
        context_pack.metadata.get("agent_tool_approval_capabilities", "")
    )
    capabilities = tuple(
        item.strip() for item in raw_capabilities.split(",") if item.strip()
    )
    approval_id = str(context_pack.metadata.get("agent_tool_approval_id", ""))
    if not approval_id.strip() or not capabilities:
        return None
    return AgentToolApproval.approved(
        approval_id=approval_id.strip(),
        capabilities=capabilities,
        approved_by=int(getattr(job, "owner_user_id", 0) or 0),
    )


def _approval_from_request_metadata(
    request: ComputerUseActionRequest,
) -> AgentToolApproval | None:
    raw_capabilities = request.metadata.get("agent_tool_approval_capabilities", "")
    capabilities = tuple(
        item.strip() for item in raw_capabilities.split(",") if item.strip()
    )
    approval_id = request.metadata.get("agent_tool_approval_id", "").strip()
    if not approval_id or not capabilities:
        return None
    return AgentToolApproval.approved(
        approval_id=approval_id,
        capabilities=capabilities,
        approved_by=0,
    )


def _payload_with_approval(
    request: ComputerUseActionRequest,
    approval: AgentToolApproval,
) -> dict[str, Any]:
    payload = request.to_payload()
    metadata = dict(payload.get("metadata", {}))
    metadata["agent_tool_approval_id"] = approval.approval_id
    metadata["agent_tool_approval_capabilities"] = ",".join(approval.capabilities)
    payload["metadata"] = metadata
    return payload


def _approval_required_capsule(
    *,
    request: ComputerUseActionRequest,
    decision: ComputerUseSafetyDecision,
) -> ContextCapsule:
    processed_context = "\n".join(
        (
            "# Computer-use dangerous action requires approval",
            "- event: dangerous_action_requires_approval",
            f"- action: {request.action.value}",
            f"- risk_class: {decision.risk_class.value}",
            f"- required_capability: {decision.required_capability}",
            f"- stop_condition: {decision.stop_condition}",
            "",
            f"risk_summary: {decision.risk_summary}",
            f"approval_prompt: {decision.approval_prompt}",
        )
    )
    return ContextCapsule(
        summary="Computer-use action requires approval.",
        processed_context=processed_context,
        findings=(
            Finding(
                claim=decision.reason,
                status=FindingStatus.CONFIRMED,
                confidence=0.95,
                evidence=(
                    decision.risk_class.value,
                    decision.required_capability,
                ),
            ),
        ),
        next_actions=(
            "Жвуша должна объяснить конкретный риск и запросить scoped approval.",
        ),
        markdown_report=processed_context,
    )


def _hard_stop_capsule(
    *,
    request: ComputerUseActionRequest,
    decision: ComputerUseSafetyDecision,
    artifact: str,
) -> ContextCapsule:
    processed_context = "\n".join(
        (
            "# Computer-use hard stop",
            f"- action: {request.action.value}",
            f"- capability: {_CAPABILITY_BY_ACTION[request.action]}",
            f"- stop_condition: {decision.stop_condition}",
            f"- artifact: {artifact}",
            "",
            decision.reason,
        )
    )
    return ContextCapsule(
        summary="Computer-use action hard-stopped.",
        processed_context=processed_context,
        findings=(
            Finding(
                claim=decision.reason,
                status=FindingStatus.CONFIRMED,
                confidence=0.95,
                evidence=(artifact, decision.stop_condition),
            ),
            Finding(
                claim="ToolGateway executor was not called for the final action.",
                status=FindingStatus.CONFIRMED,
                confidence=0.95,
                evidence=("hard_stop_before_execution",),
            ),
        ),
        artifacts=(artifact,),
        next_actions=(
            "Жвуша должна попросить Никиту выполнить финальное действие вручную.",
        ),
        markdown_report=processed_context,
    )


def _success_capsule(
    *,
    request: ComputerUseActionRequest,
    capability: str,
    tool_name: str,
    result: ComputerUseActionResult,
) -> ContextCapsule:
    current_url = result.metadata.get("current_url", "").strip()
    title = result.metadata.get("title", "").strip()
    page_html_artifact = result.metadata.get("page_html_artifact", "").strip()
    browser_result_artifact = result.metadata.get("browser_result_artifact", "").strip()
    screenshot_artifacts = _result_screenshot_artifacts(result)
    result_sections = _result_sections_from_metadata(result)
    lines = [
        "# Computer-use action",
        f"- action: {request.action.value}",
        f"- capability: {capability}",
        f"- tool: {tool_name}",
        f"- status: {result.status}",
    ]
    if current_url:
        lines.append(f"- current_url: {current_url}")
    if title:
        lines.append(f"- title: {title}")
    if browser_result_artifact:
        lines.append(f"- browser_result_artifact: {browser_result_artifact}")
    if screenshot_artifacts:
        lines.append("- screenshot_artifacts: " + ", ".join(screenshot_artifacts))
    if page_html_artifact:
        lines.append(f"- page_html_artifact: {page_html_artifact}")
    if result_sections:
        lines.append("- result_sections:")
        for section in result_sections[:12]:
            index = _safe_int(section.get("index"), default=0)
            text = _bounded_result_section_text(str(section.get("text", "")))
            lines.append(f"  - section {index}: {text}")
    lines.extend(("", result.message))
    processed_context = "\n".join(lines)
    summary = (
        "Computer-use action completed."
        if result.status == "completed"
        else "Computer-use action returned non-completed status."
    )
    artifacts = _result_artifacts(result)
    return ContextCapsule(
        summary=summary,
        processed_context=processed_context,
        findings=(
            Finding(
                claim=f"Computer-use tool {tool_name} returned {result.status}.",
                status=FindingStatus.CONFIRMED
                if result.status in {"completed", "configured_only", "degraded"}
                else FindingStatus.PARTIAL,
                confidence=0.9,
                evidence=(tool_name, capability),
            ),
        ),
        sources=(current_url,) if current_url else (),
        artifacts=artifacts,
        next_actions=("Передать observation Жвуше для user-facing ответа.",),
        markdown_report=processed_context,
    )


def _refusal_capsule(reason: str, *, next_action: str) -> ContextCapsule:
    return ContextCapsule(
        summary="Computer-use action refused.",
        findings=(
            Finding(
                claim=reason,
                status=FindingStatus.UNCONFIRMED,
                confidence=1.0,
            ),
        ),
        next_actions=(next_action,),
        markdown_report=f"Computer-use action refused: {reason}",
    )

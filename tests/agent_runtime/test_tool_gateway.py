"""Tool-level capability enforcement tests for Agent Runtime."""

from __future__ import annotations

import json
import socket
from typing import Any

import pytest


async def test_tool_gateway_exposes_only_allowed_tools() -> None:
    from src.agent_runtime.models import InvocationProfile
    from src.agent_runtime.tools import FunctionAgentTool, ToolGateway

    async def read_code(payload: dict[str, Any]) -> str:
        return f"read:{payload['path']}"

    async def write_files(payload: dict[str, Any]) -> str:
        return f"write:{payload['path']}"

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("read_repo_file", "read_code", read_code),
            FunctionAgentTool("write_repo_file", "write_files", write_files),
        )
    )
    profile = InvocationProfile(
        id="readonly",
        allowed_capabilities=("read_code",),
        denied_capabilities=("write_files",),
    )

    toolset = gateway.build_toolset(profile)
    assert tuple(toolset) == ("read_repo_file",)
    assert await gateway.execute(profile, "read_repo_file", {"path": "README.md"})


async def test_denied_tool_is_physically_unavailable() -> None:
    from src.agent_runtime.models import InvocationProfile
    from src.agent_runtime.tools import FunctionAgentTool, ToolDeniedError, ToolGateway

    async def write_files(payload: dict[str, Any]) -> str:
        return f"write:{payload['path']}"

    gateway = ToolGateway(
        tools=(FunctionAgentTool("write_repo_file", "write_files", write_files),)
    )
    profile = InvocationProfile(
        id="readonly",
        allowed_capabilities=("read_code",),
        denied_capabilities=("write_files",),
    )

    assert "write_repo_file" not in gateway.build_toolset(profile)
    with pytest.raises(ToolDeniedError):
        await gateway.execute(profile, "write_repo_file", {"path": ".env"})


async def test_side_effect_tool_requires_policy_approval() -> None:
    from src.agent_runtime.approvals import AgentToolApproval
    from src.agent_runtime.models import InvocationProfile
    from src.agent_runtime.tools import FunctionAgentTool, ToolDeniedError, ToolGateway

    async def restart(payload: dict[str, Any]) -> str:
        return f"restart:{payload['reason']}"

    gateway = ToolGateway(tools=(FunctionAgentTool("restart_bot", "restart", restart),))
    profile = InvocationProfile(
        id="ops",
        allowed_capabilities=("restart",),
    )

    assert "restart_bot" not in gateway.build_toolset(profile)
    with pytest.raises(ToolDeniedError):
        await gateway.execute(profile, "restart_bot", {"reason": "test"})

    approval = AgentToolApproval.approved(
        approval_id="approval-1",
        capabilities=("restart",),
        approved_by=1291112109,
    )
    assert "restart_bot" in gateway.build_toolset(profile, approval=approval)
    assert (
        await gateway.execute(
            profile,
            "restart_bot",
            {"reason": "test"},
            approval=approval,
        )
        == "restart:test"
    )


async def test_internal_gate_capabilities_still_require_tool_approval() -> None:
    from src.agent_runtime.approvals import AgentToolApproval
    from src.agent_runtime.models import InvocationProfile
    from src.agent_runtime.tools import FunctionAgentTool, ToolDeniedError, ToolGateway

    async def write_after_gate(payload: dict[str, Any]) -> str:
        return f"write:{payload['path']}"

    async def commit_after_gate(payload: dict[str, Any]) -> str:
        return f"commit:{payload['message']}"

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool(
                "write_repo_file_after_gate",
                "write_whitelisted_files_after_approval",
                write_after_gate,
            ),
            FunctionAgentTool(
                "commit_repo_after_gate",
                "commit_after_gate",
                commit_after_gate,
            ),
        )
    )
    profile = InvocationProfile(
        id="self_coding.implementation",
        allowed_capabilities=(
            "write_whitelisted_files_after_approval",
            "commit_after_gate",
        ),
    )

    assert gateway.build_toolset(profile) == {}
    with pytest.raises(ToolDeniedError):
        await gateway.execute(
            profile,
            "write_repo_file_after_gate",
            {"path": "src/app.py"},
        )

    approval = AgentToolApproval.approved(
        approval_id="approval-self-coding-side-effect",
        capabilities=("write_whitelisted_files_after_approval", "commit_after_gate"),
        approved_by=1291112109,
    )

    assert set(gateway.build_toolset(profile, approval=approval)) == {
        "write_repo_file_after_gate",
        "commit_repo_after_gate",
    }
    assert (
        await gateway.execute(
            profile,
            "commit_repo_after_gate",
            {"message": "test"},
            approval=approval,
        )
        == "commit:test"
    )


async def test_readonly_command_tool_runs_allowlisted_rg_inside_workspace(
    tmp_path,
) -> None:
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import InvocationProfile

    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "main.py").write_text(
        "def dispatch_command():\n    return 'ok'\n",
        encoding="utf-8",
    )
    gateway = build_builtin_tool_gateway(workspace_root=tmp_path)
    profile = InvocationProfile(
        id="readonly-code-search",
        allowed_capabilities=("run_readonly_commands",),
    )

    result = await gateway.execute(
        profile,
        "run_readonly_command",
        {"argv": ["rg", "-n", "dispatch_command", "src/main.py"]},
    )

    assert result["exit_code"] == 0
    assert result["cwd"] == "."
    assert "1:def dispatch_command():" in result["stdout"]
    assert result["stderr"] == ""


async def test_readonly_command_tool_rejects_unsafe_command_shapes(tmp_path) -> None:
    from src.agent_runtime.builtin_tools import ReadOnlyCommandTool

    tool = ReadOnlyCommandTool(tmp_path)

    with pytest.raises(PermissionError, match="not allowlisted"):
        await tool.execute({"argv": ["python", "-c", "print('no')"]})
    with pytest.raises(PermissionError, match="flag is not allowed"):
        await tool.execute({"argv": ["rg", "--hidden", "TOKEN"]})
    with pytest.raises(PermissionError, match="path escape"):
        await tool.execute({"argv": ["rg", "TOKEN", "../outside"]})
    with pytest.raises(PermissionError, match="cwd must not escape"):
        await tool.execute({"argv": ["rg", "TOKEN"], "cwd": ".."})


async def test_readonly_browser_gateway_can_use_project_root_for_commands(
    tmp_path,
) -> None:
    from src.agent_runtime.browser_artifacts import build_readonly_browser_tool_gateway
    from src.agent_runtime.models import InvocationProfile

    workspace_root = tmp_path / "workspace"
    project_root = tmp_path / "project"
    workspace_root.mkdir()
    source_dir = project_root / "src"
    source_dir.mkdir(parents=True)
    (source_dir / "bot.py").write_text(
        "def hermes_baseline_status():\n    return 'ok'\n",
        encoding="utf-8",
    )
    gateway = build_readonly_browser_tool_gateway(
        workspace_root=workspace_root,
        readonly_command_root=project_root,
        enable_browser_use=False,
    )
    profile = InvocationProfile(
        id="readonly-code-search",
        allowed_capabilities=("run_readonly_commands",),
    )

    result = await gateway.execute(
        profile,
        "run_readonly_command",
        {"argv": ["rg", "-n", "hermes_baseline", "src/bot.py"]},
    )

    assert result["exit_code"] == 0
    assert "1:def hermes_baseline_status():" in result["stdout"]


async def test_merge_tool_gateways_keeps_first_duplicate_tool_root(tmp_path) -> None:
    from src.agent_runtime.browser_artifacts import build_readonly_browser_tool_gateway
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import InvocationProfile
    from src.agent_runtime.tools import merge_tool_gateways

    workspace_root = tmp_path / "workspace"
    project_root = tmp_path / "project"
    workspace_root.mkdir()
    source_dir = project_root / "src"
    source_dir.mkdir(parents=True)
    (source_dir / "bot.py").write_text(
        "def hermes_baseline_import():\n    return 'ok'\n",
        encoding="utf-8",
    )
    project_gateway = build_readonly_browser_tool_gateway(
        workspace_root=workspace_root,
        readonly_command_root=project_root,
        enable_browser_use=False,
    )
    workspace_gateway = build_builtin_tool_gateway(workspace_root=workspace_root)
    combined = merge_tool_gateways(project_gateway, workspace_gateway)
    profile = InvocationProfile(
        id="readonly-code-search",
        allowed_capabilities=("run_readonly_commands",),
    )

    result = await combined.execute(
        profile,
        "run_readonly_command",
        {"argv": ["rg", "-n", "hermes_baseline", "src/bot.py"]},
    )

    assert result["exit_code"] == 0
    assert "1:def hermes_baseline_import():" in result["stdout"]


async def test_telegram_mcp_write_capability_requires_policy_approval() -> None:
    from src.agent_runtime.approvals import AgentToolApproval
    from src.agent_runtime.models import InvocationProfile
    from src.agent_runtime.tools import FunctionAgentTool, ToolDeniedError, ToolGateway

    async def send(payload: dict[str, Any]) -> str:
        return f"sent:{payload['chat']}"

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("telegram_mcp_send_message", "telegram_mcp_send", send),
        )
    )
    profile = InvocationProfile(
        id="telegram_mcp.personal_actions",
        allowed_capabilities=("telegram_mcp_send",),
    )

    assert "telegram_mcp_send_message" not in gateway.build_toolset(profile)
    with pytest.raises(ToolDeniedError):
        await gateway.execute(
            profile,
            "telegram_mcp_send_message",
            {"chat": "@example"},
        )

    approval = AgentToolApproval.approved(
        approval_id="approval-telegram-mcp",
        capabilities=("telegram_mcp_send",),
        approved_by=1291112109,
    )
    assert "telegram_mcp_send_message" in gateway.build_toolset(
        profile,
        approval=approval,
    )


async def test_desktop_control_capabilities_require_tool_gateway_approval() -> None:
    from src.agent_runtime.approvals import AgentToolApproval
    from src.agent_runtime.models import InvocationProfile
    from src.agent_runtime.tools import FunctionAgentTool, ToolDeniedError, ToolGateway

    async def pause(_payload: dict[str, Any]) -> str:
        return "paused"

    gateway = ToolGateway(
        tools=(
            FunctionAgentTool("desktop_media_pause", "desktop_media_control", pause),
        )
    )
    profile = InvocationProfile(
        id="desktop_control.convenience",
        worker="desktop_control",
        allowed_capabilities=("desktop_media_control",),
    )

    assert gateway.build_toolset(profile) == {}
    with pytest.raises(ToolDeniedError):
        await gateway.execute(profile, "desktop_media_pause", {})

    approval = AgentToolApproval.approved(
        approval_id="approval-desktop",
        capabilities=("desktop_media_control",),
        approved_by=1291112109,
    )
    assert "desktop_media_pause" in gateway.build_toolset(profile, approval=approval)


async def test_computer_use_shell_tool_requires_matching_approval(tmp_path) -> None:
    from src.agent_runtime.approvals import AgentToolApproval
    from src.agent_runtime.computer_use import build_computer_use_tool_gateway
    from src.agent_runtime.profiles import COMPUTER_USE_APPROVED_SHELL
    from src.agent_runtime.tools import ToolDeniedError

    calls: list[tuple[str, ...]] = []

    async def runner(
        argv: tuple[str, ...],
        *,
        cwd: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        del cwd, timeout_seconds
        calls.append(argv)
        return {"exit_code": 0, "stdout": "ok", "stderr": ""}

    gateway = build_computer_use_tool_gateway(
        workspace_root=tmp_path,
        shell_runner=runner,
        shell_allowed_executables=("echo",),
    )
    payload = {
        "action": "desktop_shell_command",
        "argv": ["echo", "ok"],
        "cwd": ".",
    }

    assert "desktop_shell" not in gateway.build_toolset(COMPUTER_USE_APPROVED_SHELL)
    with pytest.raises(ToolDeniedError):
        await gateway.execute(COMPUTER_USE_APPROVED_SHELL, "desktop_shell", payload)

    wrong_approval = AgentToolApproval.approved(
        approval_id="approval-submit",
        capabilities=("browser_submit",),
        approved_by=1291112109,
    )
    assert "desktop_shell" not in gateway.build_toolset(
        COMPUTER_USE_APPROVED_SHELL,
        approval=wrong_approval,
    )

    approval = AgentToolApproval.approved(
        approval_id="approval-shell",
        capabilities=("desktop.shell",),
        approved_by=1291112109,
    )
    assert "desktop_shell" in gateway.build_toolset(
        COMPUTER_USE_APPROVED_SHELL,
        approval=approval,
    )
    result = await gateway.execute(
        COMPUTER_USE_APPROVED_SHELL,
        "desktop_shell",
        payload,
        approval=approval,
    )

    assert result["status"] == "completed"
    assert calls == [("echo", "ok")]


def test_file_agent_approval_store_persists_request_and_grant(tmp_path) -> None:
    from src.agent_runtime.approvals import FileAgentApprovalStore

    store = FileAgentApprovalStore(tmp_path / "approvals")

    request = store.create_request(
        job_id="job-1",
        capability="restart",
        reason="apply new code",
        requested_by=1291112109,
        telegram_status="Жду подтверждение restart.",
    )
    loaded = store.get(request.approval_id)
    approval = store.approve(request.approval_id, approved_by=1291112109)

    assert loaded.status == "pending"
    assert loaded.telegram_status == "Жду подтверждение restart."
    assert approval.allows("restart")
    assert store.get(request.approval_id).status == "approved"


async def test_builtin_gateway_exposes_browser_read_but_not_submit(
    tmp_path,
) -> None:
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.profiles import WEB_RESEARCH_READONLY
    from src.agent_runtime.tools import FunctionAgentTool, ToolDeniedError

    async def fetch(url: str) -> str:
        return f"<html>{url}</html>"

    async def submit(payload: dict[str, Any]) -> str:
        return f"submitted:{payload['url']}"

    gateway = build_builtin_tool_gateway(
        workspace_root=tmp_path,
        web_fetcher=fetch,
        extra_tools=(
            FunctionAgentTool("browser_submit_form", "browser_submit", submit),
        ),
    )

    toolset = gateway.build_toolset(WEB_RESEARCH_READONLY)
    assert "browser_read_url" in toolset
    assert "browser_submit_form" not in toolset
    assert (
        await gateway.execute(
            WEB_RESEARCH_READONLY,
            "browser_read_url",
            {"url": "https://example.com/page"},
        )
        == "<html>https://example.com/page</html>"
    )
    with pytest.raises(ToolDeniedError):
        await gateway.execute(
            WEB_RESEARCH_READONLY,
            "browser_submit_form",
            {"url": "https://example.com/form"},
        )


async def test_builtin_gateway_exposes_web_search_only_when_configured(
    tmp_path,
) -> None:
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.profiles import WEB_RESEARCH_READONLY

    async def search(query: str, max_results: int) -> tuple[str, ...]:
        return tuple(
            f"https://example.com/{query}-{index}" for index in range(max_results)
        )

    without_search = build_builtin_tool_gateway(workspace_root=tmp_path)
    with_search = build_builtin_tool_gateway(
        workspace_root=tmp_path,
        web_searcher=search,
    )

    assert "web_search_sources" not in without_search.build_toolset(
        WEB_RESEARCH_READONLY
    )
    assert "web_search_sources" in with_search.build_toolset(WEB_RESEARCH_READONLY)
    assert await with_search.execute(
        WEB_RESEARCH_READONLY,
        "web_search_sources",
        {"query": "dreams", "max_results": 2},
    ) == ("https://example.com/dreams-0", "https://example.com/dreams-1")


def test_duckduckgo_html_parser_extracts_public_source_urls() -> None:
    from src.agent_runtime.builtin_tools import _extract_duckduckgo_result_urls

    html = """
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2Fwhatsnew%2F3.14.html&rut=abc">Python</a>
    <a href="https://duckduckgo.com/settings">settings</a>
    <a class="result-link" href="https://example.com/article">Example</a>
    <a href="http://127.0.0.1/private">private</a>
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2Fwhatsnew%2F3.14.html&rut=dup">Duplicate</a>
    """

    assert _extract_duckduckgo_result_urls(html, max_results=5) == (
        "https://docs.python.org/3/whatsnew/3.14.html",
        "https://example.com/article",
    )


def test_brave_html_parser_extracts_public_source_urls() -> None:
    from src.agent_runtime.builtin_tools import _extract_brave_result_urls

    html = """
    <a href="/search?q=Python">internal</a>
    <a href="https://cdn.search.brave.com/app.css">asset</a>
    <a href="https://docs.python.org/3/whatsnew/3.14.html">Python docs</a>
    <a href="https://www.python.org/downloads/release/python-3144/">Release</a>
    <a href="http://127.0.0.1/private">private</a>
    <a href="https://docs.python.org/3/whatsnew/3.14.html">duplicate</a>
    """

    assert _extract_brave_result_urls(html, max_results=5) == (
        "https://docs.python.org/3/whatsnew/3.14.html",
        "https://www.python.org/downloads/release/python-3144/",
    )


async def test_public_dns_guard_allows_controlled_egress_proxy_for_hostnames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.agent_runtime.builtin_tools import (
        is_public_http_url,
        verify_public_dns,
    )

    def fake_getaddrinfo(*_args: object, **_kwargs: object) -> list[object]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("198.18.11.10", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    assert is_public_http_url("https://example.com") is True
    assert is_public_http_url("https://198.18.11.10") is False
    await verify_public_dns("https://example.com")


async def test_public_dns_guard_still_rejects_private_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.agent_runtime.builtin_tools import verify_public_dns

    def fake_getaddrinfo(*_args: object, **_kwargs: object) -> list[object]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.2", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(ValueError, match="private address"):
        await verify_public_dns("https://example.com")


async def test_redirect_helper_closes_async_response_before_following(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.agent_runtime.builtin_tools import get_public_url_following_safe_redirects

    class Response:
        def __init__(
            self,
            status_code: int,
            location: str = "",
        ) -> None:
            self.status_code = status_code
            self.headers = {"location": location} if location else {}
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    class Client:
        def __init__(self) -> None:
            self.responses = [
                Response(302, "https://example.com/final"),
                Response(200),
            ]
            self.seen: list[str] = []

        async def get(self, url: str) -> Response:
            self.seen.append(url)
            return self.responses.pop(0)

    def fake_getaddrinfo(*_args: object, **_kwargs: object) -> list[object]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 443))]

    client = Client()
    first = client.responses[0]
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    response = await get_public_url_following_safe_redirects(
        client,
        "https://example.com/start",
    )

    assert response.status_code == 200
    assert first.closed is True
    assert client.seen == [
        "https://example.com/start",
        "https://example.com/final",
    ]


async def test_builtin_gateway_exposes_screenshot_and_download_only_when_configured(
    tmp_path,
) -> None:
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.profiles import WEB_RESEARCH_READONLY

    async def screenshot(url: str) -> str:
        return f"artifacts/screenshot-{url.rsplit('/', 1)[-1]}.png"

    async def download(url: str) -> str:
        return f"artifacts/download-{url.rsplit('/', 1)[-1]}"

    without_tools = build_builtin_tool_gateway(workspace_root=tmp_path)
    with_tools = build_builtin_tool_gateway(
        workspace_root=tmp_path,
        browser_screenshotter=screenshot,
        browser_downloader=download,
    )

    assert "browser_screenshot_url" not in without_tools.build_toolset(
        WEB_RESEARCH_READONLY
    )
    assert "browser_download_file" not in without_tools.build_toolset(
        WEB_RESEARCH_READONLY
    )
    assert "browser_screenshot_url" in with_tools.build_toolset(WEB_RESEARCH_READONLY)
    assert "browser_download_file" in with_tools.build_toolset(WEB_RESEARCH_READONLY)
    assert (
        await with_tools.execute(
            WEB_RESEARCH_READONLY,
            "browser_screenshot_url",
            {"url": "https://example.com/post"},
        )
        == "artifacts/screenshot-post.png"
    )
    assert (
        await with_tools.execute(
            WEB_RESEARCH_READONLY,
            "browser_download_file",
            {"url": "https://example.com/file.pdf"},
        )
        == "artifacts/download-file.pdf"
    )


async def test_builtin_gateway_drafts_form_without_submit(tmp_path) -> None:
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import InvocationProfile

    gateway = build_builtin_tool_gateway(workspace_root=tmp_path)
    profile = InvocationProfile(
        id="browser.draft_form",
        allowed_capabilities=("browser_draft_form",),
        denied_capabilities=("browser_submit",),
    )

    toolset = gateway.build_toolset(profile)
    assert "browser_draft_form" in toolset

    artifact = await gateway.execute(
        profile,
        "browser_draft_form",
        {
            "url": "https://example.com/contact",
            "method": "POST",
            "fields": {"email": "nikita@example.com", "message": "hello"},
            "purpose": "prepare a contact form draft",
        },
    )

    assert artifact.startswith("agent_runtime/browser_artifacts/form-draft-")
    draft_path = tmp_path / artifact
    assert draft_path.is_file()
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    assert draft["action_url"] == "https://example.com/contact"
    assert draft["method"] == "POST"
    assert draft["fields"] == {"email": "nikita@example.com", "message": "hello"}
    assert draft["submit_blocked"] is True
    assert draft["requires_approval_for_submit"] is True


async def test_builtin_gateway_refuses_form_draft_submit_flags(tmp_path) -> None:
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import InvocationProfile

    gateway = build_builtin_tool_gateway(workspace_root=tmp_path)
    profile = InvocationProfile(
        id="browser.draft_form",
        allowed_capabilities=("browser_draft_form",),
    )

    with pytest.raises(ValueError, match="does not submit"):
        await gateway.execute(
            profile,
            "browser_draft_form",
            {
                "url": "https://example.com/contact",
                "fields": {"message": "hello"},
                "submit": True,
            },
        )


async def test_builtin_gateway_browser_submit_requires_injected_tool_and_approval(
    tmp_path,
) -> None:
    from src.agent_runtime.approvals import AgentToolApproval
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import InvocationProfile

    calls: list[dict[str, Any]] = []

    async def submit(
        draft: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, str]:
        calls.append({"draft": draft, "payload": payload})
        return {"status": "submitted", "artifact": "agent_runtime/submits/1.json"}

    profile = InvocationProfile(
        id="browser.submit",
        allowed_capabilities=("browser_draft_form", "browser_submit"),
    )
    without_submit = build_builtin_tool_gateway(workspace_root=tmp_path)
    with_submit = build_builtin_tool_gateway(
        workspace_root=tmp_path,
        browser_submitter=submit,
    )

    draft_artifact = await with_submit.execute(
        profile,
        "browser_draft_form",
        {
            "url": "https://example.com/contact",
            "method": "POST",
            "fields": {"message": "approved submit"},
        },
    )
    approval = AgentToolApproval.approved(
        approval_id="approval-browser-submit",
        capabilities=("browser_submit",),
        approved_by=1291112109,
    )

    assert "browser_submit_form" not in without_submit.build_toolset(
        profile,
        approval=approval,
    )
    assert "browser_submit_form" not in with_submit.build_toolset(profile)
    assert "browser_submit_form" in with_submit.build_toolset(
        profile,
        approval=approval,
    )

    result = await with_submit.execute(
        profile,
        "browser_submit_form",
        {
            "draft_artifact": draft_artifact,
            "action_kind": "form_submit",
        },
        approval=approval,
    )

    assert result == {"status": "submitted", "artifact": "agent_runtime/submits/1.json"}
    assert calls[0]["draft"]["action_url"] == "https://example.com/contact"
    assert calls[0]["draft"]["fields"] == {"message": "approved submit"}


async def test_builtin_gateway_browser_submit_rejects_login_purchase_delete_kinds(
    tmp_path,
) -> None:
    from src.agent_runtime.approvals import AgentToolApproval
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import InvocationProfile

    called = False

    async def submit(
        draft: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, str]:
        nonlocal called
        del draft, payload
        called = True
        return {"status": "submitted"}

    gateway = build_builtin_tool_gateway(
        workspace_root=tmp_path,
        browser_submitter=submit,
    )
    profile = InvocationProfile(
        id="browser.submit",
        allowed_capabilities=("browser_draft_form", "browser_submit"),
    )
    draft_artifact = await gateway.execute(
        profile,
        "browser_draft_form",
        {
            "url": "https://example.com/contact",
            "fields": {"message": "approved submit"},
        },
    )
    approval = AgentToolApproval.approved(
        approval_id="approval-browser-submit",
        capabilities=("browser_submit",),
        approved_by=1291112109,
    )

    with pytest.raises(ValueError, match="separate high-risk policy"):
        await gateway.execute(
            profile,
            "browser_submit_form",
            {
                "draft_artifact": draft_artifact,
                "action_kind": "login",
            },
            approval=approval,
        )

    assert called is False


async def test_builtin_gateway_browser_high_risk_actions_are_policy_separated(
    tmp_path,
) -> None:
    from src.agent_runtime.approvals import AgentToolApproval
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import InvocationProfile
    from src.agent_runtime.tools import ToolDeniedError

    calls: list[dict[str, Any]] = []

    async def high_risk_action(
        action_kind: str,
        draft: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, str]:
        calls.append(
            {
                "action_kind": action_kind,
                "draft": draft,
                "payload": payload,
            }
        )
        return {
            "status": "completed",
            "action_kind": action_kind,
            "artifact": f"agent_runtime/browser_artifacts/{action_kind}.json",
        }

    gateway = build_builtin_tool_gateway(
        workspace_root=tmp_path,
        browser_high_risk_handlers={
            "login": high_risk_action,
            "purchase": high_risk_action,
            "publish": high_risk_action,
            "delete": high_risk_action,
            "send": high_risk_action,
        },
    )
    profile = InvocationProfile(
        id="browser.high_risk",
        allowed_capabilities=(
            "browser_draft_form",
            "login",
            "purchase",
            "publish",
            "delete",
            "send_message",
        ),
    )
    draft_artifact = await gateway.execute(
        profile,
        "browser_draft_form",
        {
            "url": "https://example.com/checkout",
            "fields": {"message": "draft only"},
        },
    )
    login_approval = AgentToolApproval.approved(
        approval_id="approval-browser-login",
        capabilities=("login",),
        approved_by=1291112109,
    )

    assert "browser_login_action" not in gateway.build_toolset(profile)
    assert "browser_login_action" in gateway.build_toolset(
        profile,
        approval=login_approval,
    )
    assert "browser_purchase_action" not in gateway.build_toolset(
        profile,
        approval=login_approval,
    )

    result = await gateway.execute(
        profile,
        "browser_login_action",
        {
            "draft_artifact": draft_artifact,
            "action_kind": "login",
        },
        approval=login_approval,
    )

    assert result["action_kind"] == "login"
    assert calls[0]["action_kind"] == "login"
    assert calls[0]["draft"]["action_url"] == "https://example.com/checkout"

    with pytest.raises(ValueError, match="handles only login"):
        await gateway.execute(
            profile,
            "browser_login_action",
            {
                "draft_artifact": draft_artifact,
                "action_kind": "purchase",
            },
            approval=login_approval,
        )
    with pytest.raises(ToolDeniedError):
        await gateway.execute(
            profile,
            "browser_purchase_action",
            {
                "draft_artifact": draft_artifact,
                "action_kind": "purchase",
            },
            approval=login_approval,
        )


async def test_builtin_gateway_blocks_workspace_path_escape(tmp_path) -> None:
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import InvocationProfile

    gateway = build_builtin_tool_gateway(workspace_root=tmp_path)
    profile = InvocationProfile(
        id="workspace.readonly",
        allowed_capabilities=("read_workspace",),
    )
    (tmp_path / "note.md").write_text("ok", encoding="utf-8")

    assert await gateway.execute(profile, "read_workspace_file", {"path": "note.md"})
    with pytest.raises(PermissionError):
        await gateway.execute(profile, "read_workspace_file", {"path": "../x"})


async def test_builtin_gateway_reads_only_whitelisted_project_files(
    tmp_path,
) -> None:
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import InvocationProfile

    workspace = tmp_path / "workspace"
    project = tmp_path / "project"
    workspace.mkdir()
    (project / "src").mkdir(parents=True)
    (project / "src" / "allowed.py").write_text("ARCHITECTURE", encoding="utf-8")
    (project / ".env").write_text("SECRET=1", encoding="utf-8")
    gateway = build_builtin_tool_gateway(
        workspace_root=workspace,
        project_root=project,
        project_read_allowed_paths=("src/allowed.py",),
    )
    profile = InvocationProfile(
        id="project.readonly",
        allowed_capabilities=("read_workspace",),
    )

    assert (
        await gateway.execute(
            profile,
            "read_project_file",
            {"path": "src/allowed.py"},
        )
        == "ARCHITECTURE"
    )
    with pytest.raises(PermissionError):
        await gateway.execute(profile, "read_project_file", {"path": ".env"})


async def test_builtin_gateway_writes_only_whitelisted_workspace_files_after_approval(
    tmp_path,
) -> None:
    from src.agent_runtime.approvals import AgentToolApproval
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import InvocationProfile
    from src.agent_runtime.tools import ToolDeniedError

    gateway = build_builtin_tool_gateway(
        workspace_root=tmp_path,
        workspace_write_allowed_paths=(
            "agent_runtime/local_file_tasks/stage-l-local-file-baseline.txt",
        ),
    )
    profile = InvocationProfile(
        id="local_file.approved_write",
        allowed_capabilities=("write_whitelisted_files_after_approval",),
    )
    payload = {
        "path": "agent_runtime/local_file_tasks/stage-l-local-file-baseline.txt",
        "content": "bounded write\n",
        "mode": "create",
    }

    assert "write_workspace_file_after_gate" not in gateway.build_toolset(profile)
    with pytest.raises(ToolDeniedError):
        await gateway.execute(profile, "write_workspace_file_after_gate", payload)

    approval = AgentToolApproval.approved(
        approval_id="approval-local-file-write",
        capabilities=("write_whitelisted_files_after_approval",),
        approved_by=1291112109,
    )
    result = await gateway.execute(
        profile,
        "write_workspace_file_after_gate",
        payload,
        approval=approval,
    )

    assert result == {
        "path": "agent_runtime/local_file_tasks/stage-l-local-file-baseline.txt",
        "mode": "create",
        "bytes_written": len(b"bounded write\n"),
    }
    assert (
        tmp_path / "agent_runtime/local_file_tasks/stage-l-local-file-baseline.txt"
    ).read_text(encoding="utf-8") == "bounded write\n"


async def test_builtin_gateway_rejects_unlisted_workspace_write_paths(
    tmp_path,
) -> None:
    from src.agent_runtime.approvals import AgentToolApproval
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import InvocationProfile

    gateway = build_builtin_tool_gateway(
        workspace_root=tmp_path,
        workspace_write_allowed_paths=("agent_runtime/local_file_tasks/allowed.txt",),
    )
    profile = InvocationProfile(
        id="local_file.approved_write",
        allowed_capabilities=("write_whitelisted_files_after_approval",),
    )
    approval = AgentToolApproval.approved(
        approval_id="approval-local-file-write",
        capabilities=("write_whitelisted_files_after_approval",),
        approved_by=1291112109,
    )

    with pytest.raises(PermissionError):
        await gateway.execute(
            profile,
            "write_workspace_file_after_gate",
            {
                "path": "agent_runtime/local_file_tasks/../secrets.txt",
                "content": "nope",
            },
            approval=approval,
        )
    with pytest.raises(PermissionError):
        await gateway.execute(
            profile,
            "write_workspace_file_after_gate",
            {"path": ".env", "content": "SECRET=leak"},
            approval=approval,
        )

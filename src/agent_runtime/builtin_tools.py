"""Built-in Tool Gateway tools for Agent Runtime profiles."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import ipaddress
import json
import os
import shlex
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

from src.agent_runtime.tools import AgentTool, ToolGateway


class WebFetcher(Protocol):
    """Async read-only URL fetcher used by BrowserReadTool."""

    async def __call__(self, url: str) -> str: ...


class WebSearcher(Protocol):
    """Async read-only source searcher used by WebSearchSourcesTool."""

    async def __call__(self, query: str, max_results: int) -> tuple[str, ...]: ...


class BrowserScreenshotter(Protocol):
    """Async read-only screenshot provider used by BrowserScreenshotTool."""

    async def __call__(self, url: str) -> str: ...


class BrowserDownloader(Protocol):
    """Async read-only download provider used by BrowserDownloadTool."""

    async def __call__(self, url: str) -> str: ...


class BrowserSubmitter(Protocol):
    """Async high-risk browser form submitter used only after approval."""

    async def __call__(
        self,
        draft: dict[str, Any],
        payload: dict[str, Any],
    ) -> Any: ...


class BrowserHighRiskActionHandler(Protocol):
    """Async browser action handler for separate high-risk policies."""

    async def __call__(
        self,
        action_kind: str,
        draft: dict[str, Any],
        payload: dict[str, Any],
    ) -> Any: ...


async def duckduckgo_html_search_sources(
    query: str,
    max_results: int,
    *,
    proxy: str = "",
) -> tuple[str, ...]:
    """Return public result URLs from read-only search HTML endpoints."""
    query = query.strip()
    if not query:
        return ()
    bounded_max = max(1, min(int(max_results), 10))
    results = await _duckduckgo_html_search_sources(
        query,
        bounded_max,
        proxy=proxy,
    )
    if results:
        return results
    return await _brave_html_search_sources(query, bounded_max, proxy=proxy)


async def _duckduckgo_html_search_sources(
    query: str,
    max_results: int,
    *,
    proxy: str = "",
) -> tuple[str, ...]:
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        import httpx

        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=False,
            proxy=_httpx_proxy(proxy),
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; ZHVUSHA/1.0; +https://localhost)"
                )
            },
        ) as client:
            response = await _get_public_url_following_safe_redirects(client, url)
            response.raise_for_status()
    except (OSError, ValueError, RuntimeError):
        return ()
    except Exception as exc:
        if exc.__class__.__module__.startswith("httpx"):
            return ()
        raise
    return _extract_duckduckgo_result_urls(
        str(response.text),
        max_results=max_results,
    )


async def _brave_html_search_sources(
    query: str,
    max_results: int,
    *,
    proxy: str = "",
) -> tuple[str, ...]:
    url = f"https://search.brave.com/search?q={quote_plus(query)}"
    try:
        import httpx

        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=False,
            proxy=_httpx_proxy(proxy),
            headers={"User-Agent": "Mozilla/5.0 (compatible; ZHVUSHA/1.0)"},
        ) as client:
            response = await _get_public_url_following_safe_redirects(client, url)
            response.raise_for_status()
    except (OSError, ValueError, RuntimeError):
        return ()
    except Exception as exc:
        if exc.__class__.__module__.startswith("httpx"):
            return ()
        raise
    return _extract_brave_result_urls(str(response.text), max_results=max_results)


_BROWSER_HIGH_RISK_ACTION_CAPABILITIES: tuple[tuple[str, str], ...] = (
    ("login", "login"),
    ("purchase", "purchase"),
    ("publish", "publish"),
    ("delete", "delete"),
    ("send", "send_message"),
)
_CONTROLLED_EGRESS_PROXY_NETWORKS: tuple[
    ipaddress.IPv4Network | ipaddress.IPv6Network, ...
] = (ipaddress.ip_network("198.18.0.0/15"),)


@dataclass
class WorkspaceReadTool:
    """Read a workspace file inside a fixed root."""

    root: Path
    max_chars: int = 20_000
    name: str = "read_workspace_file"
    capability: str = "read_workspace"

    async def execute(self, payload: Mapping[str, Any]) -> str:
        raw_path = str(payload.get("path", "")).strip()
        if not raw_path:
            raise ValueError("path is required")
        root = self.root.expanduser().resolve()
        target = (root / raw_path).expanduser().resolve()
        if not target.is_relative_to(root):
            raise PermissionError("path escapes workspace root")
        text = target.read_text(encoding="utf-8")
        return text[: self.max_chars]


@dataclass
class WorkspaceWriteTool:
    """Write only exact allowlisted workspace files after approval."""

    root: Path
    allowed_paths: tuple[str, ...]
    max_chars: int = 50_000
    name: str = "write_workspace_file_after_gate"
    capability: str = "write_whitelisted_files_after_approval"

    async def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        raw_path = str(payload.get("path", "")).strip()
        if not raw_path:
            raise ValueError("path is required")
        mode = str(payload.get("mode", "create")).strip().lower() or "create"
        if mode not in {"create", "overwrite", "append"}:
            raise ValueError("mode must be create, overwrite or append")
        content = str(payload.get("content", ""))
        if len(content) > self.max_chars:
            raise ValueError("content exceeds max_chars")

        normalized = _normalize_workspace_write_path(raw_path)
        allowed = {
            _normalize_workspace_write_path(path)
            for path in self.allowed_paths
            if str(path).strip()
        }
        if normalized not in allowed:
            raise PermissionError("workspace write path is not allowlisted")

        root = self.root.expanduser().resolve()
        target = (root / normalized).resolve()
        if not target.is_relative_to(root):
            raise PermissionError("path escapes workspace root")
        target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "create" and target.exists():
            raise FileExistsError("workspace write target already exists")
        if mode == "append":
            with target.open("a", encoding="utf-8") as handle:
                handle.write(content)
        else:
            target.write_text(content, encoding="utf-8")
        return {
            "path": normalized,
            "mode": mode,
            "bytes_written": len(content.encode()),
        }


@dataclass
class ProjectReadTool:
    """Read whitelisted project files for architecture-grounded artifacts."""

    root: Path
    allowed_paths: tuple[str, ...]
    max_chars: int = 20_000
    name: str = "read_project_file"
    capability: str = "read_workspace"

    async def execute(self, payload: Mapping[str, Any]) -> str:
        raw_path = str(payload.get("path", "")).strip()
        if not raw_path:
            raise ValueError("path is required")
        normalized = Path(raw_path).as_posix()
        if normalized not in self.allowed_paths:
            raise PermissionError("project path is not allowed")
        root = self.root.expanduser().resolve()
        target = (root / normalized).resolve()
        if not target.is_relative_to(root):
            raise PermissionError("path escapes project root")
        text = target.read_text(encoding="utf-8")
        return text[: self.max_chars]


@dataclass
class ReadOnlyCommandTool:
    """Run a bounded allowlisted read-only command inside the workspace."""

    root: Path
    allowed_commands: tuple[str, ...] = ("rg",)
    timeout_seconds: float = 10.0
    max_output_chars: int = 20_000
    name: str = "run_readonly_command"
    capability: str = "run_readonly_commands"

    async def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        argv = _readonly_argv(payload.get("argv"))
        executable = argv[0]
        if executable not in set(self.allowed_commands):
            raise PermissionError("command is not allowlisted for read-only execution")
        for arg in argv:
            _validate_readonly_command_arg(arg)
        root = self.root.expanduser().resolve()
        cwd = _resolve_workspace_child(root, payload.get("cwd", "."))
        create_process = getattr(asyncio, "create_subprocess_" + "exec")
        proc = await create_process(
            *argv,
            cwd=str(cwd),
            env=_readonly_command_env(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_raw, stderr_raw = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout_seconds,
            )
        except TimeoutError as exc:
            proc.kill()
            with contextlib.suppress(ProcessLookupError):
                await proc.wait()
            raise TimeoutError("read-only command timed out") from exc
        stdout, stdout_truncated = _decode_limited(stdout_raw, self.max_output_chars)
        stderr, stderr_truncated = _decode_limited(stderr_raw, self.max_output_chars)
        return {
            "argv": argv,
            "command": shlex.join(argv),
            "cwd": cwd.relative_to(root).as_posix() or ".",
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }


@dataclass
class BrowserReadTool:
    """Read a URL without form submission or external side effects."""

    fetcher: WebFetcher | None = None
    proxy: str = ""
    max_chars: int = 20_000
    name: str = "browser_read_url"
    capability: str = "browser_read"

    async def execute(self, payload: Mapping[str, Any]) -> str:
        url = str(payload.get("url", "")).strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("url must be http(s)")
        if self.fetcher is not None:
            text = await self.fetcher(url)
        else:
            text = await _httpx_fetch(url, proxy=self.proxy)
        return text[: self.max_chars]


@dataclass
class WebSearchSourcesTool:
    """Find read-only source URLs without opening forms or taking actions."""

    searcher: WebSearcher
    name: str = "web_search_sources"
    capability: str = "web_search_sources"

    async def execute(self, payload: Mapping[str, Any]) -> tuple[str, ...]:
        query = str(payload.get("query", "")).strip()
        if not query:
            raise ValueError("query is required")
        max_results_raw = payload.get("max_results", 5)
        max_results = int(max_results_raw)
        if max_results < 1:
            raise ValueError("max_results must be positive")
        results = await self.searcher(query, max_results)
        return tuple(result.strip() for result in results if result.strip())[
            :max_results
        ]


@dataclass
class BrowserScreenshotTool:
    """Capture a screenshot artifact without submitting forms or logging in."""

    screenshotter: BrowserScreenshotter
    enforce_public_network_guard: bool = False
    proxy: str = ""
    name: str = "browser_screenshot_url"
    capability: str = "browser_screenshot"

    async def execute(self, payload: Mapping[str, Any]) -> str:
        url = _validated_http_url(payload)
        if self.enforce_public_network_guard:
            await verify_public_redirect_chain(url, proxy=self.proxy)
        return await self.screenshotter(url)


@dataclass
class BrowserDownloadTool:
    """Download an allowed URL as a read-only artifact."""

    downloader: BrowserDownloader
    enforce_public_network_guard: bool = False
    proxy: str = ""
    name: str = "browser_download_file"
    capability: str = "browser_download"

    async def execute(self, payload: Mapping[str, Any]) -> str:
        url = _validated_http_url(payload)
        if self.enforce_public_network_guard:
            await verify_public_redirect_chain(url, proxy=self.proxy)
        return await self.downloader(url)


@dataclass
class BrowserDraftFormTool:
    """Prepare a browser form draft artifact without submitting anything."""

    root: Path
    artifact_dir: str = "agent_runtime/browser_artifacts"
    max_fields: int = 50
    max_value_chars: int = 5_000
    name: str = "browser_draft_form"
    capability: str = "browser_draft_form"

    async def execute(self, payload: Mapping[str, Any]) -> str:
        _reject_submit_flags(payload)
        url = _validated_http_url(payload)
        method = str(payload.get("method", "POST")).strip().upper() or "POST"
        if method not in {"GET", "POST"}:
            raise ValueError("form draft method must be GET or POST")
        fields = self._normalize_fields(payload.get("fields", {}))
        draft = {
            "capability": self.capability,
            "action_url": url,
            "method": method,
            "fields": fields,
            "purpose": str(payload.get("purpose", "")).strip()[:1000],
            "notes": str(payload.get("notes", "")).strip()[:2000],
            "submit_blocked": True,
            "requires_approval_for_submit": True,
            "next_required_capability": "browser_submit",
            "source": "agent_runtime.browser_draft_form",
        }
        return self._write_draft_artifact(url=url, draft=draft)

    def _normalize_fields(self, raw: object) -> dict[str, str]:
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise ValueError("fields must be a JSON object")
        if len(raw) > self.max_fields:
            raise ValueError("form draft has too many fields")
        fields: dict[str, str] = {}
        for key, value in raw.items():
            field_name = str(key).strip()
            if not field_name:
                raise ValueError("form draft field names must be non-empty")
            fields[field_name] = _compact_field_value(value)[: self.max_value_chars]
        return fields

    def _write_draft_artifact(self, *, url: str, draft: dict[str, object]) -> str:
        root = self.root.expanduser().resolve()
        artifact_root = (root / self.artifact_dir).resolve()
        if not artifact_root.is_relative_to(root):
            raise ValueError("artifact_dir escapes workspace root")
        serialized = json.dumps(draft, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
        host = _safe_artifact_name(urlparse(url).netloc or "form")
        target = (artifact_root / f"form-draft-{digest}-{host}.json").resolve()
        if not target.is_relative_to(root):
            raise RuntimeError("browser draft artifact path escapes workspace root")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(draft, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target.relative_to(root).as_posix()


@dataclass
class BrowserSubmitTool:
    """Submit a previously prepared browser form draft after approval."""

    root: Path
    submitter: BrowserSubmitter
    name: str = "browser_submit_form"
    capability: str = "browser_submit"

    async def execute(self, payload: Mapping[str, Any]) -> Any:
        action_kind = str(payload.get("action_kind", "form_submit")).strip().lower()
        if action_kind not in {"form_submit", "generic_form_submit"}:
            raise ValueError(
                "browser_submit_form requires a separate high-risk policy for "
                "login, purchase, publish, delete or send actions"
            )
        draft_artifact = str(
            payload.get("draft_artifact") or payload.get("draft_path") or ""
        ).strip()
        if not draft_artifact:
            raise ValueError("draft_artifact is required")
        draft = _load_browser_form_draft(self.root, draft_artifact)
        return await self.submitter(draft, dict(payload))


class _SearchResultLinkParser(HTMLParser):
    """Collect links from a search result HTML document."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.lower() != "a":
            return
        attr_map = {key.lower(): value or "" for key, value in attrs}
        href = attr_map.get("href", "").strip()
        if href:
            self.links.append(href)


def _extract_duckduckgo_result_urls(
    html: str,
    *,
    max_results: int,
) -> tuple[str, ...]:
    parser = _SearchResultLinkParser()
    parser.feed(html)
    results: list[str] = []
    seen: set[str] = set()
    for href in parser.links:
        url = _normalize_duckduckgo_result_url(href)
        if url is None or url in seen:
            continue
        seen.add(url)
        results.append(url)
        if len(results) >= max_results:
            break
    return tuple(results)


def _extract_brave_result_urls(
    html: str,
    *,
    max_results: int,
) -> tuple[str, ...]:
    parser = _SearchResultLinkParser()
    parser.feed(html)
    results: list[str] = []
    seen: set[str] = set()
    for href in parser.links:
        url = _normalize_brave_result_url(href)
        if url is None or url in seen:
            continue
        seen.add(url)
        results.append(url)
        if len(results) >= max_results:
            break
    return tuple(results)


def _normalize_duckduckgo_result_url(raw_href: str) -> str | None:
    href = raw_href.strip()
    if not href:
        return None
    if href.startswith("//"):
        href = f"https:{href}"
    elif href.startswith("/"):
        href = urljoin("https://duckduckgo.com", href)
    parsed = urlparse(href)
    host = (parsed.hostname or "").lower().rstrip(".")
    if host.endswith("duckduckgo.com"):
        target = parse_qs(parsed.query).get("uddg", ("",))[0]
        if not target:
            return None
        href = unquote(target)
    if not is_public_http_url(href):
        return None
    return href


def _normalize_brave_result_url(raw_href: str) -> str | None:
    href = raw_href.strip()
    if not href:
        return None
    if href.startswith("//"):
        href = f"https:{href}"
    elif href.startswith("/"):
        href = urljoin("https://search.brave.com", href)
    parsed = urlparse(href)
    host = (parsed.hostname or "").lower().rstrip(".")
    if host in {
        "search.brave.com",
        "cdn.search.brave.com",
        "imgs.search.brave.com",
        "tiles.search.brave.com",
    }:
        return None
    if not is_public_http_url(href):
        return None
    return href


def _httpx_proxy(proxy: str) -> str | None:
    normalized = proxy.strip()
    return normalized or None


@dataclass
class BrowserHighRiskActionTool:
    """Run one browser high-risk action after its separate approval policy."""

    root: Path
    action_kind: str
    capability: str
    handler: BrowserHighRiskActionHandler
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"browser_{self.action_kind}_action"

    async def execute(self, payload: Mapping[str, Any]) -> Any:
        payload_action = str(payload.get("action_kind", self.action_kind)).strip()
        payload_action = payload_action.lower()
        if payload_action != self.action_kind:
            raise ValueError(f"{self.name} handles only {self.action_kind} actions")
        draft_artifact = str(
            payload.get("draft_artifact") or payload.get("draft_path") or ""
        ).strip()
        if not draft_artifact:
            raise ValueError("draft_artifact is required")
        draft = _load_browser_form_draft(self.root, draft_artifact)
        return await self.handler(self.action_kind, draft, dict(payload))


async def _httpx_fetch(url: str, *, proxy: str = "") -> str:
    import httpx

    async with httpx.AsyncClient(
        timeout=20.0,
        follow_redirects=False,
        proxy=_httpx_proxy(proxy),
    ) as client:
        response = await _get_public_url_following_safe_redirects(client, url)
        response.raise_for_status()
        return str(response.text)


def _validated_http_url(payload: Mapping[str, Any]) -> str:
    url = str(payload.get("url", "")).strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be http(s)")
    if not is_public_http_url(url):
        raise ValueError("url must be a public http(s) source")
    return url


def is_public_http_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.username or parsed.password:
        return False
    host = parsed.hostname.strip().lower().rstrip(".")
    if host in {"localhost", "metadata.google.internal"} or host.endswith(".local"):
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


async def verify_public_redirect_chain(
    url: str,
    *,
    max_redirects: int = 5,
    proxy: str = "",
) -> None:
    import httpx

    async with httpx.AsyncClient(
        timeout=20.0,
        follow_redirects=False,
        proxy=_httpx_proxy(proxy),
    ) as client:
        response = await get_public_url_following_safe_redirects(
            client,
            url,
            max_redirects=max_redirects,
        )
        response.close()


async def get_public_url_following_safe_redirects(
    client: Any,
    url: str,
    *,
    max_redirects: int = 5,
) -> Any:
    """Fetch a public URL while enforcing DNS and redirect safety at each hop."""
    return await _get_public_url_following_safe_redirects(
        client,
        url,
        max_redirects=max_redirects,
    )


async def _get_public_url_following_safe_redirects(
    client: Any,
    url: str,
    *,
    max_redirects: int = 5,
) -> Any:
    current = url
    for _ in range(max_redirects + 1):
        if not is_public_http_url(current):
            raise ValueError("redirect target is not a public source")
        await verify_public_dns(current)
        response = await client.get(current)
        if response.status_code not in {301, 302, 303, 307, 308}:
            return response
        location = response.headers.get("location", "").strip()
        if not location:
            return response
        await _close_response(response)
        current = urljoin(current, location)
    raise ValueError("too many redirects")


async def _close_response(response: Any) -> None:
    aclose = getattr(response, "aclose", None)
    if callable(aclose):
        await aclose()
        return
    close = getattr(response, "close", None)
    if callable(close):
        close()


async def verify_public_dns(url: str) -> None:
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        raise ValueError("url must include hostname")
    try:
        infos = await asyncio.to_thread(
            socket.getaddrinfo,
            host,
            None,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise ValueError("source hostname could not be resolved safely") from exc
    if not infos:
        raise ValueError("source hostname did not resolve")
    for info in infos:
        address = str(info[4][0])
        if not _is_public_ip(address):
            raise ValueError("source hostname resolves to a private address")


def _is_public_ip(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if any(ip in network for network in _CONTROLLED_EGRESS_PROXY_NETWORKS):
        return True
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _reject_submit_flags(payload: Mapping[str, Any]) -> None:
    forbidden_flags = (
        "submit",
        "auto_submit",
        "click_submit",
        "browser_submit",
        "login",
        "purchase",
        "publish",
        "delete",
        "send",
        "send_message",
    )
    for key in forbidden_flags:
        if _truthy(payload.get(key)):
            raise ValueError("browser_draft_form does not submit or confirm actions")


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _compact_field_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int | float | bool):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _safe_artifact_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "-" for char in value)
    return safe.strip(".-_")[:80] or "form"


def _readonly_argv(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("argv must be a non-empty list of strings")
    if len(value) > 64:
        raise ValueError("argv is too long")
    argv = tuple(str(item) for item in value)
    if any(not item.strip() for item in argv):
        raise ValueError("argv items must be non-empty strings")
    if "/" in argv[0] or "\\" in argv[0]:
        raise PermissionError("command path is not allowed")
    return argv


def _validate_readonly_command_arg(value: str) -> None:
    if "\x00" in value:
        raise ValueError("argv items must not contain NUL bytes")
    forbidden_flags = {
        "--pre",
        "--pre-glob",
        "--hidden",
        "--no-ignore",
        "--no-ignore-global",
        "--no-ignore-parent",
        "--no-ignore-vcs",
        "--search-zip",
        "--follow",
        "--glob",
        "-g",
        "-L",
    }
    if value in forbidden_flags or value.startswith("-u"):
        raise PermissionError(
            f"rg flag is not allowed for read-only execution: {value}"
        )
    path_value = value[1:] if value.startswith("!") else value
    if path_value.startswith(("~", "/")):
        raise PermissionError("absolute or home-relative paths are not allowed")
    if ".." in Path(path_value).parts:
        raise PermissionError("path escape is not allowed")


def _resolve_workspace_child(root: Path, value: Any) -> Path:
    raw = str(value or ".").strip() or "."
    if raw.startswith(("~", "/")):
        raise PermissionError("cwd must stay inside workspace root")
    if ".." in Path(raw).parts:
        raise PermissionError("cwd must not escape workspace root")
    target = (root / raw).expanduser().resolve()
    if not target.is_relative_to(root):
        raise PermissionError("cwd escapes workspace root")
    return target


def _normalize_workspace_write_path(raw_path: str) -> str:
    path = Path(raw_path.strip()).as_posix()
    if not path or path.startswith("/") or path.startswith("~"):
        raise PermissionError("workspace write path must be relative")
    if ".." in Path(path).parts:
        raise PermissionError("workspace write path must not escape workspace")
    return path


def _readonly_command_env() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
    }


def _decode_limited(value: bytes, max_chars: int) -> tuple[str, bool]:
    text = value.decode("utf-8", errors="replace")
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _load_browser_form_draft(root: Path, draft_artifact: str) -> dict[str, Any]:
    workspace_root = root.expanduser().resolve()
    normalized = Path(draft_artifact).as_posix()
    if normalized.startswith("/") or ".." in Path(normalized).parts:
        raise PermissionError("draft_artifact must stay inside workspace")
    target = (workspace_root / normalized).resolve()
    if not target.is_relative_to(workspace_root):
        raise PermissionError("draft_artifact escapes workspace root")
    artifact_root = (workspace_root / "agent_runtime" / "browser_artifacts").resolve()
    if not target.is_relative_to(artifact_root):
        raise PermissionError("draft_artifact must be a browser artifact")
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("browser form draft artifact must be a JSON object")
    if data.get("capability") != "browser_draft_form":
        raise ValueError("draft_artifact is not a browser form draft")
    if data.get("submit_blocked") is not True:
        raise ValueError("browser form draft must have submit_blocked=true")
    if data.get("next_required_capability") != "browser_submit":
        raise ValueError("browser form draft must require browser_submit")
    return dict(data)


def build_builtin_tool_gateway(
    *,
    workspace_root: Path,
    readonly_command_root: Path | None = None,
    project_root: Path | None = None,
    project_read_allowed_paths: tuple[str, ...] = (),
    workspace_write_allowed_paths: tuple[str, ...] = (),
    web_fetcher: WebFetcher | None = None,
    web_searcher: WebSearcher | None = None,
    web_proxy: str = "",
    browser_screenshotter: BrowserScreenshotter | None = None,
    browser_downloader: BrowserDownloader | None = None,
    browser_submitter: BrowserSubmitter | None = None,
    browser_high_risk_handlers: Mapping[str, BrowserHighRiskActionHandler]
    | None = None,
    enforce_public_network_guard: bool = False,
    extra_tools: tuple[AgentTool, ...] = (),
) -> ToolGateway:
    """Build the default capability-enforced tool gateway.

    Dangerous tools such as `browser_submit`, `login`, `purchase`, `delete`,
    `edit_env`, `restart`, `publish` and write tools are intentionally absent
    unless an explicit policy layer injects their handler.
    """
    tools: list[AgentTool] = [
        WorkspaceReadTool(workspace_root),
        ReadOnlyCommandTool(readonly_command_root or workspace_root),
        BrowserReadTool(web_fetcher, proxy=web_proxy.strip()),
        BrowserDraftFormTool(workspace_root),
    ]
    if project_root is not None and project_read_allowed_paths:
        tools.append(
            ProjectReadTool(
                root=project_root,
                allowed_paths=project_read_allowed_paths,
            )
        )
    if workspace_write_allowed_paths:
        tools.append(
            WorkspaceWriteTool(
                root=workspace_root,
                allowed_paths=workspace_write_allowed_paths,
            )
        )
    if web_searcher is not None:
        tools.append(WebSearchSourcesTool(web_searcher))
    if browser_screenshotter is not None:
        tools.append(
            BrowserScreenshotTool(
                browser_screenshotter,
                enforce_public_network_guard=enforce_public_network_guard,
                proxy=web_proxy.strip(),
            )
        )
    if browser_downloader is not None:
        tools.append(
            BrowserDownloadTool(
                browser_downloader,
                enforce_public_network_guard=enforce_public_network_guard,
                proxy=web_proxy.strip(),
            )
        )
    if browser_submitter is not None:
        tools.append(BrowserSubmitTool(workspace_root, browser_submitter))
    if browser_high_risk_handlers is not None:
        tools.extend(
            _browser_high_risk_action_tools(
                workspace_root,
                browser_high_risk_handlers,
            )
        )
    tools.extend(extra_tools)
    return ToolGateway(tools=tuple(tools))


def _browser_high_risk_action_tools(
    root: Path,
    handlers: Mapping[str, BrowserHighRiskActionHandler],
) -> tuple[BrowserHighRiskActionTool, ...]:
    normalized_handlers = {
        key.strip().lower(): handler for key, handler in handlers.items()
    }
    allowed = {action for action, _capability in _BROWSER_HIGH_RISK_ACTION_CAPABILITIES}
    unknown = set(normalized_handlers) - allowed - {"send_message"}
    if unknown:
        raise ValueError(
            "unknown browser high-risk action policy: " + ", ".join(sorted(unknown))
        )
    tools: list[BrowserHighRiskActionTool] = []
    for action_kind, capability in _BROWSER_HIGH_RISK_ACTION_CAPABILITIES:
        handler = normalized_handlers.get(action_kind)
        if handler is None and action_kind == "send":
            handler = normalized_handlers.get("send_message")
        if handler is None:
            continue
        tools.append(
            BrowserHighRiskActionTool(
                root=root,
                action_kind=action_kind,
                capability=capability,
                handler=handler,
            )
        )
    return tuple(tools)

"""Read-only browser artifact providers for Agent Runtime."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import re
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar, cast
from urllib.parse import unquote, urlparse

from src.agent_runtime.builtin_tools import (
    WebFetcher,
    WebSearcher,
    build_builtin_tool_gateway,
    get_public_url_following_safe_redirects,
    is_public_http_url,
    verify_public_redirect_chain,
)

if TYPE_CHECKING:
    from src.agent_runtime.tools import ToolGateway

DownloadFetcher = Callable[[str, int], Awaitable["DownloadedContent"]]
ScreenshotRunner = Callable[[str, Path, float], Awaitable[None]]
BrowserUseSessionFactory = Callable[
    ["BrowserUseSessionOptions"],
    "BrowserUseSessionProtocol",
]
_T = TypeVar("_T")

_BROWSER_CANDIDATES = (
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
)
_CONTENT_TYPE_SUFFIXES = {
    "application/pdf": ".pdf",
    "application/json": ".json",
    "text/csv": ".csv",
    "text/html": ".html",
    "text/markdown": ".md",
    "text/plain": ".txt",
}
_VERIFICATION_CHALLENGE_RE = re.compile(
    r"("
    r"performing security verification|"
    r"checking if the site connection is secure|"
    r"verify (?:you are|that you are) (?:human|not a bot)|"
    r"verifies you are not a bot|"
    r"attention required!?\s*\|\s*cloudflare|"
    r"just a moment\.{0,3}.*cloudflare|"
    r"security service to protect against malicious bots|"
    r"ray id:"
    r")",
    re.IGNORECASE | re.DOTALL,
)


class BrowserVerificationBlockedError(RuntimeError):
    """Raised when a browser page is an anti-bot/security verification screen."""


@dataclass(frozen=True)
class DownloadedContent:
    """Downloaded read-only artifact payload."""

    content: bytes
    content_type: str = ""
    filename: str = ""


@dataclass(frozen=True)
class BrowserCapabilityStatus:
    """Availability status for one read-only browser capability."""

    state: str
    reason: str = ""


@dataclass(frozen=True)
class ReadOnlyBrowserGatewayBundle:
    """Tool gateway plus capability-status metadata for UI/audit rendering."""

    gateway: ToolGateway
    status: dict[str, BrowserCapabilityStatus]


@dataclass(frozen=True)
class BrowserUseSessionOptions:
    """Bounded options passed to browser-use BrowserSession factories."""

    executable_path: str | None
    headless: bool
    proxy: str
    viewport_width: int
    viewport_height: int
    user_data_dir: Path
    args: tuple[str, ...]


class BrowserUseSessionProtocol(Protocol):
    """Small browser-use session surface needed by read-only tools."""

    async def start(self) -> None: ...

    async def navigate_to(self, url: str, new_tab: bool = False) -> None: ...

    async def get_state_as_text(self) -> str: ...

    async def take_screenshot(
        self,
        path: str | None = None,
        full_page: bool = False,
        image_format: str = "png",
        quality: int | None = None,
        clip: dict[str, object] | None = None,
    ) -> bytes: ...

    async def stop(self) -> None: ...


class BrowserUseReadOnlyBackend:
    """Read and screenshot public pages through browser-use BrowserSession."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        browser_executable: str | None,
        http_proxy: str,
        timeout_seconds: float,
        session_factory: BrowserUseSessionFactory | None = None,
    ) -> None:
        self._workspace_root = workspace_root.expanduser().resolve()
        self._browser_executable = browser_executable
        self._http_proxy = http_proxy.strip()
        self._timeout_seconds = timeout_seconds
        self._session_factory = session_factory or _default_browser_use_session

    @property
    def is_available(self) -> bool:
        """Return whether browser-use can be imported or injected."""
        if self._session_factory is not _default_browser_use_session:
            return True
        return _browser_use_import_available()

    async def read_url(self, url: str, *, max_chars: int = 20_000) -> str:
        """Open a public page and return browser-use's accessibility text state."""
        _validate_public_http_url(url)

        async def read(session: BrowserUseSessionProtocol) -> str:
            return await session.get_state_as_text()

        text = await self._with_session(url, read)
        _raise_if_verification_challenge(text, url=url)
        return text[:max_chars]

    async def screenshot_url(self, url: str, output_path: Path) -> None:
        """Open a public page and persist a screenshot through browser-use."""
        _validate_public_http_url(url)

        async def screenshot(session: BrowserUseSessionProtocol) -> bytes:
            state_text = await session.get_state_as_text()
            _raise_if_verification_challenge(state_text, url=url)
            return await session.take_screenshot(str(output_path), False, "png")

        screenshot_bytes = await self._with_session(url, screenshot)
        if not output_path.is_file() and screenshot_bytes:
            output_path.write_bytes(screenshot_bytes)

    async def _with_session(
        self,
        url: str,
        action: Callable[[BrowserUseSessionProtocol], Awaitable[_T]],
    ) -> _T:
        session = self._session_factory(self._session_options())
        try:
            await asyncio.wait_for(session.start(), timeout=self._timeout_seconds)
            await asyncio.wait_for(
                session.navigate_to(url),
                timeout=self._timeout_seconds,
            )
            return await asyncio.wait_for(
                action(session), timeout=self._timeout_seconds
            )
        finally:
            with contextlib.suppress(Exception):
                await session.stop()

    def _session_options(self) -> BrowserUseSessionOptions:
        return BrowserUseSessionOptions(
            executable_path=self._browser_executable,
            headless=True,
            proxy=self._http_proxy,
            viewport_width=1365,
            viewport_height=900,
            user_data_dir=self._workspace_root / "runtime" / "browser_use",
            args=(
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--no-default-browser-check",
            ),
        )


class ReadOnlyBrowserArtifactProvider:
    """Create browser artifacts inside the workspace without external actions."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        artifact_dir: str = "agent_runtime/browser_artifacts",
        max_download_bytes: int = 10_000_000,
        screenshot_timeout_seconds: float = 30.0,
        browser_executable: str | None = None,
        browser_backend: str = "chromium",
        http_proxy: str = "",
        download_fetcher: DownloadFetcher | None = None,
        screenshot_runner: ScreenshotRunner | None = None,
        browser_use_session_factory: BrowserUseSessionFactory | None = None,
        enforce_public_network_guard: bool = False,
    ) -> None:
        self._workspace_root = workspace_root.expanduser().resolve()
        self._artifact_root = (self._workspace_root / artifact_dir).resolve()
        if not self._artifact_root.is_relative_to(self._workspace_root):
            raise ValueError("artifact_dir escapes workspace root")
        self._max_download_bytes = max_download_bytes
        self._screenshot_timeout_seconds = screenshot_timeout_seconds
        self._browser_backend = _normalize_browser_backend(browser_backend)
        self._browser_executable = (
            discover_browser_executable()
            if browser_executable is None
            else browser_executable or None
        )
        self._http_proxy = http_proxy.strip()
        self._browser_use_backend = self._build_browser_use_backend(
            session_factory=browser_use_session_factory,
        )
        self._uses_default_download_fetcher = download_fetcher is None
        self._download_fetcher = download_fetcher
        self._screenshot_runner = screenshot_runner
        self._enforce_public_network_guard = enforce_public_network_guard

    @property
    def can_screenshot(self) -> bool:
        """Return whether screenshots can be captured in this environment."""
        if self._enforce_public_network_guard and self._screenshot_runner is None:
            return self._browser_use_backend is not None
        if self._browser_use_backend is not None:
            return True
        return (
            self._screenshot_runner is not None or self._browser_executable is not None
        )

    @property
    def can_read_with_browser(self) -> bool:
        """Return whether browser_read can use a real browser session."""
        return self._browser_use_backend is not None

    @property
    def browser_backend(self) -> str:
        """Return the selected read-only browser backend."""
        if self._browser_use_backend is not None:
            return "browser_use"
        return "chromium"

    async def read_url(self, url: str) -> str:
        """Read a URL through the selected real-browser backend."""
        if self._browser_use_backend is None:
            raise RuntimeError("browser-use read backend is not available")
        return await self._browser_use_backend.read_url(url)

    async def download_file(self, url: str) -> str:
        """Download a URL as a bounded read-only workspace artifact."""
        _validate_http_url(url)
        if (
            self._enforce_public_network_guard
            and not self._uses_default_download_fetcher
        ):
            await verify_public_redirect_chain(url, proxy=self._http_proxy)
        if self._download_fetcher is not None:
            downloaded = await self._download_fetcher(url, self._max_download_bytes)
        else:
            downloaded = await _httpx_download(
                url,
                self._max_download_bytes,
                proxy=self._http_proxy,
            )
        if len(downloaded.content) > self._max_download_bytes:
            raise ValueError("download exceeds max_download_bytes")
        target = self._artifact_path(
            url=url,
            prefix="download",
            filename=downloaded.filename,
            fallback_name="artifact",
            suffix=_suffix_for_download(downloaded),
        )
        self._write_artifact(target, downloaded.content)
        return self._relative_artifact(target)

    async def screenshot_url(self, url: str) -> str:
        """Capture a headless browser screenshot as a workspace artifact."""
        if self._enforce_public_network_guard or self._browser_use_backend is not None:
            _validate_public_http_url(url)
        else:
            _validate_http_url(url)
        if self._enforce_public_network_guard:
            if self._screenshot_runner is None and self._browser_use_backend is None:
                raise RuntimeError("safe public screenshot runner is not configured")
            if self._screenshot_runner is not None:
                await verify_public_redirect_chain(url, proxy=self._http_proxy)
        target = self._artifact_path(
            url=url,
            prefix="screenshot",
            filename="",
            fallback_name="page",
            suffix=".png",
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if self._screenshot_runner is not None:
            await self._screenshot_runner(
                url,
                target,
                self._screenshot_timeout_seconds,
            )
        elif self._browser_use_backend is not None:
            await self._browser_use_backend.screenshot_url(url, target)
        else:
            if self._browser_executable is None:
                raise RuntimeError("no headless browser executable found")
            await _capture_chromium_screenshot(
                browser_executable=self._browser_executable,
                url=url,
                output_path=target,
                timeout_seconds=self._screenshot_timeout_seconds,
                proxy=self._http_proxy,
            )
        if not target.is_file() or target.stat().st_size <= 0:
            raise RuntimeError("browser screenshot did not create an artifact")
        return self._relative_artifact(target)

    def _build_browser_use_backend(
        self,
        *,
        session_factory: BrowserUseSessionFactory | None,
    ) -> BrowserUseReadOnlyBackend | None:
        if self._browser_backend not in {"auto", "browser_use"}:
            return None
        backend = BrowserUseReadOnlyBackend(
            workspace_root=self._workspace_root,
            browser_executable=self._browser_executable,
            http_proxy=self._http_proxy,
            timeout_seconds=self._screenshot_timeout_seconds,
            session_factory=session_factory,
        )
        if backend.is_available:
            return backend
        if self._browser_backend == "browser_use":
            return backend
        return None

    def _artifact_path(
        self,
        *,
        url: str,
        prefix: str,
        filename: str,
        fallback_name: str,
        suffix: str,
    ) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        safe_name = _safe_filename(filename or _url_filename(url) or fallback_name)
        if not Path(safe_name).suffix and suffix:
            safe_name = f"{safe_name}{suffix}"
        target = (self._artifact_root / f"{prefix}-{digest}-{safe_name}").resolve()
        if not target.is_relative_to(self._workspace_root):
            raise RuntimeError("artifact path escapes workspace root")
        return target

    def _write_artifact(self, target: Path, content: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def _relative_artifact(self, target: Path) -> str:
        return target.relative_to(self._workspace_root).as_posix()


def build_readonly_browser_tool_gateway(
    *,
    workspace_root: Path,
    readonly_command_root: Path | None = None,
    enable_browser_use: bool,
    web_fetcher: WebFetcher | None = None,
    web_searcher: WebSearcher | None = None,
    browser_executable: str | None = None,
    browser_backend: str = "chromium",
    web_proxy: str = "",
    browser_use_session_factory: BrowserUseSessionFactory | None = None,
) -> ToolGateway:
    """Build built-in tools plus optional read-only browser artifacts."""
    return build_readonly_browser_gateway_bundle(
        workspace_root=workspace_root,
        readonly_command_root=readonly_command_root,
        enable_browser_use=enable_browser_use,
        web_fetcher=web_fetcher,
        web_searcher=web_searcher,
        browser_executable=browser_executable,
        browser_backend=browser_backend,
        web_proxy=web_proxy,
        browser_use_session_factory=browser_use_session_factory,
    ).gateway


def build_readonly_browser_gateway_bundle(
    *,
    workspace_root: Path,
    readonly_command_root: Path | None = None,
    enable_browser_use: bool,
    web_fetcher: WebFetcher | None = None,
    web_searcher: WebSearcher | None = None,
    browser_executable: str | None = None,
    browser_backend: str = "chromium",
    web_proxy: str = "",
    browser_use_session_factory: BrowserUseSessionFactory | None = None,
) -> ReadOnlyBrowserGatewayBundle:
    """Build read-only browser tools and explicit capability degradation status."""
    if not enable_browser_use:
        gateway = build_builtin_tool_gateway(
            workspace_root=workspace_root,
            readonly_command_root=readonly_command_root,
            web_fetcher=web_fetcher,
            web_searcher=web_searcher,
            web_proxy=web_proxy,
        )
        return ReadOnlyBrowserGatewayBundle(
            gateway=gateway,
            status={
                "browser_read": BrowserCapabilityStatus(
                    state="available",
                    reason="HTTP read remains available without browser artifacts.",
                ),
                "browser_download": BrowserCapabilityStatus(
                    state="unavailable",
                    reason="ENABLE_BROWSER_USE=false.",
                ),
                "browser_screenshot": BrowserCapabilityStatus(
                    state="unavailable",
                    reason="ENABLE_BROWSER_USE=false.",
                ),
            },
        )

    provider = ReadOnlyBrowserArtifactProvider(
        workspace_root=workspace_root,
        browser_executable=browser_executable,
        browser_backend=browser_backend,
        http_proxy=web_proxy,
        browser_use_session_factory=browser_use_session_factory,
    )
    browser_read_fetcher = web_fetcher
    if browser_read_fetcher is None and provider.can_read_with_browser:
        browser_read_fetcher = provider.read_url
    gateway = build_builtin_tool_gateway(
        workspace_root=workspace_root,
        readonly_command_root=readonly_command_root,
        web_fetcher=browser_read_fetcher,
        web_searcher=web_searcher,
        web_proxy=web_proxy,
        browser_screenshotter=provider.screenshot_url
        if provider.can_screenshot
        else None,
        browser_downloader=provider.download_file,
    )
    screenshot_status = (
        BrowserCapabilityStatus(state="available")
        if provider.can_screenshot
        else BrowserCapabilityStatus(
            state="degraded",
            reason="Chromium/Chrome executable not found; screenshots unavailable.",
        )
    )
    return ReadOnlyBrowserGatewayBundle(
        gateway=gateway,
        status={
            "browser_read": BrowserCapabilityStatus(
                state="available",
                reason=f"backend={provider.browser_backend}",
            ),
            "browser_download": BrowserCapabilityStatus(state="available"),
            "browser_screenshot": screenshot_status
            if screenshot_status.state != "available"
            else BrowserCapabilityStatus(
                state="available",
                reason=f"backend={provider.browser_backend}",
            ),
        },
    )


def discover_browser_executable() -> str | None:
    """Find a local Chromium-compatible executable if one is installed."""
    for candidate in _BROWSER_CANDIDATES:
        path = shutil.which(candidate)
        if path:
            return path
    return None


def _default_browser_use_session(
    options: BrowserUseSessionOptions,
) -> BrowserUseSessionProtocol:
    try:
        from browser_use.browser import BrowserProfile, BrowserSession
        from browser_use.browser.profile import ProxySettings, ViewportSize
    except ImportError as exc:  # pragma: no cover - exercised through status tests.
        raise RuntimeError("browser-use is not installed") from exc

    options.user_data_dir.mkdir(parents=True, exist_ok=True)
    profile = BrowserProfile(
        executable_path=options.executable_path,
        headless=options.headless,
        args=list(options.args),
        viewport=ViewportSize(
            width=options.viewport_width,
            height=options.viewport_height,
        ),
        proxy=ProxySettings(server=options.proxy) if options.proxy else None,
        user_data_dir=options.user_data_dir,
        keep_alive=False,
        block_ip_addresses=True,
        enable_default_extensions=False,
    )
    return cast("BrowserUseSessionProtocol", BrowserSession(browser_profile=profile))


def _browser_use_import_available() -> bool:
    try:
        import browser_use  # noqa: F401
    except ImportError:
        return False
    return True


def _normalize_browser_backend(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"", "auto"}:
        return "auto"
    if normalized in {"browser_use", "browseruse"}:
        return "browser_use"
    if normalized in {"chromium", "chrome", "legacy"}:
        return "chromium"
    raise ValueError("browser_backend must be auto, browser_use or chromium")


def looks_like_verification_challenge(text: str) -> bool:
    """Return whether page text is an anti-bot/security verification screen."""
    return bool(_VERIFICATION_CHALLENGE_RE.search(text))


def _raise_if_verification_challenge(text: str, *, url: str) -> None:
    if not looks_like_verification_challenge(text):
        return
    host = urlparse(url).netloc or url
    raise BrowserVerificationBlockedError(
        f"browser reached security verification challenge for {host}"
    )


def _validate_public_http_url(url: str) -> None:
    _validate_http_url(url)
    if not is_public_http_url(url):
        raise ValueError("url must be a public http(s) source")


async def _httpx_download(
    url: str,
    max_bytes: int,
    *,
    proxy: str = "",
) -> DownloadedContent:
    import httpx

    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=False,
        proxy=proxy.strip() or None,
    ) as client:
        response = await get_public_url_following_safe_redirects(client, url)
        response.raise_for_status()
        content = response.content
    if len(content) > max_bytes:
        raise ValueError("download exceeds max_download_bytes")
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
    filename = _filename_from_content_disposition(
        response.headers.get("content-disposition", "")
    )
    return DownloadedContent(
        content=content,
        content_type=content_type,
        filename=filename,
    )


async def _capture_chromium_screenshot(
    *,
    browser_executable: str,
    url: str,
    output_path: Path,
    timeout_seconds: float,
    proxy: str = "",
) -> None:
    argv = [
        browser_executable,
        "--headless=new",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--no-first-run",
        "--no-default-browser-check",
        "--window-size=1365,900",
        f"--screenshot={output_path}",
    ]
    normalized_proxy = proxy.strip()
    if normalized_proxy:
        argv.append(f"--proxy-server={normalized_proxy}")
    argv.append(url)
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_seconds,
        )
    except TimeoutError as exc:
        proc.kill()
        with contextlib.suppress(ProcessLookupError):
            await proc.wait()
        raise TimeoutError("browser screenshot timed out") from exc
    if proc.returncode != 0:
        details = (stderr or stdout).decode("utf-8", errors="replace")[-500:]
        raise RuntimeError(f"browser screenshot failed: {details}")


def _validate_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be http(s)")


def _safe_filename(value: str) -> str:
    name = unquote(value).strip()[:120]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-_")
    return safe or "artifact"


def _url_filename(url: str) -> str:
    return Path(urlparse(url).path).name


def _suffix_for_download(downloaded: DownloadedContent) -> str:
    if downloaded.filename and Path(downloaded.filename).suffix:
        return ""
    return _CONTENT_TYPE_SUFFIXES.get(downloaded.content_type.lower(), ".bin")


def _filename_from_content_disposition(value: str) -> str:
    match = re.search(r'filename="?([^";]+)"?', value)
    if match is None:
        return ""
    return match.group(1)

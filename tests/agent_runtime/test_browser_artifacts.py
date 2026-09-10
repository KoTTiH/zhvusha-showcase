"""Read-only browser artifact provider tests."""

from __future__ import annotations

from pathlib import Path

import pytest


async def test_readonly_browser_provider_downloads_file_inside_workspace(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.browser_artifacts import (
        DownloadedContent,
        ReadOnlyBrowserArtifactProvider,
    )

    async def fetch(url: str, max_bytes: int) -> DownloadedContent:
        assert url == "https://example.com/report.pdf?token=1"
        assert max_bytes > 0
        return DownloadedContent(
            content=b"%PDF-1.4",
            content_type="application/pdf",
            filename="report.pdf",
        )

    provider = ReadOnlyBrowserArtifactProvider(
        workspace_root=tmp_path,
        download_fetcher=fetch,
    )

    artifact = await provider.download_file("https://example.com/report.pdf?token=1")

    assert artifact.startswith("agent_runtime/browser_artifacts/download-")
    assert artifact.endswith("-report.pdf")
    path = tmp_path / artifact
    assert path.is_file()
    assert path.read_bytes() == b"%PDF-1.4"


async def test_readonly_browser_provider_screenshot_uses_headless_runner(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.browser_artifacts import ReadOnlyBrowserArtifactProvider

    captured: dict[str, object] = {}

    async def runner(url: str, output_path: Path, timeout_seconds: float) -> None:
        captured["url"] = url
        captured["timeout"] = timeout_seconds
        output_path.write_bytes(b"png")

    provider = ReadOnlyBrowserArtifactProvider(
        workspace_root=tmp_path,
        screenshot_runner=runner,
    )

    artifact = await provider.screenshot_url("https://example.com/post")

    assert captured["url"] == "https://example.com/post"
    assert artifact.startswith("agent_runtime/browser_artifacts/screenshot-")
    assert artifact.endswith("-post.png")
    assert (tmp_path / artifact).read_bytes() == b"png"


async def test_guarded_browser_provider_does_not_use_raw_chromium_screenshot(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.browser_artifacts import ReadOnlyBrowserArtifactProvider

    provider = ReadOnlyBrowserArtifactProvider(
        workspace_root=tmp_path,
        browser_executable="/usr/bin/chromium",
        enforce_public_network_guard=True,
    )

    assert provider.can_screenshot is False
    with pytest.raises(RuntimeError, match="safe public screenshot runner"):
        await provider.screenshot_url("https://example.com/post")


async def test_guarded_browser_provider_blocks_private_source_before_runner(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.browser_artifacts import ReadOnlyBrowserArtifactProvider

    called = False

    async def runner(url: str, output_path: Path, timeout_seconds: float) -> None:
        nonlocal called
        called = True
        output_path.write_bytes(b"png")

    provider = ReadOnlyBrowserArtifactProvider(
        workspace_root=tmp_path,
        screenshot_runner=runner,
        enforce_public_network_guard=True,
    )

    with pytest.raises(ValueError, match="public"):
        await provider.screenshot_url("http://127.0.0.1/private")
    assert called is False


async def test_browser_use_backend_reads_page_state_and_saves_screenshot(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.browser_artifacts import (
        BrowserUseSessionOptions,
        build_readonly_browser_gateway_bundle,
    )
    from src.agent_runtime.profiles import WEB_RESEARCH_READONLY

    events: list[str] = []
    options_seen: list[BrowserUseSessionOptions] = []

    class FakeBrowserUseSession:
        async def start(self) -> None:
            events.append("start")

        async def navigate_to(self, url: str, new_tab: bool = False) -> None:
            events.append(f"navigate:{url}:{new_tab}")

        async def get_state_as_text(self) -> str:
            events.append("state")
            return "Browser-use page state for Example"

        async def take_screenshot(
            self,
            path: str | None = None,
            full_page: bool = False,
            image_format: str = "png",
            quality: int | None = None,
            clip: dict[str, object] | None = None,
        ) -> bytes:
            del full_page, image_format, quality, clip
            events.append("screenshot")
            payload = b"browser-use-png"
            if path is not None:
                Path(path).write_bytes(payload)
            return payload

        async def stop(self) -> None:
            events.append("stop")

    def session_factory(options: BrowserUseSessionOptions) -> FakeBrowserUseSession:
        options_seen.append(options)
        return FakeBrowserUseSession()

    bundle = build_readonly_browser_gateway_bundle(
        workspace_root=tmp_path,
        enable_browser_use=True,
        browser_backend="browser_use",
        browser_executable="/usr/bin/chromium",
        web_proxy="http://127.0.0.1:7897",
        browser_use_session_factory=session_factory,
    )

    page_state = await bundle.gateway.execute(
        WEB_RESEARCH_READONLY,
        "browser_read_url",
        {"url": "https://example.com/post"},
    )
    artifact = await bundle.gateway.execute(
        WEB_RESEARCH_READONLY,
        "browser_screenshot_url",
        {"url": "https://example.com/post"},
    )

    assert page_state == "Browser-use page state for Example"
    assert artifact.startswith("agent_runtime/browser_artifacts/screenshot-")
    assert (tmp_path / artifact).read_bytes() == b"browser-use-png"
    assert bundle.status["browser_read"].reason == "backend=browser_use"
    assert bundle.status["browser_screenshot"].reason == "backend=browser_use"
    assert options_seen[0].proxy == "http://127.0.0.1:7897"
    assert options_seen[0].executable_path == "/usr/bin/chromium"
    assert events.count("start") == 2
    assert events.count("stop") == 2


async def test_browser_use_backend_rejects_security_verification_pages(
    tmp_path: Path,
) -> None:
    import pytest
    from src.agent_runtime.browser_artifacts import (
        BrowserUseSessionOptions,
        BrowserVerificationBlockedError,
        build_readonly_browser_gateway_bundle,
    )
    from src.agent_runtime.profiles import WEB_RESEARCH_READONLY

    events: list[str] = []

    class FakeBrowserUseSession:
        async def start(self) -> None:
            events.append("start")

        async def navigate_to(self, url: str, new_tab: bool = False) -> None:
            events.append(f"navigate:{url}:{new_tab}")

        async def get_state_as_text(self) -> str:
            events.append("state")
            return (
                "www.dotabuff.com\n"
                "Performing security verification\n"
                "This website verifies you are not a bot.\n"
                "Ray ID: test"
            )

        async def take_screenshot(
            self,
            path: str | None = None,
            full_page: bool = False,
            image_format: str = "png",
            quality: int | None = None,
            clip: dict[str, object] | None = None,
        ) -> bytes:
            del path, full_page, image_format, quality, clip
            events.append("screenshot")
            return b"must-not-write"

        async def stop(self) -> None:
            events.append("stop")

    def session_factory(options: BrowserUseSessionOptions) -> FakeBrowserUseSession:
        del options
        return FakeBrowserUseSession()

    bundle = build_readonly_browser_gateway_bundle(
        workspace_root=tmp_path,
        enable_browser_use=True,
        browser_backend="browser_use",
        browser_executable="/usr/bin/chromium",
        browser_use_session_factory=session_factory,
    )

    with pytest.raises(BrowserVerificationBlockedError, match="security verification"):
        await bundle.gateway.execute(
            WEB_RESEARCH_READONLY,
            "browser_read_url",
            {"url": "https://www.dotabuff.com/search?q=kereexa"},
        )
    with pytest.raises(BrowserVerificationBlockedError, match="security verification"):
        await bundle.gateway.execute(
            WEB_RESEARCH_READONLY,
            "browser_screenshot_url",
            {"url": "https://www.dotabuff.com/search?q=kereexa"},
        )

    assert "screenshot" not in events
    assert not list((tmp_path / "agent_runtime" / "browser_artifacts").glob("*.png"))


def test_readonly_browser_gateway_is_feature_flagged(tmp_path: Path) -> None:
    from src.agent_runtime.browser_artifacts import build_readonly_browser_tool_gateway
    from src.agent_runtime.profiles import WEB_RESEARCH_READONLY

    disabled = build_readonly_browser_tool_gateway(
        workspace_root=tmp_path,
        enable_browser_use=False,
    )
    enabled = build_readonly_browser_tool_gateway(
        workspace_root=tmp_path,
        enable_browser_use=True,
        browser_executable="/usr/bin/chromium",
    )

    assert "browser_download_file" not in disabled.build_toolset(WEB_RESEARCH_READONLY)
    assert "browser_screenshot_url" not in disabled.build_toolset(WEB_RESEARCH_READONLY)
    assert "browser_download_file" in enabled.build_toolset(WEB_RESEARCH_READONLY)
    assert "browser_screenshot_url" in enabled.build_toolset(WEB_RESEARCH_READONLY)


def test_readonly_browser_capabilities_report_degraded_without_chromium(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.browser_artifacts import (
        build_readonly_browser_gateway_bundle,
    )

    bundle = build_readonly_browser_gateway_bundle(
        workspace_root=tmp_path,
        enable_browser_use=True,
        browser_executable="",
    )

    assert bundle.status["browser_read"].state == "available"
    assert bundle.status["browser_download"].state == "available"
    assert bundle.status["browser_screenshot"].state == "degraded"
    assert "Chromium" in bundle.status["browser_screenshot"].reason

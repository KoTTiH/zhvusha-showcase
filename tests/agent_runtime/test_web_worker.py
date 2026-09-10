"""Read-only web research worker tests."""

from __future__ import annotations


async def test_web_worker_reads_urls_through_tool_gateway(tmp_path) -> None:
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import AgentJob, ContextPack
    from src.agent_runtime.profiles import WEB_RESEARCH_READONLY
    from src.agent_runtime.workers.web import WebResearchWorkerBackend

    async def fetch(url: str) -> str:
        return f"<html><title>Example</title><p>source for {url}</p></html>"

    gateway = build_builtin_tool_gateway(
        workspace_root=tmp_path,
        web_fetcher=fetch,
    )
    worker = WebResearchWorkerBackend(tool_gateway=gateway)
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:web",
        fingerprint="web",
        kind="web_research",
        profile=WEB_RESEARCH_READONLY,
        context_pack=ContextPack(
            user_request="Проверь источник https://example.com/post",
        ),
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    assert capsule.sources == ("https://example.com/post",)
    assert "source for https://example.com/post" in capsule.processed_context
    assert capsule.findings[0].evidence == ("https://example.com/post",)
    assert "1 источник" in capsule.summary


async def test_web_worker_reports_no_urls_without_external_action(tmp_path) -> None:
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import AgentJob, ContextPack
    from src.agent_runtime.profiles import WEB_RESEARCH_READONLY
    from src.agent_runtime.workers.web import WebResearchWorkerBackend

    gateway = build_builtin_tool_gateway(workspace_root=tmp_path)
    worker = WebResearchWorkerBackend(tool_gateway=gateway)
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:web-empty",
        fingerprint="web-empty",
        kind="web_research",
        profile=WEB_RESEARCH_READONLY,
        context_pack=ContextPack(user_request="проверь источник без ссылки"),
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    assert capsule.sources == ()
    assert "не нашла URL" in capsule.summary
    assert capsule.findings[0].status.value == "unconfirmed"


async def test_web_worker_searches_sources_when_no_url_is_provided(tmp_path) -> None:
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import AgentJob, ContextPack
    from src.agent_runtime.profiles import WEB_RESEARCH_READONLY
    from src.agent_runtime.workers.web import WebResearchWorkerBackend

    async def search(query: str, max_results: int) -> tuple[str, ...]:
        assert "Anthropic Dreams" in query
        assert max_results == 5
        return ("https://example.com/dreams",)

    async def fetch(url: str) -> str:
        return f"source text for {url}"

    gateway = build_builtin_tool_gateway(
        workspace_root=tmp_path,
        web_fetcher=fetch,
        web_searcher=search,
    )
    worker = WebResearchWorkerBackend(tool_gateway=gateway)
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:web-search",
        fingerprint="web-search",
        kind="web_research",
        profile=WEB_RESEARCH_READONLY,
        context_pack=ContextPack(user_request="проверь Anthropic Dreams"),
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    assert capsule.sources == ("https://example.com/dreams",)
    assert "source text for https://example.com/dreams" in capsule.processed_context


async def test_web_worker_continues_after_unreadable_search_result(tmp_path) -> None:
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import AgentJob, ContextPack
    from src.agent_runtime.profiles import WEB_RESEARCH_READONLY
    from src.agent_runtime.workers.web import WebResearchWorkerBackend

    async def search(query: str, max_results: int) -> tuple[str, ...]:
        del query, max_results
        return ("https://example.com/down", "https://example.com/up")

    async def fetch(url: str) -> str:
        if url.endswith("/down"):
            raise RuntimeError("connect failed")
        return f"source text for {url}"

    gateway = build_builtin_tool_gateway(
        workspace_root=tmp_path,
        web_fetcher=fetch,
        web_searcher=search,
    )
    worker = WebResearchWorkerBackend(tool_gateway=gateway)
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:web-search-partial",
        fingerprint="web-search-partial",
        kind="web_research",
        profile=WEB_RESEARCH_READONLY,
        context_pack=ContextPack(user_request="проверь Anthropic Dreams"),
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    assert capsule.sources == ("https://example.com/up",)
    assert "source text for https://example.com/up" in capsule.processed_context
    assert any("connect failed" in finding.claim for finding in capsule.findings)


async def test_web_worker_attaches_optional_browser_artifacts(tmp_path) -> None:
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import AgentJob, ContextPack
    from src.agent_runtime.profiles import WEB_RESEARCH_READONLY
    from src.agent_runtime.workers.web import WebResearchWorkerBackend

    async def fetch(url: str) -> str:
        return f"source text for {url}"

    async def screenshot(url: str) -> str:
        return (
            f"agent_runtime/browser_artifacts/screenshot-{url.rsplit('/', 1)[-1]}.png"
        )

    async def download(url: str) -> str:
        return f"agent_runtime/browser_artifacts/download-{url.rsplit('/', 1)[-1]}"

    gateway = build_builtin_tool_gateway(
        workspace_root=tmp_path,
        web_fetcher=fetch,
        browser_screenshotter=screenshot,
        browser_downloader=download,
    )
    worker = WebResearchWorkerBackend(tool_gateway=gateway)
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:web-artifacts",
        fingerprint="web-artifacts",
        kind="web_research",
        profile=WEB_RESEARCH_READONLY,
        context_pack=ContextPack(user_request="Проверь https://example.com/report.pdf"),
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    assert capsule.artifacts == (
        "agent_runtime/browser_artifacts/screenshot-report.pdf.png",
        "agent_runtime/browser_artifacts/download-report.pdf",
    )
    assert any("Скриншот источника" in finding.claim for finding in capsule.findings)
    assert any("Файл источника скачан" in finding.claim for finding in capsule.findings)


async def test_web_worker_degrades_optional_browser_artifact_failures(tmp_path) -> None:
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import AgentJob, ContextPack
    from src.agent_runtime.profiles import WEB_RESEARCH_READONLY
    from src.agent_runtime.workers.web import WebResearchWorkerBackend

    async def fetch(url: str) -> str:
        return f"source text for {url}"

    async def screenshot(url: str) -> str:
        del url
        raise RuntimeError("browser screenshot did not create an artifact")

    gateway = build_builtin_tool_gateway(
        workspace_root=tmp_path,
        web_fetcher=fetch,
        browser_screenshotter=screenshot,
    )
    worker = WebResearchWorkerBackend(tool_gateway=gateway)
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:web-artifacts-degraded",
        fingerprint="web-artifacts-degraded",
        kind="web_research",
        profile=WEB_RESEARCH_READONLY,
        context_pack=ContextPack(user_request="Проверь https://example.com/report"),
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    assert capsule.sources == ("https://example.com/report",)
    assert capsule.artifacts == ()
    assert "source text for https://example.com/report" in capsule.processed_context


async def test_web_worker_keeps_screenshot_when_browser_read_fails(tmp_path) -> None:
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import AgentJob, ContextPack
    from src.agent_runtime.profiles import WEB_RESEARCH_READONLY
    from src.agent_runtime.workers.web import WebResearchWorkerBackend

    async def fetch(url: str) -> str:
        del url
        raise RuntimeError("connect failed")

    async def screenshot(url: str) -> str:
        return (
            f"agent_runtime/browser_artifacts/screenshot-{url.rsplit('/', 1)[-1]}.png"
        )

    gateway = build_builtin_tool_gateway(
        workspace_root=tmp_path,
        web_fetcher=fetch,
        browser_screenshotter=screenshot,
    )
    worker = WebResearchWorkerBackend(tool_gateway=gateway)
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:web-read-failed-screenshot-ok",
        fingerprint="web-read-failed-screenshot-ok",
        kind="web_research",
        profile=WEB_RESEARCH_READONLY,
        context_pack=ContextPack(user_request="Проверь https://example.com/article"),
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    assert capsule.sources == ("https://example.com/article",)
    assert capsule.artifacts == (
        "agent_runtime/browser_artifacts/screenshot-article.png",
    )
    assert "Скриншот" in capsule.summary
    assert any("не прочитан" in finding.claim for finding in capsule.findings)


async def test_web_worker_rejects_security_verification_text(tmp_path) -> None:
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import AgentJob, ContextPack
    from src.agent_runtime.profiles import WEB_RESEARCH_READONLY
    from src.agent_runtime.workers.web import WebResearchWorkerBackend

    async def fetch(url: str) -> str:
        del url
        return (
            "www.dotabuff.com\n"
            "Performing security verification\n"
            "This website verifies you are not a bot.\n"
            "Ray ID: test"
        )

    async def screenshot(url: str) -> str:
        raise AssertionError(f"must not screenshot blocked source: {url}")

    gateway = build_builtin_tool_gateway(
        workspace_root=tmp_path,
        web_fetcher=fetch,
        browser_screenshotter=screenshot,
    )
    worker = WebResearchWorkerBackend(tool_gateway=gateway)
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:web-verification-text",
        fingerprint="web-verification-text",
        kind="web_research",
        profile=WEB_RESEARCH_READONLY,
        context_pack=ContextPack(
            user_request="Проверь https://www.dotabuff.com/search?q=kereexa"
        ),
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    assert capsule.sources == ()
    assert capsule.artifacts == ()
    assert "security verification" in capsule.summary
    assert "Source blocked by security verification" in capsule.processed_context
    assert any(finding.status.value == "rejected" for finding in capsule.findings)


async def test_web_worker_reports_security_verification_screenshot_block(
    tmp_path,
) -> None:
    from src.agent_runtime.browser_artifacts import BrowserVerificationBlockedError
    from src.agent_runtime.builtin_tools import build_builtin_tool_gateway
    from src.agent_runtime.models import AgentJob, ContextPack
    from src.agent_runtime.profiles import WEB_RESEARCH_READONLY
    from src.agent_runtime.workers.web import WebResearchWorkerBackend

    async def fetch(url: str) -> str:
        del url
        raise RuntimeError("connect failed")

    async def screenshot(url: str) -> str:
        raise BrowserVerificationBlockedError(
            f"browser reached security verification challenge for {url}"
        )

    gateway = build_builtin_tool_gateway(
        workspace_root=tmp_path,
        web_fetcher=fetch,
        browser_screenshotter=screenshot,
    )
    worker = WebResearchWorkerBackend(tool_gateway=gateway)
    job = AgentJob.new(
        owner_user_id=1,
        chat_id=2,
        source_message_id="tg:web-verification-screenshot",
        fingerprint="web-verification-screenshot",
        kind="web_research",
        profile=WEB_RESEARCH_READONLY,
        context_pack=ContextPack(
            user_request="Проверь https://www.dotabuff.com/search?q=kereexa"
        ),
    )

    capsule = await worker.run(job=job, context_pack=job.context_pack)

    assert capsule.sources == ()
    assert capsule.artifacts == ()
    assert "security verification" in capsule.summary
    assert any("security verification" in finding.claim for finding in capsule.findings)

"""Read-only web research worker backed by Tool Gateway."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from src.agent_runtime.browser_artifacts import looks_like_verification_challenge
from src.agent_runtime.models import (
    ContextCapsule,
    Finding,
    FindingStatus,
)
from src.agent_runtime.tools import ToolDeniedError, ToolNotFoundError

if TYPE_CHECKING:
    from src.agent_runtime.models import AgentJob, ContextPack
    from src.agent_runtime.tools import ToolGateway

_URL_RE = re.compile(r"https?://[^\s<>()\"']+", re.IGNORECASE)
_DOWNLOAD_SUFFIXES = {
    ".csv",
    ".doc",
    ".docx",
    ".jpeg",
    ".jpg",
    ".json",
    ".md",
    ".pdf",
    ".png",
    ".txt",
    ".webp",
    ".xls",
    ".xlsx",
    ".zip",
}


class WebResearchWorkerBackend:
    """Read URLs through capability-enforced browser_read tools only."""

    name = "web_research"

    def __init__(self, *, tool_gateway: ToolGateway, max_sources: int = 5) -> None:
        self._tool_gateway = tool_gateway
        self._max_sources = max_sources

    async def run(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
    ) -> ContextCapsule:
        """Read URLs from the context and return a source-backed capsule."""
        urls = _extract_urls(
            (
                context_pack.user_request,
                *context_pack.chat_context,
                *context_pack.attachments,
                *job.followups,
            )
        )[: self._max_sources]
        if not urls:
            urls = await self._search_sources(job=job, context_pack=context_pack)
        if not urls:
            return _no_sources_capsule("не нашла URL для read-only web research.")

        return await self._read_urls(job=job, urls=urls[: self._max_sources])

    async def cancel(self, job_id: str) -> bool:
        """No long-lived process is held by this worker."""
        del job_id
        return False

    async def _search_sources(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
    ) -> tuple[str, ...]:
        try:
            results = await self._tool_gateway.execute(
                job.profile,
                "web_search_sources",
                {
                    "query": context_pack.user_request,
                    "max_results": self._max_sources,
                },
            )
        except (ToolDeniedError, ToolNotFoundError):
            return ()
        return tuple(str(result).strip() for result in results if str(result).strip())

    async def _read_urls(
        self,
        *,
        job: AgentJob,
        urls: tuple[str, ...],
    ) -> ContextCapsule:
        sections: list[str] = []
        findings: list[Finding] = []
        artifacts: list[str] = []
        read_urls: list[str] = []
        artifact_urls: list[str] = []
        for url in urls:
            try:
                text = await self._tool_gateway.execute(
                    job.profile,
                    "browser_read_url",
                    {"url": url},
                )
            except Exception as exc:
                findings.append(
                    Finding(
                        claim=(
                            "Источник найден, но не прочитан через "
                            f"browser_read_url: {url} ({_error_summary(exc)})"
                        ),
                        status=FindingStatus.UNCONFIRMED,
                        confidence=0.4,
                        evidence=(url,),
                    )
                )
                (
                    source_artifacts,
                    artifact_findings,
                ) = await self._optional_browser_artifacts(
                    job=job,
                    url=url,
                )
                artifacts.extend(source_artifacts)
                findings.extend(artifact_findings)
                if source_artifacts:
                    artifact_urls.append(url)
                    sections.append(
                        _artifact_only_section(
                            url=url,
                            error=_error_summary(exc),
                            artifacts=source_artifacts,
                        )
                    )
                continue
            text = str(text).strip()
            if looks_like_verification_challenge(text):
                findings.append(_verification_challenge_finding(url))
                sections.append(_verification_challenge_section(url))
                continue
            sections.append(f"# Source: {url}\n{text}")
            read_urls.append(url)
            findings.append(
                Finding(
                    claim=f"Источник прочитан через browser_read_url: {url}",
                    status=FindingStatus.CONFIRMED,
                    confidence=0.9,
                    evidence=(url,),
                )
            )
            (
                source_artifacts,
                artifact_findings,
            ) = await self._optional_browser_artifacts(
                job=job,
                url=url,
            )
            artifacts.extend(source_artifacts)
            findings.extend(artifact_findings)

        processed_context = "\n\n".join(sections)
        source_urls = (*read_urls, *artifact_urls)
        if not source_urls:
            if _has_verification_challenge(findings):
                summary = (
                    "Сайт показал security verification/challenge; целевую "
                    "страницу и скриншот read-only получить не удалось."
                )
                processed_context = "\n\n".join(sections)
                return ContextCapsule(
                    summary=summary,
                    processed_context=processed_context,
                    findings=tuple(findings),
                    sources=(),
                    artifacts=tuple(artifacts),
                    next_actions=(
                        "Попросить пользователя открыть сайт вручную или дать "
                        "другой источник без anti-bot verification.",
                    ),
                    markdown_report=f"{summary}\n\n{processed_context}",
                )
            return ContextCapsule(
                summary="Не смогла прочитать найденные источники read-only.",
                findings=tuple(findings),
                sources=(),
                next_actions=(
                    "Повторить позже или передать конкретные доступные URL.",
                ),
                markdown_report="Не смогла прочитать найденные источники read-only.",
            )
        if read_urls:
            summary = f"Прочитала {len(read_urls)} источник(а) read-only."
        else:
            summary = (
                "Скриншот источника сохранён, но текст страницы read-only "
                "прочитать не удалось."
            )
        artifact_report = ""
        if artifacts:
            artifact_report = "\n\nArtifacts:\n" + "\n".join(
                f"- {artifact}" for artifact in artifacts
            )
        return ContextCapsule(
            summary=summary,
            processed_context=processed_context,
            findings=tuple(findings),
            sources=tuple(source_urls),
            artifacts=tuple(artifacts),
            next_actions=("Передать прочитанный контекст Жвуше для синтеза ответа.",),
            markdown_report=f"{summary}\n\n{processed_context}{artifact_report}",
        )

    async def _optional_browser_artifacts(
        self,
        *,
        job: AgentJob,
        url: str,
    ) -> tuple[tuple[str, ...], tuple[Finding, ...]]:
        artifacts: list[str] = []
        findings: list[Finding] = []
        screenshot = await self._try_optional_tool(
            job=job,
            tool_name="browser_screenshot_url",
            url=url,
        )
        if isinstance(screenshot, Exception):
            if _is_verification_challenge_error(screenshot):
                findings.append(_verification_challenge_finding(url))
            return tuple(artifacts), tuple(findings)
        if screenshot is not None:
            artifacts.append(screenshot)
            findings.append(
                Finding(
                    claim=f"Скриншот источника сохранён: {url}",
                    status=FindingStatus.CONFIRMED,
                    confidence=0.85,
                    evidence=(url, screenshot),
                )
            )

        if _looks_like_downloadable_url(url):
            downloaded = await self._try_optional_tool(
                job=job,
                tool_name="browser_download_file",
                url=url,
            )
            if isinstance(downloaded, Exception):
                return tuple(artifacts), tuple(findings)
            if downloaded is not None:
                artifacts.append(downloaded)
                findings.append(
                    Finding(
                        claim=f"Файл источника скачан read-only: {url}",
                        status=FindingStatus.CONFIRMED,
                        confidence=0.85,
                        evidence=(url, downloaded),
                    )
                )
        return tuple(artifacts), tuple(findings)

    async def _try_optional_tool(
        self,
        *,
        job: AgentJob,
        tool_name: str,
        url: str,
    ) -> str | Exception | None:
        try:
            result = await self._tool_gateway.execute(
                job.profile,
                tool_name,
                {"url": url},
            )
        except (ToolDeniedError, ToolNotFoundError):
            return None
        except Exception as exc:
            return exc
        artifact = str(result).strip()
        return artifact or None


def _extract_urls(parts: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        for match in _URL_RE.findall(part):
            url = match.rstrip(".,;:!?)]}")
            if url in seen:
                continue
            seen.add(url)
            result.append(url)
    return tuple(result)


def _no_sources_capsule(summary: str) -> ContextCapsule:
    return ContextCapsule(
        summary=summary,
        findings=(
            Finding(
                claim="В запросе нет URL, и web_search_sources не дал источники.",
                status=FindingStatus.UNCONFIRMED,
                confidence=1.0,
            ),
        ),
        next_actions=("Пришли ссылку или подключи read-only web search provider.",),
        markdown_report=summary,
    )


def _artifact_only_section(
    *,
    url: str,
    error: str,
    artifacts: tuple[str, ...],
) -> str:
    artifact_lines = "\n".join(f"- {artifact}" for artifact in artifacts)
    return (
        f"# Source artifact only: {url}\n"
        f"Read failed: {error}\n"
        "Artifacts:\n"
        f"{artifact_lines}"
    )


def _verification_challenge_finding(url: str) -> Finding:
    return Finding(
        claim=(
            "Источник открыл anti-bot/security verification вместо целевой "
            f"страницы: {url}"
        ),
        status=FindingStatus.REJECTED,
        confidence=0.95,
        evidence=(url,),
    )


def _verification_challenge_section(url: str) -> str:
    return (
        f"# Source blocked by security verification: {url}\n"
        "Страница не является целевым источником: сайт показал anti-bot/security "
        "verification. Read-only агент не проходит такую проверку и не должен "
        "выдавать verification screen как результат."
    )


def _has_verification_challenge(findings: list[Finding]) -> bool:
    return any(
        _finding_mentions_verification_challenge(finding) for finding in findings
    )


def _finding_mentions_verification_challenge(finding: Finding) -> bool:
    return _verification_marker(finding.claim)


def _is_verification_challenge_error(exc: Exception) -> bool:
    return _verification_marker(_error_summary(exc))


def _verification_marker(text: str) -> bool:
    lowered = text.lower()
    return (
        "security verification" in lowered
        or "anti-bot" in lowered
        or "challenge" in lowered
        or "not a bot" in lowered
        or "cloudflare" in lowered
    )


def _looks_like_downloadable_url(url: str) -> bool:
    return Path(urlparse(url).path.lower()).suffix in _DOWNLOAD_SUFFIXES


def _error_summary(exc: Exception) -> str:
    text = str(exc).strip()
    if text:
        return text
    return exc.__class__.__name__

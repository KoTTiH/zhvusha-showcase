"""Tool-gated image artifacts for channel visuals."""

from __future__ import annotations

import asyncio
import hashlib
import re
import shutil
from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlparse

from src.llm.protocols import LLMGatewayProtocol, LLMImageRequest

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping
    from pathlib import Path

    LocalCardRenderer = Callable[[str, Path, float], Awaitable[None]]


@dataclass
class ChannelVisualImageTool:
    """Generate one visual artifact through the Agent Runtime Tool Gateway."""

    workspace_root: Path
    llm: LLMGatewayProtocol
    artifact_dir: str = "agent_runtime/channel_visual_artifacts"
    name: str = "channel_visual_generate_image"
    capability: str = "channel_visual_image_generation"

    async def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("prompt is required")
        caption = str(payload.get("caption", "")).strip()[:1024]
        response = await self.llm.generate_image(
            LLMImageRequest(
                prompt=prompt,
                caller="agent_runtime.channel_visual",
            )
        )
        target = self._artifact_path(prompt=prompt, mime_type=response.mime_type)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.image)
        workspace_root = self.workspace_root.expanduser().resolve()
        return {
            "status": "ready",
            "asset_path": target.relative_to(workspace_root).as_posix(),
            "caption": caption,
            "prompt": prompt,
            "model": response.model,
            "mime_type": response.mime_type,
            "revised_prompt": response.revised_prompt,
        }

    def _artifact_path(self, *, prompt: str, mime_type: str) -> Path:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        suffix = ".png" if mime_type == "image/png" else ".bin"
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", prompt[:48]).strip(".-_")
        filename = f"generated-{digest}-{name or 'visual'}{suffix}"
        target = (self._root() / filename).resolve()
        if not target.is_relative_to(self._root()):
            raise RuntimeError("artifact path escapes workspace root")
        return target

    def _root(self) -> Path:
        root = self.workspace_root.expanduser().resolve()
        artifact_root = (root / self.artifact_dir).resolve()
        if not artifact_root.is_relative_to(root):
            raise ValueError("artifact_dir escapes workspace root")
        return artifact_root


@dataclass
class ChannelVisualLocalCardTool:
    """Render a deterministic local card when source screenshots are unusable."""

    workspace_root: Path
    artifact_dir: str = "agent_runtime/channel_visual_artifacts"
    browser_executable: str | None = None
    render_timeout_seconds: float = 30.0
    renderer: LocalCardRenderer | None = None
    name: str = "channel_visual_generate_card"
    capability: str = "channel_visual_image_generation"

    async def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        title = _compact_single_line(str(payload.get("title", ""))) or "Визуал"
        body = _compact_single_line(str(payload.get("body", "")))
        source_url = str(payload.get("source_url", "")).strip()
        caption = str(payload.get("caption", "")).strip()[:1024]
        html = _render_local_card_html(
            title=title,
            body=body,
            source_url=source_url,
        )
        target = self._artifact_path(
            title=title,
            body=body,
            source_url=source_url,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        await self._render(html=html, target=target)
        if not target.is_file() or target.stat().st_size <= 0:
            raise RuntimeError("local visual card did not create an artifact")
        workspace_root = self.workspace_root.expanduser().resolve()
        return {
            "status": "ready",
            "asset_path": target.relative_to(workspace_root).as_posix(),
            "caption": caption or "Визуальная карточка к посту",
            "source_url": source_url,
            "model": "local-chromium-card",
            "mime_type": "image/png",
        }

    async def _render(self, *, html: str, target: Path) -> None:
        if self.renderer is not None:
            await self.renderer(html, target, self.render_timeout_seconds)
            return
        browser = self.browser_executable or _discover_browser_executable()
        if not browser:
            raise RuntimeError("no Chromium-compatible executable found")
        await _render_html_card_with_chromium(
            browser_executable=browser,
            html=html,
            output_path=target,
            timeout_seconds=self.render_timeout_seconds,
        )

    def _artifact_path(self, *, title: str, body: str, source_url: str) -> Path:
        digest = hashlib.sha256(f"{title}\n{body}\n{source_url}".encode()).hexdigest()[
            :12
        ]
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", title[:48]).strip(".-_")
        target = self._root() / f"fallback-card-{digest}-{name or 'visual'}.png"
        target = target.resolve()
        if not target.is_relative_to(self._root()):
            raise RuntimeError("artifact path escapes workspace root")
        return target

    def _root(self) -> Path:
        root = self.workspace_root.expanduser().resolve()
        artifact_root = (root / self.artifact_dir).resolve()
        if not artifact_root.is_relative_to(root):
            raise ValueError("artifact_dir escapes workspace root")
        return artifact_root


async def _render_html_card_with_chromium(
    *,
    browser_executable: str,
    html: str,
    output_path: Path,
    timeout_seconds: float,
) -> None:
    url = "data:text/html;charset=utf-8," + quote(html)
    proc = await asyncio.create_subprocess_exec(
        browser_executable,
        "--headless=new",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--no-first-run",
        "--no-default-browser-check",
        "--hide-scrollbars",
        "--window-size=1365,812",
        f"--screenshot={output_path}",
        url,
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
        await proc.wait()
        raise TimeoutError("local card render timed out") from exc
    if proc.returncode != 0:
        details = (stderr or stdout).decode("utf-8", errors="replace")[-500:]
        raise RuntimeError(f"local card render failed: {details}")


def _render_local_card_html(*, title: str, body: str, source_url: str) -> str:
    headline = _fit_headline(title)
    subtitle = _fit_subtitle(body)
    source_label = _source_label(source_url)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
*{{box-sizing:border-box}}
html,body{{margin:0;width:1365px;height:812px}}
body{{
  background:#111318;
  color:#f4f1ea;
  font-family:Inter,Arial,sans-serif;
  overflow:hidden;
}}
.wrap{{
  position:relative;
  width:1365px;
  height:812px;
  padding:68px 84px;
  background:
    radial-gradient(circle at 18% 16%,#2f6f61 0,#182824 24%,transparent 43%),
    linear-gradient(135deg,#101216 0%,#202028 54%,#2a241d 100%);
}}
.source{{display:flex;gap:14px;align-items:center;color:#b8c0b4;font-size:28px}}
.mark{{
  width:52px;height:52px;border:2px solid #d8d2c4;border-radius:50%;
  display:grid;place-items:center;font-weight:700;color:#f4f1ea;
}}
.title{{
  margin-top:84px;font-size:86px;line-height:.98;font-weight:760;
  max-width:820px;letter-spacing:0;
}}
.title span{{color:#8bd8b3}}
.sub{{
  margin-top:32px;font-size:32px;line-height:1.26;
  max-width:800px;color:#ded9ce;
}}
.panel{{
  position:absolute;right:82px;top:108px;width:360px;height:620px;
  border-radius:44px;background:#efe9dd;color:#151515;
  box-shadow:0 30px 80px rgba(0,0,0,.38);
  padding:28px;border:10px solid #2d2b2a;
}}
.phonebar{{height:30px;display:flex;justify-content:center}}
.pill{{width:92px;height:8px;border-radius:9px;background:#171717}}
.apphead{{font-size:24px;font-weight:760;margin-top:18px}}
.card{{
  margin-top:22px;border:1px solid #d5cbbb;border-radius:24px;
  padding:20px;background:#fffdf8;
}}
.row{{display:flex;align-items:center;gap:12px;margin:17px 0;font-size:19px;color:#2f312f}}
.dot{{width:13px;height:13px;border-radius:50%;background:#2f8f69}}
.terminal{{
  margin-top:24px;background:#171717;color:#9ff1bd;border-radius:20px;
  padding:18px;font-family:monospace;font-size:16px;line-height:1.45;
}}
.foot{{position:absolute;left:84px;bottom:38px;width:820px;display:flex;gap:14px;flex-wrap:wrap}}
.tag{{font-size:22px;color:#17201b;background:#ccebd5;border-radius:999px;padding:11px 17px}}
</style>
</head>
<body>
<main class="wrap">
  <div class="source"><div class="mark">AI</div><div>{escape(source_label)}</div></div>
  <h1 class="title">{headline}</h1>
  <p class="sub">{escape(subtitle)}</p>
  <section class="panel">
    <div class="phonebar"><div class="pill"></div></div>
    <div class="apphead">Codex session</div>
    <div class="card">
      <div class="row"><div class="dot"></div><strong>Waiting for approval</strong></div>
      <div class="row">diff · tests · terminal</div>
      <div class="row">reply from phone</div>
    </div>
    <div class="terminal">$ pytest -q<br>36 passed<br><br>agent: нужен выбор<br>human: approve</div>
  </section>
  <div class="foot">
    <div class="tag">remote control</div>
    <div class="tag">agent view</div>
    <div class="tag">context не скисает</div>
  </div>
</main>
</body>
</html>"""


def _discover_browser_executable() -> str | None:
    for candidate in (
        "chromium",
        "chromium-browser",
        "google-chrome",
        "google-chrome-stable",
    ):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def _compact_single_line(text: str) -> str:
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def _fit_headline(title: str) -> str:
    compact = title.strip()[:54] or "визуал к посту"
    words = compact.split()
    if len(words) >= 3:
        split_at = max(1, len(words) // 2)
        first = escape(" ".join(words[:split_at]))
        second = escape(" ".join(words[split_at:]))
        return f"{first} <span>{second}</span>"
    return escape(compact)


def _fit_subtitle(body: str) -> str:
    if not body:
        return "Локальная карточка к посту, когда внешний скриншот недоступен."
    sentence = body.split(". ", 1)[0].strip()
    return sentence[:170]


def _source_label(source_url: str) -> str:
    host = urlparse(source_url).hostname if source_url else ""
    if not host:
        return "source-backed visual"
    return f"{host.removeprefix('www.')} · fallback visual"

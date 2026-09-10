#!/usr/bin/env python3
"""CLI image generator for ZHVUSHA channel visuals.

The bot's CLI image adapter passes the prompt through stdin and
``ZHVUSHA_IMAGE_*`` environment variables. This wrapper renders a deterministic
local PNG with Chromium, so image generation can run without an image API.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

from src.agent_runtime.image_artifacts import (
    _render_html_card_with_chromium,
    _render_local_card_html,
)


def _read_prompt() -> str:
    return (
        os.environ.get("ZHVUSHA_IMAGE_PROMPT", "").strip()
        or sys.stdin.read().strip()
        or "Визуальная карточка к посту Жвуши"
    )


def _browser() -> str:
    for candidate in (
        "chromium",
        "chromium-browser",
        "google-chrome",
        "google-chrome-stable",
    ):
        path = shutil.which(candidate)
        if path:
            return path
    raise RuntimeError("Chromium-compatible executable was not found")


async def _main() -> None:
    output_raw = os.environ.get("ZHVUSHA_IMAGE_OUTPUT", "").strip()
    if not output_raw:
        raise RuntimeError("ZHVUSHA_IMAGE_OUTPUT is required")

    prompt = _read_prompt()
    output_path = Path(output_raw).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timeout = float(os.environ.get("ZHVUSHA_IMAGE_TIMEOUT_SECONDS", "30") or "30")
    html = _render_local_card_html(
        title=prompt,
        body=prompt,
        source_url="",
    )
    await _render_html_card_with_chromium(
        browser_executable=_browser(),
        html=html,
        output_path=output_path,
        timeout_seconds=timeout,
    )


if __name__ == "__main__":
    asyncio.run(_main())

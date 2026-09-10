"""CLI-backed image generation adapter for optional channel visuals."""

from __future__ import annotations

import asyncio
import os
import shlex
import tempfile
from pathlib import Path

from src.llm.protocols import LLMError, LLMImageRequest, LLMImageResponse


class CLIImageGenerator:
    """Run a local/subscription-backed CLI to produce one image artifact."""

    def __init__(
        self,
        *,
        command: str,
        model: str = "",
        size: str = "1024x1024",
        timeout_seconds: float = 300.0,
    ) -> None:
        self._argv = shlex.split(command)
        if not self._argv:
            raise ValueError("image generation CLI command is required")
        self._model = model
        self._size = size
        self._timeout_seconds = max(1.0, timeout_seconds)

    async def generate_image(self, request: LLMImageRequest) -> LLMImageResponse:
        model = request.model or self._model or "cli"
        size = request.size or self._size
        with tempfile.TemporaryDirectory(prefix="zhvusha-image-cli-") as tmp_dir:
            output_path = Path(tmp_dir) / "output.png"
            env = {
                **os.environ,
                "ZHVUSHA_IMAGE_PROMPT": request.prompt,
                "ZHVUSHA_IMAGE_OUTPUT": str(output_path),
                "ZHVUSHA_IMAGE_MODEL": model,
                "ZHVUSHA_IMAGE_SIZE": size,
                "ZHVUSHA_IMAGE_CALLER": request.caller,
            }
            proc = await asyncio.create_subprocess_exec(
                *self._argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(request.prompt.encode("utf-8")),
                    timeout=self._timeout_seconds,
                )
            except TimeoutError as exc:
                proc.kill()
                await proc.wait()
                raise LLMError("image generation CLI timed out") from exc
            if proc.returncode != 0:
                details = (stderr or stdout).decode("utf-8", errors="replace")[-500:]
                raise LLMError(f"image generation CLI failed: {details}")

            if output_path.is_file() and output_path.stat().st_size > 0:
                image = output_path.read_bytes()
                mime_type = _mime_type_from_suffix(output_path.suffix)
            elif stdout:
                image = stdout
                mime_type = "image/png"
            else:
                raise LLMError("image generation CLI returned no image")

        return LLMImageResponse(
            image=image,
            model=model,
            mime_type=mime_type,
            revised_prompt="",
        )


def _mime_type_from_suffix(suffix: str) -> str:
    normalized = suffix.lower()
    if normalized in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if normalized == ".webp":
        return "image/webp"
    return "image/png"

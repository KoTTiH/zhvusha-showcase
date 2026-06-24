"""OpenAI Image API adapter for optional channel visuals."""

from __future__ import annotations

import base64
from typing import Any

from src.llm.protocols import (
    LLMError,
    LLMImageRequest,
    LLMImageResponse,
    ProviderUnavailableError,
)


class OpenAIImageGenerator:
    """Small adapter around the configured OpenAI Images API model."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        size: str = "1024x1024",
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._size = size
        self._base_url = base_url.rstrip("/")

    async def generate_image(self, request: LLMImageRequest) -> LLMImageResponse:
        if not self._api_key or not (request.model or self._model):
            raise ProviderUnavailableError("OpenAI image generation is not configured")

        import httpx

        model = request.model or self._model
        payload: dict[str, Any] = {
            "model": model,
            "prompt": request.prompt,
            "size": request.size or self._size,
            "n": 1,
            "response_format": "b64_json",
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self._base_url}/images/generations",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenAI image generation failed: {exc}") from exc

        data = response.json()
        try:
            item = data["data"][0]
            encoded = item["b64_json"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("OpenAI image generation returned no image") from exc

        return LLMImageResponse(
            image=base64.b64decode(encoded),
            model=str(model),
            mime_type="image/png",
            revised_prompt=str(item.get("revised_prompt", "")),
        )

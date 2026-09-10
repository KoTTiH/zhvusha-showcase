"""Google Gemini adapter — used primarily for vision (image description).

Private adapter. External modules import via ``LLMRouter`` and
``LLMGatewayProtocol`` from ``src.llm.protocols``.
"""

from __future__ import annotations

from typing import Any

import structlog

from src.core.config import get_settings
from src.llm.base import BaseLLMAdapter
from src.llm.protocols import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
    LLMVisionRequest,
)

logger = structlog.get_logger()


def _get_genai() -> Any:
    """Lazy import ``google.genai`` to avoid errors when the extra is absent."""
    import google.genai as genai

    return genai


class GeminiAdapter(BaseLLMAdapter):
    """Adapter for the Google Gemini API.

    Used primarily for vision (image description). Text generation tiers
    default to Codex CLI unless explicitly overridden in configuration.
    """

    name: str = "gemini"
    default_model: str = "gemini-2.5-flash-lite"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        settings = get_settings()
        model = request.model or self.default_model

        genai = _get_genai()
        client: Any = genai.Client(api_key=settings.google_api_key)

        config_kwargs: dict[str, Any] = {}
        if request.system:
            config_kwargs["system_instruction"] = request.system

        config = genai.types.GenerateContentConfig(**config_kwargs)

        logger.info(
            "llm_request",
            adapter=self.name,
            model=model,
            prompt_len=len(request.prompt),
            has_system=bool(request.system),
            caller=request.caller,
        )

        response = await client.aio.models.generate_content(
            model=model,
            contents=request.prompt,
            config=config,
        )

        text: str = response.text or ""

        logger.info(
            "llm_response",
            adapter=self.name,
            model=model,
            response_len=len(text),
        )

        # Gemini's free tier does not surface token usage in a stable shape;
        # the router accounts for Gemini calls coarsely via record_gemini_call().
        return LLMResponse(text=text, model=model, usage=LLMUsage())

    async def describe_images(self, request: LLMVisionRequest) -> LLMResponse:
        """Describe images using the multimodal Gemini API.

        Returns an ``LLMResponse`` whose ``text`` holds the description.
        """
        settings = get_settings()
        model = settings.vision_model

        genai = _get_genai()
        client: Any = genai.Client(api_key=settings.google_api_key)

        parts: list[Any] = [request.prompt]
        for img in request.images:
            parts.append(genai.types.Part.from_bytes(data=img, mime_type="image/jpeg"))

        logger.info(
            "vision_request",
            adapter=self.name,
            model=model,
            num_images=len(request.images),
            caller=request.caller,
        )

        response = await client.aio.models.generate_content(
            model=model,
            contents=parts,
        )

        text: str = response.text or ""

        logger.info(
            "vision_response",
            adapter=self.name,
            model=model,
            response_len=len(text),
        )

        return LLMResponse(text=text, model=model, usage=LLMUsage())

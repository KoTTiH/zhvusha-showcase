"""Generic OpenAI-compatible chat adapter for additional providers."""

from __future__ import annotations

import httpx
import structlog

from src.core.config import Settings, get_settings
from src.llm.base import BaseLLMAdapter
from src.llm.protocols import LLMError, LLMRequest, LLMResponse, LLMUsage

logger = structlog.get_logger()
_DEFAULT_TIMEOUT_SECONDS = 60.0


class OpenAICompatibleChatAdapter(BaseLLMAdapter):
    """Adapter for providers exposing ``/chat/completions``."""

    default_model = ""

    def __init__(
        self,
        *,
        name: str,
        api_key_attr: str,
        base_url_attr: str,
    ) -> None:
        self.name = name
        self._api_key_attr = api_key_attr
        self._base_url_attr = base_url_attr
        self._client: httpx.AsyncClient | None = None

    def _settings(self) -> Settings:
        return get_settings()

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            settings = self._settings()
            api_key = str(getattr(settings, self._api_key_attr))
            base_url = str(getattr(settings, self._base_url_attr)).rstrip("/")
            self._client = httpx.AsyncClient(
                base_url=base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=_DEFAULT_TIMEOUT_SECONDS,
            )
        return self._client

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if not request.model:
            raise LLMError(f"{self.name} received empty model")
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        payload: dict[str, object] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": 4096,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        try:
            response = await self._get_client().post("/chat/completions", json=payload)
            response.raise_for_status()
        except Exception as exc:
            logger.error(
                "openai_compatible_provider_error",
                provider=self.name,
                model=request.model,
                error=str(exc)[:500],
            )
            raise LLMError(f"{self.name} API error: {exc}") from exc
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"Empty response from {self.name}")
        text = choices[0].get("message", {}).get("content", "") or ""
        usage_raw = data.get("usage") or {}
        usage = LLMUsage(
            input_tokens=int(usage_raw.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage_raw.get("completion_tokens", 0) or 0),
        )
        return LLMResponse(text=text, model=request.model, usage=usage)


class NvidiaNIMAdapter(OpenAICompatibleChatAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="nvidia_nim",
            api_key_attr="nvidia_nim_api_key",
            base_url_attr="nvidia_nim_base_url",
        )


class HuggingFaceRouterAdapter(OpenAICompatibleChatAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="huggingface",
            api_key_attr="huggingface_api_key",
            base_url_attr="huggingface_base_url",
        )


class MiniMaxAdapter(OpenAICompatibleChatAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="minimax",
            api_key_attr="minimax_api_key",
            base_url_attr="minimax_base_url",
        )

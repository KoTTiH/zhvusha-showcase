"""OpenRouter adapter — OpenAI-compatible HTTP client.

Private adapter. External modules import via ``LLMRouter`` and
``LLMGatewayProtocol`` from ``src.llm.protocols``.

OpenRouter aggregates many providers under a single OpenAI-compatible
``/chat/completions`` endpoint. Models are addressed as ``vendor/model``,
e.g. ``deepseek/deepseek-chat``, ``meta-llama/llama-3.3-70b-instruct``.
The model id is passed through unchanged — the adapter does not curate
which OpenRouter models exist.
"""

from __future__ import annotations

import httpx
import structlog

from src.core.config import get_settings
from src.llm.base import BaseLLMAdapter
from src.llm.protocols import (
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMUsage,
)

logger = structlog.get_logger()

_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_TIMEOUT_SECONDS = 60.0


def _extract_error_message(response: httpx.Response) -> str:
    """Pull a human-readable reason from an OpenRouter error response.

    OpenRouter wraps errors as ``{"error": {"message": "...", "code": ...}}``
    in the body. ``raise_for_status`` only carries URL + status, which hides
    the *why* (insufficient credits, rate limit on free tier, model down,
    etc.). We try to surface that message; fall back to a body snippet.
    """
    try:
        data = response.json()
    except Exception:
        return response.text[:200]
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
    return str(data)[:200]


class OpenRouterAdapter(BaseLLMAdapter):
    """Adapter for OpenRouter (https://openrouter.ai/).

    OpenAI-compatible chat completions over plain HTTP — no SDK required.
    The router resolves the model id before calling ``generate``; the adapter
    treats ``request.model`` as authoritative.
    """

    name: str = "openrouter"
    # Router resolves the api_id via ``src.llm.providers``; the adapter
    # rejects empty model rather than guessing.
    default_model: str = ""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            settings = get_settings()
            self._client = httpx.AsyncClient(
                base_url=_BASE_URL,
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=_DEFAULT_TIMEOUT_SECONDS,
            )
        return self._client

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if not request.model:
            raise LLMError("OpenRouterAdapter received empty model")

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

        logger.info(
            "llm_request",
            adapter=self.name,
            model=request.model,
            prompt_len=len(request.prompt),
            has_system=bool(request.system),
            caller=request.caller,
        )

        try:
            response = await self._get_client().post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            reason = _extract_error_message(exc.response)
            logger.error(
                "openrouter_api_error",
                status=exc.response.status_code,
                reason=reason,
                model=request.model,
            )
            raise LLMError(
                f"OpenRouter API error {exc.response.status_code}: {reason}"
            ) from exc
        except Exception as exc:
            logger.error("openrouter_api_error", error=str(exc)[:500])
            raise LLMError(f"OpenRouter API error: {exc}") from exc

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("Empty response from OpenRouter (no choices)")
        text = choices[0].get("message", {}).get("content", "") or ""

        usage_raw = data.get("usage") or {}
        usage = LLMUsage(
            input_tokens=int(usage_raw.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage_raw.get("completion_tokens", 0) or 0),
        )

        logger.info(
            "llm_response",
            adapter=self.name,
            model=request.model,
            response_len=len(text),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )

        return LLMResponse(text=text, model=request.model, usage=usage)

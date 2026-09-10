"""Voice gateway normalization for the shared ZHVUSHA message loop."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.skills.base import AgentContext


class VoiceTranscript(BaseModel):
    """Transcript produced by an STT layer before Жвуша sees the message."""

    text: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    language: str = ""
    audio_ref: str = ""


class VoiceInputEnvelope(BaseModel):
    """Normalized voice input before it enters the normal chat pipeline."""

    user_id: int
    chat_id: int | None
    mode: Literal["personal", "assistant", "social"]
    message_id: int | None = None
    transcript: VoiceTranscript
    gateway: str = "voice"


class NormalizedVoiceMessage(BaseModel):
    """Voice input converted into the same message contract as text chat."""

    text: str
    user_id: int
    chat_id: int | None
    mode: Literal["personal", "assistant", "social"]
    message_id: int | None = None
    dispatch_allowed: bool
    needs_confirmation: bool
    confirmation_prompt: str = ""
    metadata: dict[str, object] = Field(default_factory=dict)

    def to_agent_context(self, base_context: AgentContext) -> AgentContext:
        """Attach voice provenance to an existing AgentContext."""
        return replace(
            base_context,
            user_id=self.user_id,
            chat_id=self.chat_id,
            mode=self.mode,
            message_id=self.message_id,
            metadata={**base_context.metadata, **self.metadata},
        )


class VoiceOutputEnvelope(BaseModel):
    """Text response prepared for optional TTS rendering."""

    text: str
    reply_to_source: str = "voice"
    renderer: str = "voice_output"
    dialogue_owner: str = "zhvusha"


class VoiceGatewayNormalizer:
    """Normalize voice transcripts without creating a separate assistant."""

    def __init__(self, *, min_confidence: float = 0.75) -> None:
        self._min_confidence = min_confidence

    def normalize(
        self,
        envelope: VoiceInputEnvelope,
        *,
        base_context: AgentContext,
    ) -> NormalizedVoiceMessage:
        """Return a text message plus voice metadata for the existing pipeline."""
        if base_context.user_id != envelope.user_id:
            raise ValueError("voice envelope user_id does not match base context")
        if base_context.chat_id != envelope.chat_id:
            raise ValueError("voice envelope chat_id does not match base context")
        text = " ".join(envelope.transcript.text.split())
        low_confidence = envelope.transcript.confidence < self._min_confidence
        metadata: dict[str, object] = {
            "source": "voice",
            "voice_gateway": envelope.gateway,
            "voice_confidence": envelope.transcript.confidence,
            "voice_language": envelope.transcript.language,
            "voice_audio_ref": envelope.transcript.audio_ref,
            "voice_low_confidence": low_confidence,
            "needs_confirmation_if_low_confidence": low_confidence,
            "voice_dialogue_owner": "zhvusha",
        }
        prompt = f"Подтверди голосовую команду: «{text}»." if low_confidence else ""
        return NormalizedVoiceMessage(
            text=text,
            user_id=envelope.user_id,
            chat_id=envelope.chat_id,
            mode=envelope.mode,
            message_id=envelope.message_id,
            dispatch_allowed=not low_confidence,
            needs_confirmation=low_confidence,
            confirmation_prompt=prompt,
            metadata=metadata,
        )


class VoiceOutputRenderer:
    """Prepare an existing Жвуша text reply for a TTS layer."""

    def render_text_reply(
        self,
        text: str,
        *,
        reply_to_source: str = "voice",
    ) -> VoiceOutputEnvelope:
        """Wrap a text reply without changing dialogue ownership."""
        return VoiceOutputEnvelope(text=text, reply_to_source=reply_to_source)

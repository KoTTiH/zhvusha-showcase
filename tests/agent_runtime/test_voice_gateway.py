"""Voice gateway normalization contracts."""

from __future__ import annotations


def test_voice_input_becomes_agent_context_metadata_without_new_personality() -> None:
    from src.agent_runtime.voice_gateway import (
        VoiceGatewayNormalizer,
        VoiceInputEnvelope,
        VoiceTranscript,
    )
    from src.skills.base import AgentContext

    base = AgentContext(
        user_id=12345,
        chat_id=12345,
        mode="personal",
        message_id=77,
        metadata={"dialogue_context": "known context"},
    )
    envelope = VoiceInputEnvelope(
        user_id=12345,
        chat_id=12345,
        mode="personal",
        message_id=77,
        transcript=VoiceTranscript(
            text="Проверь Kubernetes ingress",
            confidence=0.92,
            language="ru",
            audio_ref="telegram:file:voice-77",
        ),
        gateway="telegram_voice",
    )

    normalized = VoiceGatewayNormalizer().normalize(envelope, base_context=base)
    context = normalized.to_agent_context(base)

    assert normalized.dispatch_allowed is True
    assert normalized.text == "Проверь Kubernetes ingress"
    assert context.metadata["source"] == "voice"
    assert context.metadata["voice_gateway"] == "telegram_voice"
    assert context.metadata["voice_confidence"] == 0.92
    assert context.metadata["voice_audio_ref"] == "telegram:file:voice-77"
    assert context.metadata["voice_low_confidence"] is False
    assert context.metadata["voice_dialogue_owner"] == "zhvusha"
    assert context.metadata["dialogue_context"] == "known context"


def test_low_confidence_voice_input_requires_confirmation_before_dispatch() -> None:
    from src.agent_runtime.voice_gateway import (
        VoiceGatewayNormalizer,
        VoiceInputEnvelope,
        VoiceTranscript,
    )
    from src.skills.base import AgentContext

    base = AgentContext(user_id=12345, chat_id=12345, mode="personal", message_id=78)
    envelope = VoiceInputEnvelope(
        user_id=12345,
        chat_id=12345,
        mode="personal",
        message_id=78,
        transcript=VoiceTranscript(
            text="отправь сообщение",
            confidence=0.51,
            language="ru",
            audio_ref="telegram:file:voice-78",
        ),
        gateway="telegram_voice",
    )

    normalized = VoiceGatewayNormalizer(min_confidence=0.75).normalize(
        envelope,
        base_context=base,
    )
    context = normalized.to_agent_context(base)

    assert normalized.dispatch_allowed is False
    assert normalized.needs_confirmation is True
    assert "Подтверди голосовую команду" in normalized.confirmation_prompt
    assert context.metadata["voice_low_confidence"] is True
    assert context.metadata["needs_confirmation_if_low_confidence"] is True


def test_voice_output_envelope_is_renderer_not_dialogue_owner() -> None:
    from src.agent_runtime.voice_gateway import VoiceOutputRenderer

    envelope = VoiceOutputRenderer().render_text_reply(
        "Готово.",
        reply_to_source="voice",
    )

    assert envelope.text == "Готово."
    assert envelope.reply_to_source == "voice"
    assert envelope.dialogue_owner == "zhvusha"
    assert envelope.renderer == "voice_output"

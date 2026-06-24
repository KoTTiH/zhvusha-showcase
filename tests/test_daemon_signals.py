"""Tests for Signal model serialization."""

from __future__ import annotations

from datetime import UTC, datetime

from src.daemon.signals import Signal


class TestSignal:
    def test_default_values(self) -> None:
        sig = Signal()
        assert sig.source == ""
        assert sig.priority == "normal"
        assert sig.signal_type == ""
        assert sig.requires_response is False
        assert sig.ttl_minutes is None
        assert sig.id  # UUID generated

    def test_custom_values(self) -> None:
        sig = Signal(
            source="telegram_chat",
            priority="critical",
            signal_type="user_message",
            payload={"text": "hello"},
            requires_response=True,
            ttl_minutes=30,
        )
        assert sig.source == "telegram_chat"
        assert sig.priority == "critical"
        assert sig.payload["text"] == "hello"
        assert sig.ttl_minutes == 30

    def test_to_dict(self) -> None:
        sig = Signal(
            source="kwork",
            priority="normal",
            signal_type="new_project",
            payload={"title": "Bot"},
        )
        d = sig.to_dict()
        assert d["source"] == "kwork"
        assert d["priority"] == "normal"
        assert '"title": "Bot"' in d["payload"]
        assert d["requires_response"] == "False"
        assert d["ttl_minutes"] == ""

    def test_roundtrip(self) -> None:
        original = Signal(
            source="cron",
            priority="background",
            signal_type="scheduled",
            payload={"task": "morning"},
            requires_response=False,
            ttl_minutes=60,
        )
        d = original.to_dict()
        restored = Signal.from_dict(d)

        assert restored.source == original.source
        assert restored.priority == original.priority
        assert restored.signal_type == original.signal_type
        assert restored.payload == original.payload
        assert restored.requires_response == original.requires_response
        assert restored.ttl_minutes == original.ttl_minutes

    def test_from_dict_with_missing_fields(self) -> None:
        sig = Signal.from_dict({"source": "test"})
        assert sig.source == "test"
        assert sig.priority == "normal"
        assert sig.payload == {}

    def test_from_dict_with_bytes(self) -> None:
        """Redis returns bytes — from_dict should handle that."""
        sig = Signal.from_dict(
            {
                "source": "test",
                "priority": "critical",
                "signal_type": "msg",
                "payload": '{"k": "v"}',
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "requires_response": "True",
                "ttl_minutes": "15",
            }
        )
        assert sig.source == "test"
        assert sig.requires_response is True
        assert sig.ttl_minutes == 15

    def test_stream_entry_id_defaults_to_none(self) -> None:
        sig = Signal()
        assert sig.stream_entry_id is None

    def test_stream_entry_id_not_serialized(self) -> None:
        """stream_entry_id is runtime-only, not sent to Redis."""
        sig = Signal(source="test")
        sig.stream_entry_id = "1234-0"
        d = sig.to_dict()
        assert "stream_entry_id" not in d

    def test_stream_entry_id_not_in_repr(self) -> None:
        sig = Signal(source="test")
        sig.stream_entry_id = "1234-0"
        assert "stream_entry_id" not in repr(sig)

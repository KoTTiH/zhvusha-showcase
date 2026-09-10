"""Telegram message utilities.

Compatibility wrapper: lower layers import from ``src.utils.telegram``.
"""

from __future__ import annotations

from src.utils.telegram import send_long_message
from src.utils.text import _split_text

__all__ = ["_split_text", "send_long_message"]

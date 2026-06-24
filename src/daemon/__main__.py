"""Entry point for Zhvusha daemon.

Usage:
    python -m src.daemon
"""

from __future__ import annotations

import asyncio

from src.daemon.main import run_daemon

if __name__ == "__main__":
    asyncio.run(run_daemon())

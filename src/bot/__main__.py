"""Entry point for Zhvusha bot.

Usage:
    python -m src.bot
"""

from __future__ import annotations

import asyncio
import faulthandler
import signal

from src.bot.main import main

if __name__ == "__main__":
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    asyncio.run(main())

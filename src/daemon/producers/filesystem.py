"""Filesystem watcher producer using watchdog."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.daemon.signals import Signal

if TYPE_CHECKING:
    from pathlib import Path

    from src.daemon.stream import SignalStream
    from src.daemon.ticker import AdaptiveTicker


class FilesystemProducer:
    """Watches workspace/codebase for file changes."""

    def __init__(
        self,
        stream: SignalStream,
        ticker: AdaptiveTicker,
        watch_paths: list[Path] | None = None,
    ) -> None:
        self._stream = stream
        self._ticker = ticker
        self._watch_paths = watch_paths or []
        self._observer: Any = None

    async def start(self) -> None:
        """Start the filesystem watcher."""
        if not self._watch_paths:
            return

        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer

            # Capture the running event loop NOW (async context) so the
            # watchdog thread callback can schedule coroutines on it.
            loop = asyncio.get_running_loop()

            class Handler(FileSystemEventHandler):
                def __init__(self, producer: FilesystemProducer) -> None:
                    self._producer = producer

                def on_modified(self, event: object) -> None:
                    src_path = getattr(event, "src_path", "")
                    if not getattr(event, "is_directory", False):
                        signal = Signal(
                            source="filesystem",
                            priority="background",
                            signal_type="file_changed",
                            payload={"path": str(src_path)},
                        )
                        coro = self._producer._stream.push(signal)
                        try:
                            loop.call_soon_threadsafe(loop.create_task, coro)
                            loop.call_soon_threadsafe(self._producer._ticker.wake)
                        except RuntimeError:
                            coro.close()

            observer = Observer()
            handler = Handler(self)
            for path in self._watch_paths:
                observer.schedule(handler, str(path), recursive=True)
            observer.start()
            self._observer = observer
        except ImportError:
            pass  # watchdog not installed

    async def stop(self) -> None:
        """Stop the filesystem watcher."""
        if self._observer is not None:
            self._observer.stop()
            await asyncio.to_thread(self._observer.join, timeout=5.0)

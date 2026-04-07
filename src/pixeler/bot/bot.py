import time
from abc import ABC, abstractmethod
from typing import Union

from pixeler.bot.bot_status import BotStatus
from pixeler.bot.bot_thread import BotThread
from pixeler.window.win32_window import Win32Window
from pixeler.window.window import Window


class Bot(ABC):
    """
    Abstract base class for automation bots.

    Subclass Bot and implement step() with a single iteration of your bot's
    logic. The framework manages the loop, threading, and lifecycle — you do
    not need to write a while loop.

    Lifecycle:
        on_start()  →  step() [repeating]  →  on_stop()

    Example::

        class MyBot(Bot):
            def step(self):
                screenshot = self.window.screenshot()
                # ... analyse and act ...

        bot = MyBot(window=Win32Window("My Game"))
        bot.start()
        # ... later ...
        bot.stop()
    """

    def __init__(self, window: Union[Window, Win32Window] = None):
        self.window = window
        self.status = BotStatus.STOPPED
        self._thread: BotThread | None = None
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Abstract / override-able methods
    # ------------------------------------------------------------------

    @abstractmethod
    def step(self):
        """
        A single iteration of the bot's main logic.

        Called repeatedly by the bot thread while status is RUNNING.
        Do not write a while loop here — return normally to continue,
        or call self.stop() to end the bot.
        """

    def on_start(self):
        """Called once on the main thread immediately before the loop starts."""

    def on_stop(self):
        """Called once on the main thread immediately after the loop stops."""

    # ------------------------------------------------------------------
    # Lifecycle control (call from main thread or from within step())
    # ------------------------------------------------------------------

    def start(self):
        """Start the bot thread. No-op if already running."""
        if self.status != BotStatus.STOPPED:
            self.log("Bot is already running or paused.")
            return
        self.log("Starting...")
        self._start_time = time.time()
        self.status = BotStatus.RUNNING
        self.on_start()
        self._thread = BotThread(self)
        self._thread.start()

    def stop(self):
        """Stop the bot thread and join it. No-op if already stopped."""
        if self.status == BotStatus.STOPPED:
            self.log("Bot is not running.")
            return
        self.log(f"Stopping... (ran for {self.elapsed():.1f}s)")
        self.status = BotStatus.STOPPED
        if self._thread is not None:
            self._thread.stop()
            self._thread.join(timeout=5)
            self._thread = None
        if self.window is not None:
            self.window.close()
        self.on_stop()

    def pause(self):
        """Pause the step() loop without stopping the thread."""
        if self.status == BotStatus.RUNNING:
            self.status = BotStatus.PAUSED
            self.log("Paused.")

    def resume(self):
        """Resume from a paused state."""
        if self.status == BotStatus.PAUSED:
            self.status = BotStatus.RUNNING
            self.log("Resumed.")

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        return self.status == BotStatus.RUNNING

    def is_paused(self) -> bool:
        return self.status == BotStatus.PAUSED

    def elapsed(self) -> float:
        """Seconds since the bot was last started."""
        return time.time() - self._start_time

    def log(self, message: str):
        print(f"[{time.strftime('%H:%M:%S')}] {type(self).__name__}: {message}")

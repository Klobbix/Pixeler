import threading
import time
import traceback


class BotThread(threading.Thread):
    """
    Daemon thread that drives a Bot's step() loop cooperatively.

    Cooperative stopping (via a threading.Event) is used instead of async
    exception injection, so the thread exits cleanly even when step() is
    blocked inside a C extension call.
    """

    def __init__(self, bot):
        super().__init__(daemon=True)
        self._bot = bot
        self._stop_event = threading.Event()

    def run(self):
        from pixeler.bot.bot_status import BotStatus
        self._bot.log(f"Thread started (id={self.ident})")
        try:
            while not self._stop_event.is_set():
                if self._bot.status == BotStatus.PAUSED:
                    time.sleep(0.05)
                    continue
                self._bot.step()
        except Exception:
            self._bot.log("Unhandled exception in step():\n" + traceback.format_exc())
            # Signal stop without joining — we ARE the bot thread, so join() would deadlock
            self._stop_event.set()
            self._bot.status = BotStatus.STOPPED
            if self._bot.window is not None:
                self._bot.window.close()
            self._bot.on_stop()
        finally:
            self._bot.log(f"Thread stopped (id={self.ident})")

    def stop(self):
        """Signal the loop to exit after the current step() returns."""
        self._stop_event.set()

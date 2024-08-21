"""
Focuses on an open Notepad window and resizes it.
"""
import cv2

from src.pixeler.bot.bot import Bot
from src.pixeler.input.window import Window


class WindowTracking(Bot):
    def on_start(self):
        pass

    def on_stop(self):
        cv2.destroyAllWindows()

    def loop(self):
        self.window.focus()
        self.window.resize(100, 100)
        self.stop()


if __name__ == '__main__':
    example = WindowTracking(Window.from_title_contains("Notepad"))
    example.start()

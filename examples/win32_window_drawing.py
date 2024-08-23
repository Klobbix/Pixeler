"""
Focuses on an open Notepad window and draws shapes on top of it using the win32 API.
This only works on Windows OS.
"""

import cv2

from src.pixeler.bot.bot import Bot
from src.pixeler.vision.color import GREEN
from src.pixeler.window.win32_window import Win32Window


class Win32WindowDrawing(Bot):
    def on_start(self):
        self.window.focus()

    def on_stop(self):
        cv2.destroyAllWindows()

    def loop(self):
        while True:
            self.window.draw_text(text="Testing out a string", color=GREEN, x=100, y=0)
            self.window.draw_rectangle((100, 100), (250, 250), color=GREEN)


if __name__ == '__main__':
    window = Win32Window("Notepad")
    example = Win32WindowDrawing(window)
    example.start()

"""
Focuses on an open Notepad window and draws shapes on top of it.
This only works on Windows OS.
"""
import cv2

from src.pixeler.bot.bot import Bot
from src.pixeler.input.win32_window import Win32Window
from src.pixeler.vision.color import GREEN


class Win32WindowDrawing(Bot):
    def on_start(self):
        self.window.focus()

    def on_stop(self):
        cv2.destroyAllWindows()

    def loop(self):
        while True:
            self.window.draw_text(text="Testing out a string", color=GREEN)


if __name__ == '__main__':
    example = Win32WindowDrawing(Win32Window.from_title_contains("Notepad"))
    example.start()

"""
Focuses on an open Notepad window and writes into it.
"""
import cv2

import src.pixeler.input.keyboard as keyboard
from src.pixeler.bot.bot import Bot
from src.pixeler.window.window import Window


class NotepadWriting(Bot):
    def on_start(self):
        print(f"Window positioned at: {self.window.position()}")
        self.window.focus()

    def on_stop(self):
        cv2.destroyAllWindows()

    def loop(self):
        keyboard.write("This is a test sentence!")
        keyboard.press("enter")
        self.stop()


if __name__ == '__main__':
    example = NotepadWriting(Window("Notepad"))
    example.start()

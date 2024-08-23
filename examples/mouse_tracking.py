"""
Focuses on an open Notepad window and resizes it.
"""

from src.pixeler.bot.bot import Bot
from src.pixeler.input import mouse


class WindowTracking(Bot):
    def on_start(self):
        pass

    def on_stop(self):
        pass

    def loop(self):
        while True:
            print(mouse.position())

if __name__ == '__main__':
    example = WindowTracking()
    example.start()

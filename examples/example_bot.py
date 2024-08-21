import cv2

from src.pixeler.bot.bot import Bot


class ExampleBot(Bot):
    def on_start(self):
        pass

    def on_stop(self):
        pass

    def loop(self):
        while True:
            # Loops until ESC is pressed
            k = cv2.waitKey(1)
            if k == 27:
                self.stop()


if __name__ == '__main__':
    example = ExampleBot()
    example.start()

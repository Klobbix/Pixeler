"""
Creates a window that can be colored on.
"""
import cv2
import numpy as np

import src.pixeler.vision.oir as oir
from src.pixeler.bot.bot import Bot
from src.pixeler.vision.color import GREEN


class ScreenTracking(Bot):
    def on_start(self):
        pass

    def on_stop(self):
        cv2.destroyAllWindows()

    def loop(self):
        canvas = np.zeros((300, 300, 3), dtype="uint8")
        cv2.imshow("Canvas", canvas)
        while True:
            # print(ocr.extract_text(mat))
            # Loops until ESC is pressed
            k = cv2.waitKey(1)
            if k == 27:
                self.stop()
            # Listen for 'r' key
            if k == 114:
                oir.draw_rectangle(canvas, (0, 0), 100, 100, GREEN)
                cv2.imshow("Canvas", canvas)
            # Listen for 'c' key to clear canvas
            if k == 99:
                canvas = np.zeros((300, 300, 3), dtype="uint8")
                cv2.imshow("Canvas", canvas)


if __name__ == '__main__':
    example = ScreenTracking()
    example.start()

"""
Creates a window that captures the user's monitor for a given bounding box.
"""
import cv2
import numpy as np
from mss import mss

from src.pixeler.bot.bot import Bot
import src.pixeler.vision.ocr as ocr
import src.pixeler.vision.utils as utils


class ScreenTracking(Bot):
    def on_start(self):
        pass

    def on_stop(self):
        cv2.destroyAllWindows()

    def loop(self):
        bounding_box = {'top': 100, 'left': 0, 'width': 400, 'height': 300}
        sct = mss()
        while True:
            sct_img = sct.grab(bounding_box)
            cv2.imshow('screen', np.array(sct_img))
            mat = utils.mss_to_cv2(sct_img)
            print(ocr.extract_text(mat))
            # Loops until ESC is pressed
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                self.stop()


if __name__ == '__main__':
    example = ScreenTracking()
    example.start()

from src.pixeler.bot.bot import Bot
from src.pixeler.vision.utils import *


class ObjectColorTracking(Bot):
    def on_start(self):
        pass

    def on_stop(self):
        cv2.destroyAllWindows()

    def loop(self):
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        while 1:
            # Take each frame
            _, frame = cap.read()

            # Convert BGR to HSV
            hsv = convert(frame, cv2.COLOR_BGR2HSV)

            # define range of color in HSV
            color = Color([100, 50, 100], [130, 255, 255])

            # Threshold the HSV image to get only blue colors
            mask = mask_to_color(hsv, color)

            # Bitwise-AND mask and original image
            res = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow('frame', frame)
            cv2.imshow('mask', mask)
            cv2.imshow('res', res)
            # Loops until ESC is pressed
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                self.stop()


if __name__ == '__main__':
    example = ObjectColorTracking()
    example.start()

"""
Focuses on an open Notepad window and draws a rectangle around a specified template image.
This example provides active object detection from a given template image.
This only works on Windows OS.
"""
from pathlib import Path

import cv2
from mss import mss

from src.pixeler.bot.bot import Bot
from src.pixeler.input.win32_window import Win32Window
from src.pixeler.vision.color import GREEN
from src.pixeler.vision.oir import find_template_in_image, draw_rectangle
from src.pixeler.vision.utils import load_mat_from_file, convert_to_gray


class NotepadObjectFindingCv(Bot):
    logo = None

    def on_start(self):
        self.window.focus()
        # Load a template screenshot of a sprite or other captured image.
        self.logo = load_mat_from_file(Path("images\\template.png"))

    def on_stop(self):
        cv2.destroyAllWindows()

    def loop(self):
        sct = mss()
        h, w = self.logo.shape
        while True:
            # Actively take a screenshot of the focused window
            mat_raw = self.window.screenshot(sct)
            # Convert to gray to match template logo
            mat_gray = convert_to_gray(mat_raw)
            cv2.imshow("ff", mat_raw)
            # Find the logo in the screenshot
            loc, confidence = find_template_in_image(mat_gray, self.logo)
            print(f"loc: {loc}, confidence: {confidence}")
            if confidence >= 0.45:
                # Adjust the size of the found template
                bottom_right = (loc[0] + w, loc[1] + h)
                print(f"Drawing rectangle from {loc} to {bottom_right}")
                draw_rectangle(mat_raw, loc, h, w, GREEN)
                cv2.imshow("ff", mat_raw)

            # Loops until ESC is pressed
            k = cv2.waitKey(1)
            if k == 27:
                self.stop()
                sct.close()


if __name__ == '__main__':
    example = NotepadObjectFindingCv(Win32Window.from_title_contains("Notepad"))
    example.start()

"""
Focuses on an open Notepad window and draws a rectangle around a specified template image.
This example provides active object detection from a given template image.
This only works on Windows OS.
"""
from pathlib import Path

import cv2

from src.pixeler.bot.bot import Bot
from src.pixeler.vision.color import GREEN, Color
from src.pixeler.vision.oir import find_template_in_image, draw_rectangle
from src.pixeler.vision.utils import load_mat_from_file, convert_to_gray
from src.pixeler.window.win32_window import Win32Window


class NotepadObjectFindingWin32(Bot):
    logo = None

    def on_start(self):
        self.window.focus()
        # Load a template screenshot of a sprite or other captured image.
        self.logo = load_mat_from_file(Path("images\\template.png"))

    def on_stop(self):
        cv2.destroyAllWindows()

    def loop(self):
        h, w = self.logo.shape
        while True:
            # Actively take a screenshot of the focused window
            mat_raw = self.window.screenshot()
            # Convert to gray to match template logo
            mat_gray = convert_to_gray(mat_raw)
            cv2.imshow("ff", mat_raw)
            # Find the logo in the screenshot
            loc, confidence = find_template_in_image(mat_gray, self.logo)
            #print(f"loc: {loc}, confidence: {confidence}")
            if confidence >= 0.85:
                # Adjust the size of the found template
                bottom_right = (loc[0] + w, loc[1] + h)
                draw_rectangle(mat_raw, loc, h, w, GREEN)
                #self.window.draw_rectangle(loc, bottom_right, GREEN)
                self.window.draw_text("Testing", GREEN, 100,20)
                self.window.draw_rectangle_outline((100, 100), (250, 250), color=GREEN)
                cv2.imshow("ff", mat_raw)

            # Loops until ESC is pressed
            k = cv2.waitKey(1)
            if k == 27:
                self.stop()


if __name__ == '__main__':
    example = NotepadObjectFindingWin32(Win32Window("Notepad"))
    example.start()

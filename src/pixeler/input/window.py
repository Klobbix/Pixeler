import cv2
import numpy as np
import pywinctl
from mss import mss
from pygetwindow import BaseWindow
from pywinbox import Point


class Window:
    def __init__(self, handle: BaseWindow):
        self.handle = handle

    @classmethod
    def from_title(cls, title: str) -> 'Window':
        """
        Creates a new Window instance from an exact title.
        :param title: The title of the window.
        :return: A new Window instance.
        """
        windows = pywinctl.getWindowsWithTitle(title)
        if windows:
            return cls(windows[0])
        else:
            raise Exception("No window with title: " + title)

    @classmethod
    def from_title_contains(cls, title: str) -> 'Window':
        """
       Creates a new Window instance from a title that contains a given text.
       :param title: The title of the window to search for.
       :return: A new Window instance.
       """
        windows = pywinctl.getWindowsWithTitle(title, condition=pywinctl.Re.CONTAINS)
        if windows:
            return cls(windows[0])
        else:
            raise Exception("No window with title: " + title)

    def __destroy(self):
        pass

    def focus(self):
        """
        Focuses on the instanced window object.
        :return:
        """
        if self.handle:
            self.handle.activate()

    def maximize(self):
        """
        Maximizes the instanced window object.
        :return:
        """
        if self.handle:
            self.handle.maximize()

    def minimize(self):
        """
        Minimizes the instanced window object.
        :return:
        """
        if self.handle:
            self.handle.minimize()

    def move(self, x: int, y: int):
        """
        Moves the instanced window object.
        :param x: The x position of the window.
        :param y: The y position of the window.
        :return:
        """
        if self.handle:
            self.handle.moveTo(x, y)

    def close(self):
        """
        Closes this window.
        This may trigger "Are you sure you want to quit?" dialogs or other actions that prevent the window from actually closing.
        This is identical to clicking the X button on the window.
        :return:
        """
        if self.handle:
            self.handle.close()

    def position(self) -> Point:
        """
        Returns the current position of the window.
        :return: The position of the window.
        """
        if self.handle:
            return self.handle.position

    def resize(self, width: int, height: int):
        """
        Resizes the window to the given width and height.
        :param width: The width of the window.
        :param height: The height of the window.
        :return:
        """
        if self.handle:
            self.handle.size = (width, height)

    def screenshot(self, sct: mss) -> cv2.Mat:
        """
        Takes a screenshot of the window and returns it.
        :param sct: The mss sct object
        :return: A Mat of the window.
        """
        if self.handle:
            box = {'top': self.handle.top, 'left': self.handle.left, 'width': self.handle.size.width,
                   'height': self.handle.height}
            shot = sct.grab(box)
            return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

    def screenshot_monitor(self, sct: mss, monitor: int = 1) -> cv2.Mat:
        """
        Takes a screenshot of the entire monitor and returns it.
        :param sct: The mss sct object
        :return: A Mat of the window.
        """
        if self.handle:
            # Get information of monitor 2
            mon = sct.monitors[monitor]

            # The screen part to capture
            box = {
                "top": mon["top"] + 100,  # 100px from the top
                "left": mon["left"] + 100,  # 100px from the left
                "width": mon["width"],
                "height": mon["height"],
                "mon": monitor,
            }
            shot = sct.grab(box)
            return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

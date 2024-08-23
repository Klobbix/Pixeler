import cv2
import numpy as np
import pywinctl
from mss import mss
from pywinbox import Point

from src.pixeler.window.abstract_window import AbstractWindow


class Window(AbstractWindow):
    def __init__(self, title: str):
        self.mss = mss()
        self.handle = self.__from_title(title)

    def __from_title(self, title: str, condition: int = pywinctl.Re.CONTAINS):
        """
        Creates a new BaseWindow instance from a title.
        :param title: The title of the window.
        :param condition: Window finding flags
        :return: A new Window instance.
        """
        windows = pywinctl.getWindowsWithTitle(title, condition=condition)
        if windows:
            return windows[0]
        else:
            raise Exception("No window with title: " + title)

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
        self.mss.close()
        if self.handle:
            self.handle.close()

    def width(self):
        """
        Returns the width of the window.
        :return: The width of the window.
        """
        if self.handle:
            return self.handle.width

    def height(self):
        """
        Returns the height of the window.
        :return: The height of the window.
        """
        if self.handle:
            return self.handle.height

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

    def screenshot(self) -> cv2.Mat:
        """
        Takes a screenshot of the window and returns it.
        :return: A Mat of the window.
        """
        if self.handle:
            box = {'top': self.handle.top, 'left': self.handle.left, 'width': self.handle.size.width,
                   'height': self.handle.height}
            shot = self.mss.grab(box)
            return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

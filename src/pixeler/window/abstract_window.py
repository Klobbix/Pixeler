from abc import ABC, abstractmethod

import cv2


class AbstractWindow(ABC):
    @abstractmethod
    def focus(self):
        """
        Focuses on the instanced window object.
        """
        pass

    @abstractmethod
    def maximize(self):
        """
        Maximizes the instanced window object.
        """
        pass

    @abstractmethod
    def minimize(self):
        """
        Minimizes the instanced window object.
        """
        pass

    @abstractmethod
    def move(self, x: int, y: int):
        """
        Moves the instanced window object.
        :param x: The x position of the window.
        :param y: The y position of the window.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Closes the window.
        """
        pass

    @abstractmethod
    def position(self):
        """
        Returns the current top-left position of the window.
        :return: The position of the window.
        """
        pass

    @abstractmethod
    def width(self):
        """
        Returns the width of the window.
        :return: The width of the window.
        """
        pass

    @abstractmethod
    def height(self):
        """
        Returns the height of the window.
        :return: The height of the window.
        """
        pass

    @abstractmethod
    def resize(self, width: int, height: int):
        """
        Resizes the window to the given width and height.
        :param width: The width of the window.
        :param height: The height of the window.
        """
        pass

    @abstractmethod
    def screenshot(self) -> cv2.Mat:
        """
        Takes a screenshot of the window and returns it.
        :return: A Mat of the window.
        """
        pass

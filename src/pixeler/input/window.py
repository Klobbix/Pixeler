import pywinctl
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

    def focus(self):
        """
        Focuses on the instanced window object.
        :return:
        """
        if self.handle:
            self.handle.activate()

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

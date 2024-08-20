import pywinctl
from pygetwindow import BaseWindow
from pywinbox import Point


class Window:
    def __init__(self, handle: BaseWindow):
        self.handle = handle

    @classmethod
    def from_title(cls, title: str) -> 'Window':
        windows = pywinctl.getWindowsWithTitle(title)
        if windows:
            return cls(windows[0])
        else:
            raise Exception("No window with title: " + title)

    def focus(self) -> bool:
        if self.handle:
            return self.handle.activate()

    def close(self) -> bool:
        if self.handle:
            return self.handle.close()

    def position(self) -> Point:
        if self.handle:
            return self.handle.position

    def resize(self, width: int, height: int):
        if self.handle:
            self.handle.size = (width, height)

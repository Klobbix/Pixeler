from typing import Union

import pywinctl
from pygetwindow import Win32Window, MacOSWindow
from pywinbox import Point
from pywinctl._pywinctl_linux import LinuxWindow


class Window:
    def __init__(self, handle: Union[Win32Window, LinuxWindow, MacOSWindow]):
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

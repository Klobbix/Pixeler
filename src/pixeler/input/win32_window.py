import pywinctl
import win32api
import win32con
import win32gui
from pygetwindow import BaseWindow

from src.pixeler.input.window import Window
from src.pixeler.vision.color import Color


class Win32Window(Window):
    def __init__(self, handle: BaseWindow):
        super().__init__(handle)
        self.hwnd = None

    @classmethod
    def from_title(cls, title: str) -> 'Win32Window':
        """
        Creates a new Win32Window instance for a Windows OS window from an exact title.
        :param title: The title of the window.
        :return: A new Win32Window instance.
        """
        windows = pywinctl.getWindowsWithTitle(title)
        if windows:
            return cls(windows[0])
        else:
            raise Exception("No window with title: " + title)

    @classmethod
    def from_title_contains(cls, title: str) -> 'Win32Window':
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

    def get_hwnd(self):
        """ Get the handle of the currently active (foreground) window """
        if self.hwnd is None:
            self.focus()
            self.hwnd = win32gui.GetForegroundWindow()
        return self.hwnd

    def draw_text(self, text: str, color: Color, flags: int = win32con.DT_LEFT | win32con.DT_TOP):
        # Set the device context
        hdc = win32gui.GetDC(self.get_hwnd())

        # Set the text color (using RGB values)
        win32gui.SetTextColor(hdc, win32api.RGB(color.lower[0], color.lower[1], color.lower[2]))

        # Set the background mode to transparent
        win32gui.SetBkMode(hdc, win32con.TRANSPARENT)

        # Draw the text at the specified position
        win32gui.DrawText(hdc, text, -1, win32gui.GetClientRect(self.get_hwnd()), flags)

        # Release the device context (important to avoid memory leaks)
        win32gui.ReleaseDC(self.get_hwnd(), hdc)

"""
Class for targeting Windows OS specific windows.
"""
import cv2
import numpy as np
import pywinctl
import win32api
import win32con
import win32gui
from mss import mss
from pygetwindow import BaseWindow

from src.pixeler.input.window import Window
from src.pixeler.vision.color import Color


class Win32Window(Window):
    def __init__(self, handle: BaseWindow):
        super().__init__(handle)
        self.hwnd = None
        self.hdc = None
        self.shapes = dict()

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

    def __destroy(self):
        win32gui.ReleaseDC(self.get_hwnd(), self.get_hdc())

    def get_hwnd(self):
        """ Get the handle of the currently active (foreground) window """
        if self.hwnd is None:
            self.focus()
            self.hwnd = win32gui.GetForegroundWindow()
        return self.hwnd

    def get_hdc(self):
        """ Get the device context of the currently active (foreground) window """
        if self.hdc is None:
            self.hdc = win32gui.GetDC(self.get_hwnd())
        return self.hdc

    def get_window_rect(self) -> (int, int, int, int):
        return win32gui.GetWindowRect(self.get_hwnd())

    def screenshot(self, sct: mss) -> cv2.Mat:
        """
        Takes a screenshot of the window and returns it.
        :param sct: The mss sct object
        :return: A Mat of the window.
        """
        if self.handle:
            # Get the window's client rectangle (excluding title bar and borders)
            client_rect = win32gui.GetClientRect(self.get_hwnd())
            client_top_left = self.handle.position
            client_bottom_right = win32gui.ClientToScreen(self.get_hwnd(), (client_rect[2], client_rect[3]))

            # Calculate the width and height of the client area
            window_width = client_bottom_right[0] - client_top_left[0]
            window_height = client_bottom_right[1] - client_top_left[1]

            # Define the box for capturing
            box = {
                'top': client_top_left[1],
                'left': client_top_left[0],
                'width': window_width,
                'height': window_height
            }

            # Capture the screenshot of the defined box
            shot = sct.grab(box)

            # Convert the screenshot to a format usable by OpenCV
            return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)


def draw_text(self, text: str, color: Color, flags: int = win32con.DT_LEFT | win32con.DT_TOP):
    hdc = self.get_hdc()
    # Set the text color (using RGB values)
    win32gui.SetTextColor(hdc, win32api.RGB(color.lower[0], color.lower[1], color.lower[2]))

    # Set the background mode to transparent
    win32gui.SetBkMode(hdc, win32con.TRANSPARENT)

    # Draw the text at the specified position
    win32gui.DrawText(hdc, text, -1, win32gui.GetClientRect(self.get_hwnd()), flags)


def draw_rectangle(self, top_left: tuple, bottom_right: tuple, color: Color):
    if len(top_left) != 2 and len(bottom_right) != 2:
        raise ValueError("Coordinates must be of length 2 (X, Y).")

    hdc = self.get_hdc()
    # Create a pen with the desired color
    pen = win32gui.CreatePen(win32con.PS_SOLID, 1, win32api.RGB(color.lower[0], color.lower[1], color.lower[2]))
    win32gui.SelectObject(hdc, pen)

    win32gui.Rectangle(hdc, top_left[0], top_left[1], bottom_right[0], bottom_right[1])
    # Clean up resources
    win32gui.DeleteObject(pen)


def draw_rectangle_outline(self, top_left: tuple, bottom_right: tuple, color: Color):
    if len(top_left) != 2 and len(bottom_right) != 2:
        raise ValueError("Coordinates must be of length 2 (X, Y).")

    hdc = win32gui.GetDC(self.get_hwnd())

    # Create a pen with the desired color
    pen = win32gui.CreatePen(win32con.PS_SOLID, 3, win32api.RGB(color.lower[0], color.lower[1], color.lower[2]))
    win32gui.SelectObject(hdc, pen)

    # Set the brush to NULL_BRUSH to avoid filling the rectangle
    null_brush = win32gui.GetStockObject(win32con.NULL_BRUSH)
    brush = win32gui.SelectObject(hdc, null_brush)

    # Draw the rectangle outline on the device context
    win32gui.Rectangle(hdc, top_left[0], top_left[1], bottom_right[0], bottom_right[1])

    # Clean up resources
    win32gui.DeleteObject(pen)
    win32gui.DeleteObject(null_brush)
    win32gui.DeleteObject(brush)
    win32gui.ReleaseDC(self.hwnd, hdc)


def clear(self):
    win32gui.InvalidateRect(self.get_hwnd(), None, True)
    win32gui.UpdateWindow(self.get_hwnd())


def clear_area(self, top_left: tuple, bottom_right: tuple):
    win32gui.InvalidateRect(self.get_hwnd(), (top_left[0], top_left[1], bottom_right[0], bottom_right[1]), True)
    win32gui.UpdateWindow(self.get_hwnd())

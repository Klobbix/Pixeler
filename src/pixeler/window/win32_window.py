"""
Class for targeting Windows OS specific windows.
"""

import cv2
import numpy as np
import win32api
import win32con
import win32gui
from mss import mss

from src.pixeler.vision.color import Color
from src.pixeler.window.abstract_window import AbstractWindow
from src.pixeler.window.overlay import Overlay


class Win32Window(AbstractWindow):
    def __init__(self, title: str):
        self.hwnd = self.__from_title(title)
        self.overlay = None
        self.hdc = None
        self.mss = None
        self.rectangles = {}
        self.pens = {}
        self.brushes = {}

    def __from_title(cls, title: str):
        """
        Finds a window whose title contains the given substring.
        :param title: The title to search for.
        :return: The handle (HWND) of the window if found, None otherwise.
        """

        def enum_windows_proc(hwnd, result_list):
            """Callback function for win32gui.EnumWindows."""
            window_title = win32gui.GetWindowText(hwnd)
            if title.lower() in window_title.lower():
                result_list.append(hwnd)

        # List to store matching window handles
        found_windows = []

        # Enumerate all top-level windows and apply the callback function
        win32gui.EnumWindows(enum_windows_proc, found_windows)

        if found_windows:
            print(found_windows)
            # Return the first matching window handle, or modify to return all
            return found_windows[0]
        else:
            return None

    def create_overlay(self):
        self.overlay = Overlay(self.hwnd)

    def focus(self):
        win32gui.SetForegroundWindow(self.hwnd)

    def maximize(self):
        win32gui.ShowWindow(self.hwnd, win32con.SW_MAXIMIZE)

    def minimize(self):
        win32gui.ShowWindow(self.hwnd, win32con.SW_MINIMIZE)

    def move(self, x: int, y: int):
        # Windows 10 has an invisible border of 7 pixels
        win32gui.MoveWindow(self.hwnd, x - 7, y, self.width(), self.height(), True)

    def close(self):
        if self.mss:
            self.mss.close()
        for pen in self.pens:
            win32gui.DeleteObject(pen)
        for brush in self.brushes:
            win32gui.DeleteObject(brush)
        win32gui.ReleaseDC(self.hwnd, self.hdc)

    def width(self):
        """
        Returns the width of the window.
        :return: The width of the window.
        """
        pos = self.position()
        return pos[2] - pos[0]

    def height(self):
        """
        Returns the height of the window.
        :return: The height of the window.
        """
        pos = self.position()
        return pos[3] - pos[1]

    def position(self) -> (int, int, int, int):
        return win32gui.GetWindowRect(self.hwnd)

    def resize(self, width: int, height: int):
        pos = self.position()
        win32gui.MoveWindow(self.hwnd, pos[0] - 7, pos[1], width, height, True)

    def get_hdc(self):
        """ Get the device context of the currently active (foreground) window """
        if self.hdc is None:
            self.hdc = win32gui.GetDC(self.hwnd)
        return self.hdc

    def screenshot(self) -> cv2.Mat:
        """
        Takes a screenshot of the window and returns it.
        :param sct: The mss sct object
        :return: A Mat of the window.
        """
        # Get the window's client rectangle (excluding title bar and borders)
        if self.mss is None:
            self.mss = mss()
        pos = self.position()

        # Define the box for capturing
        box = {
            'top': pos[1],
            'left': pos[0],
            'width': self.width(),
            'height': self.height()
        }

        # Capture the screenshot of the defined box
        shot = self.mss.grab(box)

        # Convert the screenshot to a format usable by OpenCV
        return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

    def get_cached_pen(self, color: Color):
        color_tuple = (color.lower[0], color.lower[1], color.lower[2])
        if color_tuple not in self.pens:
            pen = win32gui.CreatePen(win32con.PS_SOLID, 1, win32api.RGB(*color_tuple))
            self.pens[color_tuple] = pen
        return self.pens[color_tuple]

    def get_cached_brush(self, color: Color):
        color_tuple = (color.lower[0], color.lower[1], color.lower[2])
        if color_tuple not in self.brushes:
            brush = win32gui.CreateSolidBrush(win32api.RGB(*color_tuple))
            self.brushes[color_tuple] = brush
        return self.brushes[color_tuple]

    def draw_text(self, text: str, color: Color, x: int, y: int, flags: int = 0):
        pos = win32gui.GetClientRect(self.hwnd)
        adjusted = (pos[0] + x, pos[1] + y, pos[2], pos[3])

        hdc = self.get_hdc()
        # Set the text color (using RGB values)
        win32gui.SetTextColor(hdc, win32api.RGB(color.lower[0], color.lower[1], color.lower[2]))

        # Set the background mode to transparent
        win32gui.SetBkMode(hdc, win32con.TRANSPARENT)

        # Draw the text at the specified position
        win32gui.DrawText(hdc, text, -1, adjusted, flags)

    def draw_rectangle(self, top_left: tuple, bottom_right: tuple, color: Color):
        if len(top_left) != 2 and len(bottom_right) != 2:
            raise ValueError("Coordinates must be of length 2 (X, Y).")

        key = (top_left, bottom_right, color)
        if key in self.rectangles:
            return

        self.rectangles[key] = True
        hdc = self.get_hdc()
        pen = self.get_cached_pen(color)
        brush = self.get_cached_brush(color)

        win32gui.SelectObject(hdc, pen)
        win32gui.SelectObject(hdc, brush)

        win32gui.Rectangle(hdc, top_left[0], top_left[1], bottom_right[0], bottom_right[1])

    def draw_rectangle_outline(self, top_left: tuple, bottom_right: tuple, color: Color):
        if len(top_left) != 2 and len(bottom_right) != 2:
            raise ValueError("Coordinates must be of length 2 (X, Y).")

        key = (top_left, bottom_right, color)
        if key in self.rectangles:
            return

        self.rectangles[key] = True
        hdc = self.get_hdc()
        pen = self.get_cached_pen(color)
        null_brush = win32gui.GetStockObject(win32con.NULL_BRUSH)

        win32gui.SelectObject(hdc, pen)
        win32gui.SelectObject(hdc, null_brush)

        # Draw the rectangle outline on the device context
        win32gui.Rectangle(hdc, top_left[0], top_left[1], bottom_right[0], bottom_right[1])

        # Clean up resources
        win32gui.DeleteObject(null_brush)

    def clear_screen(self):
        win32gui.InvalidateRect(self.hwnd, None, True)
        win32gui.UpdateWindow(self.hwnd)

    def clear_area(self, top_left: tuple, bottom_right: tuple):
        win32gui.InvalidateRect(self.hwnd, (top_left[0], top_left[1], bottom_right[0], bottom_right[1]), True)
        win32gui.UpdateWindow(self.hwnd)

    def clear_drawn_cache(self):
        self.rectangles.clear()

    def redraw_rectangles(self):
        for rect_key in self.rectangles:
            top_left, bottom_right, color = rect_key
            self.draw_rectangle(top_left, bottom_right, Color(*color))

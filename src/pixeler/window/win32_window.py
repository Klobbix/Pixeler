"""
Win32-specific window handle using the native HWND.

Prefer this over Window when you need the Overlay, direct GDI drawing onto
the game window itself, or precise control over Win32 window behaviour.
"""

import ctypes as _ctypes

import cv2
import numpy as np
import win32con
import win32gui
from mss import mss

from pixeler.window.abstract_window import AbstractWindow

# Make the process per-monitor DPI aware so that GetWindowRect, GDI drawing,
# and MSS screen capture all operate in the same physical-pixel coordinate
# space.  Must happen before any window or GDI operations in the process.
#
# We try the three APIs in order from newest to oldest. The older calls
# return HRESULTs (not exceptions) when the process has already been marked
# DPI-aware by someone else, so a non-zero return on SetProcessDpiAwareness
# is not a real failure — we just move on.
def _init_dpi_awareness() -> None:
    # Windows 10 1703+: DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 (-4)
    try:
        if _ctypes.windll.user32.SetProcessDpiAwarenessContext(-4):
            return
    except (AttributeError, OSError):
        pass
    # Windows 8.1+: PROCESS_PER_MONITOR_DPI_AWARE (2)
    try:
        _ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except (AttributeError, OSError):
        pass
    # Windows Vista+: system-DPI aware
    try:
        _ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


_init_dpi_awareness()


class Win32Window(AbstractWindow):

    def __init__(self, title: str):
        self.hwnd: int = self._hwnd_from_title(title)
        self._mss: mss | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hwnd_from_title(title: str) -> int:
        """Find the first top-level window whose title contains *title*."""
        found: list[int] = []

        def _callback(hwnd: int, _) -> bool:
            if title.lower() in win32gui.GetWindowText(hwnd).lower():
                found.append(hwnd)
            return True

        win32gui.EnumWindows(_callback, None)
        if not found:
            raise ValueError(f"No window found with title containing '{title}'")
        return found[0]

    def _mss_grab(self) -> cv2.Mat:
        if self._mss is None:
            self._mss = mss()
        x, y = win32gui.ClientToScreen(self.hwnd, (0, 0))
        _, _, w, h = win32gui.GetClientRect(self.hwnd)
        box = {'top': y, 'left': x, 'width': w, 'height': h}
        shot = self._mss.grab(box)
        return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

    # ------------------------------------------------------------------
    # AbstractWindow implementation
    # ------------------------------------------------------------------

    def focus(self) -> None:
        win32gui.SetForegroundWindow(self.hwnd)

    def maximize(self) -> None:
        win32gui.ShowWindow(self.hwnd, win32con.SW_MAXIMIZE)

    def minimize(self) -> None:
        win32gui.ShowWindow(self.hwnd, win32con.SW_MINIMIZE)

    def move(self, x: int, y: int) -> None:
        win32gui.MoveWindow(self.hwnd, x, y, self.width(), self.height(), True)

    def close(self) -> None:
        if self._mss is not None:
            self._mss.close()
            self._mss = None

    def position(self) -> tuple[int, int, int, int]:
        return win32gui.GetWindowRect(self.hwnd)

    def width(self) -> int:
        l, t, r, b = self.position()
        return r - l

    def height(self) -> int:
        l, t, r, b = self.position()
        return b - t

    def resize(self, width: int, height: int) -> None:
        l, t, _, _ = self.position()
        win32gui.MoveWindow(self.hwnd, l, t, width, height, True)

    def is_visible(self) -> bool:
        return (win32gui.IsWindowVisible(self.hwnd)
                and not win32gui.IsIconic(self.hwnd))

    def title(self) -> str:
        return win32gui.GetWindowText(self.hwnd)

    def screenshot(self) -> cv2.Mat:
        if not self.is_visible():
            raise RuntimeError(
                f"Cannot screenshot window '{self.title()}' — it is minimised or hidden"
            )
        return self._mss_grab()

    # ------------------------------------------------------------------
    # Overlay factory
    # ------------------------------------------------------------------

    def create_overlay(self) -> 'Overlay':
        """
        Create and start a transparent overlay window over this window.

        Returns a ready-to-use Overlay instance. Call begin_frame() /
        drawing methods / end_frame() in your bot's step() loop.
        """
        from pixeler.window.overlay import Overlay
        overlay = Overlay(self.hwnd)
        overlay.start()
        return overlay

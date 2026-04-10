import cv2
import numpy as np
import pywinctl
from mss import mss

from pixeler.window.abstract_window import AbstractWindow


class Window(AbstractWindow):
    """
    Cross-platform window handle using pywinctl + mss for screen capture.
    Use this when you do not need Win32-specific drawing or overlays.
    """

    def __init__(self, title: str):
        self._mss = mss()
        self.handle = self._from_title(title)

    def _from_title(self, title: str, condition: int = pywinctl.Re.CONTAINS):
        windows = pywinctl.getWindowsWithTitle(title, condition=condition)
        if not windows:
            raise ValueError(f"No window found with title containing '{title}'")
        return windows[0]

    # ------------------------------------------------------------------
    # AbstractWindow implementation
    # ------------------------------------------------------------------

    def focus(self) -> None:
        self.handle.activate()

    def maximize(self) -> None:
        self.handle.maximize()

    def minimize(self) -> None:
        self.handle.minimize()

    def move(self, x: int, y: int) -> None:
        self.handle.moveTo(x, y)

    def close(self) -> None:
        self._mss.close()
        self.handle.close()

    def position(self) -> tuple[int, int, int, int]:
        r = self.handle.rect
        return (r.left, r.top, r.right, r.bottom)

    def width(self) -> int:
        return self.handle.width

    def height(self) -> int:
        return self.handle.height

    def resize(self, width: int, height: int) -> None:
        self.handle.size = (width, height)

    def is_visible(self) -> bool:
        return self.handle.isActive and not self.handle.isMinimized

    def title(self) -> str:
        return self.handle.title

    def screenshot(self) -> cv2.Mat:
        """
        Capture the window contents. Returns the last known frame if the
        window is currently minimised (mss cannot grab a minimised window).
        """
        if self.handle.isMinimized:
            raise RuntimeError(
                f"Cannot screenshot minimised window '{self.handle.title}'"
            )
        box = {
            'top': self.handle.top,
            'left': self.handle.left,
            'width': self.handle.size.width,
            'height': self.handle.height,
        }
        shot = self._mss.grab(box)
        return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

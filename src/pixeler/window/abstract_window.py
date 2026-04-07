from abc import ABC, abstractmethod

import cv2


class AbstractWindow(ABC):

    @abstractmethod
    def focus(self) -> None:
        """Bring the window to the foreground."""

    @abstractmethod
    def maximize(self) -> None:
        """Maximise the window."""

    @abstractmethod
    def minimize(self) -> None:
        """Minimise the window."""

    @abstractmethod
    def move(self, x: int, y: int) -> None:
        """Move the window's top-left corner to (x, y) in screen coordinates."""

    @abstractmethod
    def close(self) -> None:
        """Release resources and, where applicable, close the window."""

    @abstractmethod
    def position(self) -> tuple[int, int, int, int]:
        """Return the window's screen rect as (left, top, right, bottom)."""

    @abstractmethod
    def width(self) -> int:
        """Return the window width in pixels."""

    @abstractmethod
    def height(self) -> int:
        """Return the window height in pixels."""

    @abstractmethod
    def resize(self, width: int, height: int) -> None:
        """Resize the window to the given dimensions."""

    @abstractmethod
    def screenshot(self) -> cv2.Mat:
        """Capture and return the window contents as a BGR cv2.Mat."""

    @abstractmethod
    def is_visible(self) -> bool:
        """Return True if the window exists and is not minimised."""

    @abstractmethod
    def title(self) -> str:
        """Return the current window title string."""

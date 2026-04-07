from typing import List, Optional

import cv2
import mss
import numpy as np
import pyautogui

import pixeler.math.random as rd
from pixeler.math.point import Point


class Rectangle:
    """Axis-aligned bounding rectangle defined by top-left corner (x, y) and dimensions (w, h)."""

    def __init__(self, x: float, y: float, w: float, h: float):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_coords(cls, x: int, y: int, width: int, height: int) -> 'Rectangle':
        monitor = pyautogui.size()
        width = min(width, monitor.width - x)
        height = min(height, monitor.height - y)
        return cls(x, y, width, height)

    @classmethod
    def from_point(cls, point: Point, width: int, height: int) -> 'Rectangle':
        monitor = pyautogui.size()
        width = min(width, monitor.width - point.x)
        height = min(height, monitor.height - point.y)
        return cls(point.x, point.y, width, height)

    @classmethod
    def from_points(cls, start_point: Point, end_point: Point) -> 'Rectangle':
        return cls(
            start_point.x,
            start_point.y,
            end_point.x - start_point.x,
            end_point.y - start_point.y,
        )

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def get_center(self) -> Point:
        return Point(self.x + self.w / 2, self.y + self.h / 2)

    def get_top_left(self) -> Point:
        return Point(self.x, self.y)

    def get_top_right(self) -> Point:
        return Point(self.x + self.w, self.y)

    def get_bottom_left(self) -> Point:
        return Point(self.x, self.y + self.h)

    def get_bottom_right(self) -> Point:
        return Point(self.x + self.w, self.y + self.h)

    def area(self) -> float:
        return self.w * self.h

    def contains_point(self, point: Point) -> bool:
        return (self.x <= point.x <= self.x + self.w and
                self.y <= point.y <= self.y + self.h)

    def intersects(self, other: 'Rectangle') -> bool:
        """Return True if this rectangle overlaps with *other*."""
        return (self.x < other.x + other.w and
                self.x + self.w > other.x and
                self.y < other.y + other.h and
                self.y + self.h > other.y)

    def intersection(self, other: 'Rectangle') -> Optional['Rectangle']:
        """
        Return the overlapping rectangle, or None if they do not intersect.
        """
        x = max(self.x, other.x)
        y = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)
        if x2 <= x or y2 <= y:
            return None
        return Rectangle(x, y, x2 - x, y2 - y)

    def union(self, other: 'Rectangle') -> 'Rectangle':
        """Return the smallest rectangle that contains both rectangles."""
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        x2 = max(self.x + self.w, other.x + other.w)
        y2 = max(self.y + self.h, other.y + other.h)
        return Rectangle(x, y, x2 - x, y2 - y)

    def expand(self, amount: float) -> 'Rectangle':
        """Return a rectangle grown by *amount* pixels on all sides."""
        return Rectangle(self.x - amount, self.y - amount,
                         self.w + amount * 2, self.h + amount * 2)

    def shrink(self, amount: float) -> 'Rectangle':
        """Return a rectangle shrunk by *amount* pixels on all sides."""
        return self.expand(-amount)

    def scale(self, factor: float) -> 'Rectangle':
        """Return a rectangle scaled about its center by *factor*."""
        cx, cy = self.get_center()
        new_w = self.w * factor
        new_h = self.h * factor
        return Rectangle(cx - new_w / 2, cy - new_h / 2, new_w, new_h)

    # ------------------------------------------------------------------
    # Random sampling
    # ------------------------------------------------------------------

    def random_point(self, custom_seeds: List[List[int]] = None) -> Point:
        """
        Return a random point inside this rectangle using the seeded
        truncated-normal distribution (biased toward the centre, like a
        human targeting a UI element).
        """
        if custom_seeds is None:
            center = self.get_center()
            custom_seeds = rd.random_seeds(mod=int(center.x + center.y))
        x, y = rd.random_point_in(self.x, self.y, self.w, self.h, custom_seeds)
        return Point(x, y)

    # ------------------------------------------------------------------
    # Screen capture
    # ------------------------------------------------------------------

    def screenshot(self, save_path: str = None) -> cv2.Mat:
        with mss.mss() as sct:
            monitor = {
                "top": int(self.y),
                "left": int(self.x),
                "width": int(self.w),
                "height": int(self.h),
            }
            sct_img = sct.grab(monitor)
            img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
            if save_path:
                cv2.imwrite(save_path, img)
            return img

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.w, "height": self.h}

    def __repr__(self) -> str:
        return f"Rectangle(x={self.x}, y={self.y}, w={self.w}, h={self.h})"

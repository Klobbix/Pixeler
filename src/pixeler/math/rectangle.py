from typing import List

import cv2
import mss
import numpy as np
import pyautogui

import src.pixeler.math.random as rd
from src.pixeler.math.point import Point


class Rectangle:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    @classmethod
    def from_coords(cls, x: int, y: int, width: int, height: int) -> 'Rectangle':
        # Get monitor dimensions
        monitor = pyautogui.size()
        max_width = monitor.width
        max_height = monitor.height

        # Ensure the width and height do not exceed monitor dimensions
        width = min(width, max_width - x)
        height = min(height, max_height - y)
        return cls(x, y, width, height)

    @classmethod
    def from_point(cls, point: Point, width: int, height: int) -> 'Rectangle':
        # Get monitor dimensions
        monitor = pyautogui.size()
        max_width = monitor.width
        max_height = monitor.height

        # Ensure the width and height do not exceed monitor dimensions
        width = min(width, max_width - point.x)
        height = min(height, max_height - point.y)
        return cls(point.x, point.y, width, height)

    @classmethod
    def from_points(cls, start_point: Point, end_point: Point) -> 'Rectangle':
        return cls(
            start_point.x,
            start_point.y,
            end_point.x - start_point.x,
            end_point.y - start_point.y,
        )

    def random_point(self, custom_seeds: List[List[int]] = None) -> Point:
        if custom_seeds is None:
            center = self.get_center()
            custom_seeds = rd.random_seeds(mod=(center[0] + center[1]))
        x, y = rd.random_point_in(self.x, self.y, self.w, self.h, custom_seeds)
        return Point(x, y)

    def get_center(self) -> Point:
        return Point(self.x + self.w // 2, self.y + self.h // 2)

    def get_top_left(self) -> Point:
        return Point(self.x, self.y)

    def get_top_right(self) -> Point:
        return Point(self.x + self.w, self.y)

    def get_bottom_left(self) -> Point:
        return Point(self.x, self.y + self.h)

    def get_bottom_right(self) -> Point:
        return Point(self.x + self.w, self.y + self.h)

    def contains_point(self, point: Point) -> bool:
        return (self.x <= point.x <= self.x + self.w) and (self.y <= point.y <= self.y + self.h)

    def screenshot(self, save_path: str = None) -> cv2.Mat:
        with mss.mss() as sct:
            # Define the bounding box for the region to capture
            monitor = {
                "top": self.y,
                "left": self.x,
                "width": self.w,
                "height": self.h
            }
            # Capture the region
            sct_img = sct.grab(monitor)

            # Convert the raw image to a numpy array (OpenCV format)
            img_np = np.array(sct_img)

            # Convert the color from BGRA to BGR (if necessary)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)

            # Save the image if a save_path is provided
            if save_path:
                cv2.imwrite(save_path, img_np)

            return img_np

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "width": self.w,
            "height": self.h,
        }

    def __str__(self):
        return f"Rectangle(x={self.x}, y={self.y}, w={self.w}, h={self.h})"

    def __repr__(self):
        return self.__str__()

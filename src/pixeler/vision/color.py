from typing import Sequence

import numpy as np


class Color:
    def __init__(self, lower: Sequence[float], upper: Sequence[float] = None):
        """
        Defines a color or range of colors. This class converts RGB colors to BGR to satisfy OpenCV's color format.
        Args:
            lower: The lower bound of the color range [R, G, B].
            upper: The upper bound of the color range [R, G, B]. Exclude this arg if you're defining a solid color.
        """
        self.lower = np.array(lower[::-1])
        self.upper = np.array(upper[::-1]) if upper else np.array(lower[::-1])

"""Solid colors"""
BLACK = Color([0, 0, 0])
BLUE = Color([0, 0, 255])
CYAN = Color([0, 255, 255])
GREEN = Color([0, 255, 0])
ORANGE = Color([255, 144, 64])
PINK = Color([255, 0, 231])
PURPLE = Color([170, 0, 255])
RED = Color([255, 0, 0])
WHITE = Color([255, 255, 255])
YELLOW = Color([255, 255, 0])
from typing import Sequence

import cv2
import numpy as np


class Color:
    """
    Solid color for **drawing** (overlay, cv2 drawing functions).

    Stores values in BGR order as required by OpenCV and GDI.
    Initialise with RGB — the conversion is handled internally.

    Example::

        RED   = Color([255, 0, 0])    # RGB → stored as BGR [0, 0, 255]
        RANGE = Color([200, 0, 0], [255, 80, 80])  # red range
    """

    def __init__(self, lower: Sequence[float], upper: Sequence[float] = None):
        self.lower = np.array(lower[::-1], dtype=np.float32)
        self.upper = np.array(upper[::-1] if upper else lower[::-1], dtype=np.float32)

    def to_bgr_tuple(self) -> tuple[int, int, int]:
        """Return the lower bound as a plain (B, G, R) int tuple."""
        return (int(self.lower[0]), int(self.lower[1]), int(self.lower[2]))


class ColorFilter:
    """
    HSV color range for **detecting** pixels in a screenshot.

    HSV is far more robust than BGR/RGB for color detection because hue is
    independent of brightness and shadow — the same in-game object stays
    the same hue even under different lighting conditions.

    OpenCV HSV ranges: H ∈ [0, 179],  S ∈ [0, 255],  V ∈ [0, 255]

    Example::

        # Detect red health bars
        red_filter = ColorFilter.from_rgb(200, 30, 30, hue_tol=15)

        # Detect a known HSV range directly
        grass = ColorFilter((35, 60, 60), (85, 255, 255))

        # Apply to a screenshot
        mask = red_filter.mask(screenshot)
    """

    def __init__(self,
                 lower_hsv: tuple[int, int, int],
                 upper_hsv: tuple[int, int, int]):
        self.lower = np.array(lower_hsv, dtype=np.uint8)
        self.upper = np.array(upper_hsv, dtype=np.uint8)

    @classmethod
    def from_bgr(cls,
                 bgr: tuple[int, int, int],
                 hue_tol: int = 10,
                 sat_tol: int = 55,
                 val_tol: int = 55) -> 'ColorFilter':
        """
        Build a filter from a BGR sample color with per-channel tolerances.

        :param bgr:     Sample color in (B, G, R) order.
        :param hue_tol: ± tolerance applied to the hue channel [0–179].
        :param sat_tol: ± tolerance applied to the saturation channel.
        :param val_tol: ± tolerance applied to the value channel.
        """
        pixel = np.uint8([[list(bgr)]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        lower = (max(0, h - hue_tol), max(0, s - sat_tol), max(0, v - val_tol))
        upper = (min(179, h + hue_tol), min(255, s + sat_tol), min(255, v + val_tol))
        return cls(lower, upper)

    @classmethod
    def from_rgb(cls,
                 r: int, g: int, b: int,
                 hue_tol: int = 10,
                 sat_tol: int = 55,
                 val_tol: int = 55) -> 'ColorFilter':
        """Build a filter from an RGB sample color."""
        return cls.from_bgr((b, g, r), hue_tol=hue_tol, sat_tol=sat_tol, val_tol=val_tol)

    def mask(self, image: cv2.Mat) -> cv2.Mat:
        """
        Return a binary mask where white pixels match this color range.

        :param image: BGR image (e.g. from window.screenshot()).
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, self.lower, self.upper)

    def mask_cleaned(self, image: cv2.Mat, kernel_size: int = 5) -> cv2.Mat:
        """
        Like mask(), but applies morphological opening to remove noise
        (isolated pixels and thin lines that aren't real regions).
        """
        raw = self.mask(image)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        return cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)


# ---------------------------------------------------------------------------
# Pre-defined solid colors (RGB input → stored as BGR)
# ---------------------------------------------------------------------------

BLACK  = Color([0,   0,   0])
BLUE   = Color([0,   0,   255])
CYAN   = Color([0,   255, 255])
GREEN  = Color([0,   255, 0])
ORANGE = Color([255, 144, 64])
PINK   = Color([255, 0,   231])
PURPLE = Color([170, 0,   255])
RED    = Color([255, 0,   0])
WHITE  = Color([255, 255, 255])
YELLOW = Color([255, 255, 0])

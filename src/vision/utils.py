from functools import lru_cache
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def load_mat_from_file(path: Path, flags: int = cv2.IMREAD_GRAYSCALE) -> cv2.Mat:
    return cv2.imread(str(path), flags)


@lru_cache(maxsize=8)
def convert_rgb_to_hsv(r: float, g: float, b: float) -> Tuple[int, int, int]:
    return cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_BGR2HSV)


def get_hsv_bounds(hsv: Tuple[int, int, int]) -> (Tuple[int, int, int], Tuple[int, int, int]):
    lower = (hsv[0] - 10, 100, 100)
    upper = (hsv[0] + 10, 255, 255)
    return lower, upper

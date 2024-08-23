from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from mss.screenshot import ScreenShot

from src.pixeler.vision.color import Color


def load_mat_from_file(path: Path, flags: int = cv2.IMREAD_GRAYSCALE) -> cv2.Mat:
    """
    Loads an image from a given path and reads it as a cv Mat.
    :param path: The path to the image file
    :param flags: Flags to apply to the Mat. Defaults to GRAYSCALE.
    :return: A cv Mat
    """
    return cv2.imread(str(path), flags)


def mss_to_cv2(screenshot: ScreenShot) -> cv2.Mat:
    """
    Converts a mss screenshot to a cv Mat.
    :param screenshot: The screenshot to convert.
    :return: A cv Mat
    """
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)


def edge_detect(mat: cv2.Mat, threshold1: int = 200, threshold2: int = 300) -> cv2.Mat:
    """
    Basic edge detection.
    :param mat: The mat
    :param threshold1: The threshold for edge detection
    :param threshold2: The threshold for edge detection
    :return:
    """
    return cv2.Canny(mat, threshold1, threshold2)


def convert(mat: cv2.Mat, code: int) -> cv2.Mat:
    """
    Convert Mat from one color-space type to another
    :param mat: The cv Mat
    :param code: The color space (cv2.COLOR_XXX)
    :return: A new Mat
    """
    return cv2.cvtColor(mat, code)


def convert_to_gray(img: cv2.Mat) -> cv2.Mat:
    """
    Converts the Mat to GRAY for CV analysis.
    :param img: The Mat to convert
    :return: A cv Mat
    """
    return convert(img, cv2.COLOR_BGR2GRAY)


def mask_to_color(mat: cv2.Mat, color: Color) -> cv2.Mat:
    """
    Thresholds the image to get only a given color
    :param mat: The cv Mat
    :param color: The color to mask
    :return: A masked Mat from the given color
    """
    return cv2.inRange(mat, color.lower, color.upper)


def convert_rgb_to_hsv(r: float, g: float, b: float) -> cv2.Mat:
    """
    Converts RGB to HSV
    :param r: Red value [0-255]
    :param g: Green value [0-255]
    :param b: Blue value [0-255]
    :return: HSV encoded Mat
    """
    return cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_BGR2HSV)


def get_hsv_bounds(hsv: cv2.Mat) -> (Tuple[int, int, int], Tuple[int, int, int]):
    """
    Returns the bounds of the hsv image
    :param hsv: The cv mat
    :return:The upper and lower bounds
    """
    lower = (hsv[0] - 10, 100, 100)
    upper = (hsv[0] + 10, 255, 255)
    return lower, upper

"""
Optical Image Recognition functions for image recognition and drawing.
"""
from typing import Sequence

import cv2

from src.vision.color import Color


def mask_by_color(image: cv2.Mat, color: Color) -> cv2.Mat:
    """
    Overlays an existing image with the shapes found by a given color.
    :param image: The original image
    :param color: The color to mask from
    :return: A masked Image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, color.lower, color.upper)


def find_template_in_image(image: cv2.Mat, template: cv2.Mat, method=cv2.TM_CCOEFF_NORMED) -> (Sequence[int], cv2.Mat):
    """
    Search for the template image within the given image using template matching.

    :param image: The larger image in which to search (as a cv2 Mat).
    :param template: The template image to search for (as a cv2 Mat). Ensure that the dimensions of this Mat are less than or equal to template.
    :param method: The matching method to use (default is cv2.TM_CCOEFF_NORMED).
    :return: The top-left corner (x, y) of the best match location and the match result.
    """
    # Perform template matching
    result = cv2.matchTemplate(image, template, method)

    # Get the minimum and maximum values and their locations
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, the best match is the minimum value
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        best_match_location = min_loc
    else:
        best_match_location = max_loc

    return best_match_location, result


def draw_rectangle(image: cv2.Mat, top_left: tuple, height: float, width: float, color: Color,
                   thickness: int = 2, line_type: int = cv2.FILLED):
    """
    Draws a rectangle around the given image.
    :param image: The larger image in which to draw (as a cv2 Mat).
    :param top_left: The top-left corner (x, y) of the rectangle to draw (as a tuple).
    :param height: The height of the rectangle.
    :param width: The width of the rectangle.
    :param color: The color of the rectangle as a tuple (RGB).
    :param thickness: The thickness of the rectangle.
    :param line_type: The line type of the rectangle.
    :return: A Mat of the drawn rectangle.
    """
    if len(top_left) != 2:
        raise ValueError("top_left must be of length 2 (X, Y).")
    bottom_right = (top_left[0] + width, top_left[1] + height)
    cv2.rectangle(img=image, pt1=top_left, pt2=bottom_right, color=color.lower, thickness=thickness,
                  lineType=line_type)

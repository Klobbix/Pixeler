"""
Optical Image Recognition functions for image recognition and drawing.
"""
from typing import Sequence

import cv2

from src.pixeler.vision.color import Color


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

    return best_match_location, max_val


def draw_rectangle(image: cv2.Mat, top_left: tuple, height: float, width: float, color: Color,
                   thickness: int = 2, line_type: int = cv2.FILLED):
    """
    Draws a rectangle on the given image coordinates.
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
    cv2.rectangle(img=image, pt1=top_left, pt2=bottom_right, color=color.lower.tolist(), thickness=thickness,
                  lineType=line_type)


def draw_line(image: cv2.Mat, start: tuple, end: tuple, color: Color,
              thickness: int = 2, line_type: int = cv2.FILLED):
    """
    Draws a line on the given image.
    :param image: The larger image in which to draw (as a cv2 Mat).
    :param start: The start point of the line.
    :param end: The end point of the line.
    :param color: The color of the line as a tuple (RGB).
    :param thickness: The thickness of the line.
    :param line_type: The line type of the line.
    :return: A Mat of the drawn line.
    """
    if len(start) != 2 and len(end) != 2:
        raise ValueError("Coordinates must be of length 2 (X, Y).")
    cv2.line(img=image, pt1=start, pt2=end, color=color.lower.tolist(), thickness=thickness,
             lineType=line_type)


def draw_arrowed_line(image: cv2.Mat, start: tuple, end: tuple, color: Color,
                      thickness: int = 2, line_type: int = cv2.FILLED):
    """
    Draws a line on the given image.
    :param image: The larger image in which to draw (as a cv2 Mat).
    :param start: The start point of the line.
    :param end: The end point of the line.
    :param color: The color of the line as a tuple (RGB).
    :param thickness: The thickness of the line.
    :param line_type: The line type of the line.
    :return: A Mat of the drawn line.
    """
    if len(start) != 2 and len(end) != 2:
        raise ValueError("Coordinates must be of length 2 (X, Y).")
    cv2.arrowedLine(img=image, pt1=start, pt2=end, color=color.lower.tolist(), thickness=thickness,
                    line_type=line_type)


def draw_circle(image: cv2.Mat, center: tuple, radius: int, color: Color,
                thickness: int = 2, line_type: int = cv2.FILLED):
    """
    Draws a circle on the given image.
    :param image: The larger image in which to draw (as a cv2 Mat).
    :param center: The center point of the circle.
    :param radius: The radius of the circle.
    :param color: The color of the circle as a tuple (RGB).
    :param thickness: The thickness of the circle.
    :param line_type: The line type of the circle.
    :return: A Mat of the drawn circle.
    """
    if len(center) != 2:
        raise ValueError("Coordinates must be of length 2 (X, Y).")
    cv2.circle(img=image, center=center, radius=radius, color=color.lower.tolist(), thickness=thickness,
               lineType=line_type)


def draw_ellipse(image: cv2.Mat, center: tuple, axes: tuple, angle: float, start_angle: float, end_angle: float,
                 color: Color, thickness: int = 2, line_type: int = cv2.FILLED):
    """
    Draws a line on the given image.
    :param image: The larger image in which to draw (as a cv2 Mat).
    :param center: The center point of the ellipse.
    :param axes: The axes of the ellipse.
    :param angle: The angle of the ellipse.
    :param start_angle: The start angle of the ellipse.
    :param end_angle: The end angle of the ellipse.
    :param color: The color of the ellipse as a tuple (RGB).
    :param thickness: The thickness of the ellipse.
    :param line_type: The line type of the ellipse.
    :return: A Mat of the drawn ellipse.
    """
    if len(center) != 2 and len(axes) != 2:
        raise ValueError("Coordinates must be of length 2 (X, Y).")
    cv2.ellipse(img=image, center=center, axes=axes, angle=angle, startAngle=start_angle, endAngle=end_angle,
                color=color.lower.tolist(), thickness=thickness,
                lineType=line_type)


def draw_polylines(image: cv2.Mat, points: Sequence[cv2.Mat], closed: bool,
                   color: Color, thickness: int = 2, line_type: int = cv2.FILLED):
    """
    Draws a line on the given image.
    :param image: The larger image in which to draw (as a cv2 Mat).
    :param points: The points of the polylines.
    :param closed: If the polylines are closed.
    :param color: The color of the polylines as a tuple (RGB).
    :param thickness: The thickness of the polylines.
    :param line_type: The line type of the polylines.
    :return: A Mat of the drawn polylines.
    """
    cv2.polylines(img=image, pts=points, isClosed=closed,
                  color=color.lower.tolist(), thickness=thickness,
                  lineType=line_type)


def fill_poly(image: cv2.Mat, points: Sequence[cv2.Mat],
              color: Color, line_type: int = cv2.FILLED):
    """
    Draws a line on the given image.
    :param image: The larger image in which to draw (as a cv2 Mat).
    :param points: The points of the polylines.
    :param closed: If the polylines are closed.
    :param color: The color of the polylines as a tuple (RGB).
    :param thickness: The thickness of the polylines.
    :param line_type: The line type of the polylines.
    :return: A Mat of the drawn polylines.
    """
    cv2.fillPoly(img=image, pts=points, color=color.lower.tolist(), lineType=line_type)

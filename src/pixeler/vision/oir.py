"""
Drawing primitives for annotating cv2.Mat images.

These functions draw shapes onto an OpenCV image in memory (for debug saves,
not live screen drawing). For live on-screen annotation use the Overlay class.
"""

from typing import Sequence

import cv2

from pixeler.vision.color import Color


def draw_rectangle(image: cv2.Mat,
                   top_left: tuple,
                   height: float,
                   width: float,
                   color: Color,
                   thickness: int = 2,
                   line_type: int = cv2.LINE_AA) -> None:
    """Draw a rectangle onto *image* in place."""
    if len(top_left) != 2:
        raise ValueError("top_left must be (x, y).")
    bottom_right = (int(top_left[0] + width), int(top_left[1] + height))
    cv2.rectangle(image, (int(top_left[0]), int(top_left[1])), bottom_right,
                  color.to_bgr_tuple(), thickness=thickness, lineType=line_type)


def draw_line(image: cv2.Mat,
              start: tuple,
              end: tuple,
              color: Color,
              thickness: int = 2,
              line_type: int = cv2.LINE_AA) -> None:
    """Draw a line onto *image* in place."""
    cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])),
             color.to_bgr_tuple(), thickness=thickness, lineType=line_type)


def draw_arrowed_line(image: cv2.Mat,
                      start: tuple,
                      end: tuple,
                      color: Color,
                      thickness: int = 2,
                      line_type: int = cv2.LINE_AA) -> None:
    """Draw an arrowed line onto *image* in place."""
    cv2.arrowedLine(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])),
                    color.to_bgr_tuple(), thickness=thickness, line_type=line_type)


def draw_circle(image: cv2.Mat,
                center: tuple,
                radius: int,
                color: Color,
                thickness: int = 2,
                line_type: int = cv2.LINE_AA) -> None:
    """Draw a circle onto *image* in place."""
    cv2.circle(image, (int(center[0]), int(center[1])), radius,
               color.to_bgr_tuple(), thickness=thickness, lineType=line_type)


def draw_ellipse(image: cv2.Mat,
                 center: tuple,
                 axes: tuple,
                 angle: float,
                 start_angle: float,
                 end_angle: float,
                 color: Color,
                 thickness: int = 2,
                 line_type: int = cv2.LINE_AA) -> None:
    """Draw an ellipse arc onto *image* in place."""
    cv2.ellipse(image, (int(center[0]), int(center[1])), axes,
                angle, start_angle, end_angle,
                color.to_bgr_tuple(), thickness=thickness, lineType=line_type)


def draw_polylines(image: cv2.Mat,
                   points: Sequence[cv2.Mat],
                   closed: bool,
                   color: Color,
                   thickness: int = 2,
                   line_type: int = cv2.LINE_AA) -> None:
    """Draw a polyline onto *image* in place."""
    cv2.polylines(image, points, isClosed=closed,
                  color=color.to_bgr_tuple(), thickness=thickness, lineType=line_type)


def fill_poly(image: cv2.Mat,
              points: Sequence[cv2.Mat],
              color: Color,
              line_type: int = cv2.LINE_AA) -> None:
    """Fill a polygon onto *image* in place."""
    cv2.fillPoly(image, points, color=color.to_bgr_tuple(), lineType=line_type)


def draw_text(image: cv2.Mat,
              text: str,
              position: tuple,
              color: Color,
              font_scale: float = 0.6,
              thickness: int = 1,
              font: int = cv2.FONT_HERSHEY_SIMPLEX) -> None:
    """Draw a text label onto *image* in place."""
    cv2.putText(image, text, (int(position[0]), int(position[1])),
                font, font_scale, color.to_bgr_tuple(), thickness, cv2.LINE_AA)

import math
import random

import cv2
import mss
import numpy as np

from pixeler.math.point import Point


class Circle:
    """
    Circular region on screen, defined by a centre point and radius (pixels).

    Useful for round UI elements, minimap circles, health orbs, etc.
    """

    def __init__(self, center: Point, radius: int):
        self.center = center
        self.radius = radius

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def contains_point(self, point: Point) -> bool:
        """Return True if *point* lies inside or on the circle boundary."""
        return self.center.distance_to(point) <= self.radius

    def area(self) -> float:
        return math.pi * self.radius ** 2

    def bounding_rectangle(self) -> 'Rectangle':
        """Return the axis-aligned bounding rectangle of this circle."""
        from pixeler.math.rectangle import Rectangle
        return Rectangle(
            self.center.x - self.radius,
            self.center.y - self.radius,
            self.radius * 2,
            self.radius * 2,
        )

    # ------------------------------------------------------------------
    # Random sampling
    # ------------------------------------------------------------------

    def random_point(self) -> Point:
        """
        Return a uniformly distributed random point inside the circle.

        Uses the sqrt-radius method to avoid the clustering-at-centre bias
        that arises from naive angle+radius sampling.
        """
        angle = random.uniform(0, 2 * math.pi)
        r = self.radius * math.sqrt(random.random())
        return Point(
            int(round(self.center.x + r * math.cos(angle))),
            int(round(self.center.y + r * math.sin(angle))),
        )

    def random_point_normal(self, sigma_ratio: float = 0.35) -> Point:
        """
        Return a normally-distributed random point biased toward the centre,
        resembling how humans aim at circular targets.

        :param sigma_ratio: Standard deviation as a fraction of the radius.
        """
        sigma = self.radius * sigma_ratio
        while True:
            dx = np.random.normal(0, sigma)
            dy = np.random.normal(0, sigma)
            if math.hypot(dx, dy) <= self.radius:
                return Point(
                    int(round(self.center.x + dx)),
                    int(round(self.center.y + dy)),
                )

    # ------------------------------------------------------------------
    # Screen capture
    # ------------------------------------------------------------------

    def screenshot(self, save_path: str = None) -> cv2.Mat:
        """Capture and return only the pixels inside the circle (background masked black)."""
        from pixeler.math.rectangle import Rectangle
        rect = self.bounding_rectangle()
        with mss.mss() as sct:
            monitor = {
                "top": int(rect.y),
                "left": int(rect.x),
                "width": int(rect.w),
                "height": int(rect.h),
            }
            sct_img = sct.grab(monitor)
            img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

        # Mask outside the circle to black
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (self.radius, self.radius), self.radius, 255, -1)
        result = cv2.bitwise_and(img, img, mask=mask)

        if save_path:
            cv2.imwrite(save_path, result)
        return result

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"cx": self.center.x, "cy": self.center.y, "radius": self.radius}

    def __repr__(self) -> str:
        return f"Circle(center={self.center}, radius={self.radius})"

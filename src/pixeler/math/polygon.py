import random
from typing import List

import cv2
import mss
import numpy as np

from pixeler.math.point import Point


class Polygon:
    """
    Arbitrary convex or concave polygon defined by an ordered list of vertices.

    Useful for non-rectangular interactive regions: inventory grids, minimap
    zones, irregular UI panels, etc.
    """

    def __init__(self, vertices: List[Point]):
        if len(vertices) < 3:
            raise ValueError("A polygon requires at least 3 vertices.")
        self.vertices = vertices

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def contains_point(self, point: Point) -> bool:
        """
        Ray-casting containment test. Works for both convex and concave
        polygons; points exactly on an edge are treated as inside.
        """
        x, y = point.x, point.y
        n = len(self.vertices)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = self.vertices[i].x, self.vertices[i].y
            xj, yj = self.vertices[j].x, self.vertices[j].y
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def centroid(self) -> Point:
        """Return the arithmetic centroid (average of all vertices)."""
        n = len(self.vertices)
        return Point(
            sum(v.x for v in self.vertices) / n,
            sum(v.y for v in self.vertices) / n,
        )

    def bounding_rectangle(self) -> 'Rectangle':
        """Return the axis-aligned bounding rectangle of this polygon."""
        from pixeler.math.rectangle import Rectangle
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        x, y = min(xs), min(ys)
        return Rectangle(x, y, max(xs) - x, max(ys) - y)

    def area(self) -> float:
        """Signed area via the shoelace formula. Returns the absolute value."""
        n = len(self.vertices)
        total = 0.0
        for i in range(n):
            j = (i + 1) % n
            total += self.vertices[i].x * self.vertices[j].y
            total -= self.vertices[j].x * self.vertices[i].y
        return abs(total) / 2.0

    def perimeter(self) -> float:
        n = len(self.vertices)
        return sum(self.vertices[i].distance_to(self.vertices[(i + 1) % n])
                   for i in range(n))

    # ------------------------------------------------------------------
    # Random sampling
    # ------------------------------------------------------------------

    def random_point(self, max_attempts: int = 1000) -> Point:
        """
        Return a uniformly random point inside the polygon via rejection
        sampling from the bounding rectangle.

        :param max_attempts: Safety limit — raises RuntimeError if the
            polygon is so thin that sampling keeps missing.
        """
        rect = self.bounding_rectangle()
        for _ in range(max_attempts):
            candidate = Point(
                random.uniform(rect.x, rect.x + rect.w),
                random.uniform(rect.y, rect.y + rect.h),
            )
            if self.contains_point(candidate):
                return Point(int(round(candidate.x)), int(round(candidate.y)))
        raise RuntimeError(
            f"Could not find a point inside polygon after {max_attempts} attempts. "
            "The polygon may be degenerate."
        )

    # ------------------------------------------------------------------
    # Screen capture
    # ------------------------------------------------------------------

    def screenshot(self, save_path: str = None) -> cv2.Mat:
        """Capture pixels inside the polygon; areas outside are masked black."""
        rect = self.bounding_rectangle()
        rx, ry = int(rect.x), int(rect.y)
        with mss.mss() as sct:
            monitor = {
                "top": ry, "left": rx,
                "width": int(rect.w), "height": int(rect.h),
            }
            sct_img = sct.grab(monitor)
            img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

        # Build mask in local (bounding-rect-relative) coordinates
        pts = np.array([(int(v.x - rx), int(v.y - ry)) for v in self.vertices],
                       dtype=np.int32)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        result = cv2.bitwise_and(img, img, mask=mask)

        if save_path:
            cv2.imwrite(save_path, result)
        return result

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"vertices": [{"x": v.x, "y": v.y} for v in self.vertices]}

    def __repr__(self) -> str:
        return f"Polygon(vertices={self.vertices})"

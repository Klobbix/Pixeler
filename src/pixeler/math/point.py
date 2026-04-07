import math
from typing import NamedTuple


class Point(NamedTuple):
    """
    Immutable 2-D screen coordinate.

    Behaves as a (x, y) tuple so existing unpacking and indexing code is
    unaffected, while also supporting arithmetic, geometry helpers, and
    interpolation that are useful throughout bot automation logic.
    """
    x: float
    y: float

    # ------------------------------------------------------------------
    # Arithmetic — override tuple's concatenation behaviour
    # ------------------------------------------------------------------

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Point':
        return Point(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> 'Point':
        return Point(self.x / scalar, self.y / scalar)

    def __neg__(self) -> 'Point':
        return Point(-self.x, -self.y)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def distance_to(self, other: 'Point') -> float:
        """Euclidean distance between this point and another."""
        return math.hypot(other.x - self.x, other.y - self.y)

    def angle_to(self, other: 'Point') -> float:
        """
        Angle in degrees from this point to *other*, measured clockwise
        from the positive x-axis (matching screen coordinate conventions).
        """
        return math.degrees(math.atan2(other.y - self.y, other.x - self.x))

    def midpoint(self, other: 'Point') -> 'Point':
        """Midpoint between this point and another."""
        return Point((self.x + other.x) / 2, (self.y + other.y) / 2)

    def lerp(self, other: 'Point', t: float) -> 'Point':
        """
        Linearly interpolate toward *other*.

        :param t: Blend factor — 0.0 returns self, 1.0 returns other.
        """
        return Point(self.x + (other.x - self.x) * t,
                     self.y + (other.y - self.y) * t)

    def offset(self, dx: float, dy: float) -> 'Point':
        """Return a new point shifted by (dx, dy)."""
        return Point(self.x + dx, self.y + dy)

    def normalize(self) -> 'Point':
        """
        Return a unit vector in the same direction.
        Returns Point(0, 0) for the zero vector.
        """
        mag = math.hypot(self.x, self.y)
        if mag == 0:
            return Point(0.0, 0.0)
        return Point(self.x / mag, self.y / mag)

    def dot(self, other: 'Point') -> float:
        """Dot product with another point/vector."""
        return self.x * other.x + self.y * other.y

    def as_int(self) -> 'Point':
        """Return a new Point with both coordinates rounded to int."""
        return Point(int(round(self.x)), int(round(self.y)))

    def as_tuple(self) -> tuple[int, int]:
        """Return (int(x), int(y)) — compatible with win32 / OpenCV APIs."""
        return (int(self.x), int(self.y))

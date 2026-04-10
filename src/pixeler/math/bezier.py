"""
Bézier curve utilities for generating human-like mouse paths.

Quadratic curves (3 control points) suit short distances; cubic curves
(4 control points) provide more expressive arcs for longer movements.
"""

import random
from typing import List

from pixeler.math.point import Point


# ---------------------------------------------------------------------------
# Curve evaluation
# ---------------------------------------------------------------------------

def quadratic(p0: Point, p1: Point, p2: Point, t: float) -> Point:
    """
    Evaluate a quadratic Bézier at parameter *t* ∈ [0, 1].

    :param p0: Start point.
    :param p1: Control point.
    :param p2: End point.
    """
    u = 1.0 - t
    x = u * u * p0.x + 2 * u * t * p1.x + t * t * p2.x
    y = u * u * p0.y + 2 * u * t * p1.y + t * t * p2.y
    return Point(x, y)


def cubic(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    """
    Evaluate a cubic Bézier at parameter *t* ∈ [0, 1].

    :param p0: Start point.
    :param p1: First control point.
    :param p2: Second control point.
    :param p3: End point.
    """
    u = 1.0 - t
    x = (u ** 3 * p0.x
         + 3 * u * u * t * p1.x
         + 3 * u * t * t * p2.x
         + t ** 3 * p3.x)
    y = (u ** 3 * p0.y
         + 3 * u * u * t * p1.y
         + 3 * u * t * t * p2.y
         + t ** 3 * p3.y)
    return Point(x, y)


def sample_quadratic(p0: Point, p1: Point, p2: Point, steps: int) -> List[Point]:
    """Sample *steps+1* evenly-spaced points along a quadratic Bézier."""
    return [quadratic(p0, p1, p2, i / steps) for i in range(steps + 1)]


def sample_cubic(p0: Point, p1: Point, p2: Point, p3: Point, steps: int) -> List[Point]:
    """Sample *steps+1* evenly-spaced points along a cubic Bézier."""
    return [cubic(p0, p1, p2, p3, i / steps) for i in range(steps + 1)]


# ---------------------------------------------------------------------------
# Natural control-point generation
# ---------------------------------------------------------------------------

def random_cubic_control_points(start: Point, end: Point,
                                 offset_ratio: float = 0.4,
                                 spread_ratio: float = 0.35
                                 ) -> tuple[Point, Point]:
    """
    Generate two random cubic Bézier control points that produce a
    natural-looking curve between *start* and *end*.

    Control points are placed at roughly 1/3 and 2/3 along the path,
    displaced perpendicular to the straight line by a random amount.
    This prevents the perfectly straight or mirrored-arc paths that look
    robotic.

    :param offset_ratio:  How far along the start→end axis each control
                          point is placed (default 0.4 → near the ends).
    :param spread_ratio:  Maximum perpendicular displacement as a fraction
                          of the total distance.
    :returns: (cp1, cp2) — the two interior control points.
    """
    dist = start.distance_to(end)
    if dist == 0:
        return start, end

    # Unit vector along the line and its perpendicular
    dx = (end.x - start.x) / dist
    dy = (end.y - start.y) / dist
    perp_x, perp_y = -dy, dx

    max_spread = dist * spread_ratio
    # Random lateral offsets, flipped randomly so curves go either side
    spread1 = random.uniform(-max_spread, max_spread)
    spread2 = random.uniform(-max_spread, max_spread)

    t1 = random.uniform(offset_ratio * 0.6, offset_ratio * 1.4)
    t2 = random.uniform(1.0 - offset_ratio * 1.4, 1.0 - offset_ratio * 0.6)

    cp1 = Point(
        start.x + dx * dist * t1 + perp_x * spread1,
        start.y + dy * dist * t1 + perp_y * spread1,
    )
    cp2 = Point(
        start.x + dx * dist * t2 + perp_x * spread2,
        start.y + dy * dist * t2 + perp_y * spread2,
    )
    return cp1, cp2


def natural_path(start: Point, end: Point, steps: int = 40) -> List[Point]:
    """
    Return a list of points forming a human-like curved path from *start*
    to *end*, sampled from a cubic Bézier with randomly-generated control
    points.

    :param steps: Number of intermediate samples (higher = smoother).
    """
    cp1, cp2 = random_cubic_control_points(start, end)
    return sample_cubic(start, cp1, cp2, end, steps)


# ---------------------------------------------------------------------------
# Arc length utilities
# ---------------------------------------------------------------------------

def arc_length(points: List[Point]) -> float:
    """Approximate arc length of a sampled curve as the sum of chord lengths."""
    return sum(points[i].distance_to(points[i + 1]) for i in range(len(points) - 1))


def resample_uniform(points: List[Point], n: int) -> List[Point]:
    """
    Re-sample a polyline so that *n* points are spaced by equal arc length.
    Useful for driving mouse movement at a constant pixel-per-step rate.
    """
    if len(points) < 2:
        return list(points)

    total = arc_length(points)
    if total == 0:
        return [points[0]] * n

    step = total / (n - 1)
    result = [points[0]]
    accumulated = 0.0
    i = 0

    for _ in range(n - 2):
        target = step * len(result)
        while i < len(points) - 2:
            seg = points[i].distance_to(points[i + 1])
            if accumulated + seg >= target:
                break
            accumulated += seg
            i += 1
        seg = points[i].distance_to(points[i + 1])
        if seg == 0:
            result.append(points[i])
        else:
            t = (target - accumulated) / seg
            result.append(points[i].lerp(points[i + 1], t))

    result.append(points[-1])
    return result

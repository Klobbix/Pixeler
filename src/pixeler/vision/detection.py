"""
Screen detection: color-based region finding and template matching.

Two complementary approaches:
- **Color detection** — fast, zero setup. Find game objects by their distinctive
  HSV color (health bars, minimap dots, highlighted items).
- **Template matching** — find a reference image inside a screenshot. Best for
  UI elements with a fixed appearance (buttons, icons, cursors).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

from pixeler.math.point import Point
from pixeler.math.rectangle import Rectangle
from pixeler.vision.color import ColorFilter


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ColorRegion:
    """A contiguous region of pixels that matched a ColorFilter."""
    center: Point
    rect: Rectangle
    area: int

    def __lt__(self, other: 'ColorRegion') -> bool:
        return self.area < other.area


@dataclass
class TemplateMatch:
    """A single template match result."""
    top_left: Point
    confidence: float
    template_w: int
    template_h: int

    @property
    def bottom_right(self) -> Point:
        return Point(self.top_left.x + self.template_w,
                     self.top_left.y + self.template_h)

    @property
    def center(self) -> Point:
        return Point(self.top_left.x + self.template_w // 2,
                     self.top_left.y + self.template_h // 2)

    @property
    def rect(self) -> Rectangle:
        return Rectangle(self.top_left.x, self.top_left.y,
                         self.template_w, self.template_h)


# ---------------------------------------------------------------------------
# Color detection
# ---------------------------------------------------------------------------

def find_color_regions(image: cv2.Mat,
                       color_filter: ColorFilter,
                       min_area: int = 50,
                       clean_noise: bool = True) -> List[ColorRegion]:
    """
    Find all contiguous pixel regions that match *color_filter*, sorted by
    area descending (largest first).

    :param image:        BGR screenshot from window.screenshot().
    :param color_filter: HSV color range to search for.
    :param min_area:     Ignore regions smaller than this (pixels²).
    :param clean_noise:  Apply morphological opening to remove noise before
                         contour detection.
    :returns: List of ColorRegion, largest first.
    """
    mask = (color_filter.mask_cleaned(image) if clean_noise
            else color_filter.mask(image))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions: List[ColorRegion] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Centroid via image moments (more accurate than bounding box centre)
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            cx, cy = x + w // 2, y + h // 2
        else:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

        regions.append(ColorRegion(
            center=Point(cx, cy),
            rect=Rectangle(x, y, w, h),
            area=int(area),
        ))

    regions.sort(key=lambda r: r.area, reverse=True)
    return regions


def find_largest_color_region(image: cv2.Mat,
                               color_filter: ColorFilter,
                               min_area: int = 50) -> ColorRegion | None:
    """
    Return the single largest region matching *color_filter*, or None if
    nothing is found.
    """
    regions = find_color_regions(image, color_filter, min_area=min_area)
    return regions[0] if regions else None


def sample_color_at(image: cv2.Mat, x: int, y: int) -> tuple[int, int, int]:
    """
    Return the BGR color of the pixel at (x, y).

    Useful for calibrating a ColorFilter against an unknown object.
    """
    b, g, r = image[y, x]
    return (int(b), int(g), int(r))


def color_percentage(image: cv2.Mat, color_filter: ColorFilter) -> float:
    """
    Return the fraction [0.0–1.0] of pixels in *image* that match
    *color_filter*.  Useful for tracking e.g. health-bar fill level.
    """
    mask = color_filter.mask(image)
    total = mask.shape[0] * mask.shape[1]
    if total == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / total


# ---------------------------------------------------------------------------
# Template matching
# ---------------------------------------------------------------------------

def find_template(image: cv2.Mat,
                  template: cv2.Mat,
                  method: int = cv2.TM_CCOEFF_NORMED,
                  threshold: float = 0.8,
                  scale: float = 1.0) -> TemplateMatch | None:
    """
    Find the best match of *template* inside *image*.

    :param image:     BGR screenshot to search within.
    :param template:  BGR reference image to look for.
    :param method:    OpenCV matching method (default: TM_CCOEFF_NORMED).
    :param threshold: Minimum confidence [0–1] to accept a match.
    :param scale:     Resize the template by this factor before matching.
                      Use to compensate for DPI scaling or window resizing.
                      1.0 = no scaling (default).
    :returns: TemplateMatch if found above threshold, else None.
    """
    if scale != 1.0:
        h, w = template.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        template = cv2.resize(template, (new_w, new_h))

    h, w = template.shape[:2]
    result = cv2.matchTemplate(image, template, method)

    if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
        min_val, _, min_loc, _ = cv2.minMaxLoc(result)
        confidence = 1.0 - min_val  # invert so higher = better
        best_loc = min_loc
    else:
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        confidence = float(max_val)
        best_loc = max_loc

    if confidence < threshold:
        return None

    return TemplateMatch(
        top_left=Point(best_loc[0], best_loc[1]),
        confidence=confidence,
        template_w=w,
        template_h=h,
    )


def find_template_multiscale(
    image: cv2.Mat,
    template: cv2.Mat,
    method: int = cv2.TM_CCOEFF_NORMED,
    threshold: float = 0.8,
    scale_range: tuple[float, float] = (0.5, 2.0),
    scale_steps: int = 20,
) -> TemplateMatch | None:
    """
    Find *template* inside *image* by trying a range of scale factors and
    returning the best match above *threshold*.

    Useful when the game window may be at a different size than when the
    template was captured (e.g. DPI scaling, window resize).

    :param image:       BGR screenshot to search within.
    :param template:    BGR reference image to look for.
    :param method:      OpenCV matching method (default: TM_CCOEFF_NORMED).
    :param threshold:   Minimum confidence [0–1] to accept a match.
    :param scale_range: ``(min_scale, max_scale)`` range to search.
    :param scale_steps: Number of evenly-spaced scales to try within the range.
    :returns: Best TemplateMatch found above threshold, else None.
    """
    best: TemplateMatch | None = None
    for s in np.linspace(scale_range[0], scale_range[1], int(scale_steps)):
        match = find_template(image, template, method=method, threshold=0.0, scale=float(s))
        if match is not None and (best is None or match.confidence > best.confidence):
            best = match
    if best is None or best.confidence < threshold:
        return None
    return best


def _rotate_mat(mat: cv2.Mat, angle: float) -> cv2.Mat:
    """
    Rotate *mat* by *angle* degrees counter-clockwise, expanding the canvas
    so no corners are clipped.
    """
    h, w = mat.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += new_w / 2.0 - cx
    M[1, 2] += new_h / 2.0 - cy
    return cv2.warpAffine(mat, M, (new_w, new_h))


def find_template_multiangle(
    image: cv2.Mat,
    template: cv2.Mat,
    method: int = cv2.TM_CCOEFF_NORMED,
    threshold: float = 0.8,
    angle_range: tuple[float, float] = (0.0, 360.0),
    angle_steps: int = 36,
    scale: float = 1.0,
) -> TemplateMatch | None:
    """
    Find *template* inside *image* by trying a range of rotation angles and
    returning the best match above *threshold*.

    Useful when the sprite may appear at different orientations (e.g. a
    character facing left vs right, or a spinning icon).

    The template is rotated at each step — the canvas is expanded to prevent
    corner clipping, so the returned ``TemplateMatch`` dimensions reflect the
    rotated bounding box.  The ``center`` property is still accurate.

    :param image:        BGR screenshot to search within.
    :param template:     BGR reference image to look for.
    :param method:       OpenCV matching method (default: TM_CCOEFF_NORMED).
    :param threshold:    Minimum confidence [0–1] to accept a match.
    :param angle_range:  ``(start_deg, end_deg)`` rotation range to search.
                         ``(0, 360)`` covers full rotation (default).
                         ``(-45, 45)`` covers a ±45° tilt.
    :param angle_steps:  Number of evenly-spaced angles to try (default: 36,
                         i.e. every 10° for a full rotation).
    :param scale:        Fixed scale factor applied before rotation.
    :returns: Best TemplateMatch found above threshold, else None.
    """
    best: TemplateMatch | None = None
    for angle in np.linspace(angle_range[0], angle_range[1], int(angle_steps), endpoint=False):
        rotated = _rotate_mat(template, float(angle))
        match = find_template(image, rotated, method=method, threshold=0.0, scale=scale)
        if match is not None and (best is None or match.confidence > best.confidence):
            best = match
    if best is None or best.confidence < threshold:
        return None
    return best


def find_all_templates(image: cv2.Mat,
                       template: cv2.Mat,
                       threshold: float = 0.8,
                       method: int = cv2.TM_CCOEFF_NORMED) -> List[TemplateMatch]:
    """
    Find **all** non-overlapping occurrences of *template* in *image*.

    Uses a suppress-and-rescan approach: after each match is found, a
    region equal to the template size is zeroed out of the result map so
    the next scan finds a different location.

    :param threshold: Minimum confidence to include a match.
    :returns: List of TemplateMatch sorted by confidence descending.
    """
    h, w = template.shape[:2]
    result = cv2.matchTemplate(image, template, method)
    result_copy = result.copy()
    matches: List[TemplateMatch] = []

    while True:
        _, max_val, _, max_loc = cv2.minMaxLoc(result_copy)
        if max_val < threshold:
            break

        matches.append(TemplateMatch(
            top_left=Point(max_loc[0], max_loc[1]),
            confidence=float(max_val),
            template_w=w,
            template_h=h,
        ))

        # Suppress this region so the next iteration finds a different peak
        x1 = max(0, max_loc[0] - w // 2)
        y1 = max(0, max_loc[1] - h // 2)
        x2 = min(result_copy.shape[1], max_loc[0] + w // 2)
        y2 = min(result_copy.shape[0], max_loc[1] + h // 2)
        result_copy[y1:y2, x1:x2] = 0.0

    matches.sort(key=lambda m: m.confidence, reverse=True)
    return matches


def load_template(path: str | Path,
                  grayscale: bool = False) -> cv2.Mat:
    """
    Load a template image from disk.

    :param grayscale: Load as grayscale (faster matching, less memory).
                      The scene image must also be grayscale when using this.
    """
    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(f"Template image not found: {path}")
    return img


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def load_templates(
    paths: str | Path | List[str | Path],
    grayscale: bool = False,
) -> List[cv2.Mat]:
    """
    Load one or more template images from disk and return them as a list.

    Accepts:
    - A single image path.
    - A directory path — all image files directly inside it are loaded
      (non-recursive, sorted by filename for determinism).
    - A list of any mix of the above.

    :param paths:     A single path, a directory path, or a list of paths.
    :param grayscale: Load as grayscale (faster matching, less memory).
    :returns: List of loaded ``cv2.Mat`` objects.
    :raises FileNotFoundError: If a path does not exist or a directory is empty.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    resolved: List[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            images = sorted(
                f for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
            )
            if not images:
                raise FileNotFoundError(f"No image files found in directory: {p}")
            resolved.extend(images)
        else:
            resolved.append(p)

    return [load_template(p, grayscale=grayscale) for p in resolved]

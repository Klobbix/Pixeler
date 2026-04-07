"""
Image preprocessing utilities.

These functions prepare raw screenshots for vision analysis. Applying the
right preprocessing before template matching or OCR can dramatically
improve accuracy on low-resolution, anti-aliased, or noisy game text/sprites.
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from mss.screenshot import ScreenShot

from pixeler.vision.color import Color


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def load_mat_from_file(path: Path, flags: int = cv2.IMREAD_COLOR) -> cv2.Mat:
    """Load an image from *path* and return it as a BGR cv2.Mat."""
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def mss_to_cv2(screenshot: ScreenShot) -> cv2.Mat:
    """Convert an mss ScreenShot to a BGR cv2.Mat."""
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)


def to_gray(image: cv2.Mat) -> cv2.Mat:
    """Convert a BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_hsv(image: cv2.Mat) -> cv2.Mat:
    """Convert a BGR image to HSV."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def convert(image: cv2.Mat, code: int) -> cv2.Mat:
    """Convert *image* between colour spaces using any cv2.COLOR_* code."""
    return cv2.cvtColor(image, code)


# ---------------------------------------------------------------------------
# Preprocessing for OCR
# ---------------------------------------------------------------------------

def upscale(image: cv2.Mat, factor: float = 2.0) -> cv2.Mat:
    """
    Upscale *image* by *factor* using cubic interpolation.

    Tesseract accuracy degrades significantly for text smaller than ~20px
    tall. Upscaling 2–4× is the single most impactful OCR improvement for
    game UIs with small fonts.
    """
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w * factor), int(h * factor)),
                      interpolation=cv2.INTER_CUBIC)


def apply_clahe(image: cv2.Mat,
                clip_limit: float = 2.0,
                tile_size: int = 8) -> cv2.Mat:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a
    grayscale image to improve local contrast.

    Works better than global histogram equalisation for game screenshots
    where parts of the image are very bright (UI glow effects) and parts
    are dark.
    """
    if len(image.shape) == 3:
        image = to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_size, tile_size))
    return clahe.apply(image)


def binarize(image: cv2.Mat,
             method: str = 'otsu',
             invert: bool = False) -> cv2.Mat:
    """
    Threshold a grayscale image to pure black-and-white.

    :param method: ``'otsu'`` for automatic threshold (recommended for most
                   game text), or ``'adaptive'`` for uneven-lighting scenes.
    :param invert: Invert the result (use when text is bright on dark background).
    """
    if len(image.shape) == 3:
        image = to_gray(image)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if method == 'otsu':
        _, binary = cv2.threshold(image, 0, 255, flag | cv2.THRESH_OTSU)
    elif method == 'adaptive':
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            flag, blockSize=11, C=2
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'otsu' or 'adaptive'.")
    return binary


def denoise(image: cv2.Mat, strength: int = 10) -> cv2.Mat:
    """
    Remove noise from a grayscale image using Non-Local Means denoising.

    Use after binarise when there are stray pixels around text characters.
    """
    if len(image.shape) == 3:
        image = to_gray(image)
    return cv2.fastNlMeansDenoising(image, None, h=strength)


def preprocess_for_ocr(image: cv2.Mat,
                        scale: float = 2.0,
                        invert: bool = False) -> cv2.Mat:
    """
    Standard preprocessing pipeline for game screenshot OCR.

    Steps: upscale → grayscale → CLAHE → Otsu threshold → denoise.

    :param scale:  Upscale factor (2.0 is a good default for game UIs).
    :param invert: True when text is light on dark background (most game UIs).
    :returns: Binary (black/white) image ready for pytesseract.
    """
    img = upscale(image, scale)
    img = apply_clahe(img)
    img = binarize(img, method='otsu', invert=invert)
    img = denoise(img)
    return img


# ---------------------------------------------------------------------------
# Edge and contour helpers
# ---------------------------------------------------------------------------

def edge_detect(image: cv2.Mat,
                threshold1: int = 100,
                threshold2: int = 200) -> cv2.Mat:
    """Apply Canny edge detection."""
    gray = to_gray(image) if len(image.shape) == 3 else image
    return cv2.Canny(gray, threshold1, threshold2)


def find_contours(mask: cv2.Mat) -> list:
    """
    Find external contours in a binary mask.

    :returns: List of contour arrays (compatible with cv2.boundingRect etc.).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def largest_contour(mask: cv2.Mat):
    """
    Return the largest contour in *mask* by area, or None if mask is empty.
    """
    contours = find_contours(mask)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


# ---------------------------------------------------------------------------
# HSV utilities
# ---------------------------------------------------------------------------

def convert_rgb_to_hsv(r: float, g: float, b: float) -> Tuple[int, int, int]:
    """
    Convert an RGB colour to its OpenCV HSV representation.

    :returns: (H, S, V) as a plain tuple. H ∈ [0,179], S,V ∈ [0,255].
    """
    pixel = np.uint8([[[int(b), int(g), int(r)]]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    return (int(hsv[0]), int(hsv[1]), int(hsv[2]))


def get_hsv_bounds(h: int, s: int, v: int,
                   hue_tol: int = 10,
                   sat_tol: int = 50,
                   val_tol: int = 50
                   ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Compute lower and upper HSV bounds around a target (H, S, V) value.

    Convenience wrapper; prefer ColorFilter.from_rgb() for full integration.
    """
    lower = (max(0, h - hue_tol), max(0, s - sat_tol), max(0, v - val_tol))
    upper = (min(179, h + hue_tol), min(255, s + sat_tol), min(255, v + val_tol))
    return lower, upper


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def draw_debug_regions(image: cv2.Mat, regions, color=(0, 255, 0)) -> cv2.Mat:
    """
    Draw bounding rectangles for a list of ColorRegion or TemplateMatch
    objects onto a copy of *image* for visual debugging.
    """
    debug = image.copy()
    for region in regions:
        r = region.rect
        cv2.rectangle(debug, (int(r.x), int(r.y)),
                      (int(r.x + r.w), int(r.y + r.h)), color, 2)
    return debug

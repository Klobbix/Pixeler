"""
Optical Character Recognition for reading game UI text.

Tesseract performs poorly on raw game screenshots because game text is usually:
- Small (10–18px font height)
- Anti-aliased onto a complex background
- Sometimes white on dark, sometimes dark on light

Always preprocess before calling extract_text(). The convenience function
read_text() does this automatically.
"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from pytesseract import pytesseract

from pixeler.math.rectangle import Rectangle
from pixeler.vision.utils import preprocess_for_ocr


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class Word:
    """A single recognized word with its position and confidence score."""
    text: str
    rect: Rectangle
    confidence: float


# ---------------------------------------------------------------------------
# Core OCR functions
# ---------------------------------------------------------------------------

def read_text(image: cv2.Mat,
              preprocess: bool = True,
              invert: bool = False,
              scale: float = 2.0,
              config: str = "") -> str:
    """
    Extract all text from *image* as a plain string.

    :param preprocess: Apply the standard game-OCR preprocessing pipeline
                       (upscale, CLAHE, threshold, denoise). Recommended for
                       all game screenshots.
    :param invert:     Set True when text is light on a dark background
                       (most game UIs). Ignored when preprocess=False.
    :param scale:      Upscale factor used during preprocessing.
    :param config:     Additional Tesseract config flags.
    :returns: Stripped string of all recognized text.
    """
    img = preprocess_for_ocr(image, scale=scale, invert=invert) if preprocess else image
    if len(img.shape) == 2:
        # Tesseract expects RGB; convert grayscale → RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return pytesseract.image_to_string(img, config=config).strip()


def read_number(image: cv2.Mat,
                preprocess: bool = True,
                invert: bool = False,
                scale: float = 2.0,
                decimal: bool = False) -> float | None:
    """
    Extract a numeric value from *image*.

    Restricts Tesseract to digit characters for far better accuracy on
    health bars, cooldown timers, gold counts, etc.

    :param decimal: Allow a decimal point (for values like "99.5").
    :returns: Parsed float, or None if no number could be read.
    """
    whitelist = "0123456789." if decimal else "0123456789"
    config = f"--psm 7 -c tessedit_char_whitelist={whitelist}"
    raw = read_text(image, preprocess=preprocess, invert=invert,
                    scale=scale, config=config)
    cleaned = raw.strip().replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def read_words(image: cv2.Mat,
               preprocess: bool = True,
               invert: bool = False,
               scale: float = 2.0,
               min_confidence: float = 50.0) -> List[Word]:
    """
    Extract individual words with their bounding boxes and confidence scores.

    Useful for finding where specific text appears on screen so you can click
    on it or track it.

    :param min_confidence: Discard words below this Tesseract confidence [0–100].
    :returns: List of Word objects.
    """
    img = preprocess_for_ocr(image, scale=scale, invert=invert) if preprocess else image
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    data = pytesseract.image_to_data(img_rgb, output_type=pytesseract.Output.DICT,
                                      config="--psm 6")

    scale_inv = 1.0 / scale if preprocess else 1.0
    words: List[Word] = []

    for i, text in enumerate(data['text']):
        text = text.strip()
        if not text:
            continue
        conf = float(data['conf'][i])
        if conf < min_confidence:
            continue
        x = int(data['left'][i] * scale_inv)
        y = int(data['top'][i] * scale_inv)
        w = int(data['width'][i] * scale_inv)
        h = int(data['height'][i] * scale_inv)
        words.append(Word(
            text=text,
            rect=Rectangle(x, y, w, h),
            confidence=conf,
        ))

    return words


def find_text_position(image: cv2.Mat,
                        target: str,
                        preprocess: bool = True,
                        invert: bool = False,
                        case_sensitive: bool = False) -> Rectangle | None:
    """
    Search for *target* text in *image* and return its bounding rectangle.

    :param case_sensitive: Match case exactly when True.
    :returns: Rectangle of the matching word, or None if not found.
    """
    words = read_words(image, preprocess=preprocess, invert=invert)
    needle = target if case_sensitive else target.lower()
    for word in words:
        hay = word.text if case_sensitive else word.text.lower()
        if needle in hay:
            return word.rect
    return None


# ---------------------------------------------------------------------------
# Low-level wrappers (thin pass-through to pytesseract)
# ---------------------------------------------------------------------------

def extract_text(image: cv2.Mat, config: str = "") -> str:
    """
    Raw pytesseract string extraction without preprocessing.

    Prefer read_text() which applies the preprocessing pipeline automatically.
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return pytesseract.image_to_string(img_rgb, config=config)


def extract_boxes(image: cv2.Mat, config: str = "") -> str:
    """Return character-level bounding boxes as a raw tesseract string."""
    return pytesseract.image_to_boxes(image, config=config)


def extract_osd(image: cv2.Mat, config: str = "") -> str:
    """Return orientation and script detection information."""
    return pytesseract.image_to_osd(image, config=config)


def detect_objects(image: cv2.Mat,
                   cascade_file: str,
                   scale_factor: float = 1.1,
                   min_neighbors: int = 5) -> np.ndarray:
    """
    Haar cascade object detection (faces, eyes, etc.).

    :param cascade_file: Path to an OpenCV Haar cascade XML file.
    :returns: Array of (x, y, w, h) bounding rectangles.
    """
    cascade = cv2.CascadeClassifier(cascade_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cascade.detectMultiScale(gray, scaleFactor=scale_factor,
                                    minNeighbors=min_neighbors)

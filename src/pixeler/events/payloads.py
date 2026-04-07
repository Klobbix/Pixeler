"""
Typed payload dataclasses for all built-in Pixeler events.

Every event emitted by ``ScreenAnalyzer`` or ``GameBot`` carries one of these
as ``event.data``.  Handlers should type-annotate the payload for IDE support::

    def on_enemy(event: Event) -> None:
        payload: DetectionPayload = event.data
        print(f"Enemy at {payload.center} conf={payload.confidence:.2f}")

Payload catalogue
-----------------
Vision
    ``DetectionPayload``    —  YOLO / ORB object detection result.
    ``ColorRegionPayload``  —  Color-filter region match.
    ``TemplateMatchPayload`` — Template-matching result.
    ``OCRPayload``           — Text read from a screen region.

Bot lifecycle
    ``BotLifecyclePayload`` —  Status change (started / stopped / paused).

Input
    ``MouseActionPayload``   — Mouse action performed by the bot.
    ``KeyboardActionPayload`` — Keyboard action performed by the bot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from pixeler.bot.bot_status import BotStatus
from pixeler.math.point import Point
from pixeler.math.rectangle import Rectangle
from pixeler.vision.classifier import Detection
from pixeler.vision.detection import ColorRegion, TemplateMatch
from pixeler.vision.ocr import Word


# ---------------------------------------------------------------------------
# Vision payloads
# ---------------------------------------------------------------------------

@dataclass
class DetectionPayload:
    """
    Payload for ``detection.<class_name>`` events.

    Emitted once per detected class per frame.  When multiple objects of the
    same class are found, they are all included in ``detections`` and
    ``best`` points to the highest-confidence one.

    Attributes:
        class_name:  The detected class label (e.g. ``"enemy"``).
        best:        The single highest-confidence ``Detection`` in this frame.
        detections:  All detections of this class in this frame, sorted by
                     confidence descending.
        confidence:  Shortcut to ``best.confidence``.
        center:      Shortcut to ``best.center`` (``Point``).
        rect:        Shortcut to ``best.box`` (``Rectangle``).
    """
    class_name: str
    best: Detection
    detections: List[Detection] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        return self.best.confidence

    @property
    def center(self) -> Point:
        return self.best.center

    @property
    def rect(self) -> Rectangle:
        return self.best.box


@dataclass
class ColorRegionPayload:
    """
    Payload for ``color.<filter_name>`` events.

    Emitted when one or more contiguous pixel regions match a ``ColorFilter``.

    Attributes:
        filter_name: The user-assigned label for the ``ColorFilter``
                     (e.g. ``"health_low"``).
        largest:     The single largest matching region (most significant hit).
        regions:     All matching regions, sorted by area descending.
        center:      Shortcut to ``largest.center``.
        rect:        Shortcut to ``largest.rect``.
        area:        Shortcut to ``largest.area``.
    """
    filter_name: str
    largest: ColorRegion
    regions: List[ColorRegion] = field(default_factory=list)

    @property
    def center(self) -> Point:
        return self.largest.center

    @property
    def rect(self) -> Rectangle:
        return self.largest.rect

    @property
    def area(self) -> int:
        return self.largest.area


@dataclass
class TemplateMatchPayload:
    """
    Payload for ``template.<template_name>`` events.

    Emitted when a reference image is found in the current frame.

    Attributes:
        template_name: The user-assigned label for the template
                       (e.g. ``"login_button"``).
        match:         The ``TemplateMatch`` result (position + confidence).
        confidence:    Shortcut to ``match.confidence``.
        center:        Shortcut to ``match.center``.
        rect:          Shortcut to ``match.rect``.
    """
    template_name: str
    match: TemplateMatch

    @property
    def confidence(self) -> float:
        return self.match.confidence

    @property
    def center(self) -> Point:
        return self.match.center

    @property
    def rect(self) -> Rectangle:
        return self.match.rect


@dataclass
class OCRPayload:
    """
    Payload for ``ocr.<region_name>`` events.

    Emitted after reading text from a named screen region.

    Attributes:
        region_name: The user-assigned label for the OCR region
                     (e.g. ``"health_bar"``, ``"chat_box"``).
        text:        Full extracted text string (stripped).
        words:       Individual ``Word`` results with positions and confidence.
        number:      Parsed numeric value if the region contains a number,
                     else None.  Populated by the analyzer for regions
                     added via ``add_ocr(..., numeric=True)``.
    """
    region_name: str
    text: str
    words: List[Word] = field(default_factory=list)
    number: Optional[float] = None


# ---------------------------------------------------------------------------
# Bot lifecycle payload
# ---------------------------------------------------------------------------

@dataclass
class BotLifecyclePayload:
    """
    Payload for ``bot.started``, ``bot.stopped``, ``bot.paused``,
    and ``bot.resumed`` events.

    Attributes:
        status:  The new ``BotStatus`` after the transition.
        elapsed: Seconds the bot has been running (0.0 on start).
    """
    status: BotStatus
    elapsed: float = 0.0


# ---------------------------------------------------------------------------
# Input payloads
# ---------------------------------------------------------------------------

@dataclass
class MouseActionPayload:
    """
    Payload for ``input.mouse.*`` events.

    Attributes:
        action:   One of ``"click"``, ``"right_click"``, ``"double_click"``,
                  ``"move"``, ``"scroll"``.
        position: Screen coordinate where the action occurred.
        button:   Mouse button involved (``"left"``, ``"right"``, ``"middle"``).
                  Empty string for scroll/move events.
        clicks:   Scroll amount (positive = up, negative = down).
                  0 for non-scroll events.
    """
    action: str
    position: Point
    button: str = "left"
    clicks: int = 0


@dataclass
class KeyboardActionPayload:
    """
    Payload for ``input.key.*`` events.

    Attributes:
        action:       One of ``"press"``, ``"write"``, ``"hotkey"``.
        key_or_text:  The key name, typed text, or hotkey combination.
    """
    action: str
    key_or_text: str

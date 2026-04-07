"""
ScreenAnalyzer — per-frame detector pipeline that emits events onto an EventBus.

Each registered detector runs against the current screenshot every time
``analyze()`` is called (once per ``GameBot.step()``).  When a detector finds
something it emits a typed event; modules receive those events and act.

Throttling
----------
Every detector has an optional ``throttle_s`` (seconds).  If the detector
fired less than ``throttle_s`` seconds ago the entire run is skipped — this
prevents expensive YOLO/OCR calls from running every 16 ms frame while cheap
color filters can still run at full speed.

Detector types and their events
--------------------------------
YOLO (``add_yolo``)
    Emits one ``detection.<class_name>`` event per class found in the frame.
    A single YOLO pass detects all classes; the result is split by class so
    modules can subscribe narrowly.

ORB (``add_orb``)
    Emits ``detection.<name>`` when the reference sprite is found.

Color (``add_color``)
    Emits ``color.<name>`` when one or more pixel regions match the filter.

Template (``add_template``)
    Emits ``template.<name>`` when the reference image is found above the
    confidence threshold.

OCR (``add_ocr``)
    Emits ``ocr.<name>`` when the named screen region contains readable text.
    Set ``numeric=True`` to also populate ``OCRPayload.number``.

Usage::

    from pixeler.events.event_bus import EventBus
    from pixeler.bot.screen_analyzer import ScreenAnalyzer
    from pixeler.vision.classifier import YOLOClassifier
    from pixeler.vision.color import ColorFilter
    from pixeler.math.rectangle import Rectangle

    bus      = EventBus()
    analyzer = ScreenAnalyzer(bus)

    analyzer.add_yolo("detector", YOLOClassifier("models/game.onnx", ["enemy", "loot"]))
    analyzer.add_color("health_low", ColorFilter.from_rgb(220, 30, 30, hue_tol=10))
    analyzer.add_ocr("health_text", Rectangle(10, 10, 80, 18), throttle_s=0.5, numeric=True)

    # In the bot loop:
    events = analyzer.analyze(screenshot)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from pixeler.events.event_bus import Event, EventBus
from pixeler.events import event_names as names
from pixeler.events.payloads import (
    ColorRegionPayload,
    DetectionPayload,
    OCRPayload,
    TemplateMatchPayload,
)
from pixeler.math.rectangle import Rectangle
from pixeler.vision.classifier import Detection, ORBMatcher, YOLOClassifier
from pixeler.vision.color import ColorFilter
from pixeler.vision.detection import (
    find_color_regions,
    find_template,
    load_template,
)
from pixeler.vision.ocr import read_number, read_text, read_words


# ---------------------------------------------------------------------------
# Internal entry type
# ---------------------------------------------------------------------------

@dataclass
class _Entry:
    """Internal descriptor for one registered detector."""
    name: str
    kind: str       # "yolo" | "orb" | "color" | "template" | "ocr"
    obj: Any        # the detector object (see kind-specific notes below)
    throttle_s: float
    enabled: bool
    # kind-specific config
    threshold: float = 0.8   # template: min confidence
    numeric: bool = False    # ocr: also run read_number()
    # throttle state (monotonic seconds; 0.0 → always fires on first call)
    last_emit: float = field(default=0.0)

    def is_ready(self) -> bool:
        """Return True if the throttle window has elapsed."""
        if self.throttle_s <= 0.0:
            return True
        return (time.monotonic() - self.last_emit) >= self.throttle_s

    def mark_emitted(self) -> None:
        self.last_emit = time.monotonic()


# ---------------------------------------------------------------------------
# ScreenAnalyzer
# ---------------------------------------------------------------------------

class ScreenAnalyzer:
    """
    Runs a configurable pipeline of detectors each frame and emits events.

    :param bus: The ``EventBus`` that events are emitted onto.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._entries: List[_Entry] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_yolo(
        self,
        name: str,
        classifier: YOLOClassifier,
        throttle_s: float = 0.0,
    ) -> None:
        """
        Register a YOLO multi-class detector.

        One ``detection.<class_name>`` event is emitted per class detected in
        each frame.  All detections of the same class are bundled into a single
        ``DetectionPayload`` so modules see the full picture at once.

        :param name:        Logical label for this detector (used in logs).
        :param classifier:  A ready ``YOLOClassifier`` (model already loaded).
        :param throttle_s:  Minimum seconds between detector runs.  YOLO is
                            expensive — 0.05–0.1 s is a good starting point.
        """
        self._add("yolo", name, classifier, throttle_s)

    def add_orb(
        self,
        name: str,
        matcher: ORBMatcher,
        throttle_s: float = 0.0,
    ) -> None:
        """
        Register an ORB feature matcher.

        Emits ``detection.<name>`` when the reference sprite is found.

        :param name:      Also used as the detection class name in the event.
        :param matcher:   A ready ``ORBMatcher`` (reference already loaded).
        :param throttle_s: Minimum seconds between detector runs.
        """
        self._add("orb", name, matcher, throttle_s)

    def add_color(
        self,
        name: str,
        color_filter: ColorFilter,
        throttle_s: float = 0.0,
    ) -> None:
        """
        Register a color-region detector.

        Emits ``color.<name>`` when one or more pixel regions match the filter.

        :param name:         Filter label — used as the event name suffix and
                             ``ColorRegionPayload.filter_name``.
        :param color_filter: An ``ColorFilter`` instance (HSV range).
        :param throttle_s:   Minimum seconds between detector runs.
        """
        self._add("color", name, color_filter, throttle_s)

    def add_template(
        self,
        name: str,
        template_path: Path | str,
        threshold: float = 0.8,
        throttle_s: float = 0.0,
    ) -> None:
        """
        Register a template matcher.

        The template image is loaded from disk once at registration time and
        cached.  Emits ``template.<name>`` when the reference is found above
        ``threshold``.

        :param name:           Template label — used as the event name suffix.
        :param template_path:  Path to the reference image file.
        :param threshold:      Minimum ``TM_CCOEFF_NORMED`` confidence [0–1].
        :param throttle_s:     Minimum seconds between detector runs.
        """
        template_mat = load_template(template_path)
        entry = self._add("template", name, template_mat, throttle_s)
        entry.threshold = threshold

    def add_ocr(
        self,
        name: str,
        region: Rectangle,
        throttle_s: float = 1.0,
        numeric: bool = False,
    ) -> None:
        """
        Register an OCR region reader.

        Crops ``region`` from the screenshot each run and reads it with
        Tesseract.  Emits ``ocr.<name>`` whenever text is found.

        OCR is the most expensive detector — keep ``throttle_s`` ≥ 0.5 s.

        :param name:      Region label — used as the event name suffix and
                          ``OCRPayload.region_name``.
        :param region:    ``Rectangle`` defining the screen area to read.
                          Coordinates are relative to the window screenshot
                          (not the full screen).
        :param throttle_s: Minimum seconds between OCR runs (default: 1.0 s).
        :param numeric:   If True, also attempt ``read_number()`` and populate
                          ``OCRPayload.number``.
        """
        entry = self._add("ocr", name, region, throttle_s)
        entry.numeric = numeric

    # ------------------------------------------------------------------
    # Detector management
    # ------------------------------------------------------------------

    def enable(self, name: str) -> None:
        """Enable a previously disabled detector by name."""
        entry = self._get(name)
        if entry:
            entry.enabled = True

    def disable(self, name: str) -> None:
        """Disable a detector without removing it."""
        entry = self._get(name)
        if entry:
            entry.enabled = False

    def remove(self, name: str) -> None:
        """Remove a detector entirely."""
        self._entries = [e for e in self._entries if e.name != name]

    def names(self) -> List[str]:
        """Return the names of all registered detectors."""
        return [e.name for e in self._entries]

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(self, screenshot: cv2.Mat) -> List[Event]:
        """
        Run all enabled, non-throttled detectors against *screenshot*.

        Events are both returned **and** emitted onto the bus.  Call this
        once per bot step from ``GameBot.step()``.

        :param screenshot: BGR ``cv2.Mat`` from ``window.screenshot()``.
        :returns: List of all events emitted this frame.
        """
        emitted: List[Event] = []

        for entry in self._entries:
            if not entry.enabled:
                continue
            if not entry.is_ready():
                continue

            try:
                new_events = self._run_entry(entry, screenshot)
            except Exception as exc:
                print(
                    f"[ScreenAnalyzer] Error in detector '{entry.name}' "
                    f"({entry.kind}): {type(exc).__name__}: {exc}"
                )
                continue

            if new_events:
                entry.mark_emitted()
                self._bus.emit_many(new_events)
                emitted.extend(new_events)

        return emitted

    # ------------------------------------------------------------------
    # Per-kind dispatch
    # ------------------------------------------------------------------

    def _run_entry(self, entry: _Entry, screenshot: cv2.Mat) -> List[Event]:
        if entry.kind == "yolo":
            return self._run_yolo(entry, screenshot)
        if entry.kind == "orb":
            return self._run_orb(entry, screenshot)
        if entry.kind == "color":
            return self._run_color(entry, screenshot)
        if entry.kind == "template":
            return self._run_template(entry, screenshot)
        if entry.kind == "ocr":
            return self._run_ocr(entry, screenshot)
        return []

    def _run_yolo(self, entry: _Entry, screenshot: cv2.Mat) -> List[Event]:
        classifier: YOLOClassifier = entry.obj
        detections: List[Detection] = classifier.detect(screenshot)
        if not detections:
            return []

        # Group by class name; emit one event per class found
        by_class: Dict[str, List[Detection]] = {}
        for d in detections:
            by_class.setdefault(d.class_name, []).append(d)

        events: List[Event] = []
        for class_name, group in by_class.items():
            # Sort by confidence desc; best = highest confidence
            group.sort(key=lambda d: d.confidence, reverse=True)
            payload = DetectionPayload(
                class_name=class_name,
                best=group[0],
                detections=group,
            )
            events.append(Event(
                name=names.detection(class_name),
                data=payload,
                source=entry.name,
            ))
        return events

    def _run_orb(self, entry: _Entry, screenshot: cv2.Mat) -> List[Event]:
        matcher: ORBMatcher = entry.obj
        detection = matcher.find(screenshot)
        if detection is None:
            return []

        # Give the detection the entry's name as its class
        detection = Detection(
            class_id=detection.class_id,
            class_name=entry.name,
            confidence=detection.confidence,
            box=detection.box,
        )
        payload = DetectionPayload(
            class_name=entry.name,
            best=detection,
            detections=[detection],
        )
        return [Event(name=names.detection(entry.name), data=payload, source=entry.name)]

    def _run_color(self, entry: _Entry, screenshot: cv2.Mat) -> List[Event]:
        color_filter: ColorFilter = entry.obj
        regions = find_color_regions(screenshot, color_filter)
        if not regions:
            return []

        payload = ColorRegionPayload(
            filter_name=entry.name,
            largest=regions[0],
            regions=regions,
        )
        return [Event(name=names.color(entry.name), data=payload, source=entry.name)]

    def _run_template(self, entry: _Entry, screenshot: cv2.Mat) -> List[Event]:
        template_mat: cv2.Mat = entry.obj
        match = find_template(screenshot, template_mat, threshold=entry.threshold)
        if match is None:
            return []

        payload = TemplateMatchPayload(template_name=entry.name, match=match)
        return [Event(name=names.template(entry.name), data=payload, source=entry.name)]

    def _run_ocr(self, entry: _Entry, screenshot: cv2.Mat) -> List[Event]:
        region: Rectangle = entry.obj
        crop = _crop(screenshot, region)
        if crop is None or crop.size == 0:
            return []

        text = read_text(crop)
        if not text:
            return []

        words = read_words(crop)
        number: Optional[float] = None
        if entry.numeric:
            number = read_number(crop)

        payload = OCRPayload(
            region_name=entry.name,
            text=text,
            words=words,
            number=number,
        )
        return [Event(name=names.ocr(entry.name), data=payload, source=entry.name)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add(
        self,
        kind: str,
        name: str,
        obj: Any,
        throttle_s: float,
    ) -> _Entry:
        entry = _Entry(
            name=name,
            kind=kind,
            obj=obj,
            throttle_s=throttle_s,
            enabled=True,
        )
        self._entries.append(entry)
        return entry

    def _get(self, name: str) -> Optional[_Entry]:
        for e in self._entries:
            if e.name == name:
                return e
        return None

    def __repr__(self) -> str:
        summary = ", ".join(f"{e.name}({e.kind})" for e in self._entries)
        return f"ScreenAnalyzer([{summary}])"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _crop(image: cv2.Mat, region: Rectangle) -> Optional[cv2.Mat]:
    """
    Crop *region* from *image*.  Returns None if the region falls entirely
    outside the image bounds.
    """
    h, w = image.shape[:2]
    x = max(0, int(region.x))
    y = max(0, int(region.y))
    x2 = min(w, int(region.x + region.w))
    y2 = min(h, int(region.y + region.h))
    if x2 <= x or y2 <= y:
        return None
    return image[y:y2, x:x2]

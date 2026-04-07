"""
Standard event name constants for the Pixeler event system.

Naming convention
-----------------
Events use a dot-separated hierarchy:  ``<category>.<subcategory>``

Modules subscribe to a **prefix wildcard** to catch a whole category, or to a
specific name to filter tightly::

    # Receive every detection event regardless of class
    bus.subscribe("detection.*", my_handler)

    # Receive only enemy detections
    bus.subscribe(names.detection("enemy"), my_handler)

    # Receive every event on the bus
    bus.subscribe("*", log_all)

Builder functions
-----------------
For events that are parameterized by a name (detection class, color filter
label, template name, OCR region), use the builder functions below instead of
hand-rolling f-strings.  This keeps event names consistent between the emitter
(``ScreenAnalyzer``) and the subscriber (``GameModule``).

Example::

    from pixeler.events import event_names as names

    bus.subscribe(names.detection("enemy"), on_enemy)
    bus.subscribe(names.color("health_low"), on_health_low)
    bus.subscribe(names.ocr("health_bar"), on_health_text)
"""

# ---------------------------------------------------------------------------
# Category prefixes  (use with ".*" wildcard subscriptions)
# ---------------------------------------------------------------------------

#: Root prefix for YOLO / ORB object detections.
DETECTION = "detection"

#: Root prefix for color-region detections.
COLOR = "color"

#: Root prefix for template-match detections.
TEMPLATE = "template"

#: Root prefix for OCR region reads.
OCR = "ocr"

#: Root prefix for input action events.
INPUT = "input"

# ---------------------------------------------------------------------------
# Bot lifecycle  (exact names — no parameterization needed)
# ---------------------------------------------------------------------------

BOT_STARTED = "bot.started"
BOT_STOPPED = "bot.stopped"
BOT_PAUSED  = "bot.paused"
BOT_RESUMED = "bot.resumed"

# ---------------------------------------------------------------------------
# Input events  (exact names)
# ---------------------------------------------------------------------------

MOUSE_CLICK  = "input.mouse.click"
MOUSE_MOVE   = "input.mouse.move"
MOUSE_SCROLL = "input.mouse.scroll"
KEY_PRESS    = "input.key.press"
KEY_WRITE    = "input.key.write"
KEY_HOTKEY   = "input.key.hotkey"

# ---------------------------------------------------------------------------
# Builder functions  (emit and subscribe with the same call)
# ---------------------------------------------------------------------------

def detection(class_name: str) -> str:
    """
    Return the event name for a specific detected class.

    ``detection("enemy")``  →  ``"detection.enemy"``

    Subscribe with ``"detection.*"`` to catch all classes.
    """
    return f"detection.{class_name}"


def color(filter_name: str) -> str:
    """
    Return the event name for a named color-filter match.

    ``color("health_low")``  →  ``"color.health_low"``

    Subscribe with ``"color.*"`` to catch all color events.
    """
    return f"color.{filter_name}"


def template(template_name: str) -> str:
    """
    Return the event name for a named template match.

    ``template("login_button")``  →  ``"template.login_button"``
    """
    return f"template.{template_name}"


def ocr(region_name: str) -> str:
    """
    Return the event name for a named OCR region read.

    ``ocr("health_bar")``  →  ``"ocr.health_bar"``

    Subscribe with ``"ocr.*"`` to catch all OCR events.
    """
    return f"ocr.{region_name}"

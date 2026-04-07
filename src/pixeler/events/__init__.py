"""
Pixeler event system — synchronous publish/subscribe bus with typed payloads.

Quick reference::

    from pixeler.events import EventBus, Event, event_names as names
    from pixeler.events.payloads import DetectionPayload

    bus = EventBus()

    # Subscribe to a specific class
    bus.subscribe(names.detection("enemy"), lambda e: print(e.data.center))

    # Subscribe to all detections
    bus.subscribe("detection.*", lambda e: print(e.name, e.data.confidence))

    # Subscribe to everything
    bus.subscribe("*", lambda e: print(e.name))

    # Emit
    bus.emit(Event(names.detection("enemy"), data=DetectionPayload(...)))
"""

from pixeler.events.event_bus import Event, EventBus, EventHandler
from pixeler.events import event_names
from pixeler.events import payloads

__all__ = [
    "Event",
    "EventBus",
    "EventHandler",
    "event_names",
    "payloads",
]

"""
Synchronous publish/subscribe event bus.

Design choices
--------------
- **Synchronous dispatch** — handlers run on the same thread that calls
  ``emit()``.  Because ``GameBot.step()`` runs on a single daemon thread,
  this keeps timing deterministic and avoids thread-safety complexity.
- **Wildcard subscriptions** — subscribing to ``"detection.*"`` matches any
  event whose name starts with ``"detection."``, so modules can listen to a
  whole category without knowing every individual class name.
- **Fault isolation** — exceptions raised inside a handler are caught, logged,
  and skipped.  One broken handler never stops the rest of the dispatch.

Usage::

    bus = EventBus()

    def on_enemy(event: Event):
        print(f"Enemy at {event.data.center}")

    bus.subscribe("detection.enemy", on_enemy)
    bus.subscribe("detection.*", lambda e: print(f"Any detection: {e.name}"))

    bus.emit(Event("detection.enemy", data=payload))
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    """
    An immutable event record dispatched through the bus.

    :param name:   Dot-separated event name, e.g. ``"detection.enemy"``,
                   ``"ocr.health_bar"``, ``"bot.started"``.
    :param data:   Payload attached to the event — typically a typed dataclass
                   from ``pixeler.events.payloads``, but any value is accepted.
    :param source: Optional string identifying who emitted the event
                   (e.g. the detector name).  Useful for filtering/logging.
    :param ts:     Timestamp (``time.monotonic()``) set automatically at
                   construction time.
    """
    name: str
    data: Any = field(default=None, compare=False)
    source: str = ""
    ts: float = field(default_factory=time.monotonic, compare=False)


EventHandler = Callable[[Event], None]


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class EventBus:
    """
    Synchronous publish/subscribe event bus.

    Subscription patterns
    ~~~~~~~~~~~~~~~~~~~~~
    Exact name
        ``subscribe("detection.enemy", handler)``
        Only receives ``"detection.enemy"`` events.

    Prefix wildcard (``.*``)
        ``subscribe("detection.*", handler)``
        Receives any event whose name starts with ``"detection."`` —
        including ``"detection.enemy"``, ``"detection.npc"``, etc.
        Also matches the bare prefix ``"detection"`` itself.

    Catch-all
        ``subscribe("*", handler)``
        Receives every event emitted on this bus.
    """

    def __init__(self) -> None:
        # pattern → ordered list of handlers
        self._subs: Dict[str, List[EventHandler]] = {}

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(self, pattern: str, handler: EventHandler) -> None:
        """
        Register *handler* to be called when an event matching *pattern* is
        emitted.  Registering the same handler for the same pattern twice is
        a no-op.

        :param pattern: Exact event name or wildcard (e.g. ``"detection.*"``).
        :param handler: Callable that accepts a single ``Event`` argument.
        """
        handlers = self._subs.setdefault(pattern, [])
        if handler not in handlers:
            handlers.append(handler)

    def unsubscribe(self, pattern: str, handler: EventHandler) -> None:
        """
        Remove *handler* from the subscription for *pattern*.

        Silently does nothing if the handler was not registered.
        """
        handlers = self._subs.get(pattern)
        if handlers:
            try:
                handlers.remove(handler)
            except ValueError:
                pass

    def unsubscribe_all(self, pattern: str) -> None:
        """Remove every handler registered for *pattern*."""
        self._subs.pop(pattern, None)

    def clear(self) -> None:
        """Remove all subscriptions from the bus."""
        self._subs.clear()

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def emit(self, event: Event) -> None:
        """
        Dispatch *event* to all matching handlers synchronously.

        Handlers are called in subscription order.  Exceptions raised inside a
        handler are caught, printed, and skipped — they never prevent other
        handlers from running.

        :param event: The ``Event`` to dispatch.
        """
        for pattern, handlers in list(self._subs.items()):
            if not handlers:
                continue
            if self._matches(pattern, event.name):
                for handler in list(handlers):
                    try:
                        handler(event)
                    except Exception as exc:
                        print(
                            f"[EventBus] Handler error for '{event.name}' "
                            f"(pattern='{pattern}'): {type(exc).__name__}: {exc}"
                        )

    def emit_many(self, events: List[Event]) -> None:
        """
        Dispatch a sequence of events in order.

        Convenience wrapper for emitting all results from a single
        ``ScreenAnalyzer.analyze()`` call without calling ``emit`` in a loop.
        """
        for event in events:
            self.emit(event)

    # ------------------------------------------------------------------
    # Pattern matching
    # ------------------------------------------------------------------

    @staticmethod
    def _matches(pattern: str, event_name: str) -> bool:
        """
        Return True if *event_name* satisfies *pattern*.

        Rules:
        - ``"*"``            matches everything.
        - ``"foo.*"``        matches ``"foo"``, ``"foo.bar"``, ``"foo.bar.baz"``.
        - ``"foo.bar"``      matches only ``"foo.bar"`` exactly.
        """
        if pattern == "*":
            return True
        if pattern == event_name:
            return True
        if pattern.endswith(".*"):
            prefix = pattern[:-2]           # strip trailing ".*"
            return (event_name == prefix
                    or event_name.startswith(prefix + "."))
        return False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def patterns(self) -> List[str]:
        """Return a list of all registered subscription patterns."""
        return [p for p, h in self._subs.items() if h]

    def handler_count(self, pattern: str) -> int:
        """Return the number of handlers registered for *pattern*."""
        return len(self._subs.get(pattern, []))

    def __repr__(self) -> str:
        total = sum(len(h) for h in self._subs.values())
        return f"EventBus(patterns={len(self._subs)}, handlers={total})"

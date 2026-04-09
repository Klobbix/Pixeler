"""
GameModule — base class for all game-specific automation plugins.

A module encapsulates one self-contained piece of bot behaviour (combat,
looting, banking, crafting, etc.).  Multiple modules can run inside a single
``GameBot`` simultaneously, each subscribing to the events it cares about.

How to write a module
---------------------
1. Subclass ``GameModule`` and set a unique ``name``.
2. Decorate handler methods with ``@listens("event.name")`` — they are
   subscribed automatically when the module is registered.
3. Optionally override ``on_register(bus, bot)`` for imperative setup
   (e.g. storing a ``bot`` reference).
4. Optionally override ``on_deregister()`` to clean up state when the bot stops.

Example::

    from pixeler.modules.base_module import GameModule, listens
    from pixeler.events.payloads import DetectionPayload

    class CombatModule(GameModule):
        name = "combat"

        def on_register(self, bus, bot):
            self._bot = bot  # store for use in handlers

        @listens("detection.enemy")
        def _attack(self, event):
            payload: DetectionPayload = event.data
            self.log(f"Attacking enemy at {payload.center}")
            from pixeler.input.mouse import move_and_right_click
            move_and_right_click(*payload.center)

        @listens("color.health_low")
        def _eat_food(self, event):
            from pixeler.input.keyboard import press
            press("1")

        @listens("detection.*")
        def _log_any(self, event):
            self.log(f"Detection: {event.name}")

        def on_deregister(self):
            self.log("Combat module stopped.")
"""

from __future__ import annotations

import inspect
import time
from abc import ABC
from typing import TYPE_CHECKING, Callable, List

from pixeler.events.event_bus import Event, EventBus, EventHandler

if TYPE_CHECKING:
    from pixeler.bot.game_bot import GameBot


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def listens(*patterns: str) -> Callable:
    """
    Decorator that marks a ``GameModule`` method as an event handler.

    The decorated method is automatically subscribed to the given event
    pattern(s) when the module is registered with a ``GameBot`` — no
    ``self.on(...)`` call needed inside ``on_register``.

    One or more patterns may be supplied::

        @listens("detection.enemy")
        def _on_enemy(self, event): ...

        @listens("color.health_low", "ocr.health_bar")
        def _on_health(self, event): ...

    Patterns follow the same rules as ``EventBus.subscribe()``:
    exact names, prefix wildcards (``"detection.*"``), or ``"*"``.
    """
    def decorator(fn: Callable) -> Callable:
        # Store patterns directly on the function object so they survive
        # the class body and bound-method lookup later.
        existing: List[str] = getattr(fn, "_listens_patterns", [])
        fn._listens_patterns = existing + list(patterns)
        return fn
    return decorator


class GameModule(ABC):
    """
    Abstract base class for game-specific automation modules.

    Attributes
    ----------
    name : str
        Unique identifier for this module.  Must be set on every subclass.
        ``GameBot.add_module()`` raises ``ValueError`` on duplicate names.
    description : str
        Optional human-readable description shown in logs.
    """

    name: str = "unnamed_module"
    description: str = ""

    def __init__(self) -> None:
        # Set by GameBot.add_module() before on_register() is called
        self._bus: EventBus | None = None
        self._bot: GameBot | None = None
        self._subscribed: list[tuple[str, EventHandler]] = []

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_register(self, bus: EventBus, bot: GameBot) -> None:
        """
        Called once when the module is added to a ``GameBot``.

        Override to subscribe to events via ``self.on()`` or to store a
        reference to *bot* for access to the window and input helpers.
        Handlers decorated with ``@listens`` are subscribed automatically
        before this method is called, so ``on_register`` is only needed for
        imperative setup::

            def on_register(self, bus, bot):
                self._bot = bot  # store bot for use in handlers

        :param bus: The shared ``EventBus`` for this bot.
        :param bot: The ``GameBot`` this module is registered on.
        """

    def on_deregister(self) -> None:
        """
        Called when the module is removed from its ``GameBot``.

        Override to clean up timers, state, or resources.  The default
        implementation unsubscribes all handlers registered via ``self.on()``.
        """

    # ------------------------------------------------------------------
    # Convenience helpers (available inside on_register and handlers)
    # ------------------------------------------------------------------

    def on(self, event_name: str, handler: EventHandler) -> None:
        """
        Subscribe *handler* to *event_name* on this module's bus.

        Subscriptions made through this helper are tracked and automatically
        unsubscribed when ``on_deregister()`` is called.

        :param event_name: Exact name or wildcard pattern (``"detection.*"``).
        :param handler:    Callable that accepts a single ``Event``.
        :raises RuntimeError: if called before ``on_register()`` (no bus set).
        """
        if self._bus is None:
            raise RuntimeError(
                "self.on() must be called from within on_register(). "
                "The bus is not yet available at construction time."
            )
        self._bus.subscribe(event_name, handler)
        self._subscribed.append((event_name, handler))

    def off(self, event_name: str, handler: EventHandler) -> None:
        """
        Unsubscribe a specific handler registered via ``self.on()``.

        Useful for temporarily disabling a behaviour without deregistering
        the whole module.
        """
        if self._bus is not None:
            self._bus.unsubscribe(event_name, handler)
        self._subscribed = [
            (n, h) for n, h in self._subscribed
            if not (n == event_name and h is handler)
        ]

    def emit(self, event: Event) -> None:
        """
        Emit an event onto the shared bus from within a module.

        Useful for inter-module communication — one module can emit a custom
        event that another module subscribes to.
        """
        if self._bus is None:
            raise RuntimeError("Cannot emit before on_register() is called.")
        self._bus.emit(event)

    def log(self, message: str) -> None:
        """Print a timestamped log line prefixed with the module name."""
        print(f"[{time.strftime('%H:%M:%S')}] [{self.name}] {message}")

    # ------------------------------------------------------------------
    # Internal lifecycle (called by GameBot, not by user code)
    # ------------------------------------------------------------------

    def _autowire(self) -> None:
        """
        Subscribe all ``@listens``-decorated methods on this instance.

        Called by ``GameBot.add_module()`` after ``_bus`` is set, before
        ``on_register()``.  User code should not call this directly.
        """
        for _, method in inspect.getmembers(self, predicate=inspect.ismethod):
            patterns: List[str] = getattr(method, "_listens_patterns", [])
            for pattern in patterns:
                self.on(pattern, method)

    def _deregister(self) -> None:
        """
        Internal cleanup called by ``GameBot.remove_module()``.

        Unsubscribes all tracked handlers, then calls the user-facing
        ``on_deregister()`` hook.
        """
        if self._bus is not None:
            for event_name, handler in self._subscribed:
                self._bus.unsubscribe(event_name, handler)
        self._subscribed.clear()
        self.on_deregister()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"

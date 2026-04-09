"""
GameBot — the top-level automation bot for game modules.

``GameBot`` extends ``Bot`` with three owned subsystems:

- **EventBus** — synchronous pub/sub for all game events.
- **ScreenAnalyzer** — runs detectors every step and fires events.
- **Module registry** — ``GameModule`` plugins that subscribe to events and act.

Users never override ``step()`` on a ``GameBot``.  Instead they:

1. Register detectors on ``bot.analyzer``.
2. Write ``GameModule`` subclasses and add them via ``bot.add_module()``.
3. Call ``bot.start()``.

Lifecycle events
----------------
The bus receives lifecycle events automatically:

- ``"bot.started"``  immediately after ``on_start()`` completes.
- ``"bot.stopped"``  immediately before ``on_stop()`` completes.
- ``"bot.paused"``   when ``bot.pause()`` is called.
- ``"bot.resumed"``  when ``bot.resume()`` is called.

Usage::

    from pixeler.bot.game_bot import GameBot
    from pixeler.window.win32_window import Win32Window
    from pixeler.vision.classifier import YOLOClassifier
    from pixeler.vision.color import ColorFilter
    from pixeler.math.rectangle import Rectangle
    from my_game.combat_module import CombatModule

    bot = GameBot(window=Win32Window("MyGame"))
    bot.analyzer.add_yolo("detector", YOLOClassifier("models/game.onnx", ["enemy","loot"]))
    bot.analyzer.add_color("health_low", ColorFilter.from_rgb(220, 30, 30, hue_tol=10))
    bot.analyzer.add_ocr("gold", Rectangle(10, 50, 120, 18), throttle_s=1.0, numeric=True)
    bot.add_module(CombatModule())
    bot.start()
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from pixeler.bot.bot import Bot
from pixeler.bot.bot_status import BotStatus
from pixeler.bot.screen_analyzer import ScreenAnalyzer
from pixeler.events.event_bus import Event, EventBus
from pixeler.events import event_names as names
from pixeler.events.payloads import BotLifecyclePayload
from pixeler.window.abstract_window import AbstractWindow
from pixeler.window.win32_window import Win32Window
from pixeler.window.window import Window


class GameBot(Bot):
    """
    Automation bot with an event bus, vision pipeline, and module registry.

    :param window: The game window to capture screenshots from.  Pass a
                   ``Win32Window`` for overlay support, or ``Window`` for
                   cross-platform use.  May be ``None`` if you handle
                   screenshots yourself by overriding ``_get_screenshot()``.
    """

    def __init__(self, window: Optional[Union[Window, Win32Window]] = None) -> None:
        super().__init__(window=window)
        self._bus = EventBus()
        self._analyzer = ScreenAnalyzer(self._bus)
        self._modules: Dict[str, "GameModule"] = {}  # name → module  # noqa: F821

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def bus(self) -> EventBus:
        """The shared ``EventBus`` for this bot and all its modules."""
        return self._bus

    @property
    def analyzer(self) -> ScreenAnalyzer:
        """
        The ``ScreenAnalyzer`` that runs detectors every step.

        Register detectors here before calling ``start()``::

            bot.analyzer.add_yolo("detector", YOLOClassifier(...))
            bot.analyzer.add_color("hp_low", ColorFilter.from_rgb(...))
        """
        return self._analyzer

    # ------------------------------------------------------------------
    # Module management
    # ------------------------------------------------------------------

    def add_module(self, module: "GameModule") -> None:  # noqa: F821
        """
        Register a ``GameModule`` and wire it into the event bus.

        Calls ``module.on_register(bus, self)`` immediately.  Duplicate
        module names raise ``ValueError``.

        :param module: A ``GameModule`` subclass instance.
        """
        if module.name in self._modules:
            raise ValueError(
                f"A module named '{module.name}' is already registered. "
                "Set a unique GameModule.name on your class."
            )
        module._bus = self._bus       # give the module its bus reference
        module._bot = self            # give the module its bot reference
        module._autowire()            # subscribe @listens-decorated methods
        module.on_register(self._bus, self)
        self._modules[module.name] = module
        self.log(f"Module '{module.name}' registered.")

    def remove_module(self, name: str) -> None:
        """
        Deregister a module by name and call its ``on_deregister()`` hook.

        Silently does nothing if the module is not registered.
        """
        module = self._modules.pop(name, None)
        if module is not None:
            try:
                module._deregister()
            except Exception as exc:
                self.log(f"Module '{name}' on_deregister error: {exc}")
            self.log(f"Module '{name}' removed.")

    def get_module(self, name: str) -> Optional["GameModule"]:  # noqa: F821
        """Return the registered module with *name*, or None."""
        return self._modules.get(name)

    # ------------------------------------------------------------------
    # Bot lifecycle overrides
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        self._bus.emit(Event(
            name=names.BOT_STARTED,
            data=BotLifecyclePayload(status=BotStatus.RUNNING, elapsed=0.0),
            source="GameBot",
        ))

    def on_stop(self) -> None:
        # Deregister all modules cleanly
        for name in list(self._modules.keys()):
            self.remove_module(name)
        self._bus.emit(Event(
            name=names.BOT_STOPPED,
            data=BotLifecyclePayload(status=BotStatus.STOPPED, elapsed=self.elapsed()),
            source="GameBot",
        ))

    def pause(self) -> None:
        super().pause()
        if self.status == BotStatus.PAUSED:
            self._bus.emit(Event(
                name=names.BOT_PAUSED,
                data=BotLifecyclePayload(status=BotStatus.PAUSED, elapsed=self.elapsed()),
                source="GameBot",
            ))

    def resume(self) -> None:
        super().resume()
        if self.status == BotStatus.RUNNING:
            self._bus.emit(Event(
                name=names.BOT_RESUMED,
                data=BotLifecyclePayload(status=BotStatus.RUNNING, elapsed=self.elapsed()),
                source="GameBot",
            ))

    # ------------------------------------------------------------------
    # Core step — sealed (do not override in subclasses)
    # ------------------------------------------------------------------

    def step(self) -> None:
        """
        One iteration of the bot loop.

        Takes a screenshot, runs all detectors via ``ScreenAnalyzer``, and
        emits the resulting events onto the bus.  Modules receive those events
        synchronously and perform their automation actions.

        **Do not override this method.**  Write a ``GameModule`` instead.
        """
        if self.window is None:
            return
        screenshot = self.window.screenshot()
        self._analyzer.analyze(screenshot)

"""
OSRS — Widget Module

Responsibilities:
- Move mouse and click a widget if it is on the screen.

Events consumed:
    "detection.widget"       — YOLO detected a widget on screen.

To use::

    bot.add_module(WidgetModule())
"""

from __future__ import annotations

import time

from pixeler.events.event_bus import Event, EventBus
from pixeler.events.payloads import ColorRegionPayload, DetectionPayload, OCRPayload
from pixeler.modules.base_module import GameModule


class WidgetModule(GameModule):
    name = "widget"
    description = "Click widgets when on screen."

    def __init__(
        self
    ) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def on_register(self, bus: EventBus, bot) -> None:
        self._bot = bot
        self.on("detection.buttons",  self._on_widget)

    def on_deregister(self) -> None:
        self.log("Widget module stopped.")

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _on_widget(self, event: Event) -> None:
        """Click the widget on the screen"""
        payload: DetectionPayload = event.data
        enemy = payload.best
        self.log(
            f"Widget '{enemy.class_name}' at {enemy.center} "
            f"(conf={enemy.confidence:.0%}) — clicking"
        )

        # Focus the window and press the attack skill
        self._bot.window.focus()
        from pixeler.input.mouse import move_and_click
        move_and_click(enemy.center.x, enemy.center.y, False)
        time.sleep(3.0)

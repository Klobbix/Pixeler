"""
MapleStory — Loot Module

Responsibilities:
- Pick up loot items detected on the ground by the YOLO model.
- Avoid spamming the loot key when the same item is detected across multiple
  consecutive frames (debounce by position).

Events consumed:
    "detection.loot_item"   — YOLO detected a droppable item on screen.

To use::

    bot.add_module(LootModule(loot_key="z", cooldown_s=1.0))
"""

from __future__ import annotations

import time
import math

from src.pixeler.events.event_bus import Event, EventBus
from src.pixeler.events.payloads import DetectionPayload
from src.pixeler.math.point import Point
from src.pixeler.modules.base_module import GameModule


class LootModule(GameModule):
    """
    Picks up detected loot items by moving to them and pressing the loot key.

    Position debouncing prevents re-looting the same item every frame: after
    pressing the loot key at a position, that spot is "cooled down" for
    ``cooldown_s`` seconds.

    :param loot_key:   Key bound to pick-up / loot in-game (default ``"z"``).
    :param cooldown_s: Seconds before the same screen position can be looted
                       again.  Items that don't disappear quickly won't be
                       spammed.
    :param proximity:  Pixel radius within which two detections are considered
                       the "same" item for debounce purposes.
    """

    name = "loot"
    description = "Pick up detected loot items from the ground."

    def __init__(
        self,
        loot_key: str = "z",
        cooldown_s: float = 1.5,
        proximity: int = 40,
    ) -> None:
        super().__init__()
        self._loot_key = loot_key
        self._cooldown_s = cooldown_s
        self._proximity = proximity

        # Maps screen position → last loot timestamp for debouncing
        self._loot_history: list[tuple[Point, float]] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def on_register(self, bus: EventBus, bot) -> None:
        self._bot = bot
        self.on("detection.loot_item", self._on_loot)

    def on_deregister(self) -> None:
        self.log("Loot module stopped.")

    # ------------------------------------------------------------------
    # Handler
    # ------------------------------------------------------------------

    def _on_loot(self, event: Event) -> None:
        """Loot each detected item that isn't currently on cooldown."""
        payload: DetectionPayload = event.data
        self._expire_history()

        for detection in payload.detections:
            pos = detection.center
            if self._on_cooldown(pos):
                continue

            self.log(
                f"Looting '{detection.class_name}' at {pos} "
                f"(conf={detection.confidence:.0%}) — pressing [{self._loot_key}]"
            )
            self._bot.window.focus()
            from src.pixeler.input.mouse import move_and_click
            from src.pixeler.input.keyboard import press
            move_and_click(pos.x, pos.y)
            press(self._loot_key)
            self._loot_history.append((pos, time.monotonic()))

    # ------------------------------------------------------------------
    # Debounce helpers
    # ------------------------------------------------------------------

    def _expire_history(self) -> None:
        """Remove entries that have passed their cooldown window."""
        cutoff = time.monotonic() - self._cooldown_s
        self._loot_history = [(p, t) for p, t in self._loot_history if t > cutoff]

    def _on_cooldown(self, pos: Point) -> bool:
        """Return True if *pos* is within proximity of a recently looted spot."""
        for past_pos, _ in self._loot_history:
            dx = pos.x - past_pos.x
            dy = pos.y - past_pos.y
            if math.hypot(dx, dy) < self._proximity:
                return True
        return False

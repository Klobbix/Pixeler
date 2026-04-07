"""
MapleStory — Combat Module

Responsibilities:
- Attack the nearest visible enemy using a skill key.
- Eat a potion when health drops below a threshold (detected by color or OCR).
- Track cooldowns so skills aren't spammed every frame the event fires.

Events consumed:
    "detection.enemy"       — YOLO detected an enemy on screen.
    "color.health_low"      — HSV color filter found a critically-low HP bar.
    "ocr.health_text"       — Tesseract read a numeric HP value from the HUD.

To use::

    bot.add_module(CombatModule(attack_key="ctrl", potion_key="1", hp_threshold=30.0))
"""

from __future__ import annotations

import time

from src.pixeler.events.event_bus import Event, EventBus
from src.pixeler.events.payloads import ColorRegionPayload, DetectionPayload, OCRPayload
from src.pixeler.modules.base_module import GameModule


class CombatModule(GameModule):
    """
    Attacks nearby enemies and uses health potions when HP is low.

    :param attack_key:    Keyboard key bound to your primary attack / skill.
    :param potion_key:    Keyboard key bound to health potions.
    :param hp_threshold:  Numeric HP value below which a potion is used.
                          Requires an OCR region named ``"health_text"`` added
                          to the analyzer with ``numeric=True``.
    :param attack_cooldown_s:  Minimum seconds between attack key presses.
    :param potion_cooldown_s:  Minimum seconds between potion uses (avoid
                               wasting multiple potions on the same hit).
    """

    name = "combat"
    description = "Attack enemies; use health potions when HP is critical."

    def __init__(
        self,
        attack_key: str = "ctrl",
        potion_key: str = "1",
        hp_threshold: float = 30.0,
        attack_cooldown_s: float = 0.6,
        potion_cooldown_s: float = 3.0,
    ) -> None:
        super().__init__()
        self._attack_key = attack_key
        self._potion_key = potion_key
        self._hp_threshold = hp_threshold
        self._attack_cd = attack_cooldown_s
        self._potion_cd = potion_cooldown_s

        self._last_attack: float = 0.0
        self._last_potion: float = 0.0
        self._current_hp: float | None = None  # updated by OCR events

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def on_register(self, bus: EventBus, bot) -> None:
        self._bot = bot
        self.on("detection.enemy",  self._on_enemy)
        self.on("color.health_low", self._on_health_low_color)
        self.on("ocr.health_text",  self._on_health_ocr)

    def on_deregister(self) -> None:
        self.log("Combat module stopped.")

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _on_enemy(self, event: Event) -> None:
        """Attack the highest-confidence enemy detected this frame."""
        if not self._attack_ready():
            return

        payload: DetectionPayload = event.data
        enemy = payload.best
        self.log(
            f"Enemy '{enemy.class_name}' at {enemy.center} "
            f"(conf={enemy.confidence:.0%}) — attacking with [{self._attack_key}]"
        )

        # Focus the window and press the attack skill
        self._bot.window.focus()
        from src.pixeler.input.keyboard import press
        press(self._attack_key)
        self._last_attack = time.monotonic()

    def _on_health_low_color(self, event: Event) -> None:
        """
        Color filter detected a critically-low health bar.

        This fires even when OCR can't read the exact number — useful as a
        fast, preprocessing-free fallback.
        """
        payload: ColorRegionPayload = event.data
        self.log(
            f"HP bar color critical — region area={payload.area}px² — "
            f"using potion [{self._potion_key}]"
        )
        self._use_potion()

    def _on_health_ocr(self, event: Event) -> None:
        """
        OCR read a numeric HP value from the HUD.

        Updates the tracked HP and triggers a potion if below threshold.
        """
        payload: OCRPayload = event.data
        if payload.number is None:
            return

        self._current_hp = payload.number
        if self._current_hp < self._hp_threshold:
            self.log(
                f"HP = {self._current_hp:.0f} (threshold {self._hp_threshold:.0f}) "
                f"— using potion [{self._potion_key}]"
            )
            self._use_potion()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _attack_ready(self) -> bool:
        return (time.monotonic() - self._last_attack) >= self._attack_cd

    def _use_potion(self) -> None:
        if (time.monotonic() - self._last_potion) < self._potion_cd:
            return  # still on cooldown — don't double-pot

        self._bot.window.focus()
        from src.pixeler.input.keyboard import press
        press(self._potion_key)
        self._last_potion = time.monotonic()

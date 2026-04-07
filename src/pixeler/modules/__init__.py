"""
Pixeler module system — base class for game-specific automation plugins.

Quick reference::

    from pixeler.modules.base_module import GameModule
    from pixeler.events.payloads import DetectionPayload

    class CombatModule(GameModule):
        name = "combat"

        def on_register(self, bus, bot):
            self._bot = bot
            self.on("detection.enemy", self._attack)

        def _attack(self, event):
            payload: DetectionPayload = event.data
            from pixeler.input.mouse import move_and_right_click
            move_and_right_click(*payload.center)
"""

from pixeler.modules.base_module import GameModule

__all__ = ["GameModule"]

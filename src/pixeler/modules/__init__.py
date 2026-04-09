"""
Pixeler module system — base class for game-specific automation plugins.

Quick reference::

    from pixeler.modules.base_module import GameModule, listens
    from pixeler.events.payloads import DetectionPayload

    class CombatModule(GameModule):
        name = "combat"

        def on_register(self, bus, bot):
            self._bot = bot

        @listens("detection.enemy")
        def _attack(self, event):
            payload: DetectionPayload = event.data
            from pixeler.input.mouse import move_and_right_click
            move_and_right_click(*payload.center)
"""

from pixeler.modules.base_module import GameModule, listens

__all__ = ["GameModule", "listens"]

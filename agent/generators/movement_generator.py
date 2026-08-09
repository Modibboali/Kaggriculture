"""Legal movement actions for a unit.

Movement is special among action families: moving off the edge of the board is
a no-op and locked tiles are passable, so every direction is *always* legal for
every unit. The generator therefore needs no state and returns a fixed set.
"""

from __future__ import annotations

from ..actions import MovementAction
from ..state import Direction


class MovementGenerator:
    """Generates the four always-legal movement actions."""

    def generate(self) -> tuple[MovementAction, ...]:
        """The four cardinal moves in canonical order (N/S/E/W)."""
        return tuple(MovementAction(direction=direction) for direction in Direction)

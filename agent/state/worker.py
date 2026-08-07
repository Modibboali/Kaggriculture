"""A movable farm worker (the main farmer or a hired hand)."""

from __future__ import annotations

from dataclasses import dataclass

from .inventory import Inventory
from .position import Position


@dataclass(frozen=True, slots=True)
class Worker:
    """A single unit that can be given one action per turn.

    ``id`` is stable within a snapshot: the main farmer always has id 0 and
    hired hands are numbered from 1. Hands are re-hired each day, so ids are
    not guaranteed to be stable across days.
    """

    id: int
    position: Position
    inventory: Inventory
    is_main_farmer: bool

    def moved_to(self, position: Position) -> "Worker":
        """Return a copy of this worker standing at ``position``."""
        return Worker(self.id, position, self.inventory, self.is_main_farmer)

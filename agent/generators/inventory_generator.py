"""Legal shed interactions (PICKUP / DROP) for a unit.

The shed sits at the board centre and is not a tile; a unit must stand on one
of the four tiles immediately around it to interact. PICKUP draws from the
player's shed inventory, DROP pushes the unit's own inventory into the shed.
"""

from __future__ import annotations

from ..actions import Action, DropAction, PickupAction
from ..state import Farm, Inventory, ItemType, Position


def _is_shed_adjacent(board_size: int, position: Position) -> bool:
    """Whether a unit stands orthogonally adjacent to the central shed.

    The shed is at the board centre; the four tiles around it are
    (half-1, half-1), (half, half-1), (half-1, half), (half, half), where
    ``half = board_size // 2``.
    """
    half = board_size // 2
    return position in (
        Position(half - 1, half - 1),
        Position(half, half - 1),
        Position(half - 1, half),
        Position(half, half),
    )


class InventoryGenerator:
    """Generates shed PICKUP / DROP actions for one unit."""

    def generate(
        self,
        farm: Farm,
        position: Position,
        worker_inventory: Inventory,
        shed_inventory: Inventory,
    ) -> tuple[Action, ...]:
        """The legal shed actions for a unit standing at ``position``."""
        if not _is_shed_adjacent(farm.board_size, position):
            return ()

        actions: list[Action] = []
        # One atomic PICKUP per distinct item present in the shed. The
        # environment defaults the picked-up quantity to 1; larger quantities
        # are compositions of this atomic action.
        for item in ItemType:
            if shed_inventory.contains(item):
                actions.append(PickupAction(item=item))
        # DROP of an empty inventory is a no-op, so it is not legal.
        if worker_inventory.total_items() > 0:
            actions.append(DropAction())
        return tuple(actions)

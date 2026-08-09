"""Legal animal actions for a unit standing on a coop or pasture.

PLACE puts an animal the unit is carrying onto an empty matching structure
(GOOSE -> coop, COW/SHEEP -> pasture). FEED / HARVEST / CARE /
COLLECT_FERTILIZER operate on an animal already present on the tile.
"""

from __future__ import annotations

from ..actions import (
    Action,
    CareAction,
    CollectFertilizerAction,
    FeedAction,
    HarvestAction,
    PlaceAction,
)
from ..state import AnimalType, Farm, Inventory, ItemType, Position
from ..state.tile import CoopTile, PastureTile


def _animal_tile_actions(tile: CoopTile | PastureTile, inventory: Inventory) -> list[Action]:
    """Actions for a tile that already contains an animal."""
    animal = tile.animal
    if animal is None:
        return []

    actions: list[Action] = []
    # Feeding requires wheat the unit is carrying; an already-fed animal
    # would make the feed a no-op, so it is not legal.
    if not animal.fed_today and inventory.contains(ItemType.WHEAT):
        actions.append(FeedAction())
    # Harvest only when there is product waiting to be collected.
    if animal.yield_units > 0:
        actions.append(HarvestAction())
    if not animal.cared_today:
        actions.append(CareAction())
    if animal.fertilizer_available:
        actions.append(CollectFertilizerAction())
    return actions


class AnimalGenerator:
    """Generates animal actions for one unit."""

    def generate(
        self,
        farm: Farm,
        position: Position,
        inventory: Inventory,
    ) -> tuple[Action, ...]:
        """The legal animal actions for a unit standing at ``position``."""
        tile = farm.tile_at(position)
        if isinstance(tile, CoopTile):
            if tile.animal is None:
                # Only a goose can be placed into an empty coop.
                if inventory.contains(ItemType.GOOSE):
                    return (PlaceAction(animal=AnimalType.GOOSE),)
                return ()
            return tuple(_animal_tile_actions(tile, inventory))

        if isinstance(tile, PastureTile):
            if tile.animal is None:
                actions: list[Action] = []
                # A pasture can hold a cow or a sheep.
                for animal in (AnimalType.COW, AnimalType.SHEEP):
                    if inventory.contains(animal.as_item):
                        actions.append(PlaceAction(animal=animal))
                return tuple(actions)
            return tuple(_animal_tile_actions(tile, inventory))

        return ()

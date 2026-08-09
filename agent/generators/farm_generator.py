"""Legal farm actions (crops, terrain, structures) for a unit.

Every action here is derived purely from the tile under the acting unit's
position, the player's shared seed pool, and the unit's carried fertilizer.
No game rules are applied beyond stating what the action targets; execution
is left to the future simulator.
"""

from __future__ import annotations

from ..actions import (
    Action,
    BuildCoopAction,
    BuildPastureAction,
    DigAction,
    FertilizeAction,
    HarvestAction,
    PlantAction,
    WaterAction,
)
from ..state import CropType, Farm, Inventory, ItemType, Position, Seeds
from ..state.tile import CoopTile, EmptyTile, PastureTile, PlantTile, WeedTile


class FarmGenerator:
    """Generates tile-based farm actions for one unit.

    Responsibilities in this class:
      * PLANT on empty tiles (one action per crop the player has seeds for).
      * BUILD_COOP / BUILD_PASTURE on empty tiles.
      * DIG on weeds, plants, and empty structures.
      * WATER / HARVEST / FERTILIZE on plants.
    """

    def generate(
        self,
        farm: Farm,
        position: Position,
        seeds: Seeds,
        inventory: Inventory,
    ) -> tuple[Action, ...]:
        """The legal farm actions for a unit standing at ``position``."""
        tile = farm.tile_at(position)
        if tile is None:
            return ()

        actions: list[Action] = []
        if isinstance(tile, EmptyTile):
            # One atomic PLANT per crop that the player has at least one
            # seed of; seeds are shared by all units (never carried).
            for crop in CropType:
                if seeds.contains(crop):
                    actions.append(PlantAction(crop=crop))
            actions.append(BuildCoopAction())
            actions.append(BuildPastureAction())

        elif isinstance(tile, WeedTile):
            actions.append(DigAction())

        elif isinstance(tile, PlantTile):
            plant = tile.plant
            # Watering an already-watered plant is a no-op, so it is not legal.
            if not plant.watered_today:
                actions.append(WaterAction())
            # Harvest only makes sense when there is yield to collect.
            if plant.yield_units > 0:
                actions.append(HarvestAction())
            # Fertilizing consumes fertilizer the unit is carrying.
            if inventory.contains(ItemType.FERTILIZER):
                actions.append(FertilizeAction())
            actions.append(DigAction())

        elif isinstance(tile, (CoopTile, PastureTile)):
            # DIG removes an empty structure; an occupied one cannot be dug.
            if tile.animal is None:
                actions.append(DigAction())

        return tuple(actions)

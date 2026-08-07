"""Actions that a farmer or a hired farm hand can be given on a turn.

Every class maps to exactly one ``ActionType`` and carries only the parameters
needed to state intent. The tile a field action applies to is not stored here:
it is implicit in the acting worker's position at execution time, which keeps
the objects small and shareable for search.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..state import AnimalType, CropType, Direction, ItemType
from .action import Action
from .action_type import ActionType


def _direction_action(direction: Direction) -> ActionType:
    """The ``ActionType`` that matches a movement direction."""
    match direction:
        case Direction.NORTH:
            return ActionType.NORTH
        case Direction.SOUTH:
            return ActionType.SOUTH
        case Direction.EAST:
            return ActionType.EAST
        case Direction.WEST:
            return ActionType.WEST


@dataclass(frozen=True, slots=True, kw_only=True)
class MovementAction(Action):
    """Move the acting unit one tile in ``direction``."""

    direction: Direction

    def __post_init__(self) -> None:
        # Keep the derived base field consistent with the explicit direction.
        object.__setattr__(self, "action_type", _direction_action(self.direction))

    def __str__(self) -> str:
        return self.direction.label


@dataclass(frozen=True, slots=True, kw_only=True)
class PlantAction(Action):
    """Plant a seed of ``crop`` on the acting unit's tile."""

    crop: CropType
    action_type: ActionType = ActionType.PLANT

    def __str__(self) -> str:
        return f"{self.action_type.label} {self.crop.label}"


@dataclass(frozen=True, slots=True, kw_only=True)
class WaterAction(Action):
    """Water the plant on the acting unit's tile."""

    action_type: ActionType = ActionType.WATER


@dataclass(frozen=True, slots=True, kw_only=True)
class HarvestAction(Action):
    """Harvest the plant or animal on the acting unit's tile."""

    action_type: ActionType = ActionType.HARVEST


@dataclass(frozen=True, slots=True, kw_only=True)
class FertilizeAction(Action):
    """Fertilize the plant on the acting unit's tile."""

    action_type: ActionType = ActionType.FERTILIZE


@dataclass(frozen=True, slots=True, kw_only=True)
class DigAction(Action):
    """Clear a plant, weed, or empty coop/pasture on the acting unit's tile."""

    action_type: ActionType = ActionType.DIG


@dataclass(frozen=True, slots=True, kw_only=True)
class BuildCoopAction(Action):
    """Build a goose coop on the acting unit's tile."""

    action_type: ActionType = ActionType.BUILD_COOP


@dataclass(frozen=True, slots=True, kw_only=True)
class BuildPastureAction(Action):
    """Build a pasture on the acting unit's tile."""

    action_type: ActionType = ActionType.BUILD_PASTURE


@dataclass(frozen=True, slots=True, kw_only=True)
class PlaceAction(Action):
    """Place ``animal`` from inventory onto the acting unit's tile.

    The structure type (coop for GOOSE, pasture for COW/SHEEP) is implicit in
    the animal and the tile at execution time, so it is not stored here.
    """

    animal: AnimalType
    action_type: ActionType = ActionType.PLACE

    def __str__(self) -> str:
        return f"{self.action_type.label} {self.animal.label}"


@dataclass(frozen=True, slots=True, kw_only=True)
class FeedAction(Action):
    """Feed the animal on the acting unit's tile with wheat."""

    action_type: ActionType = ActionType.FEED


@dataclass(frozen=True, slots=True, kw_only=True)
class CareAction(Action):
    """Care for the animal on the acting unit's tile."""

    action_type: ActionType = ActionType.CARE


@dataclass(frozen=True, slots=True, kw_only=True)
class CollectFertilizerAction(Action):
    """Collect 1 fertilizer from the animal on the acting unit's tile."""

    action_type: ActionType = ActionType.COLLECT_FERTILIZER


@dataclass(frozen=True, slots=True, kw_only=True)
class PickupAction(Action):
    """Pick up up to ``quantity`` of ``item`` from the shed into inventory.

    Requires being orthogonally adjacent to the shed; the environment defaults
    ``quantity`` to 1, so that is the default here as well.
    """

    item: ItemType
    quantity: int = 1
    action_type: ActionType = ActionType.PICKUP

    def __str__(self) -> str:
        return f"{self.action_type.label} {self.item.label} {self.quantity}"


@dataclass(frozen=True, slots=True, kw_only=True)
class DropAction(Action):
    """Drop the acting unit's entire inventory into the shed.

    Requires being orthogonally adjacent to the shed.
    """

    action_type: ActionType = ActionType.DROP

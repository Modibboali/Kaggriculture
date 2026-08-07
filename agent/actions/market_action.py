"""Market orders a player can submit for a turn.

Market orders are processed in submission order, so a ``TurnAction`` keeps
them as an ordered tuple. Each class maps to exactly one ``ActionType`` and
carries the traded entity and quantity; prices, costs, and legality are all
resolved by the future simulator, never by the action objects themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..state import AnimalType, CropType, ItemType, Quadrant
from .action import Action
from .action_type import ActionType


@dataclass(frozen=True, slots=True, kw_only=True)
class BuySeedAction(Action):
    """Buy ``quantity`` seeds of ``crop`` at the fixed seed price."""

    crop: CropType
    quantity: int
    action_type: ActionType = ActionType.BUY_SEED

    def __str__(self) -> str:
        return f"{self.action_type.label} {self.crop.label} {self.quantity}"


@dataclass(frozen=True, slots=True, kw_only=True)
class BuyAnimalAction(Action):
    """Buy ``quantity`` of ``animal`` at the fixed animal price."""

    animal: AnimalType
    quantity: int
    action_type: ActionType = ActionType.BUY_ANIMAL

    def __str__(self) -> str:
        return f"{self.action_type.label} {self.animal.label} {self.quantity}"


@dataclass(frozen=True, slots=True, kw_only=True)
class BuyProductAction(Action):
    """Buy ``quantity`` of ``item`` (only WHEAT / FERTILIZER) from the market."""

    item: ItemType
    quantity: int
    action_type: ActionType = ActionType.BUY_PRODUCT

    def __str__(self) -> str:
        return f"{self.action_type.label} {self.item.label} {self.quantity}"


@dataclass(frozen=True, slots=True, kw_only=True)
class SellAction(Action):
    """Sell ``quantity`` of ``item`` to the market at the current price."""

    item: ItemType
    quantity: int
    action_type: ActionType = ActionType.SELL

    def __str__(self) -> str:
        return f"{self.action_type.label} {self.item.label} {self.quantity}"


@dataclass(frozen=True, slots=True, kw_only=True)
class HireAction(Action):
    """Hire ``quantity`` farm hands for the current day."""

    quantity: int
    action_type: ActionType = ActionType.HIRE

    def __str__(self) -> str:
        return f"{self.action_type.label} {self.quantity}"


@dataclass(frozen=True, slots=True, kw_only=True)
class BuyLandAction(Action):
    """Unlock ``quadrant`` of land for the player's farm."""

    quadrant: Quadrant
    action_type: ActionType = ActionType.BUY_LAND

    def __str__(self) -> str:
        return f"{self.action_type.label} {self.quadrant.label}"

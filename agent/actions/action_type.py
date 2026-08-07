"""The closed set of action types the environment understands.

``ActionType`` lives in its own module so both the base ``Action`` and the
specialized subclasses can reference it without any circular imports. Every
action object in this package carries exactly one of these types.
"""

from __future__ import annotations

from enum import Enum


class ActionType(str, Enum):
    """Every action supported by the environment."""

    # Movement
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"

    # Farm / crops
    PLANT = "PLANT"
    WATER = "WATER"
    HARVEST = "HARVEST"
    FERTILIZE = "FERTILIZE"
    DIG = "DIG"

    # Terrain / structures
    BUILD_COOP = "BUILD_COOP"
    BUILD_PASTURE = "BUILD_PASTURE"

    # Animals
    PLACE = "PLACE"
    FEED = "FEED"
    CARE = "CARE"
    COLLECT_FERTILIZER = "COLLECT_FERTILIZER"

    # Inventory / shed
    PICKUP = "PICKUP"
    DROP = "DROP"

    # Market
    BUY_SEED = "BUY_SEED"
    BUY_PRODUCT = "BUY_PRODUCT"
    BUY_ANIMAL = "BUY_ANIMAL"
    SELL = "SELL"
    HIRE = "HIRE"
    BUY_LAND = "BUY_LAND"

    # Utility
    PASS = "PASS"

    @property
    def label(self) -> str:
        """The canonical command string used by the environment."""
        return str(self.value)

    @classmethod
    def from_label(cls, label: str) -> "ActionType":
        """Parse an action type label (case-insensitive)."""
        return cls(label.upper())

"""Categorical value types for the Kaggriculture domain model.

Every categorical value in the game is an ``Enum`` rather than a raw string
or integer, so the type system validates state construction and future search
code can pattern-match exhaustively. Enums that appear in a Kaggle observation
carry a ``label`` (the serialized string) and a ``from_label`` parser, so
string decoding lives in one place and the rest of the model stays
environment-independent.
"""

from __future__ import annotations

from enum import Enum


class _LabeledEnum(str, Enum):
    """Base for enums whose member value is also their serialized label."""

    @property
    def label(self) -> str:
        """The canonical string form used in observations."""
        return str(self.value)


class CropType(_LabeledEnum):
    """The five plantable crops."""

    WHEAT = "WHEAT"
    CARROT = "CARROT"
    TOMATO = "TOMATO"
    STRAWBERRY = "STRAWBERRY"
    MELON = "MELON"

    @classmethod
    def from_label(cls, label: str) -> "CropType":
        """Parse a crop label from an observation (case-insensitive)."""
        return cls(label.upper())

    @property
    def produce(self) -> "ItemType":
        """The market item this crop yields when harvested."""
        return ItemType[self.name]


class AnimalType(_LabeledEnum):
    """The three farm animals."""

    GOOSE = "GOOSE"
    COW = "COW"
    SHEEP = "SHEEP"

    @classmethod
    def from_label(cls, label: str) -> "AnimalType":
        """Parse an animal label from an observation (case-insensitive)."""
        return cls(label.upper())

    @property
    def as_item(self) -> "ItemType":
        """The animal itself as a market item (for buying / placing)."""
        return ItemType[self.name]

    @property
    def produce(self) -> "ItemType":
        """The market item this animal produces (EGG / MILK / WOOL)."""
        match self:
            case AnimalType.GOOSE:
                return ItemType.EGG
            case AnimalType.COW:
                return ItemType.MILK
            case AnimalType.SHEEP:
                return ItemType.WOOL


class StructureType(_LabeledEnum):
    """The categorical kinds a tile can take."""

    EMPTY = "EMPTY"
    WEED = "WEED"
    PLANT = "PLANT"
    COOP = "COOP"
    PASTURE = "PASTURE"
    LOCKED = "LOCKED"

    @classmethod
    def from_label(cls, label: str) -> "StructureType":
        """Parse a structure label from an observation (case-insensitive)."""
        return cls(label.upper())


class Direction(_LabeledEnum):
    """The four cardinal movement directions."""

    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"

    @classmethod
    def from_label(cls, label: str) -> "Direction":
        """Parse a direction label (case-insensitive)."""
        return cls(label.upper())

    @property
    def delta(self) -> tuple[int, int]:
        """The (dx, dy) offset when moving one tile in this direction.

        The grid uses screen coordinates: ``x`` grows east and ``y`` grows
        south, consistent with the observation's ``tiles[y][x]`` layout.
        """
        match self:
            case Direction.NORTH:
                return (0, -1)
            case Direction.SOUTH:
                return (0, 1)
            case Direction.EAST:
                return (1, 0)
            case Direction.WEST:
                return (-1, 0)

    @property
    def opposite(self) -> "Direction":
        """The direction that undoes a move in this direction."""
        match self:
            case Direction.NORTH:
                return Direction.SOUTH
            case Direction.SOUTH:
                return Direction.NORTH
            case Direction.EAST:
                return Direction.WEST
            case Direction.WEST:
                return Direction.EAST


class ItemType(_LabeledEnum):
    """Every physical, tradable object that can sit in an inventory or market.

    Seeds are deliberately excluded: they are never picked up or placed, live
    in a separate ``Seeds`` container on the player, and are bought at fixed
    (non-market) prices, so they are modelled as ``CropType`` counts instead
    of inventory items.
    """

    WHEAT = "WHEAT"
    CARROT = "CARROT"
    TOMATO = "TOMATO"
    STRAWBERRY = "STRAWBERRY"
    MELON = "MELON"
    EGG = "EGG"
    MILK = "MILK"
    WOOL = "WOOL"
    GOOSE = "GOOSE"
    COW = "COW"
    SHEEP = "SHEEP"
    FERTILIZER = "FERTILIZER"

    @classmethod
    def from_label(cls, label: str) -> "ItemType":
        """Parse an item label from an observation (case-insensitive)."""
        return cls(label.upper())


class Quadrant(_LabeledEnum):
    """The four 5x5 quadrants a player can unlock."""

    NW = "NW"
    NE = "NE"
    SW = "SW"
    SE = "SE"

    @classmethod
    def from_label(cls, label: str) -> "Quadrant":
        """Parse a quadrant label from an observation (case-insensitive)."""
        return cls(label.upper())


class ShopType(_LabeledEnum):
    """The town shops that can unlock over the season."""

    BAKERY = "BAKERY"
    PIZZA_SHOP = "PIZZA_SHOP"
    BRUNCH_SPOT = "BRUNCH_SPOT"
    YARN_STORE = "YARN_STORE"
    ICE_CREAM_SHOP = "ICE_CREAM_SHOP"
    PET_CAFE = "PET_CAFE"
    SMOOTHIE_SHOP = "SMOOTHIE_SHOP"
    FARMERS_MARKET = "FARMERS_MARKET"

    @classmethod
    def from_label(cls, label: str) -> "ShopType":
        """Parse a shop label; observation labels may use spaces ("PIZZA SHOP")."""
        return cls(label.upper().replace(" ", "_"))

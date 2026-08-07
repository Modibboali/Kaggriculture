"""Kaggriculture internal state model (domain layer).

This package deliberately contains no game rules, no search, and no Kaggle
environment calls: it only defines the immutable value objects a future
MuZero / AlphaZero-style planner will operate on. The single
environment-aware entry point is ``GameState.from_observation``.
"""

from .enums import (
    AnimalType,
    CropType,
    Direction,
    ItemType,
    Quadrant,
    ShopType,
    StructureType,
)
from .farm import Farm
from .game_state import GameState
from .inventory import Inventory, Seeds
from .market import Market
from .player import PlayerState
from .position import Position
from .tile import (
    EMPTY_TILE,
    LOCKED_TILE,
    WEED_TILE,
    AnimalState,
    CoopTile,
    EmptyTile,
    LockedTile,
    PastureTile,
    PlantState,
    PlantTile,
    Tile,
    WeedTile,
)
from .town import Town
from .worker import Worker

__all__ = [
    "AnimalState",
    "AnimalType",
    "CoopTile",
    "CropType",
    "Direction",
    "EMPTY_TILE",
    "EmptyTile",
    "Farm",
    "GameState",
    "Inventory",
    "ItemType",
    "LOCKED_TILE",
    "LockedTile",
    "Market",
    "PastureTile",
    "PlantState",
    "PlantTile",
    "PlayerState",
    "Position",
    "Quadrant",
    "Seeds",
    "ShopType",
    "StructureType",
    "Tile",
    "Town",
    "WEED_TILE",
    "WeedTile",
    "Worker",
]

"""The farm grid and its tile payloads.

A tile is one of a small, closed set of states. It is modelled as a
discriminated union of frozen dataclasses -- never dictionaries -- so search
code can dispatch on the concrete type or on ``tile_type`` and the type
checker can verify exhaustive handling.

``PlantState`` and ``AnimalState`` are the typed payloads carried by plant /
coop / pasture tiles. They live in this module because they are tile content,
keeping the package file layout aligned with its spec.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .enums import AnimalType, CropType, StructureType


@dataclass(frozen=True, slots=True)
class PlantState:
    """Growth / production bookkeeping for one planted crop."""

    crop: CropType
    planted_day: int
    watered_today: bool
    consecutive_unwatered: int
    yield_units: int
    fertilized_until_day: int
    max_lifespan_step: int


@dataclass(frozen=True, slots=True)
class AnimalState:
    """Production bookkeeping for one animal in a coop or pasture."""

    animal: AnimalType
    placed_day: int
    yield_units: int
    fed_today: bool
    consecutive_unfed: int
    cared_today: bool
    fertilizer_available: bool
    pending_care_bonus: int


class Tile(ABC):
    """Base of the tile discriminated union."""

    __slots__ = ()

    @property
    @abstractmethod
    def tile_type(self) -> StructureType:
        """The categorical kind of this tile (see ``StructureType``)."""
        ...


@dataclass(frozen=True, slots=True)
class EmptyTile(Tile):
    """An unlocked tile with nothing on it."""

    @property
    def tile_type(self) -> StructureType:
        return StructureType.EMPTY


@dataclass(frozen=True, slots=True)
class LockedTile(Tile):
    """A tile inside a quadrant the player has not bought yet."""

    @property
    def tile_type(self) -> StructureType:
        return StructureType.LOCKED


@dataclass(frozen=True, slots=True)
class WeedTile(Tile):
    """A weed blocking the tile; it must be dug before the tile can be used."""

    @property
    def tile_type(self) -> StructureType:
        return StructureType.WEED


@dataclass(frozen=True, slots=True)
class PlantTile(Tile):
    """A tile occupied by a crop."""

    plant: PlantState

    @property
    def tile_type(self) -> StructureType:
        return StructureType.PLANT


@dataclass(frozen=True, slots=True)
class CoopTile(Tile):
    """A goose coop, optionally occupied by a goose."""

    animal: AnimalState | None

    @property
    def tile_type(self) -> StructureType:
        return StructureType.COOP


@dataclass(frozen=True, slots=True)
class PastureTile(Tile):
    """A pasture, optionally occupied by a cow or a sheep."""

    animal: AnimalState | None

    @property
    def tile_type(self) -> StructureType:
        return StructureType.PASTURE


# Canonical singletons for field-less tiles. Reusing the same frozen
# instances keeps snapshots cheap and lets search caches share hashes.
EMPTY_TILE = EmptyTile()
LOCKED_TILE = LockedTile()
WEED_TILE = WeedTile()

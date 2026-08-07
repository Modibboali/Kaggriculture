"""Immutable typed count containers.

``Inventory`` counts physical items (``ItemType``) and ``Seeds`` counts
unplanted seeds (``CropType``). Both are value objects: every mutating
operation returns a *new* instance, so snapshots can share them freely and
search caches can use them as dictionary keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypeVar

from .enums import CropType, ItemType

_K = TypeVar("_K")


def _positive_counts(items: Mapping[_K, int]) -> dict[_K, int]:
    """Drop zero-count entries so equal multisets compare equal."""
    return {key: count for key, count in items.items() if count > 0}


@dataclass(frozen=True, slots=True, eq=False)
class Inventory:
    """An immutable multiset of ``ItemType`` counts."""

    items: Mapping[ItemType, int]

    def __post_init__(self) -> None:
        # Normalize so zero-count entries never participate in equality or
        # hashing, keeping value semantics well-defined.
        object.__setattr__(self, "items", _positive_counts(self.items))

    @classmethod
    def empty(cls) -> "Inventory":
        """An empty inventory."""
        return cls({})

    def get(self, item: ItemType) -> int:
        """The count of ``item`` (0 when absent)."""
        return self.items.get(item, 0)

    def contains(self, item: ItemType) -> bool:
        """Whether at least one unit of ``item`` is present."""
        return self.get(item) > 0

    def total_items(self) -> int:
        """Total number of units across all items."""
        return sum(self.items.values())

    def add(self, item: ItemType, count: int = 1) -> "Inventory":
        """Return a copy with ``count`` more units of ``item``."""
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        updated = dict(self.items)
        updated[item] = self.get(item) + count
        return Inventory(updated)

    def remove(self, item: ItemType, count: int = 1) -> "Inventory":
        """Return a copy with ``count`` fewer units of ``item``.

        Clamps at zero: enforcing availability is the caller's responsibility
        (the future game-logic layer), not the value object's.
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        updated = dict(self.items)
        updated[item] = self.get(item) - count
        return Inventory(updated)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Inventory):
            return NotImplemented
        return self.items == other.items

    def __hash__(self) -> int:
        return hash(frozenset(self.items.items()))


@dataclass(frozen=True, slots=True, eq=False)
class Seeds:
    """An immutable multiset of ``CropType`` seed counts."""

    counts: Mapping[CropType, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", _positive_counts(self.counts))

    @classmethod
    def empty(cls) -> "Seeds":
        """No seeds."""
        return cls({})

    def get(self, crop: CropType) -> int:
        """The seed count for ``crop`` (0 when absent)."""
        return self.counts.get(crop, 0)

    def contains(self, crop: CropType) -> bool:
        """Whether at least one seed of ``crop`` is available."""
        return self.get(crop) > 0

    def total(self) -> int:
        """Total number of seeds across all crops."""
        return sum(self.counts.values())

    def add(self, crop: CropType, count: int = 1) -> "Seeds":
        """Return a copy with ``count`` more seeds of ``crop``."""
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        updated = dict(self.counts)
        updated[crop] = self.get(crop) + count
        return Seeds(updated)

    def remove(self, crop: CropType, count: int = 1) -> "Seeds":
        """Return a copy with ``count`` fewer seeds of ``crop`` (clamped at 0)."""
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        updated = dict(self.counts)
        updated[crop] = self.get(crop) - count
        return Seeds(updated)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Seeds):
            return NotImplemented
        return self.counts == other.counts

    def __hash__(self) -> int:
        return hash(frozenset(self.counts.items()))

"""The shared market state (inventory + price table)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .enums import ItemType
from .inventory import Inventory


@dataclass(frozen=True, slots=True, eq=False)
class Market:
    """The shared market: product inventory and per-item sell prices.

    Holds no pricing logic: ``prices`` is an opaque lookup table supplied by
    the environment. Computing or predicting prices is the job of the future
    environment / logic layer, not the domain model.
    """

    inventory: Inventory
    prices: Mapping[ItemType, int]

    def price(self, item: ItemType) -> int:
        """The current unit sell price of ``item``.

        Raises ``KeyError`` if the environment did not supply a price for
        ``item`` (per the environment spec it supplies every product).
        """
        return self.prices[item]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Market):
            return NotImplemented
        return self.inventory == other.inventory and self.prices == other.prices

    def __hash__(self) -> int:
        return hash((self.inventory, frozenset(self.prices.items())))

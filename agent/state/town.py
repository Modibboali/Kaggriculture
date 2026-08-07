"""The shared town state (which shops have unlocked)."""

from __future__ import annotations

from dataclasses import dataclass

from .enums import ShopType


@dataclass(frozen=True, slots=True)
class Town:
    """Shared town state: the shops unlocked so far this season."""

    unlocked_shops: frozenset[ShopType]

    def has_shop(self, shop: ShopType) -> bool:
        """Whether ``shop`` has unlocked."""
        return shop in self.unlocked_shops

"""Per-player state (public farm plus private shed / seeds / inventories)."""

from __future__ import annotations

from dataclasses import dataclass

from .farm import Farm
from .inventory import Inventory, Seeds
from .worker import Worker


@dataclass(frozen=True, slots=True)
class PlayerState:
    """Everything known about one player.

    ``farm`` is the public part (money, tiles, worker positions). ``inventory``
    is the shed, ``seeds`` the unplanted seeds, and ``workers`` the full roster
    (main farmer first, then hired hands) carrying the private inventories.

    For an opponent, the private fields are empty because they are not
    observable; the search layer must handle that information asymmetry.
    """

    farm: Farm
    inventory: Inventory
    seeds: Seeds
    workers: tuple[Worker, ...]

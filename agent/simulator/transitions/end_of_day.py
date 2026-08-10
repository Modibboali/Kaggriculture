"""End-of-day processor: the deterministic day-boundary transition.

Mirrors the environment's ``_end_of_day`` exactly, in order:

1. daily plant refresh (via :class:`CropLifecycleTransition`),
2. seeded weed spawns on empty tiles,
3. drop every worker's carried inventory into the shed (capacity-limited,
   overflow discarded),
4. reset the farmer to the default spawn and clear hands / hires / carried
   inventories,
5. unlock a shop on the unlock schedule (seeded, deterministic).

The PRNG is the environment's ``random.Random((seed * 1_000_003) ^ day)`` and
is consumed in exactly the same order (weeds per player, then the shop pick),
so multi-day replays reproduce turn-for-turn.
"""

from __future__ import annotations

import random
from dataclasses import replace

from ...state import (
    EmptyTile,
    Farm,
    GameState,
    Inventory,
    PlayerState,
    Position,
    ShopType,
    Tile,
    Town,
    WEED_TILE,
    Worker,
)
from ..game_config import GameConfig
from . import with_player
from .crop_lifecycle import CropLifecycleTransition


def default_spawn(board_size: int) -> Position:
    """First free shed-access tile in the NW quadrant (the env's ``_default_spawn``)."""
    half = board_size // 2
    access = [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]
    for x, y in access:
        if y < half and x < half:
            return Position(x, y)
    return Position(0, 0)


class EndOfDayProcessor:
    """Applies the deterministic end-of-day transition for both players."""

    def __init__(self, config: GameConfig) -> None:
        self._config = config
        self._crop_lifecycle = CropLifecycleTransition(config)

    def process(self, state: GameState) -> GameState:
        """Return ``state`` after the end-of-day transition for the current day."""
        day = state.day
        rng = random.Random((self._config.seed * 1_000_003) ^ day)

        state = self._crop_lifecycle.daily_refresh(state)
        state = self._crop_lifecycle.refresh_animals(state)
        for player in (0, 1):
            state = self._spawn_weeds(state, player, rng)
            state = self._drop_inventories(state, player)
            state = self._reset_farmer(state, player)
        return self._unlock_shop(state, day, rng)

    def _spawn_weeds(self, state: GameState, player: int, rng: random.Random) -> GameState:
        farm = state.players[player].farm
        new_rows = []
        changed = False
        for row in farm.tiles:
            new_row: list[Tile] = []
            for tile in row:
                if isinstance(tile, EmptyTile) and rng.random() < self._config.weed_spawn_chance:
                    new_row.append(WEED_TILE)
                    changed = True
                else:
                    new_row.append(tile)
            new_rows.append(tuple(new_row))
        if not changed:
            return state
        new_farm = Farm(
            money=farm.money,
            tiles=tuple(new_rows),
            farmer=farm.farmer,
            workers=farm.workers,
            unlocked_quadrants=farm.unlocked_quadrants,
            hires_today=farm.hires_today,
        )
        player_state = state.players[player]
        return with_player(state, player, replace(player_state, farm=new_farm))

    def _drop_inventories(self, state: GameState, player: int) -> GameState:
        """Move every worker's carried inventory into the shed (capacity-capped)."""
        player_state = state.players[player]
        shed = dict(player_state.inventory.items)
        capacity = self._config.shed_capacity
        new_workers: list[Worker] = []
        changed = False
        for worker in player_state.workers:
            carried = dict(worker.inventory.items)
            if not carried:
                new_workers.append(worker)
                continue
            changed = True
            for item, count in list(carried.items()):
                room = max(0, capacity - sum(shed.values()))
                take = min(count, room)
                if take > 0:
                    shed[item] = shed.get(item, 0) + take
                del carried[item]
            new_workers.append(replace(worker, inventory=Inventory(carried)))
        if not changed:
            return state
        farm = player_state.farm
        new_farm = Farm(
            money=farm.money,
            tiles=farm.tiles,
            farmer=new_workers[0],
            workers=tuple(new_workers[1:]),
            unlocked_quadrants=farm.unlocked_quadrants,
            hires_today=farm.hires_today,
        )
        new_player = replace(
            player_state,
            farm=new_farm,
            inventory=Inventory(shed),
            workers=tuple(new_workers),
        )
        return with_player(state, player, new_player)

    def _reset_farmer(self, state: GameState, player: int) -> GameState:
        """Reset the farmer to the default spawn, clearing hands and hires."""
        player_state = state.players[player]
        farm = player_state.farm
        farmer = Worker(0, default_spawn(farm.board_size), Inventory.empty(), True)
        new_farm = Farm(
            money=farm.money,
            tiles=farm.tiles,
            farmer=farmer,
            workers=(),
            unlocked_quadrants=farm.unlocked_quadrants,
            hires_today=0,
        )
        new_player = replace(player_state, farm=new_farm, workers=(farmer,))
        return with_player(state, player, new_player)

    def _unlock_shop(self, state: GameState, day: int, rng: random.Random) -> GameState:
        next_day = day + 1
        interval = max(1, self._config.town_shop_unlock_interval)
        if not (next_day > 0 and next_day % interval == 0):
            return state
        town = state.town
        remaining = [shop for shop in ShopType if shop not in town.unlocked_shops]
        if not remaining:
            return state
        choice = rng.choice(sorted(remaining, key=lambda shop: shop.value))
        new_town = Town(frozenset((*town.unlocked_shops, choice)))
        return replace(state, town=new_town)

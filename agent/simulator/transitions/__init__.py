"""Shared helpers for the transition handlers.

Small, immutable state-rebuild utilities used across the specialized
transitions. All operations return new domain objects and never mutate the
inputs.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

from ...state import Farm, GameState, Inventory, Market, PlayerState, Position, Seeds, Tile, Worker


_Players = tuple[PlayerState, PlayerState]


def shed_access_tiles(board_size: int) -> tuple[Position, ...]:
    """The four inner-corner tiles around the shed, in NWSE order."""
    half = board_size // 2
    return (
        Position(half - 1, half - 1),
        Position(half, half - 1),
        Position(half - 1, half),
        Position(half, half),
    )


def is_shed_adjacent(position: Position, board_size: int) -> bool:
    """Whether ``position`` is one of the shed-access tiles."""
    return position in shed_access_tiles(board_size)


def with_player(state: GameState, player: int, player_state: PlayerState) -> GameState:
    """Return ``state`` with ``players[player]`` replaced by ``player_state``."""
    players = list(state.players)
    players[player] = player_state
    return replace(state, players=cast(_Players, tuple(players)))


def with_market(state: GameState, market: Market) -> GameState:
    """Return ``state`` with the market replaced."""
    return replace(state, market=market)


def with_tile(state: GameState, player: int, position: Position, tile: Tile) -> GameState:
    """Return ``state`` with the tile at ``position`` replaced for ``player``.

    ``Farm.replace_tile`` keeps the same worker entities, so the farm's
    ``farmer``/``workers`` stay consistent with ``PlayerState.workers``.
    """
    player_state = state.players[player]
    new_farm = player_state.farm.replace_tile(position, tile)
    return with_player(state, player, replace(player_state, farm=new_farm))


def set_worker(state: GameState, player: int, worker_id: int, worker: Worker) -> GameState:
    """Replace one worker in both the farm board and the player's roster."""
    player_state = state.players[player]
    farm = player_state.farm
    new_workers = tuple(worker if w.id == worker_id else w for w in player_state.workers)
    new_farm = Farm(
        money=farm.money,
        tiles=farm.tiles,
        farmer=new_workers[0],
        workers=new_workers[1:],
        unlocked_quadrants=farm.unlocked_quadrants,
        hires_today=farm.hires_today,
    )
    new_player = replace(player_state, farm=new_farm, workers=new_workers)
    return with_player(state, player, new_player)


def set_player_money(player_state: PlayerState, money: int) -> PlayerState:
    """Return ``player_state`` with its farm's money updated."""
    return replace(player_state, farm=replace(player_state.farm, money=money))


def set_player_shed(player_state: PlayerState, inventory: Inventory) -> PlayerState:
    """Return ``player_state`` with its shed inventory updated."""
    return replace(player_state, inventory=inventory)


def set_player_seeds(player_state: PlayerState, seeds: Seeds) -> PlayerState:
    """Return ``player_state`` with its seed counts updated."""
    return replace(player_state, seeds=seeds)


def worker_by_id(farm: Farm, worker_id: int) -> Worker | None:
    """The worker with ``worker_id`` (0 is the main farmer) or ``None``."""
    if worker_id == 0:
        return farm.farmer
    for worker in farm.workers:
        if worker.id == worker_id:
            return worker
    return None


def move_worker(state: GameState, player: int, worker_id: int, position: Position) -> GameState:
    """Move a worker, keeping ``farm.farmer``/``farm.workers`` consistent with
    ``PlayerState.workers`` (which carry the private inventories)."""
    player_state = state.players[player]
    farm = player_state.farm
    new_farm = farm.move_worker(worker_id, position)
    new_workers = tuple(
        worker.moved_to(position) if worker.id == worker_id else worker
        for worker in player_state.workers
    )
    new_player = replace(player_state, farm=new_farm, workers=new_workers)
    return with_player(state, player, new_player)

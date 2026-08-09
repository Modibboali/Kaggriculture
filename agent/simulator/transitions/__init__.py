"""Shared helpers for the transition handlers.

Small, immutable state-rebuild utilities used across the specialized
transitions. All operations return new domain objects and never mutate the
inputs.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

from ...state import Farm, GameState, Inventory, Market, PlayerState, Position, Seeds, Worker


_Players = tuple[PlayerState, PlayerState]


def with_player(state: GameState, player: int, player_state: PlayerState) -> GameState:
    """Return ``state`` with ``players[player]`` replaced by ``player_state``."""
    players = list(state.players)
    players[player] = player_state
    return replace(state, players=cast(_Players, tuple(players)))


def with_market(state: GameState, market: Market) -> GameState:
    """Return ``state`` with the market replaced."""
    return replace(state, market=market)


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

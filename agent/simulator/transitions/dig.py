"""Dig transition: DIG.

Verified against the official environment (``_apply_unit_action`` "DIG"):

* requires the acting unit to stand on a tile that is neither empty nor LOCKED;
* removes a plant (with NO yield), a weed, or an empty structure, turning the
  tile empty;
* a tile containing an animal is a no-op (animals are out of scope for now, but
  the guard mirrors the environment);
* digging an empty or LOCKED tile is a silent no-op.
"""

from __future__ import annotations

from ...actions import DigAction, TurnAction
from ...state import (
    CoopTile,
    EmptyTile,
    GameState,
    LockedTile,
    PastureTile,
    EMPTY_TILE,
)
from ..game_config import GameConfig
from . import with_tile, worker_by_id


class DigTransition:
    """Applies DIG for a player's farmer and hired hands."""

    def __init__(self, config: GameConfig) -> None:
        self._config = config

    def apply(self, state: GameState, player: int, turn_action: TurnAction) -> GameState:
        state = self._apply_unit(state, player, 0, turn_action.farmer_action)
        for index, action in enumerate(turn_action.worker_actions):
            state = self._apply_unit(state, player, index + 1, action)
        return state

    def _apply_unit(
        self, state: GameState, player: int, worker_id: int, action: object
    ) -> GameState:
        if not isinstance(action, DigAction):
            return state
        return self._dig(state, player, worker_id)

    def _dig(self, state: GameState, player: int, worker_id: int) -> GameState:
        player_state = state.players[player]
        farm = player_state.farm
        worker = worker_by_id(farm, worker_id)
        if worker is None:
            return state
        tile = farm.tile_at(worker.position)
        if tile is None:
            return state
        if isinstance(tile, EmptyTile):
            return state
        if isinstance(tile, LockedTile):
            return state
        if isinstance(tile, (CoopTile, PastureTile)) and tile.animal is not None:
            return state
        return with_tile(state, player, worker.position, EMPTY_TILE)

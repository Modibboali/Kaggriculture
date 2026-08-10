"""Structure transitions: BUILD_COOP / BUILD_PASTURE.

Verified against the official environment (``_apply_unit_action`` "BUILD_COOP"
/ "BUILD_PASTURE"):

* requires the acting unit to stand on an empty (unlocked, unoccupied) tile;
* costs nothing and consumes nothing;
* sets the tile to a COOP (goose) or PASTURE (cow / sheep) structure;
* building on any non-empty tile (plant, weed, structure, or an already built
  structure) is a silent no-op.

Structures are empty until an animal is placed on them and survive day
transitions.
"""

from __future__ import annotations

from ...actions import BuildCoopAction, BuildPastureAction, TurnAction
from ...state import CoopTile, EmptyTile, GameState, PastureTile
from ..game_config import GameConfig
from . import with_tile, worker_by_id


class StructureTransition:
    """Applies BUILD_COOP / BUILD_PASTURE for a player's farmer and hands."""

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
        if isinstance(action, BuildCoopAction):
            return self._build(state, player, worker_id, CoopTile(None))
        if isinstance(action, BuildPastureAction):
            return self._build(state, player, worker_id, PastureTile(None))
        return state

    def _build(self, state: GameState, player: int, worker_id: int, structure: object) -> GameState:
        assert isinstance(structure, (CoopTile, PastureTile))
        player_state = state.players[player]
        farm = player_state.farm
        worker = worker_by_id(farm, worker_id)
        if worker is None:
            return state
        if not isinstance(farm.tile_at(worker.position), EmptyTile):
            return state
        return with_tile(state, player, worker.position, structure)

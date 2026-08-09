"""Movement transition: NORTH / SOUTH / EAST / WEST.

Verified against the official environment: the board uses screen coordinates
(y grows south), moves off the edge are no-ops, and movement onto LOCKED tiles
is allowed. The acting worker's position is updated without mutating the input
state.
"""

from __future__ import annotations

from ...actions import MovementAction, TurnAction
from ...state import GameState, Position
from ..game_config import GameConfig
from . import move_worker, worker_by_id


class MovementTransition:
    """Applies the movement action of a player's farmer and hired hands."""

    def __init__(self, config: GameConfig) -> None:
        self._config = config

    def apply(self, state: GameState, player: int, turn_action: TurnAction) -> GameState:
        """Process movement for the farmer (id 0) then each hired hand."""
        state = self._apply_unit(state, player, 0, turn_action.farmer_action)
        for index, action in enumerate(turn_action.worker_actions):
            state = self._apply_unit(state, player, index + 1, action)
        return state

    def _apply_unit(
        self,
        state: GameState,
        player: int,
        worker_id: int,
        action: MovementAction | object,
    ) -> GameState:
        if not isinstance(action, MovementAction):
            return state
        player_state = state.players[player]
        worker = worker_by_id(player_state.farm, worker_id)
        if worker is None:
            return state
        dx, dy = action.direction.delta
        nx, ny = worker.position.x + dx, worker.position.y + dy
        size = self._config.board_size
        if not (0 <= nx < size and 0 <= ny < size):
            return state
        return move_worker(state, player, worker_id, Position(nx, ny))

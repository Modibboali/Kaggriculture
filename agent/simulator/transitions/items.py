"""PICKUP / DROP transitions: shed <-> carried inventory movement.

Verified against the official environment (``_apply_unit_action`` "PICKUP" /
"DROP"). These are supporting infrastructure for FERTILIZE (which consumes from
the carried inventory) and generally for moving goods between the shed and the
farmer's carried inventory.

* PICKUP requires standing on a shed-access tile; takes up to ``quantity`` of
  ``item`` from the shed into the carried inventory.
* DROP requires standing on a shed-access tile; moves the entire carried
  inventory into the shed (capacity-limited, overflow discarded).
"""

from __future__ import annotations

from dataclasses import replace

from ...actions import DropAction, PickupAction, TurnAction
from ...state import GameState, Inventory
from ..game_config import GameConfig
from . import is_shed_adjacent, set_worker, with_player, worker_by_id


class PickupDropTransition:
    """Applies PICKUP / DROP for a player's farmer and hired hands."""

    def __init__(self, config: GameConfig) -> None:
        self._config = config

    def apply(self, state: GameState, player: int, turn_action: TurnAction) -> GameState:
        state = self._apply_unit(state, player, 0, turn_action.farmer_action)
        for index, action in enumerate(turn_action.worker_actions):
            state = self._apply_unit(state, player, index + 1, action)
        return state

    def _apply_unit(
        self,
        state: GameState,
        player: int,
        worker_id: int,
        action: PickupAction | DropAction | object,
    ) -> GameState:
        if isinstance(action, PickupAction):
            return self._pickup(state, player, worker_id, action)
        if isinstance(action, DropAction):
            return self._drop(state, player, worker_id)
        return state

    def _pickup(
        self, state: GameState, player: int, worker_id: int, action: PickupAction
    ) -> GameState:
        player_state = state.players[player]
        farm = player_state.farm
        worker = worker_by_id(farm, worker_id)
        if worker is None:
            return state
        if not is_shed_adjacent(worker.position, farm.board_size):
            return state
        quantity = action.quantity
        if quantity <= 0:
            return state
        available = player_state.inventory.get(action.item)
        take = min(quantity, available)
        if take <= 0:
            return state
        new_shed = player_state.inventory.remove(action.item, take)
        new_worker = replace(worker, inventory=worker.inventory.add(action.item, take))
        state = set_worker(state, player, worker_id, new_worker)
        new_player = replace(state.players[player], inventory=new_shed)
        return with_player(state, player, new_player)

    def _drop(self, state: GameState, player: int, worker_id: int) -> GameState:
        player_state = state.players[player]
        farm = player_state.farm
        worker = worker_by_id(farm, worker_id)
        if worker is None:
            return state
        if not is_shed_adjacent(worker.position, farm.board_size):
            return state
        carried = dict(worker.inventory.items)
        if not carried:
            return state
        shed = dict(player_state.inventory.items)
        capacity = self._config.shed_capacity
        for item, count in list(carried.items()):
            room = max(0, capacity - sum(shed.values()))
            take = min(count, room)
            if take > 0:
                shed[item] = shed.get(item, 0) + take
            del carried[item]
        new_worker = replace(worker, inventory=Inventory(carried))
        state = set_worker(state, player, worker_id, new_worker)
        new_player = replace(state.players[player], inventory=Inventory(shed))
        return with_player(state, player, new_player)

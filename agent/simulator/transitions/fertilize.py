"""Fertilize transition: FERTILIZE.

Verified against the official environment (``_apply_unit_action`` "FERTILIZE"):

* requires the acting unit to stand on a plant AND to carry at least one
  FERTILIZER in its *carried* inventory (``private.inventories[idx]``) — the
  shed is not touched directly; fertilizer must first be picked up;
* consumes exactly one FERTILIZER from the carried inventory;
* sets ``fertilized_until_day = max(current, day + 2)`` (active for
  ``day``, ``day+1``, ``day+2``).

The yield effect is not applied here — it lives in the water bonus (non-ongoing
window watering becomes +2) and in the ongoing daily refresh (production becomes
+2 when watered), both of which read ``fertilized_until_day``.
"""

from __future__ import annotations

from dataclasses import replace

from ...actions import FertilizeAction, TurnAction
from ...state import GameState, ItemType, PlantTile
from ..game_config import GameConfig
from . import set_worker, with_tile, worker_by_id


class FertilizeTransition:
    """Applies FERTILIZE for a player's farmer and hired hands."""

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
        if not isinstance(action, FertilizeAction):
            return state
        return self._fertilize(state, player, worker_id)

    def _fertilize(self, state: GameState, player: int, worker_id: int) -> GameState:
        player_state = state.players[player]
        farm = player_state.farm
        worker = worker_by_id(farm, worker_id)
        if worker is None:
            return state
        tile = farm.tile_at(worker.position)
        if not isinstance(tile, PlantTile):
            return state
        if worker.inventory.get(ItemType.FERTILIZER) <= 0:
            return state

        worker_used = replace(
            worker, inventory=worker.inventory.remove(ItemType.FERTILIZER, 1)
        )
        state = set_worker(state, player, worker_id, worker_used)

        fertilized_until = max(tile.plant.fertilized_until_day, state.day + 2)
        new_plant = replace(tile.plant, fertilized_until_day=fertilized_until)
        return with_tile(state, player, worker.position, PlantTile(new_plant))

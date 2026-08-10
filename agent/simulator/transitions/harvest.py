"""Harvest transition: HARVEST.

Verified against the official environment (``_apply_unit_action`` "HARVEST"):

* for a PLANT: requires ``yield_units > 0`` and
  ``day - planted_day >= first_yield_day`` (immature crops are a silent no-op);
  one-time crops are removed (tile becomes empty), ongoing crops keep the tile
  with ``yield_units`` reset to 0 so they regrow.
* for an ANIMAL on a structure: requires ``yield_units > 0``; the product
  (EGG / MILK / WOOL) is harvested and ``yield_units`` resets to 0 (the animal
  stays and regrows).

The harvested units are added to the acting worker's *carried* inventory
(``private.inventories[idx]``), not the shed — they are dropped to the shed by
the end-of-day processor.

Harvesting an empty tile, a tile the worker is not standing on, or a
plant/animal with no yield is a silent no-op, exactly as in the environment.
"""

from __future__ import annotations

from dataclasses import replace

from ...actions import HarvestAction, TurnAction
from ...state import CoopTile, EmptyTile, GameState, PastureTile, PlantTile
from ..game_config import GameConfig
from . import set_worker, with_tile, worker_by_id


class HarvestTransition:
    """Applies HARVEST for a player's farmer and hired hands."""

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
        if not isinstance(action, HarvestAction):
            return state
        return self._harvest(state, player, worker_id)

    def _harvest(self, state: GameState, player: int, worker_id: int) -> GameState:
        player_state = state.players[player]
        farm = player_state.farm
        worker = worker_by_id(farm, worker_id)
        if worker is None:
            return state
        tile = farm.tile_at(worker.position)
        if isinstance(tile, PlantTile):
            return self._harvest_plant(state, player, worker_id, tile)
        if isinstance(tile, (CoopTile, PastureTile)) and tile.animal is not None:
            return self._harvest_animal(state, player, worker_id, tile)
        return state

    def _harvest_plant(
        self, state: GameState, player: int, worker_id: int, tile: PlantTile
    ) -> GameState:
        plant = tile.plant
        if plant.yield_units <= 0:
            return state
        spec = self._config.crops[plant.crop]
        if state.day - plant.planted_day < spec.first_yield_day:
            return state

        units = plant.yield_units
        player_state = state.players[player]
        worker = worker_by_id(player_state.farm, worker_id)
        assert worker is not None
        state = set_worker(
            state,
            player,
            worker_id,
            replace(worker, inventory=worker.inventory.add(plant.crop.produce, units)),
        )

        if spec.ongoing:
            new_tile: EmptyTile | PlantTile = PlantTile(replace(plant, yield_units=0))
        else:
            new_tile = EmptyTile()
        return with_tile(state, player, worker.position, new_tile)

    def _harvest_animal(
        self,
        state: GameState,
        player: int,
        worker_id: int,
        tile: CoopTile | PastureTile,
    ) -> GameState:
        assert tile.animal is not None
        animal = tile.animal
        if animal.yield_units <= 0:
            return state
        units = animal.yield_units
        player_state = state.players[player]
        worker = worker_by_id(player_state.farm, worker_id)
        assert worker is not None
        state = set_worker(
            state,
            player,
            worker_id,
            replace(worker, inventory=worker.inventory.add(animal.animal.produce, units)),
        )
        new_animal = replace(animal, yield_units=0)
        new_tile: CoopTile | PastureTile = (
            CoopTile(new_animal) if isinstance(tile, CoopTile) else PastureTile(new_animal)
        )
        return with_tile(state, player, worker.position, new_tile)

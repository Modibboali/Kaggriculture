"""Farming transitions: PLANT and WATER.

Verified against the official environment:

* PLANT: requires an empty owned tile and a seed; consumes one seed and
  creates a fresh plant (``yield_units`` and ``max_lifespan_step`` follow the
  environment's ``_new_plant`` exactly).
* WATER: marks ``watered_today``; for one-time crops inside the bonus window
  it adds the per-day yield bonus (fertilizer doubles it).

HARVEST / FERTILIZE / DIG / structures / animals are intentionally not yet
implemented (out of scope); unrecognized actions are no-ops, matching the
environment's handling of invalid actions.
"""

from __future__ import annotations

from dataclasses import replace

from ...actions import PlantAction, TurnAction, WaterAction
from ...state import (
    EmptyTile,
    GameState,
    PlantState,
    PlantTile,
)
from ..game_config import GameConfig
from . import with_player, worker_by_id


class FarmingTransition:
    """Applies PLANT / WATER for a player's farmer and hired hands."""

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
        action: PlantAction | WaterAction | object,
    ) -> GameState:
        if isinstance(action, PlantAction):
            return self._plant(state, player, worker_id, action)
        if isinstance(action, WaterAction):
            return self._water(state, player, worker_id)
        return state

    def _new_plant(self, crop: object, day: int) -> PlantState:
        from ...state import CropType

        assert isinstance(crop, CropType)
        spec = self._config.crops[crop]
        ongoing = spec.ongoing
        yield_units = 0 if ongoing else 1
        max_lifespan = (
            -1 if ongoing else (day + spec.max_yield_day + 1) * self._config.turns_per_day
        )
        return PlantState(
            crop=crop,
            planted_day=day,
            watered_today=False,
            consecutive_unwatered=1,
            yield_units=yield_units,
            fertilized_until_day=-1,
            max_lifespan_step=max_lifespan,
        )

    def _plant(
        self, state: GameState, player: int, worker_id: int, action: PlantAction
    ) -> GameState:
        player_state = state.players[player]
        farm = player_state.farm
        worker = worker_by_id(farm, worker_id)
        if worker is None:
            return state
        tile = farm.tile_at(worker.position)
        if not isinstance(tile, EmptyTile):
            return state
        if player_state.seeds.get(action.crop) <= 0:
            return state
        new_farm = farm.replace_tile(
            worker.position, PlantTile(self._new_plant(action.crop, state.day))
        )
        new_seeds = player_state.seeds.remove(action.crop)
        new_player = replace(player_state, farm=new_farm, seeds=new_seeds)
        return with_player(state, player, new_player)

    def _water(self, state: GameState, player: int, worker_id: int) -> GameState:
        player_state = state.players[player]
        farm = player_state.farm
        worker = worker_by_id(farm, worker_id)
        if worker is None:
            return state
        tile = farm.tile_at(worker.position)
        if not isinstance(tile, PlantTile):
            return state
        plant = tile.plant
        if plant.watered_today:
            return state
        new_plant = replace(plant, watered_today=True)
        spec = self._config.crops[plant.crop]
        if not spec.ongoing:
            age = state.day - plant.planted_day
            window_start = (spec.max_yield_day + 1) // 2
            if window_start <= age <= spec.max_yield_day:
                bonus = 2 if plant.fertilized_until_day >= state.day else 1
                new_plant = replace(
                    new_plant,
                    yield_units=min(spec.max_yield, new_plant.yield_units + bonus),
                )
        new_farm = farm.replace_tile(worker.position, PlantTile(new_plant))
        new_player = replace(player_state, farm=new_farm)
        return with_player(state, player, new_player)

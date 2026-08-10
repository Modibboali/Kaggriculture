"""Crop / animal lifecycle transitions: per-step decay and end-of-day refresh.

* ``decay`` mirrors the environment's ``_decay_plants`` (runs every step): a
  plant at or past its ``max_lifespan_step`` loses one ``yield_units`` every
  two steps, and turns into a weed at zero.
* ``daily_refresh`` mirrors ``_daily_refresh_plants`` (runs at the end of each
  day): resets ``watered_today``, updates ``consecutive_unwatered`` (>= 2
  kills the plant -> weed), and accumulates ``yield_units`` for ongoing crops
  on their production days, capping at ``max_yield`` and setting
  ``max_lifespan_step`` once the crop reaches maximum production.
* ``refresh_animals`` mirrors ``_daily_refresh_animals`` (end of day): resets
  ``fed_today``/``cared_today``, tracks ``consecutive_unfed`` (>= 2 the animal
  escapes and the structure is left empty), accumulates ``yield_units`` on
  production days (with the pending care bonus), and sets
  ``fertilizer_available`` every day.

All are deterministic and immutable; they only touch plant / structure tiles.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TypeGuard

from ...state import (
    CoopTile,
    Farm,
    GameState,
    PastureTile,
    PlantTile,
    Tile,
    WEED_TILE,
)
from ..game_config import GameConfig
from . import with_player


def _is_plant(tile: Tile) -> TypeGuard[PlantTile]:
    """Fast exact-type check for the closed tile union.

    ``Tile`` is an ABC, so ``isinstance(tile, PlantTile)`` pays an ABC
    ``__instancecheck__`` on every tile of every farm, every turn. The tile
    union is closed and the adapter always builds exact ``PlantTile``
    instances, so an exact type identity check is equivalent and much faster.
    """
    return type(tile) is PlantTile

class CropLifecycleTransition:
    """Per-step decay and per-day refresh of crop tiles."""

    def __init__(self, config: GameConfig) -> None:
        self._config = config

    def decay(self, state: GameState) -> GameState:
        """Apply ``_decay_plants`` to every farm at the current step.

        ``_decay_farm`` returns the unchanged ``Farm`` object when no plant
        needs decay, so the state is only rebuilt when something changed.
        """
        for player in (0, 1):
            farm = state.players[player].farm
            new_farm = self._decay_farm(farm, state.step)
            if new_farm is not farm:
                state = self._update_farm(state, player, new_farm)
        return state

    def daily_refresh(self, state: GameState) -> GameState:
        """Apply ``_daily_refresh_plants`` to every farm at the current day."""
        for player in (0, 1):
            farm = state.players[player].farm
            new_farm = self._refresh_farm(farm, state.day)
            if new_farm is not farm:
                state = self._update_farm(state, player, new_farm)
        return state

    def refresh_animals(self, state: GameState) -> GameState:
        """Apply ``_daily_refresh_animals`` to every farm at the current day."""
        for player in (0, 1):
            farm = state.players[player].farm
            new_farm = self._refresh_animals_farm(farm, state.day)
            if new_farm is not farm:
                state = self._update_farm(state, player, new_farm)
        return state

    @staticmethod
    def _update_farm(state: GameState, player: int, farm: Farm) -> GameState:
        player_state = state.players[player]
        return with_player(state, player, replace(player_state, farm=farm))

    def _refresh_animals_farm(self, farm: Farm, current_day: int) -> Farm:
        next_day = current_day + 1
        new_rows = []
        changed = False
        for row in farm.tiles:
            new_row = list(row)
            row_changed = False
            for index, tile in enumerate(row):
                if not isinstance(tile, (CoopTile, PastureTile)) or tile.animal is None:
                    continue
                animal = tile.animal
                consecutive = 0 if animal.fed_today else animal.consecutive_unfed + 1
                if consecutive >= 2:
                    # Animal escapes; the structure remains, empty.
                    new_row[index] = (
                        CoopTile(None) if isinstance(tile, CoopTile) else PastureTile(None)
                    )
                    row_changed = True
                    continue
                spec = self._config.animals[animal.animal]
                new_animal = replace(animal, consecutive_unfed=consecutive)
                days_since_first = next_day - animal.placed_day - spec.first_yield_day
                if days_since_first >= 0 and days_since_first % spec.interval == 0:
                    if new_animal.fed_today:
                        bonus = new_animal.pending_care_bonus
                        new_animal = replace(
                            new_animal,
                            yield_units=min(
                                spec.max_held, new_animal.yield_units + 1 + bonus
                            ),
                            pending_care_bonus=0,
                        )
                    else:
                        new_animal = replace(
                            new_animal,
                            yield_units=min(spec.max_held, new_animal.yield_units + 1),
                        )
                if new_animal.cared_today and new_animal.fed_today:
                    new_animal = replace(
                        new_animal, pending_care_bonus=new_animal.pending_care_bonus + 1
                    )
                new_animal = replace(
                    new_animal,
                    fertilizer_available=True,
                    fed_today=False,
                    cared_today=False,
                )
                new_row[index] = (
                    CoopTile(new_animal)
                    if isinstance(tile, CoopTile)
                    else PastureTile(new_animal)
                )
                row_changed = True
            new_rows.append(tuple(new_row) if row_changed else row)
            changed = changed or row_changed
        if not changed:
            return farm
        return Farm(
            money=farm.money,
            tiles=tuple(new_rows),
            farmer=farm.farmer,
            workers=farm.workers,
            unlocked_quadrants=farm.unlocked_quadrants,
            hires_today=farm.hires_today,
        )

    def _decay_farm(self, farm: Farm, step: int) -> Farm:
        """Apply decay to ``farm``, returning the same object when nothing changes."""
        new_rows = []
        changed = False
        for row in farm.tiles:
            if not any(_is_plant(tile) for tile in row):
                new_rows.append(row)
                continue
            new_row = list(row)
            row_changed = False
            for index, tile in enumerate(row):
                if not _is_plant(tile):
                    continue
                plant = tile.plant
                if plant.max_lifespan_step < 0:
                    continue
                mls = plant.max_lifespan_step
                if step >= mls and (step - mls) % 2 == 0:
                    new_yield = plant.yield_units - 1
                    new_row[index] = (
                        WEED_TILE
                        if new_yield <= 0
                        else PlantTile(replace(plant, yield_units=new_yield))
                    )
                    row_changed = True
            new_rows.append(tuple(new_row) if row_changed else row)
            changed = changed or row_changed
        if not changed:
            return farm
        return Farm(
            money=farm.money,
            tiles=tuple(new_rows),
            farmer=farm.farmer,
            workers=farm.workers,
            unlocked_quadrants=farm.unlocked_quadrants,
            hires_today=farm.hires_today,
        )

    def _refresh_farm(self, farm: Farm, current_day: int) -> Farm:
        next_day = current_day + 1
        turns_per_day = self._config.turns_per_day
        new_rows = []
        changed = False
        for row in farm.tiles:
            if not any(_is_plant(tile) for tile in row):
                new_rows.append(row)
                continue
            new_row = list(row)
            row_changed = False
            for index, tile in enumerate(row):
                if not _is_plant(tile):
                    continue
                plant = tile.plant
                was_watered = plant.watered_today
                consecutive = 0 if was_watered else plant.consecutive_unwatered + 1
                if consecutive >= 2:
                    new_row[index] = WEED_TILE
                    row_changed = True
                    continue
                spec = self._config.crops[plant.crop]
                new_plant = replace(
                    plant, watered_today=False, consecutive_unwatered=consecutive
                )
                if spec.ongoing:
                    days_since_first = (
                        next_day - plant.planted_day - spec.first_yield_day
                    )
                    if days_since_first >= 0 and days_since_first % spec.interval == 0:
                        production_count = days_since_first // spec.interval + 1
                        if production_count <= spec.max_yield:
                            fertilized = (
                                was_watered and plant.fertilized_until_day >= current_day
                            )
                            bonus = 2 if fertilized else 1
                            new_plant = replace(
                                new_plant,
                                yield_units=min(spec.max_yield, plant.yield_units + bonus),
                            )
                            if production_count == spec.max_yield:
                                new_plant = replace(
                                    new_plant,
                                    max_lifespan_step=(next_day + 1) * turns_per_day,
                                )
                new_row[index] = PlantTile(new_plant)
                row_changed = True
            new_rows.append(tuple(new_row) if row_changed else row)
            changed = changed or row_changed
        if not changed:
            return farm
        return Farm(
            money=farm.money,
            tiles=tuple(new_rows),
            farmer=farm.farmer,
            workers=farm.workers,
            unlocked_quadrants=farm.unlocked_quadrants,
            hires_today=farm.hires_today,
        )

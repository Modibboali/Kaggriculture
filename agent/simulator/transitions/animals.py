"""Animal transitions: PLACE / FEED / CARE / COLLECT_FERTILIZER.

Verified against the official environment (``_apply_unit_action``):

* PLACE: if the acting unit carries an animal and stands on a matching empty
  structure (COOP for GOOSE, PASTURE for COW/SHEEP), one animal is consumed
  from the carried inventory and the tile becomes a structure holding a fresh
  ``AnimalState``. Otherwise, if the unit is shed-adjacent, PLACE falls back
  to dropping up to one unit of the carried item into the shed.
* FEED: consumes 1 WHEAT from the carried inventory and sets ``fed_today``.
* CARE: sets ``cared_today`` (no resource).
* COLLECT_FERTILIZER: collects 1 FERTILIZER from a ``fertilizer_available``
  animal into the carried inventory and clears the flag.

All actions require the unit to stand on the animal tile and are silent no-ops
otherwise.
"""

from __future__ import annotations

from dataclasses import replace

from ...actions import (
    CareAction,
    CollectFertilizerAction,
    FeedAction,
    PlaceAction,
    TurnAction,
)
from ...state import (
    AnimalState,
    AnimalType,
    CoopTile,
    GameState,
    ItemType,
    PastureTile,
    StructureType,
)
from ..game_config import GameConfig
from . import is_shed_adjacent, set_worker, with_player, with_tile, worker_by_id

_StructureTile = CoopTile | PastureTile


def new_animal(animal: AnimalType, placed_day: int) -> AnimalState:
    """A freshly placed animal (the environment's ``_new_animal``)."""
    return AnimalState(
        animal=animal,
        placed_day=placed_day,
        yield_units=0,
        consecutive_unfed=0,
        fed_today=False,
        cared_today=False,
        fertilizer_available=False,
        pending_care_bonus=0,
    )


def _with_animal(tile: _StructureTile, animal: AnimalState | None) -> _StructureTile:
    """Return the same structure kind with ``animal``."""
    if isinstance(tile, CoopTile):
        return CoopTile(animal)
    return PastureTile(animal)


class AnimalTransition:
    """Applies PLACE / FEED / CARE / COLLECT_FERTILIZER for a player."""

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
        if isinstance(action, PlaceAction):
            return self._place(state, player, worker_id, action)
        if isinstance(action, FeedAction):
            return self._feed(state, player, worker_id)
        if isinstance(action, CareAction):
            return self._care(state, player, worker_id)
        if isinstance(action, CollectFertilizerAction):
            return self._collect_fertilizer(state, player, worker_id)
        return state

    def _place(
        self, state: GameState, player: int, worker_id: int, action: PlaceAction
    ) -> GameState:
        player_state = state.players[player]
        farm = player_state.farm
        worker = worker_by_id(farm, worker_id)
        if worker is None:
            return state
        tile = farm.tile_at(worker.position)
        item = action.animal.as_item

        # Animal placement on a matching empty structure.
        if isinstance(tile, (CoopTile, PastureTile)) and tile.animal is None:
            spec = self._config.animals[action.animal]
            matches = (
                (spec.structure == StructureType.COOP and isinstance(tile, CoopTile))
                or (spec.structure == StructureType.PASTURE and isinstance(tile, PastureTile))
            )
            if matches:
                if worker.inventory.get(item) <= 0:
                    return state
                state = set_worker(
                    state,
                    player,
                    worker_id,
                    replace(worker, inventory=worker.inventory.remove(item, 1)),
                )
                return with_tile(
                    state, player, worker.position, _with_animal(tile, new_animal(action.animal, state.day))
                )
            return state

        # Shed-drop fallback for PLACE (shed-adjacent only).
        if is_shed_adjacent(worker.position, farm.board_size):
            if worker.inventory.get(item) <= 0:
                return state
            shed = player_state.inventory
            room = max(0, self._config.shed_capacity - shed.total_items())
            if room <= 0:
                return state
            state = set_worker(
                state,
                player,
                worker_id,
                replace(worker, inventory=worker.inventory.remove(item, 1)),
            )
            new_player = replace(state.players[player], inventory=shed.add(item, 1))
            return with_player(state, player, new_player)
        return state

    def _feed(self, state: GameState, player: int, worker_id: int) -> GameState:
        player_state = state.players[player]
        farm = player_state.farm
        worker = worker_by_id(farm, worker_id)
        if worker is None:
            return state
        tile = farm.tile_at(worker.position)
        if not isinstance(tile, (CoopTile, PastureTile)) or tile.animal is None:
            return state
        if tile.animal.fed_today:
            return state
        if worker.inventory.get(ItemType.WHEAT) <= 0:
            return state
        state = set_worker(
            state, player, worker_id,
            replace(worker, inventory=worker.inventory.remove(ItemType.WHEAT, 1)),
        )
        return with_tile(
            state, player, worker.position,
            _with_animal(tile, replace(tile.animal, fed_today=True)),
        )

    def _care(self, state: GameState, player: int, worker_id: int) -> GameState:
        player_state = state.players[player]
        farm = player_state.farm
        worker = worker_by_id(farm, worker_id)
        if worker is None:
            return state
        tile = farm.tile_at(worker.position)
        if not isinstance(tile, (CoopTile, PastureTile)) or tile.animal is None:
            return state
        if tile.animal.cared_today:
            return state
        return with_tile(
            state, player, worker.position,
            _with_animal(tile, replace(tile.animal, cared_today=True)),
        )

    def _collect_fertilizer(self, state: GameState, player: int, worker_id: int) -> GameState:
        player_state = state.players[player]
        farm = player_state.farm
        worker = worker_by_id(farm, worker_id)
        if worker is None:
            return state
        tile = farm.tile_at(worker.position)
        if not isinstance(tile, (CoopTile, PastureTile)) or tile.animal is None:
            return state
        if not tile.animal.fertilizer_available:
            return state
        state = set_worker(
            state, player, worker_id,
            replace(worker, inventory=worker.inventory.add(ItemType.FERTILIZER, 1)),
        )
        return with_tile(
            state, player, worker.position,
            _with_animal(tile, replace(tile.animal, fertilizer_available=False)),
        )
